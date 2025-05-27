use glam::IVec3;
use std::sync::Arc;
use voxel_nano::jagged_tensor::JaggedTensorBuilder;

/// Initialize WGPU device and queue for testing
async fn init_wgpu() -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("Failed to find an appropriate adapter");

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            label: None,
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("Failed to create device");

    (Arc::new(device), Arc::new(queue))
}

/// Read buffer data from GPU to CPU for verification
async fn read_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    size: usize,
) -> Vec<T> {
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_buffer"),
        size: (size * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("copy_encoder"),
    });

    encoder.copy_buffer_to_buffer(
        buffer,
        0,
        &staging_buffer,
        0,
        (size * std::mem::size_of::<T>()) as u64,
    );

    queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let _ = buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::MaintainBase::Wait);

    let data = buffer_slice.get_mapped_range();
    let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    result
}

#[tokio::test]
async fn test_jagged_tensor_builder_basic() {
    let (device, queue) = init_wgpu().await;

    // Test basic construction with IVec3 data
    let input_data = vec![
        vec![IVec3::new(0, 0, 0), IVec3::new(1, 1, 1)], // batch 0: 2 elements
        vec![IVec3::new(2, 2, 2)],                      // batch 1: 1 element
        vec![
            IVec3::new(3, 3, 3),
            IVec3::new(4, 4, 4),
            IVec3::new(5, 5, 5),
        ], // batch 2: 3 elements
    ];

    let tensor = JaggedTensorBuilder::new(device.clone(), queue.clone())
        .with_ldim_2(input_data.clone())
        .build()
        .expect("Failed to build tensor");

    // Verify metadata
    assert_eq!(tensor.core().num_outer_lists(), 3);
    assert_eq!(tensor.core().num_elements(), 6);
    assert_eq!(tensor.core().ldim(), 1);

    // Read back and verify data
    let data_result = read_buffer::<IVec3>(&device, &queue, tensor.core().data_buffer(), 6).await;
    let expected_flat = vec![
        IVec3::new(0, 0, 0),
        IVec3::new(1, 1, 1), // batch 0
        IVec3::new(2, 2, 2), // batch 1
        IVec3::new(3, 3, 3),
        IVec3::new(4, 4, 4),
        IVec3::new(5, 5, 5), // batch 2
    ];
    assert_eq!(data_result, expected_flat);

    let batch_idx_result =
        read_buffer::<i32>(&device, &queue, tensor.core().batch_idx_buffer(), 6).await;
    let expected_batch_idx = vec![0, 0, 1, 2, 2, 2];
    assert_eq!(batch_idx_result, expected_batch_idx);
}

#[tokio::test]
async fn test_debug_padded_ijk() {
    let (device, queue) = init_wgpu().await;

    // Create simple test input - same as the basic test
    let input_data = vec![
        vec![IVec3::new(0, 0, 0)], // batch 0: origin point
        vec![IVec3::new(2, 2, 2)], // batch 1: corner point
    ];

    let tensor = JaggedTensorBuilder::new(device.clone(), queue.clone())
        .with_ldim_2(input_data)
        .build()
        .expect("Failed to build tensor");

    // Define bounding box: expand each point by 1 in all directions
    let bmin = IVec3::new(-1, -1, -1);
    let bmax = IVec3::new(1, 1, 1);

    println!(
        "Input coordinates: [{:?}, {:?}]",
        IVec3::new(0, 0, 0),
        IVec3::new(2, 2, 2)
    );
    println!("BBox: bmin={:?}, bmax={:?}", bmin, bmax);
    println!("Expected total padding per point: 3x3x3 = 27");
    println!("Expected total threads: 2 * 27 = 54");

    let result = tensor
        .core()
        .ops
        .padded_ijk_for_coords
        .compute(tensor.core(), bmin, bmax)
        .expect("Failed to compute padded IJK");

    println!("Output buffer size: {}", result.num_elements());

    // Read back ALL the data to see if there are any issues
    let all_output_coords =
        read_buffer::<IVec3>(&device, &queue, result.data_buffer(), result.num_elements()).await;
    let all_output_batch_idx = read_buffer::<i32>(
        &device,
        &queue,
        result.batch_idx_buffer(),
        result.num_elements(),
    )
    .await;

    // 读取list_idx缓冲区以获取调试信息（前10个包含offset值）
    let all_output_list_idx = read_buffer::<IVec3>(
        &device,
        &queue,
        result.list_idx_buffer(),
        result.num_elements(),
    )
    .await;

    // Print debug info for first 20 indices
    println!("GPU Debug: First 20 calculations:");
    for i in 0..20.min(result.num_elements()) {
        if i < 5 {
            println!(
                "  idx={}: [eidx, pad_idx, total_pad]={:?} -> coord={:?}",
                i, all_output_list_idx[i], all_output_coords[i]
            );
        } else if i < 10 {
            println!(
                "  idx={}: [x_off, y_off, z_off]={:?} -> coord={:?}",
                i, all_output_list_idx[i], all_output_coords[i]
            );
        } else if i < 15 {
            println!(
                "  idx={}: [dims_z, dims_y, dims_yz]={:?} -> coord={:?}",
                i, all_output_list_idx[i], all_output_coords[i]
            );
        } else if i < 20 {
            println!(
                "  idx={}: [dims.x, dims.y, dims.z]={:?} -> coord={:?}",
                i, all_output_list_idx[i], all_output_coords[i]
            );
        } else {
            println!(
                "  idx={}: coord={:?}, batch={}, debug={:?}",
                i, all_output_coords[i], all_output_batch_idx[i], all_output_list_idx[i]
            );
        }
    }

    // Print ALL coordinates for analysis
    println!("ALL output coordinates and batch indices:");
    for i in 0..result.num_elements() {
        println!(
            "  idx={}: coord={:?}, batch={}",
            i, all_output_coords[i], all_output_batch_idx[i]
        );
    }

    // Read back output data
    let output_coords = all_output_coords.clone();
    let output_batch_idx = all_output_batch_idx.clone();

    println!(
        "First 10 output coordinates: {:?}",
        &output_coords[..10.min(output_coords.len())]
    );
    println!(
        "First 10 batch indices: {:?}",
        &output_batch_idx[..10.min(output_batch_idx.len())]
    );

    // Separate by batch
    let mut batch_0_coords = Vec::new();
    let mut batch_1_coords = Vec::new();

    for (i, &batch) in output_batch_idx.iter().enumerate() {
        if batch == 0 {
            batch_0_coords.push(output_coords[i]);
        } else if batch == 1 {
            batch_1_coords.push(output_coords[i]);
        }
    }

    println!(
        "Batch 0 coordinates ({} total): {:?}",
        batch_0_coords.len(),
        &batch_0_coords[..10.min(batch_0_coords.len())]
    );
    println!(
        "Batch 1 coordinates ({} total): {:?}",
        batch_1_coords.len(),
        &batch_1_coords[..10.min(batch_1_coords.len())]
    );

    // Check if expected coordinates are present
    println!(
        "Batch 0 contains (-1,-1,-1): {}",
        batch_0_coords.contains(&IVec3::new(-1, -1, -1))
    );
    println!(
        "Batch 0 contains (0,0,0): {}",
        batch_0_coords.contains(&IVec3::new(0, 0, 0))
    );
    println!(
        "Batch 0 contains (1,1,1): {}",
        batch_0_coords.contains(&IVec3::new(1, 1, 1))
    );

    // Check if batch 0 has all expected coordinates for (0,0,0) + bmin=(-1,-1,-1) to bmax=(1,1,1)
    let mut expected_batch_0 = Vec::new();
    for x in -1..=1 {
        for y in -1..=1 {
            for z in -1..=1 {
                expected_batch_0.push(IVec3::new(x, y, z));
            }
        }
    }
    println!("Expected batch 0 (27 coords): {:?}", expected_batch_0);
    println!(
        "Missing from batch 0: {:?}",
        expected_batch_0
            .iter()
            .filter(|c| !batch_0_coords.contains(c))
            .collect::<Vec<_>>()
    );
    println!(
        "Extra in batch 0: {:?}",
        batch_0_coords
            .iter()
            .filter(|c| !expected_batch_0.contains(c))
            .collect::<Vec<_>>()
    );

    // Check batch 1 (around 2,2,2)
    let mut expected_batch_1 = Vec::new();
    for x in 1..=3 {
        // 2 + (-1 to 1)
        for y in 1..=3 {
            for z in 1..=3 {
                expected_batch_1.push(IVec3::new(x, y, z));
            }
        }
    }
    println!("Expected batch 1 (27 coords): {:?}", expected_batch_1);
    println!(
        "Missing from batch 1: {:?}",
        expected_batch_1
            .iter()
            .filter(|c| !batch_1_coords.contains(c))
            .collect::<Vec<_>>()
    );
    println!(
        "Extra in batch 1: {:?}",
        batch_1_coords
            .iter()
            .filter(|c| !expected_batch_1.contains(c))
            .collect::<Vec<_>>()
    );

    // Just succeed to not interfere with actual tests
    assert!(true);
}

#[tokio::test]
async fn test_padded_ijk_for_coords_basic() {
    let (device, queue) = init_wgpu().await;

    // Simple test case: 2 points, expand by 1 in each direction
    let input_data = vec![
        vec![IVec3::new(0, 0, 0)], // batch 0: origin point
        vec![IVec3::new(2, 2, 2)], // batch 1: corner point
    ];

    let tensor = JaggedTensorBuilder::new(device.clone(), queue.clone())
        .with_ldim_2(input_data)
        .build()
        .expect("Failed to build tensor");

    // Define bounding box: expand each point by 1 in all directions
    let bmin = IVec3::new(-1, -1, -1);
    let bmax = IVec3::new(1, 1, 1);

    // Expected total padding: 3x3x3 = 27 points per input point
    let _expected_total_pad = 27;

    let result = tensor
        .core()
        .ops
        .padded_ijk_for_coords
        .compute(tensor.core(), bmin, bmax)
        .expect("Failed to compute padded IJK");

    // Verify output size: 2 input points * 27 padding = 54 total output points
    assert_eq!(result.num_elements(), 54);

    // Read back output data
    let output_coords = read_buffer::<IVec3>(&device, &queue, result.data_buffer(), 54).await;
    let output_batch_idx = read_buffer::<i32>(&device, &queue, result.batch_idx_buffer(), 54).await;
    let debug_list_idx = read_buffer::<IVec3>(&device, &queue, result.list_idx_buffer(), 54).await;

    // Debug: Print first 10 raw GPU outputs
    println!("Raw GPU output (first 10):");
    for i in 0..10 {
        println!(
            "  idx={}: coord={:?}, batch={}, debug={:?}",
            i, output_coords[i], output_batch_idx[i], debug_list_idx[i]
        );
    }

    // Verify first few output coordinates for batch 0 (around origin)
    let mut batch_0_coords = Vec::new();
    let mut batch_1_coords = Vec::new();

    for (i, &batch) in output_batch_idx.iter().enumerate() {
        if batch == 0 {
            batch_0_coords.push(output_coords[i]);
        } else if batch == 1 {
            batch_1_coords.push(output_coords[i]);
        }
    }

    // Batch 0 should have 27 coordinates around (0,0,0)
    println!("Batch 0 coords (expected 27): {:?}", batch_0_coords);
    println!("Batch 1 coords (expected 27): {:?}", batch_1_coords);

    assert_eq!(batch_0_coords.len(), 27);
    assert!(batch_0_coords.contains(&IVec3::new(-1, -1, -1))); // bmin corner
    assert!(batch_0_coords.contains(&IVec3::new(0, 0, 0))); // original point
    assert!(batch_0_coords.contains(&IVec3::new(1, 1, 1))); // bmax corner

    // Batch 1 should have 27 coordinates around (2,2,2)
    assert_eq!(batch_1_coords.len(), 27);
    assert!(batch_1_coords.contains(&IVec3::new(1, 1, 1))); // 2+bmin
    assert!(batch_1_coords.contains(&IVec3::new(2, 2, 2))); // original point
    assert!(batch_1_coords.contains(&IVec3::new(3, 3, 3))); // 2+bmax
}

#[tokio::test]
async fn test_padded_ijk_for_coords_asymmetric_bbox() {
    let (device, queue) = init_wgpu().await;

    // Test with asymmetric bounding box
    let input_data = vec![
        vec![IVec3::new(5, 5, 5)], // single point
    ];

    let tensor = JaggedTensorBuilder::new(device.clone(), queue.clone())
        .with_ldim_2(input_data)
        .build()
        .expect("Failed to build tensor");

    // Asymmetric bounding box: 2x3x1 = 6 total padding
    let bmin = IVec3::new(-1, -2, 0);
    let bmax = IVec3::new(0, 0, 0);

    let result = tensor
        .core()
        .ops
        .padded_ijk_for_coords
        .compute(tensor.core(), bmin, bmax)
        .expect("Failed to compute padded IJK");

    // Verify output size: 1 input point * 6 padding = 6 total output points
    assert_eq!(result.num_elements(), 6);

    let output_coords = read_buffer::<IVec3>(&device, &queue, result.data_buffer(), 6).await;

    // Expected coordinates: (5,5,5) + all combinations of bmin..bmax offsets
    let expected_coords = vec![
        IVec3::new(4, 3, 5), // 5+(-1), 5+(-2), 5+0
        IVec3::new(4, 4, 5), // 5+(-1), 5+(-1), 5+0
        IVec3::new(4, 5, 5), // 5+(-1), 5+(0), 5+0
        IVec3::new(5, 3, 5), // 5+(0), 5+(-2), 5+0
        IVec3::new(5, 4, 5), // 5+(0), 5+(-1), 5+0
        IVec3::new(5, 5, 5), // 5+(0), 5+(0), 5+0
    ];

    // Sort both arrays for comparison since order might vary
    let mut output_sorted = output_coords;
    let mut expected_sorted = expected_coords;
    output_sorted.sort_by_key(|v| (v.x, v.y, v.z));
    expected_sorted.sort_by_key(|v| (v.x, v.y, v.z));

    assert_eq!(output_sorted, expected_sorted);
}

#[tokio::test]
async fn test_padded_ijk_for_coords_multiple_batches() {
    let (device, queue) = init_wgpu().await;

    // Test with multiple batches and multiple points per batch
    let input_data = vec![
        vec![IVec3::new(0, 0, 0), IVec3::new(10, 10, 10)], // batch 0: 2 points
        vec![IVec3::new(-5, -5, -5)],                      // batch 1: 1 point
        vec![
            IVec3::new(1, 2, 3),
            IVec3::new(7, 8, 9),
            IVec3::new(15, 16, 17),
        ], // batch 2: 3 points
    ];

    let tensor = JaggedTensorBuilder::new(device.clone(), queue.clone())
        .with_ldim_2(input_data)
        .build()
        .expect("Failed to build tensor");

    // Small bounding box: 2x2x2 = 8 padding per point
    let bmin = IVec3::new(0, 0, 0);
    let bmax = IVec3::new(1, 1, 1);

    let result = tensor
        .core()
        .ops
        .padded_ijk_for_coords
        .compute(tensor.core(), bmin, bmax)
        .expect("Failed to compute padded IJK");

    // Verify output size: 6 input points * 8 padding = 48 total output points
    assert_eq!(result.num_elements(), 48);

    let output_batch_idx = read_buffer::<i32>(&device, &queue, result.batch_idx_buffer(), 48).await;

    // Count points per batch
    let mut batch_counts = vec![0; 3];
    for &batch in &output_batch_idx {
        batch_counts[batch as usize] += 1;
    }

    // Batch 0: 2 input points * 8 padding = 16 output points
    // Batch 1: 1 input point * 8 padding = 8 output points
    // Batch 2: 3 input points * 8 padding = 24 output points
    assert_eq!(batch_counts, vec![16, 8, 24]);
}

#[tokio::test]
async fn test_padded_ijk_for_coords_edge_cases() {
    let (device, queue) = init_wgpu().await;

    // Test edge case: single point with minimal bounding box (1x1x1)
    let input_data = vec![vec![IVec3::new(100, 200, 300)]];

    let tensor = JaggedTensorBuilder::new(device.clone(), queue.clone())
        .with_ldim_2(input_data)
        .build()
        .expect("Failed to build tensor");

    // Minimal bounding box: just the point itself
    let bmin = IVec3::new(0, 0, 0);
    let bmax = IVec3::new(0, 0, 0);

    let result = tensor
        .core()
        .ops
        .padded_ijk_for_coords
        .compute(tensor.core(), bmin, bmax)
        .expect("Failed to compute padded IJK");

    // Should have exactly 1 output point (no expansion)
    assert_eq!(result.num_elements(), 1);

    let output_coords = read_buffer::<IVec3>(&device, &queue, result.data_buffer(), 1).await;
    assert_eq!(output_coords[0], IVec3::new(100, 200, 300));
}

#[tokio::test]
async fn test_builder_with_different_element_types() {
    let (device, queue) = init_wgpu().await;

    // Test with i32 elements
    let i32_data = vec![vec![42, 100], vec![200]];

    let i32_tensor = JaggedTensorBuilder::new(device.clone(), queue.clone())
        .with_ldim_2(i32_data)
        .build()
        .expect("Failed to build i32 tensor");

    assert_eq!(i32_tensor.core().num_elements(), 3);
    assert_eq!(i32_tensor.core().metadata.elem_dimensions, 1);

    // Test with f32 elements
    let f32_data = vec![vec![1.5f32, 2.7f32, 3.14f32]];

    let f32_tensor = JaggedTensorBuilder::new(device.clone(), queue.clone())
        .with_ldim_2(f32_data)
        .build()
        .expect("Failed to build f32 tensor");

    assert_eq!(f32_tensor.core().num_elements(), 3);
    assert_eq!(f32_tensor.core().metadata.elem_dimensions, 1);
}
