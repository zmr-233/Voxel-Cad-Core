// ===============================================================================
// CPU 侧方法实现
// ===============================================================================

use super::core::{JaggedIndices, JaggedMetadata, JaggedShapeCache};
// use super::elem;
use super::ops::JaggedOps;
use crate::error::ComputeError;
use crate::jagged_tensor::core::JaggedTensorCore;
use crate::jagged_tensor::{JaggedElement, JaggedTensor};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Builder for constructing JaggedTensor from CPU data using builder pattern
/// Only supports elements that implement JaggedElement trait
pub struct JaggedTensorBuilder<T: JaggedElement> {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    ldim: u8,
    nested: Vec<Vec<Vec<T>>>,
}

impl<T: JaggedElement> JaggedTensorBuilder<T> {
    /// Create a new builder with WGPU device and queue
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self {
            device,
            queue,
            ldim: 0,
            nested: Vec::new(),
        }
    }

    pub fn with_ldim_1(mut self, nested: Vec<T>) -> Self {
        self.nested = vec![vec![nested]];
        self.ldim = 1;
        self
    }

    pub fn with_ldim_2(mut self, nested: Vec<Vec<T>>) -> Self {
        self.nested = vec![nested];
        self.ldim = 2;
        self
    }

    pub fn with_ldim_3(mut self, nested: Vec<Vec<Vec<T>>>) -> Self {
        self.nested = nested;
        self.ldim = 3;
        self
    }

    /// Build the JaggedTensor<T> by uploading data to GPU
    pub fn build(self) -> Result<JaggedTensor<T>, ComputeError> {
        // Check if ldim != 0
        if self.ldim == 0 {
            return Err(ComputeError::TypeMismatch(
                "ldim must be set to 1, 2, or 3".to_string(),
            ));
        }
        let num_outer_lists = self.nested.len();
        let mut flat_data: Vec<T::Padded> = Vec::new();
        let mut batch_idx = Vec::new();
        let mut offsets = Vec::with_capacity(num_outer_lists + 1);
        let mut list_idx = Vec::new();

        let mut cur_offset: u32 = 0;

        for (bat_idx, batch) in self.nested.iter().enumerate() {
            offsets.push(glam::UVec2 {
                x: cur_offset,
                y: cur_offset + batch.len() as u32,
            });
            cur_offset += batch.len() as u32;
            for (time_idx, time) in batch.iter().enumerate() {
                for (pat_idx, patch) in time.iter().enumerate() {
                    flat_data.push(T::pad(*patch)); // 填充到对齐数据类型
                    batch_idx.push(bat_idx as i32);
                    list_idx.push(glam::UVec4 {
                        x: bat_idx as u32,
                        y: time_idx as u32,
                        z: pat_idx as u32,
                        w: 0,
                    });
                }
            }
        }

        let num_elements = flat_data.len() as u32;

        // Size
        // let data_size = elem::padded_size::<T>(num_elements);
        // let batch_idx_size = elem::padded_size::<u32>(num_elements);
        // let offsets_size = elem::padded_size::<glam::UVec2>(num_outer_lists as u32 + 1);
        // let list_idx_size = elem::padded_size::<glam::UVec3>(num_elements);

        // Create GPU buffers
        let data_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("jagged_data"),
                contents: bytemuck::cast_slice(&flat_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let batch_idx_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("batch_idx"),
                contents: bytemuck::cast_slice(&batch_idx),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let offsets_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("offsets"),
                contents: bytemuck::cast_slice(&offsets),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let list_idx_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("list_idx"),
                contents: bytemuck::cast_slice(&list_idx),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        // Build core and tensor
        let indx = JaggedIndices {
            batch_idx: batch_idx_buffer,
            offsets: offsets_buffer,
            list_idx: list_idx_buffer,
        };
        let meta = JaggedMetadata {
            num_outer_lists: num_outer_lists,
            ldim: self.ldim,
            num_elements: num_elements as usize,
            elem_stride_size: T::STRIDE_SIZE as u8,
            elem_dimensions: T::DIMENSIONS,
        };
        let cache = JaggedShapeCache {
            lshape1: None,
            lshape2: None,
            lshape3: None,
            is_dirty: true,
        };

        let ops = JaggedOps::new(&self.device)?;
        let core = JaggedTensorCore {
            data: data_buffer,
            indices: indx,
            metadata: meta,
            shape_cache: cache,
            device: self.device.clone(),
            queue: self.queue.clone(),
            ops,
        };

        let tensor =
            JaggedTensor::from_core(core).map_err(|e| ComputeError::TypeMismatch(e.to_string()))?;

        Ok(tensor)
    }
}

// ===============================================================================
// GPU 侧方法实现
// ===============================================================================
