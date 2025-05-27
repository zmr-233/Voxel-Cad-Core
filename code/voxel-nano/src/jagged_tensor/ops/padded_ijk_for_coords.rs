// src/jagged_tensor/ops/padded_ijk_for_coords.rs
//! GPU operator: 填充包围盒内 IJK 坐标
//!
//! 将输入 JaggedTensorCore 中的非规则点 (i,j,k) 按 bmin/bmax 范围膨胀为完整网格，输出新的 JaggedTensorCore。

use bytemuck::{Pod, Zeroable};
use wgpu::{ShaderStages, util::DeviceExt};

use super::JaggedElement;
use super::elem;
use crate::{error::ComputeError, jagged_tensor::core::JaggedTensorCore};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BBoxParams {
    pub bmin: glam::IVec3, // 包围盒最小点
    _padding0: u32,
    pub bmax: glam::IVec3, // 包围盒最大点
    _padding1: u32,
    pub total_pad: u32,       // 总填充量
    pub num_elems: u32,       // 输入数据的元素数量
    pub num_outer_lists: u32, // 外层列表数量
    _padding2: u32,
}

impl BBoxParams {
    pub fn new(
        bmin: glam::IVec3,
        bmax: glam::IVec3,
        total_pad: u32,
        num_elems: u32,
        num_outer_lists: u32,
    ) -> Self {
        Self {
            bmin,
            _padding0: 0,
            bmax,
            _padding1: 0,
            total_pad,
            num_elems,
            num_outer_lists,
            _padding2: 0,
        }
    }
    pub fn min_binding_size() -> wgpu::BufferSize {
        wgpu::BufferSize::new(std::mem::size_of::<Self>() as u64).unwrap()
    }
}

/// 专用 Operator 结构体，仅支持元素类型为 [i32;3]
#[derive(Clone)]
pub struct PaddedIJKForCoords {
    pipeline_a: wgpu::ComputePipeline,
    pipeline_b: wgpu::ComputePipeline,
    bind_group_layout_a: wgpu::BindGroupLayout,
    bind_group_layout_b: wgpu::BindGroupLayout,
}

impl PaddedIJKForCoords {
    pub fn new(device: &wgpu::Device) -> Result<Self, ComputeError> {
        let shader_a = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("padded_pass_a.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("padded_pass_a.wgsl").into()),
        });
        let shader_b = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("padded_pass_b.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("padded_pass_b.wgsl").into()),
        });
        let desc = |i, read_only, min_size| wgpu::BindGroupLayoutEntry {
            binding: i,
            visibility: ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: read_only,
                },
                has_dynamic_offset: false,
                min_binding_size: Some(min_size),
            },
            count: None,
        };
        // 布局：输入 data + indices(offsets,batch_idx,list_idx)、输出 IJK、输出 batch_idx、统一常量
        let bind_group_layout_a =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("dispatch_padded_ijk_layout"),
                entries: &[
                    // binding 0: JaggedTensorCore::data
                    desc(0, true, glam::IVec3::MIN_BINDING_SIZE),
                    // binding 1: JaggedIndices::batch_idx
                    desc(1, true, <u32 as JaggedElement>::MIN_BINDING_SIZE),
                    // binding 2: JaggedIndices::offsets
                    // desc(2, true, glam::IVec2::MIN_BINDING_SIZE),
                    // binging 3: JaggedIndices::list_idx
                    desc(3, true, glam::UVec3::MIN_BINDING_SIZE),
                    // Output:
                    // binding 4: JaggedTensorCore::data
                    desc(4, false, glam::IVec3::MIN_BINDING_SIZE),
                    // binding 5: JaggedIndices::batch_idx
                    desc(5, false, <u32 as JaggedElement>::MIN_BINDING_SIZE),
                    // binding 6: JaggedIndices::offsets
                    // desc(6, false, glam::IVec2::MIN_BINDING_SIZE),
                    // binging 7: JaggedIndices::list_idx
                    desc(7, false, glam::UVec3::MIN_BINDING_SIZE),
                    // binding 8: 常量统一体
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: Some(BBoxParams::min_binding_size()),
                        },
                        count: None,
                    },
                    // ⚠️注意: JaggedIndices::offsets & list_idx 需要额外的 scale_each 便利方法
                ],
            });
        let bind_group_layout_b =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("dispatch_padded_ijk_layout"),
                entries: &[
                    // binding 2: JaggedIndices::offsets
                    desc(2, true, glam::IVec2::MIN_BINDING_SIZE), // vec2<i32> in WGSL = 8 bytes, but needs 16-byte alignment
                    // binding 6: JaggedIndices::offsets
                    desc(6, false, glam::IVec2::MIN_BINDING_SIZE), // vec2<i32> in WGSL = 8 bytes, but needs 16-byte alignment
                    // binding 8: 常量统一体
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: Some(BBoxParams::min_binding_size()),
                        },
                        count: None,
                    },
                ],
            });
        let pipeline_layout_a = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("padded_a_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout_a],
            push_constant_ranges: &[],
        });
        let pipeline_layout_b = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("padded_b_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout_b],
            push_constant_ranges: &[],
        });
        let pipeline_a = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("padded_a_pipeline"),
            layout: Some(&pipeline_layout_a),
            module: &shader_a,
            entry_point: Some("cs_main"), // WGSL入口函数
            compilation_options: Default::default(),
            cache: None,
        });
        let pipeline_b = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("padded_b_pipeline"),
            layout: Some(&pipeline_layout_b),
            module: &shader_b,
            entry_point: Some("cs_main"), // WGSL入口函数
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self {
            pipeline_a,
            pipeline_b,
            bind_group_layout_a,
            bind_group_layout_b,
        })
    }

    pub fn compute(
        &self,
        core: &JaggedTensorCore,
        bmin: glam::IVec3,
        bmax: glam::IVec3,
    ) -> Result<JaggedTensorCore, ComputeError> {
        // 类型检查：仅支持 [i32;3]
        if core.metadata.elem_dimensions != <glam::IVec3 as JaggedElement>::DIMENSIONS
            || core.metadata.elem_stride_size as usize
                != <glam::IVec3 as JaggedElement>::STRIDE_SIZE
        {
            return Err(ComputeError::TypeMismatch(
                "DispatchPaddedIJKForCoords only supports [i32;3] elements".to_string(),
            ));
        }

        let device = &core.device;
        let num_elems = core.metadata.num_elements as u32;
        let dims = bmax - bmin + glam::IVec3::ONE;
        let total_pad = (dims.x * dims.y * dims.z) as u32;
        let total_threads = num_elems * total_pad;
        // 创建输出缓冲区 - 确保最小16字节大小，并用零初始化
        // ⚠️ 不能用16.max((total_threads * glam::IVec3::SIZE as u32)
        // 16.max只是确保整体16偏移，但是内部取步长仍然是 16字节
        let out_ijk_size = elem::padded_size::<glam::UVec3>(total_threads);
        let out_ijk = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("out_ijk_buffer"),
            contents: &vec![0u8; out_ijk_size as usize],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let out_bidx_size = elem::padded_size::<u32>(total_threads);
        let out_bidx = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("out_batch_idx_buffer"),
            contents: &vec![0u8; out_bidx_size as usize],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let out_offsets_size = elem::padded_size::<glam::UVec2>(total_threads);
        let out_offsets = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("out_offsets_buffer"),
            contents: &vec![0u8; out_offsets_size as usize],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let out_list_idx_size = elem::padded_size::<glam::UVec3>(total_threads);
        let out_list_idx = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("out_list_idx_buffer"),
            contents: &vec![0u8; out_list_idx_size as usize],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let bbox = BBoxParams::new(
            bmin,
            bmax,
            total_pad,
            num_elems,
            core.metadata.num_outer_lists as u32,
        );
        let bbox_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bbox_params_buffer"),
            contents: bytemuck::bytes_of(&bbox),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // 创建绑定组
        let bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dispatch_padded_ijk_bind_group"),
            layout: &self.bind_group_layout_a,
            entries: &[
                // 输入数据
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: core.data_buffer().as_entire_binding(),
                },
                // 输入 batch_idx
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: core.batch_idx_buffer().as_entire_binding(),
                },
                // 输入 offsets
                // wgpu::BindGroupEntry {
                //     binding: 2,
                //     resource: core.offsets_buffer().as_entire_binding(),
                // },
                // 输入 list_idx
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: core.list_idx_buffer().as_entire_binding(),
                },
                // 输出 out_ijk
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out_ijk.as_entire_binding(),
                },
                // 输出 out_bidx
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: out_bidx.as_entire_binding(),
                },
                // 输出 offsets
                // wgpu::BindGroupEntry {
                //     binding: 6,
                //     resource: out_offsets.as_entire_binding(),
                // },
                // 输出 list_idx
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: out_list_idx.as_entire_binding(),
                },
                // BBox 参数
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: bbox_buffer.as_entire_binding(),
                },
            ],
        });
        // 创建绑定组
        let bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dispatch_padded_ijk_bind_group"),
            layout: &self.bind_group_layout_b,
            entries: &[
                // 输入 offsets
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: core.offsets_buffer().as_entire_binding(),
                },
                // 输出 offsets
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: out_offsets.as_entire_binding(),
                },
                // BBox 参数
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: bbox_buffer.as_entire_binding(),
                },
            ],
        });

        // 计算线程组数量
        let threads_per_group: u32 = 256; // 每个工作组处理 256 个元素
        let num_groups_a = (total_threads + threads_per_group - 1) / threads_per_group;
        let num_groups_b = (bbox.num_outer_lists + threads_per_group - 1) / threads_per_group;

        // 创建命令编码器
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("padded_ijk_command_encoder"),
        });

        // 执行计算
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("padded_ijk_compute_pass"),
                timestamp_writes: None,
            });

            // 第一个 pass (padded_pass_a)
            compute_pass.set_pipeline(&self.pipeline_a);
            compute_pass.set_bind_group(0, &bind_group_a, &[]);
            compute_pass.dispatch_workgroups(num_groups_a, 1, 1);

            // 第二个 pass (padded_pass_b)
            compute_pass.set_pipeline(&self.pipeline_b);
            compute_pass.set_bind_group(0, &bind_group_b, &[]); // Bind group B uses different bindings
            compute_pass.dispatch_workgroups(num_groups_b, 1, 1);
        }

        // 提交命令并直接使用输出缓冲区构造新的 JaggedTensorCore
        core.queue.submit(std::iter::once(encoder.finish()));
        // Create a new JaggedTensorCore with updated buffers and refreshed cache
        Ok(core.with_buffers(
            out_ijk,
            out_bidx,
            out_offsets,
            out_list_idx,
            total_threads as usize,
        ))
    }
}
