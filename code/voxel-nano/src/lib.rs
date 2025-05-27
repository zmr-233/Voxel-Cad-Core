use bytemuck::{Pod, Zeroable};
mod error;
pub mod jagged_tensor;
pub use error::{ComputeError, TypeError};

// ===============================================================================
// 基础类型定义
// ===============================================================================

/// 体素数据结构
/// 对应原始 C++ 中的颜色/体素属性存储
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct VoxelData {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}
