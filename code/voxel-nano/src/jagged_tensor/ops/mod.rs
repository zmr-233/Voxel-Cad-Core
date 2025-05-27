use super::JaggedElement;
use crate::error::ComputeError;
mod padded_ijk_for_coords;

pub use padded_ijk_for_coords::PaddedIJKForCoords;

use super::elem;
// binding 0: JaggedTensorCore::data
// binding 1: JaggedIndices::batch_idx
// binding 2: JaggedIndices::offsets
// binging 3: JaggedIndices::list_idx
#[derive(Clone)]
pub struct JaggedOps {
    pub padded_ijk_for_coords: PaddedIJKForCoords,
}

impl JaggedOps {
    pub fn new(device: &wgpu::Device) -> Result<Self, ComputeError> {
        Ok(Self {
            padded_ijk_for_coords: PaddedIJKForCoords::new(device)?,
        })
    }

    // 这里可以添加各种 JaggedTensor 的操作方法
    // 例如：map, reduce, filter 等
}
