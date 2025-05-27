mod build;
mod core;
mod elem;
mod ops;
use crate::error::TypeError;
pub use build::JaggedTensorBuilder;
use bytemuck::Pod;
use core::JaggedTensorCore;
pub use elem::JaggedElement;
use std::marker::PhantomData;

/// JaggedTensor 非规则张量数据结构
/// 底层 JaggedTensorCore 已经抹除类型信息
/// 类型T主要用于计算着色器的实现
pub struct JaggedTensor<T: Pod + JaggedElement> {
    pub core: JaggedTensorCore,

    _phantom: PhantomData<T>,
}

impl<T: Pod + JaggedElement> JaggedTensor<T> {
    pub fn from_core(core: JaggedTensorCore) -> Result<Self, TypeError> {
        // 验证元素大小是否匹配
        if core.metadata.elem_stride_size != T::STRIDE_SIZE as u8 {
            return Err(TypeError::Mismatch);
        }

        // 验证元素维度是否匹配
        if core.metadata.elem_dimensions != T::DIMENSIONS as u8 {
            return Err(TypeError::Mismatch);
        }

        Ok(Self {
            core,
            _phantom: PhantomData,
        })
    }

    pub fn core(&self) -> &JaggedTensorCore {
        &self.core
    }
}
