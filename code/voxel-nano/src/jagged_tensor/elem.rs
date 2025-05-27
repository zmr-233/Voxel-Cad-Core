use bytemuck::{Pod, Zeroable};
use glam::{IVec3, UVec3, Vec3, Vec4};
use wgpu::BufferSize;

// ============================================================================
// 1. 统一的 compile-time stride 计算器
// ============================================================================
const fn gpu_stride(bytes: usize) -> usize {
    match bytes {
        0..=4 => 4,
        5..=8 => 8,
        9..=12 => 16,
        _ => ((bytes + 15) / 16) * 16,
    }
}

const fn stride_to_bufsize(n: usize) -> BufferSize {
    unsafe { BufferSize::new_unchecked(n as u64) }
}

/// 计算满足 GPU 对齐要求的缓冲区大小
#[inline]
pub fn padded_size<E: JaggedElement>(count: u32) -> usize {
    let stride = E::STRIDE_SIZE as u64;
    (count as u64).saturating_mul(stride).max(stride) as usize
}

// ============================================================================
// 2. Trait：新增关联类型与 pad/unpad 方法
// ============================================================================
pub trait JaggedElement: Pod + Zeroable + Send + Sync + 'static {
    /// 原始类型 (unpadded)
    type Unpadded: Pod + Zeroable;
    /// 对齐后的类型 (padded)
    type Padded: Pod + Zeroable;

    /// 将原始类型填充到对齐类型
    fn pad(v: Self) -> Self::Padded;
    /// 从对齐类型去除填充
    fn unpad(v: Self::Padded) -> Self;

    const WGSL_TYPE: &'static str;
    const DIMENSIONS: u8;
    const SIZE: usize = core::mem::size_of::<Self::Padded>();
    const STRIDE_SIZE: usize = gpu_stride(Self::SIZE);
    const MIN_BINDING_SIZE: BufferSize = stride_to_bufsize(Self::STRIDE_SIZE);
}

// ============================================================================
// 3. 默认实现宏：unpadded == padded
// ============================================================================
macro_rules! impl_jagged {
    ($ty:ty, $wgsl:literal, $dim:expr) => {
        impl JaggedElement for $ty {
            type Unpadded = Self;
            type Padded = Self;
            #[inline]
            fn pad(v: Self) -> Self::Padded {
                v
            }
            #[inline]
            fn unpad(v: Self::Padded) -> Self {
                v
            }
            const WGSL_TYPE: &'static str = $wgsl;
            const DIMENSIONS: u8 = $dim;
        }
    };
}

// ============================================================================
// 4. 对齐填充 Vec3 -> Vec4
// ============================================================================
macro_rules! impl_jagged_padded_vec3 {
    ($unpad:ty, $pad:ty, $wgsl:literal, $zero:expr) => {
        impl JaggedElement for $unpad {
            type Unpadded = Self;
            type Padded = $pad;
            #[inline]
            fn pad(v: Self) -> Self::Padded {
                v.extend($zero)
            }
            #[inline]
            fn unpad(v: Self::Padded) -> Self {
                v.truncate()
            }
            const WGSL_TYPE: &'static str = $wgsl;
            const DIMENSIONS: u8 = 3;
        }
    };
}

// ============================================================================
// 5. 对齐填充 [T;3] -> [T;4]
// ============================================================================
macro_rules! impl_jagged_padded_array3 {
    ($unpad:ty, $scalar:ty, $wgsl:literal) => {
        impl JaggedElement for $unpad {
            type Unpadded = Self;
            type Padded = [$scalar; 4];
            #[inline]
            fn pad(v: Self) -> Self::Padded {
                [v[0], v[1], v[2], <$scalar as Zeroable>::zeroed()]
            }
            #[inline]
            fn unpad(v: Self::Padded) -> Self {
                [v[0], v[1], v[2]]
            }
            const WGSL_TYPE: &'static str = $wgsl;
            const DIMENSIONS: u8 = 3;
        }
    };
}

// ====== 标量和常规向量 ======
impl_jagged!(i32, "i32", 1);
impl_jagged!(u32, "u32", 1);
impl_jagged!(f32, "f32", 1);
impl_jagged!([i32; 2], "vec2<i32>", 2);
impl_jagged!([f32; 4], "vec4<f32>", 4);
impl_jagged!(glam::IVec2, "vec2<i32>", 2);
impl_jagged!(glam::UVec2, "vec2<u32>", 2);
impl_jagged!(glam::Vec2, "vec2<f32>", 2);
impl_jagged!(glam::Vec4, "vec4<f32>", 4);
impl_jagged!(glam::IVec4, "vec4<i32>", 4);
impl_jagged!(glam::UVec4, "vec4<u32>", 4);

impl_jagged_padded_vec3!(Vec3, Vec4, "vec3<f32>", 0.0);
impl_jagged_padded_vec3!(IVec3, glam::IVec4, "vec3<i32>", 0);
impl_jagged_padded_vec3!(UVec3, glam::UVec4, "vec3<u32>", 0);

impl_jagged_padded_array3!([i32; 3], i32, "vec3<i32>");
impl_jagged_padded_array3!([f32; 3], f32, "vec3<f32>");
