// ===============================================================================
// 错误类型定义
// ===============================================================================

/// GPU 计算错误类型
/// 对应原始 C++ 中各种异常情况的 Rust 错误处理
#[derive(Debug, thiserror::Error)]
pub enum ComputeError {
    #[error("Shader compilation failed: {0}")]
    ShaderCompilation(String),

    #[error("Buffer creation failed: {0}")]
    BufferCreation(String),

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("GPU execution failed: {0}")]
    Execution(String),

    #[error("Type mismatch: {0}")]
    TypeMismatch(String),
}

/// 类型错误
/// 对应原始 C++ 中的类型检查错误
#[derive(Debug, thiserror::Error)]
pub enum TypeError {
    #[error("Type mismatch")]
    Mismatch,

    #[error("Unsupported type")]
    Unsupported,
}
