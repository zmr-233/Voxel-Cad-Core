use super::ops;
use crate::error::ComputeError;
use std::sync::Arc;
// ===============================================================================
// 核心数据结构 - JaggedTensorCore
// ===============================================================================
/// JaggedTensor 的核心存储结构（无类型化）
/// 对应原始 C++ JaggedTensor 的数据成员，但使用类型擦除以提高灵活性
/// ldim=3 为最高层次的嵌套列表结构
/// 目前版本只提供几种Element类型支持:
/// i32 Vec3f
#[derive(Clone)]
pub struct JaggedTensorCore {
    /// GPU Buffer: 扁平化的实际数据，对应原始 C++ 的 mData
    /// 存储所有嵌套列表中的数据元素，按线性顺序排列
    /// `mData = ['A','B','C','D','E','F','G','H']  # ΣN_i = 8`
    pub data: wgpu::Buffer,

    /// GPU 端索引信息，封装了原始 C++ 的 mBatchIdx, mOffsets, mListIdx
    pub indices: JaggedIndices,

    /// CPU 端元数据，包含形状和类型信息
    pub metadata: JaggedMetadata,

    /// CPU 端形状缓存，对应原始 C++ 的 mLShapeCache
    pub shape_cache: JaggedShapeCache,

    /// wgpu 设备句柄，用于 GPU 操作
    pub device: Arc<wgpu::Device>,

    /// wgpu 队列句柄，用于提交 GPU 命令
    pub queue: Arc<wgpu::Queue>,

    /// 一系列JaggedTensor算子
    pub ops: ops::JaggedOps,
}

/// GPU 端索引数据结构
/// 对应原始 C++ JaggedTensor 的索引相关成员：
/// - mBatchIdx: Which (linear) batch is each datum in
/// - mOffsets: Offset of each tensor in the list of lists
/// - mListIdx: LoL indexing of tensor with shape [num_tensors, ldim]
/// 这里以如下数据为例子：
/// ```python
/// # 3-层 ragged 结构： [batch][time][patch] → (数据项)
/// # 其中 A 可以是i32/Vec3f等类型
/// raw_data = [
///     [   # ────── batch 0 ──────
///         ['A', 'B'],           # t=0 有 2 个 patch
///         ['C'],                # t=1 有 1 个 patch
///     ],
///     [   # ────── batch 1 ──────
///         ['D', 'E', 'F'],      # t=0 有 3 个 patch
///         ['G', 'H'],           # t=1 有 2 个 patch
///     ]
/// ]
/// ```
#[derive(Clone)]
pub struct JaggedIndices {
    /// GPU Buffer: 批次索引数组，对应原始 C++ 的 mBatchIdx
    /// 存储每个数据元素所属的批次ID，形状为 [total_elements]
    /// 类型：i32 (对应 C++ 的 JIdxType)
    /// `mBatchIdx = [0,0,0,1,1,1,1,1] # 长度 8`
    pub batch_idx: wgpu::Buffer,

    /// GPU Buffer: 偏移量数组，对应原始 C++ 的 mOffsets
    /// 每个 batch 在 mData 里的 [start, end) 索引，形状为 [num_tensors + 1]
    /// 类型：i64 (对应 C++ 的 JOffsetsType)
    /// `mOffsets  = [[0,3], [3,8]] # 2×2`
    pub offsets: wgpu::Buffer,

    /// GPU Buffer: 列表索引映射，对应原始 C++ 的 mListIdx
    /// 存储多层嵌套列表中每个子张量的位置坐标，形状为 [num_tensors, ldim]
    /// 类型：i32 (对应 C++ 的 JLIdxType)
    /// ```python
    /// mListIdx = [
    ///     [0,0,0],  # A
    ///     [0,0,1],  # B
    ///     [0,1,0],  # C
    ///     [1,0,0],  # D
    ///     [1,0,1],  # E
    ///     [1,0,2],  # F
    ///     [1,1,0],  # G
    ///     [1,1,1],  # H
    /// ]
    /// ```
    /// ⚠️警告: 默认大小强制固定为UVec3::SIZE 实则相当于 UVec4
    pub list_idx: wgpu::Buffer,
}

/// CPU 端元数据结构
/// 对应原始 C++ JaggedTensor 的基本属性
#[derive(Clone)]
pub struct JaggedMetadata {
    /// 最外层列表数量，对应 C++ 的 mNumOuterLists
    /// 即批次大小 (batch size)
    pub num_outer_lists: usize,

    /// 嵌套层次维度，对应 C++ 通过 mListIdx.size(1) 计算的 ldim
    /// 1=List<Tensor>, 2=List<List<Tensor>>, 3=List<List<List<Tensor>>>
    /// [A, B] -> ldim=1
    /// [[A, B], [C]] -> ldim=2
    /// [[[A, B], [C]], [[D, E, F], [G, H]]] -> ldim=3
    /// 但A和B可以是Vec3f等多维数据 -- 序列存储在data中
    pub ldim: u8,

    /// 扁平化数据中的总元素数量，对应 C++ 的 mData.size(0)
    pub num_elements: usize,

    /// 单个元素的字节大小(已经展平)，用于 GPU 内存计算
    /// 注意：是考虑了对齐的大小 Vec3f -> 16, i32 -> 4
    pub elem_stride_size: u8,

    /// 单个元素的向量维度(已经展平到mdata) -- 注意与ldim区分
    /// 例如 i32 -> 1, Vec3f -> 3
    pub elem_dimensions: u8,
}

/// CPU 端形状信息缓存
/// 对应原始 C++ JaggedTensor 的 mLShapeCache 结构
/// 目的：避免重复的 GPU->CPU 数据拷贝，提升性能
// ```python
// # mLShapeCache: 缓存每一级列表中各子列表的长度，
// # 这里第 1 级是 time（每 batch 有 2 帧），
// # 第 2 级是 patch（每帧 2 个 patch），
// # 第 3 级是最内层元素数（这里每个 patch 我们简化成 1 个字母）
// mLShapeCache = {
//     "mLShape1": [2, 2],             # 每个 batch 的时间步数
//     "mLShape2": [                   # (batch,time) → patch 数
//         [2,1],                      # batch-0: t=0→2, t=1→1
//         [3,2],                      # batch-1: t=0→3, t=1→2
//     ],
//     "mLShape3": [                   # 此处每个 patch 的体素数全为 1
//         [[1,1],[1]],                # batch-0
//         [[1,1,1],[1,1]],            # batch-1
//     ],
//     'mDirty': False,  # 已经是最新缓存
// }
// ```
#[derive(Clone, Default)]
pub struct JaggedShapeCache {
    /// 一维嵌套列表的形状缓存，对应 C++ 的 mLShape1
    /// 存储每个外层列表中元素的数量
    pub lshape1: Option<Vec<usize>>,

    /// 二维嵌套列表的形状缓存，对应 C++ 的 mLShape2
    /// 存储每个外层列表中，每个内层列表的元素数量
    pub lshape2: Option<Vec<Vec<usize>>>,

    /// 三维嵌套列表的形状缓存，对应 C++ 的 mLShape3
    /// 存储每个外层列表中，每个中层列表中，每个内层列表的元素数量
    pub lshape3: Option<Vec<Vec<Vec<usize>>>>,

    /// 缓存失效标志，对应 C++ 的 mDirty
    /// 当底层数据被修改时设为 true，需要重新计算缓存
    pub is_dirty: bool,
}

// ===============================================================================
// CPU 侧方法实现
// ===============================================================================

impl JaggedTensorCore {
    pub fn new(
        data_buffer: wgpu::Buffer,
        batch_idx_buffer: wgpu::Buffer,
        offsets_buffer: wgpu::Buffer,
        list_idx_buffer: wgpu::Buffer,
        num_outer_lists: usize,
        ldim: u8,
        num_elements: usize,
        elem_size: u8,
        elem_dimensions: u8,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> Result<Self, ComputeError> {
        let ops = ops::JaggedOps::new(&device)?;
        let mut core = Self {
            data: data_buffer,
            indices: JaggedIndices {
                batch_idx: batch_idx_buffer,
                offsets: offsets_buffer,
                list_idx: list_idx_buffer,
            },
            metadata: JaggedMetadata {
                num_outer_lists,
                ldim,
                num_elements,
                elem_stride_size: elem_size,
                elem_dimensions,
            },
            shape_cache: JaggedShapeCache::default(),
            device,
            queue,
            ops,
        };
        // 初始时清空形状缓存
        core.shape_cache.clear();
        Ok(core)
    }
    /// Create a new JaggedTensorCore by replacing buffers and clearing shape cache
    pub fn with_buffers(
        &self,
        data_buffer: wgpu::Buffer,
        batch_idx_buffer: wgpu::Buffer,
        offsets_buffer: wgpu::Buffer,
        list_idx_buffer: wgpu::Buffer,
        num_elements: usize,
    ) -> Self {
        let mut new_core = self.clone();
        new_core.data = data_buffer;
        new_core.indices.batch_idx = batch_idx_buffer;
        new_core.indices.offsets = offsets_buffer;
        new_core.indices.list_idx = list_idx_buffer;
        new_core.metadata.num_elements = num_elements;
        new_core.shape_cache.clear();
        new_core
    }
    /// 获取设备引用，对应 C++ 的设备访问接口
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// 获取队列引用，对应 C++ 的队列访问接口
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// 获取数据缓冲区引用，对应 C++ 的 mData 访问
    pub fn data_buffer(&self) -> &wgpu::Buffer {
        &self.data
    }

    /// 获取批次索引缓冲区引用，对应 C++ 的 mBatchIdx 访问
    pub fn batch_idx_buffer(&self) -> &wgpu::Buffer {
        &self.indices.batch_idx
    }

    /// 获取偏移量缓冲区引用，对应 C++ 的 mOffsets 访问
    pub fn offsets_buffer(&self) -> &wgpu::Buffer {
        &self.indices.offsets
    }

    /// 获取列表索引缓冲区引用，对应 C++ 的 mListIdx 访问
    pub fn list_idx_buffer(&self) -> &wgpu::Buffer {
        &self.indices.list_idx
    }

    /// 获取元素数量，对应 C++ 的 mData.size(0)
    pub fn num_elements(&self) -> usize {
        self.metadata.num_elements
    }

    /// 获取外层列表数量，对应 C++ 的 mNumOuterLists
    pub fn num_outer_lists(&self) -> usize {
        self.metadata.num_outer_lists
    }

    /// 获取嵌套维度，对应 C++ 的 ldim()
    pub fn ldim(&self) -> u8 {
        self.metadata.ldim
    }
}

impl JaggedShapeCache {
    /// 标记缓存为脏数据，对应 C++ 的 markDirty()
    fn _mark_dirty(&mut self) {
        self.is_dirty = true;
    }

    /// 清空所有缓存，对应 C++ 的 clear()
    fn clear(&mut self) {
        self.lshape1 = None;
        self.lshape2 = None;
        self.lshape3 = None;
        // 标记为脏，保证下次访问时重建缓存
        self.is_dirty = true;
    }
}
// ===============================================================================
// 其他非重要方法
// ===============================================================================
