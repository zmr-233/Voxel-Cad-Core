// padded_pass_a.wgsl

struct BBoxParams {
    bmin: vec3<i32>,
    bmax: vec3<i32>,
    total_pad: u32,
    num_elems: u32,
    num_outer_lists: u32,
};

@group(0) @binding(8)
var<uniform> ubo: BBoxParams;

@group(0) @binding(0)
var<storage, read> data: array<vec3<i32>>;
@group(0) @binding(1)
var<storage, read> batch_idx: array<u32>;
// @group(0) @binding(2)
// var<storage, read> offsets: array<u32>;      // un used
@group(0) @binding(3)
var<storage, read> list_idx: array<vec3<i32>>; 

@group(0) @binding(4)
var<storage, read_write> out_ijk: array<vec3<i32>>;
@group(0) @binding(5)
var<storage, read_write> out_bidx: array<u32>;
// @group(0) @binding(6)
// var<storage, read_write> out_offsets: array<u32>; // un used
@group(0) @binding(7)
var<storage, read_write> out_list_idx: array<vec3<i32>>;


@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx: u32 = gid.x;
    
    // 确保从uniform buffer读取的值是一致的
    let total_pad_local = ubo.total_pad;
    let num_elems_local = ubo.num_elems;
    let total: u32 = num_elems_local * total_pad_local;
    
    if (idx >= total) {
        return;
    }

    // 原始元素索引与填充偏移量 - 使用本地变量
    let eidx: u32    = idx / total_pad_local;
    let pad_idx: u32 = idx % total_pad_local;

    // 包围盒维度计算
    let bmin: vec3<i32> = vec3<i32>(ubo.bmin.x, ubo.bmin.y, ubo.bmin.z);
    let bmax: vec3<i32> = vec3<i32>(ubo.bmax.x, ubo.bmax.y, ubo.bmax.z);
    let dims: vec3<i32> = bmax - bmin + vec3<i32>(1, 1, 1);
    let dims_z: u32  = u32(dims.z);
    let dims_y: u32  = u32(dims.y);
    let dims_yz: u32 = dims_y * dims_z;

    // 3D 偏移 (x_off, y_off, z_off) - 修正计算顺序
    let z_off: i32 = i32(pad_idx % dims_z);
    let y_off: i32 = i32((pad_idx / dims_z) % dims_y);
    let x_off: i32 = i32(pad_idx / dims_yz);

    // 读取原始坐标并添加偏移量
    let base: vec3<i32> = data[eidx];
    let offset: vec3<i32> = bmin + vec3<i32>(x_off, y_off, z_off);
    let final_coord: vec3<i32> = base + offset;

    // 写入输出缓冲 - 添加调试信息
    out_ijk[idx] = final_coord;
    out_bidx[idx] = batch_idx[eidx];
    
    // 对于剩余的位置，写入正常的list_idx
    out_list_idx[idx] = list_idx[eidx];
}
