// padded_pass_b.wgsl
struct BBoxParams {
    bmin: vec3<i32>,
    bmax: vec3<i32>,
    total_pad: u32,
    num_elems: u32,
    num_outer_lists: u32
};

@group(0) @binding(8)
var<uniform> ubo: BBoxParams;

@group(0) @binding(2)
var<storage, read> in_offsets: array<u32>;

@group(0) @binding(6)
var<storage, read_write> out_offsets: array<u32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= ubo.num_outer_lists) {
        return;
    }
    out_offsets[idx] = in_offsets[idx] * ubo.total_pad;
}
