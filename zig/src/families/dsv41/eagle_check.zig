//! The TFE head's self-test (TF_DS_EAGLE_CHECK=FILE): the head against eagle_ref.py's vectors before serving.
const std = @import("std");
const prompt = @import("prompt.zig");
const eagle = @import("eagle.zig");

const Engine = prompt.Engine;
const Eagle = eagle.Eagle;
const max_steps = eagle.max_steps;
const chunk_rows = eagle.chunk_rows;

const VecJson = struct {
    T: usize,
    steps: usize,
    dim: usize,
    layers: usize,
    taps: []const u16,
    q0: []const usize,
    drafts: []const []const i64,
    p_top1: []const []const f64,
    gap: []const []const f64,
    sizes: []const usize,
};

/// The head against eagle_ref.py's vectors in slot 0; pass: step-1 o within 3%, drafts equal until a gap < 0.05.
pub fn selfTest(eg: *Eagle, gpa: std.mem.Allocator, io: std.Io, e: *const Engine, path: []const u8) !bool {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const all = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 30));
    if (all.len < 16 or !std.mem.eql(u8, all[0..8], "TFEVECS1")) return error.NotEagleVectors;
    const n: usize = @intCast(std.mem.readInt(u64, all[8..16], .little));
    const v = try std.json.parseFromSliceLeaky(VecJson, a, all[16 .. 16 + n], .{ .ignore_unknown_fields = true });
    const D = eg.s.dim;
    const L = eg.s.n_taps;
    if (v.dim != D or v.layers != L or !std.mem.eql(u16, v.taps, eg.s.tapList()) or v.steps > max_steps) return error.EagleVectorsMismatch;
    const rows = v.T + v.steps;
    var off: usize = 16 + n;
    const taps = all[off..][0 .. rows * L * D * 2];
    off += taps.len;
    const ids_b = all[off..][0 .. rows * 4];
    off += ids_b.len;
    const o_ref = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, all[off..][0 .. v.q0.len * v.steps * D * 4]));
    const seq = try a.alloc(i32, rows);
    for (seq, 0..) |*q, i| q.* = std.mem.readInt(i32, ids_b[i * 4 ..][0..4], .little);
    const host_o = try a.alloc(u16, v.steps * D);
    var pass_ok = true;
    for (v.q0, 0..) |q0, ci| {
        eg.reset(0);
        var at: usize = 0;
        while (at < q0) {
            const k = @min(chunk_rows, eg.ring, q0 - at);
            try prompt.upload(e, eg.tin, taps[at * L * D * 2 ..].ptr, k * L * D * 2);
            try eg.absorb(e, eg.tin, L * D, &.{.{ .slot = 0, .start = at, .n = k }});
            at += k;
        }
        try eg.pass(e, &.{.{ .slot = 0, .q0 = q0, .pending = seq[q0], .seq = seq[0..q0] }}, v.steps);
        try e.d.check(e.d.api.cuMemcpyDtoH_v2(host_o.ptr, eg.dbg, v.steps * D * 2), "cuMemcpyDtoH");
        var errs: [max_steps]f32 = @splat(0);
        var p_eng: [max_steps]f32 = @splat(0);
        var same = true;
        for (0..v.steps) |j| {
            var mx: f32 = 0;
            var md: f32 = 0;
            for (0..D) |i| {
                const got: f32 = @bitCast(@as(u32, host_o[j * D + i]) << 16);
                const want = o_ref[(ci * v.steps + j) * D + i];
                mx = @max(mx, @abs(want));
                md = @max(md, @abs(got - want));
            }
            errs[j] = md / @max(mx, 1e-12);
            p_eng[j] = 1.0 / (1.0 + @exp(-(eg.confs[0][j] - eg.conf_b) / eg.conf_a));
            if (same and eg.drafts[0][j] != v.drafts[ci][j]) {
                if (v.gap[ci][j] >= 0.05) pass_ok = false;
                same = false; // the chains differ from here: later steps compare nothing
            }
            if (!same) break;
        }
        if (errs[0] > 0.03) pass_ok = false;
        std.debug.print("{{\"eagle_check\": {d}, \"rel_err\": {any}, \"drafts\": {any}, \"ref_drafts\": {any}, \"p\": {any}, \"ref_p\": {any}}}\n", .{ q0, errs[0..v.steps], eg.drafts[0][0..v.steps], v.drafts[ci], p_eng[0..v.steps], v.p_top1[ci] });
    }
    eg.reset(0);
    std.debug.print("{{\"eagle_check\": \"{s}\", \"cases\": {d}}}\n", .{ if (pass_ok) "pass" else "fail", v.q0.len });
    return pass_ok;
}
