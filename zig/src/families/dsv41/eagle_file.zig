//! The TFE head file (eagle_convert.py): its header, the tensors a head runs, and its load (Eagle.open).
const std = @import("std");
const cuda = @import("cuda");
const prompt = @import("prompt.zig");
const eagle = @import("eagle.zig");

const Engine = prompt.Engine;
const Eagle = eagle.Eagle;
const max_streams = eagle.max_streams;
const max_steps = eagle.max_steps;
const chunk_rows = eagle.chunk_rows;
const max_taps = eagle.max_taps;
const final_tap = eagle.final_tap;
const max_keys = eagle.max_keys;
const max_cand = eagle.max_cand;
const top1_parts = eagle.top1_parts;
const ring_extra = eagle.ring_extra;
const magic = "TFEHEAD1";

pub const Shape = struct {
    dim: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    inter: usize,
    rank: usize,
    vocab: usize,
    window: usize,
    rope_dim: usize,
    rope_theta: f32,
    eps: f32,
    tap_norm: bool,
    qk_norm: bool = false,
    cand_k: usize = 0, // a two_stage head's candidates (0: lowrank)
    taps: [max_taps]u16 = @splat(0),
    n_taps: usize = 0,

    pub fn tapList(s: *const Shape) []const u16 {
        return s.taps[0..s.n_taps];
    }

    /// Whether the last tap is the final state.
    pub fn final(s: *const Shape) bool {
        return s.n_taps > 0 and s.taps[s.n_taps - 1] == final_tap;
    }

    /// The layer taps (the final state left out), in the engine's order.
    pub fn layerTaps(s: *const Shape) []const u16 {
        return s.taps[0 .. s.n_taps - @intFromBool(s.final())];
    }
};

const TensorJson = struct { offset: u64, shape: []const u64, dtype: []const u8 };
const HeaderJson = struct {
    format: u32,
    kind: []const u8,
    dim: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    inter: usize,
    rank: usize,
    vocab: usize,
    window: usize,
    rope_dim: usize,
    rope_theta: f32,
    eps: f32,
    tap_norm: bool,
    qk_norm: bool = false,
    head_type: []const u8 = "lowrank",
    cand_k: usize = 0,
    taps: []const u16,
    tensors: std.json.ArrayHashMap(TensorJson),
    data_bytes: u64,
};

/// A head file's header (eagle_convert.py): its shape and the data's first byte.
pub const Header = struct {
    shape: Shape,
    json: HeaderJson,
    data_at: u64,

    /// Reads and checks the header of `path` (allocations in `a`, an arena the caller frees).
    pub fn read(a: std.mem.Allocator, io: std.Io, path: []const u8) !Header {
        var f = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer f.close(io);
        var pre: [16]u8 = undefined;
        if (try f.readPositionalAll(io, &pre, 0) != 16 or !std.mem.eql(u8, pre[0..8], magic)) return error.NotEagleHead;
        const n = std.mem.readInt(u64, pre[8..16], .little);
        if (n > 1 << 24) return error.NotEagleHead;
        const text = try a.alloc(u8, @intCast(n));
        if (try f.readPositionalAll(io, text, 16) != text.len) return error.ShortEagleHead;
        const j = try std.json.parseFromSliceLeaky(HeaderJson, a, text, .{ .ignore_unknown_fields = true });
        if (j.format != 1 or !std.mem.eql(u8, j.kind, "base")) return error.UnsupportedEagleHead;
        if (j.taps.len == 0 or j.taps.len > max_taps or j.kv_heads == 0 or j.heads % j.kv_heads != 0) return error.UnsupportedEagleHead;
        if (j.head_dim != 128 or j.rope_dim > j.head_dim or j.rope_dim % 2 != 0 or j.window + 1 > max_keys) return error.UnsupportedEagleHead;
        const two = std.mem.eql(u8, j.head_type, "two_stage");
        if (!two and !std.mem.eql(u8, j.head_type, "lowrank")) return error.UnsupportedEagleHead;
        if (two != (j.cand_k > 0) or j.cand_k > max_cand) return error.UnsupportedEagleHead;
        var s: Shape = .{ .dim = j.dim, .heads = j.heads, .kv_heads = j.kv_heads, .head_dim = j.head_dim, .inter = j.inter, .rank = j.rank, .vocab = j.vocab, .window = j.window, .rope_dim = j.rope_dim, .rope_theta = j.rope_theta, .eps = j.eps, .tap_norm = j.tap_norm, .qk_norm = j.qk_norm, .cand_k = j.cand_k };
        for (j.taps, 0..) |t, i| {
            if (t == final_tap and i + 1 != j.taps.len) return error.UnsupportedEagleHead;
            if (i > 0 and t <= j.taps[i - 1]) return error.UnsupportedEagleHead; // the engine's order: ascending, final last
            s.taps[i] = t;
        }
        s.n_taps = j.taps.len;
        return .{ .shape = s, .json = j, .data_at = 16 + n };
    }
};

/// A head linear: weights [n, k] row-major, bf16 or e4m3 with an fp32 scale a row; own: the own GEMV (always fp8).
pub const Lin = struct { w: u64 = 0, scale: u64 = 0, n: usize = 0, k: usize = 0, fp8: bool = false, own: bool = false };

/// A linear's output: bf16, fp32, or added to the bf16 out (one rounding).
pub const Out = enum(c_int) { bf16 = 0, f32 = 1, add = 2 };

/// The tensors a head needs from its file: each one's place in the file and on the device (a multiple of 256 bytes).
const Plan = struct {
    items: [40]Item = undefined,
    n: usize = 0,

    const Item = struct { name: []const u8, dtype: []const u8, shape: [2]u64, dims: usize, dest: *u64, src: u64 = 0, bytes: u64 = 0, at: u64 = 0 };

    fn add(p: *Plan, name: []const u8, dtype: []const u8, shape: []const u64, dest: *u64) void {
        var sh: [2]u64 = .{ 0, 0 };
        for (shape, 0..) |x, i| sh[i] = x;
        p.items[p.n] = .{ .name = name, .dtype = dtype, .shape = sh, .dims = shape.len, .dest = dest };
        p.n += 1;
    }

    /// A linear `base` [n, k]: "BASE.weight" bf16, or "BASE.fp8" e4m3 and "BASE.scale" fp32 [n].
    fn lin(p: *Plan, l: *Lin, comptime base: []const u8, n: usize, k: usize, fp8: bool, own: bool) void {
        l.* = .{ .n = n, .k = k, .fp8 = fp8, .own = own or fp8 };
        if (fp8) {
            p.add(base ++ ".fp8", "f8e4m3", &.{ @intCast(n), @intCast(k) }, &l.w);
            p.add(base ++ ".scale", "f32", &.{@intCast(n)}, &l.scale);
        } else p.add(base ++ ".weight", "bf16", &.{ @intCast(n), @intCast(k) }, &l.w);
    }

    /// The bytes on the device; each item's file range checked against the header.
    fn resolve(p: *Plan, h: Header) !u64 {
        var total: u64 = 0;
        for (p.items[0..p.n]) |*it| {
            const t = h.json.tensors.map.get(it.name) orelse {
                std.log.err("eagle head: no tensor {s} (an FP8 head needs eagle_convert.py's FP8 forms)", .{it.name});
                return error.MissingEagleTensor;
            };
            if (!std.mem.eql(u8, t.dtype, it.dtype) or !std.mem.eql(u64, t.shape, it.shape[0..it.dims])) {
                std.log.err("eagle head: {s} is {s} {any}, want {s} {any}", .{ it.name, t.dtype, t.shape, it.dtype, it.shape[0..it.dims] });
                return error.EagleShapeMismatch;
            }
            const es: u64 = if (std.mem.eql(u8, it.dtype, "f32")) 4 else if (std.mem.eql(u8, it.dtype, "bf16")) 2 else 1;
            var n: u64 = es;
            for (it.shape[0..it.dims]) |x| n *= x;
            if (t.offset + n > h.json.data_bytes) return error.ShortEagleHead;
            it.src = h.data_at + t.offset;
            it.bytes = n;
            it.at = total;
            total = (total + n + 255) / 256 * 256;
        }
        return total;
    }
};

/// The head of `path` (header `hdr`) on the device: the module, rings for `slots` streams, scratch from the arena.
pub fn open(gpa: std.mem.Allocator, io: std.Io, e: *const Engine, a: *prompt.Arena, path: []const u8, hdr: Header, slots: usize) !*Eagle {
    const s = hdr.shape;
    const c = e.c;
    if (s.dim != c.hidden or s.vocab != c.vocab) return error.EagleShapeMismatch;
    if (slots == 0 or slots > max_streams) return error.BadStreams;
    // a prompt gives the head its last c.window rows (the replay floor): a longer head window starts short
    if (s.window > c.window) std.log.warn("eagle head: window {d}, but a prompt gives the head only its last {d} rows", .{ s.window, c.window });
    if (!cuda.kernels.available) return error.NoKernelImages;
    const eg = try gpa.create(Eagle);
    errdefer gpa.destroy(eg);
    eg.* = undefined;
    eg.st = @splat(.{});
    eg.s = s;
    eg.conf_a = 1.0;
    eg.conf_b = 0.0;
    eg.streams = 0;
    eg.steps = 0;
    eg.passes = 0;
    eg.gpa = gpa;
    eg.io = io;
    eg.dump = null;
    eg.dump_at = 0;
    eg.graphs = null;
    if (std.c.getenv("TF_DS_EAGLE_GRAPH")) |v| {
        if (std.mem.eql(u8, std.mem.span(v), "1")) eg.graphs = .empty;
    }
    eg.tap_norms = 0;
    eg.fb_norm = 0;
    eg.q_norm = 0;
    eg.k_norm = 0;
    eg.target = 0;
    eg.topk = s.cand_k;
    if (s.cand_k > 0) if (std.c.getenv("TF_DS_EAGLE_TOPK")) |v| {
        const k = std.fmt.parseInt(usize, std.mem.span(v), 10) catch return error.BadEagleTopK;
        if (k == 0 or k > max_cand) return error.BadEagleTopK;
        eg.topk = k;
    };
    if (std.c.getenv("TF_DS_EAGLE_CONF")) |v| {
        var it = std.mem.splitScalar(u8, std.mem.span(v), ',');
        eg.conf_a = try std.fmt.parseFloat(f32, it.next() orelse return error.BadEagleConf);
        eg.conf_b = try std.fmt.parseFloat(f32, it.next() orelse "0");
    }
    // FP8: on when the file has the forms (TF_DS_EAGLE_FP8 unset), "1" requires them, "0" runs bf16
    const fp8_env = if (std.c.getenv("TF_DS_EAGLE_FP8")) |v| std.mem.span(v) else "";
    if (fp8_env.len > 0 and !std.mem.eql(u8, fp8_env, "0") and !std.mem.eql(u8, fp8_env, "1")) return error.BadEagleFp8;
    const has_fp8 = hdr.json.tensors.map.get("fc.fp8") != null;
    const fp8 = if (std.mem.eql(u8, fp8_env, "1")) true else if (std.mem.eql(u8, fp8_env, "0")) false else has_fp8;
    if (fp8_env.len == 0 and !has_fp8) std.log.info("eagle head: no FP8 forms in {s}: the bf16 linears", .{path});
    var own_all = false;
    var own_wo = true;
    if (std.c.getenv("TF_DS_EAGLE_GEMV")) |v| {
        const m = std.mem.span(v);
        if (std.mem.eql(u8, m, "all")) {
            own_all = true;
        } else if (std.mem.eql(u8, m, "cublas")) {
            own_wo = false;
        } else if (!std.mem.eql(u8, m, "wo")) return error.BadEagleGemv;
    }
    // the tensors this head runs (one form of each linear), each read from the file into one buffer
    const D = s.dim;
    const hq = s.heads * s.head_dim;
    const hkv = s.kv_heads * s.head_dim;
    const L = s.n_taps;
    const d: u64 = @intCast(D);
    var plan: Plan = .{};
    plan.add("fc_norm", "bf16", &.{d}, &eg.fc_norm);
    plan.add("e_norm", "bf16", &.{d}, &eg.e_norm);
    plan.add("g_norm", "bf16", &.{d}, &eg.g_norm);
    plan.add("attn_norm", "bf16", &.{d}, &eg.attn_norm);
    plan.add("mlp_norm", "bf16", &.{d}, &eg.mlp_norm);
    plan.add("out_norm", "bf16", &.{d}, &eg.out_norm);
    plan.lin(&eg.fc, "fc", D, L * D, fp8, own_all);
    plan.lin(&eg.w_in, "w_in", D, 2 * D, fp8, own_all);
    plan.lin(&eg.wq, "wq", hq, D, fp8, own_all);
    plan.lin(&eg.wkv, "wkv", 2 * hkv, D, fp8, own_all);
    plan.lin(&eg.wo, "wo", D, hq, fp8, own_all or own_wo);
    plan.lin(&eg.w13, "w13", 2 * s.inter, D, fp8, own_all);
    plan.lin(&eg.w2, "w2", D, s.inter, fp8, own_all or own_wo);
    plan.lin(&eg.down, "down", s.rank, D, fp8, own_all);
    plan.lin(&eg.up, "up", s.vocab, s.rank, fp8, own_all);
    if (s.tap_norm) {
        plan.add("tap_norms", "bf16", &.{ @intCast(L), d }, &eg.tap_norms);
        plan.add("fb_norm", "bf16", &.{d}, &eg.fb_norm);
    }
    if (s.qk_norm) {
        plan.add("q_norm", "bf16", &.{@intCast(s.head_dim)}, &eg.q_norm);
        plan.add("k_norm", "bf16", &.{@intCast(s.head_dim)}, &eg.k_norm);
    }
    if (s.cand_k > 0) plan.add("target_head", "bf16", &.{ @intCast(s.vocab), d }, &eg.target);
    const bytes: usize = @intCast(try plan.resolve(hdr));
    eg.buf = try cuda.DeviceBuffer.alloc(e.d, bytes);
    errdefer eg.buf.free();
    {
        var f = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer f.close(io);
        const piece = try gpa.alloc(u8, 64 << 20);
        defer gpa.free(piece);
        for (plan.items[0..plan.n]) |it| {
            var done: u64 = 0;
            while (done < it.bytes) {
                const n: usize = @intCast(@min(piece.len, it.bytes - done));
                if (try f.readPositionalAll(io, piece[0..n], it.src + done) != n) return error.ShortEagleHead;
                try eg.buf.upload(@intCast(it.at + done), piece[0..n]);
                done += n;
            }
            it.dest.* = eg.buf.ptr + it.at;
        }
    }
    if (std.c.getenv("TF_DS_EAGLE_TAPDUMP")) |v| eg.dump = try std.Io.Dir.cwd().createFile(io, std.mem.span(v), .{ .truncate = true });
    errdefer if (eg.dump) |f| f.close(io);
    eg.mod = try cuda.Module.load(e.d, cuda.kernels.dsv41_eagle);
    errdefer eg.mod.unload();
    eg.f = .{
        .rmsnorm = try eg.mod.function("tf_eagle_rmsnorm_kernel"),
        .rope = try eg.mod.function("tf_eagle_rope_kernel"),
        .attn = try eg.mod.function("tf_eagle_attn_kernel"),
        .silu_mul = try eg.mod.function("tf_eagle_silu_mul_kernel"),
        .top1 = try eg.mod.function("tf_eagle_top1_kernel"),
        .topk = try eg.mod.function("tf_eagle_topk_kernel"),
        .rescore = try eg.mod.function("tf_eagle_rescore_kernel"),
        .gemv_bf16 = try eg.mod.function("tf_eagle_gemv_bf16_kernel"),
        .gemv_fp8 = try eg.mod.function("tf_eagle_gemv_fp8_kernel"),
        .top1_part = try eg.mod.function("tf_eagle_top1_part_kernel"),
        .top1_join = try eg.mod.function("tf_eagle_top1_join_kernel"),
    };
    eg.slots = slots;
    eg.ring = s.window + ring_extra;
    const rows = slots * eg.ring;
    eg.kring = try a.take((rows + 1) * hkv * 2);
    eg.vring = try a.take((rows + 1) * hkv * 2);
    eg.gring = try a.take((rows + 1) * D * 2);
    try prompt.fill(e, eg.kring, 0, (rows + 1) * hkv * 2);
    try prompt.fill(e, eg.vring, 0, (rows + 1) * hkv * 2);
    try prompt.fill(e, eg.gring, 0, (rows + 1) * D * 2);
    const R = chunk_rows;
    const M = max_streams;
    eg.ids = try a.take(R * 8);
    eg.rrow = try a.take(R * 8);
    eg.pos = try a.take(R * 8);
    eg.tin = try a.take(R * L * D * 2);
    eg.fin = try a.take(R * D * 2);
    eg.ev = try a.take(R * D * 2);
    eg.gv = try a.take(R * D * 2);
    eg.cat = try a.take(R * 2 * D * 2);
    eg.x = try a.take(R * D * 2);
    eg.h = try a.take(R * D * 2);
    eg.kv = try a.take(R * 2 * hkv * 2);
    eg.xs = .{ try a.take(M * D * 2), try a.take(M * D * 2) };
    eg.q = try a.take(M * hq * 2);
    eg.att = try a.take(M * hq * 2);
    eg.gu = try a.take(M * 2 * s.inter * 2);
    eg.y = try a.take(M * s.inter * 2);
    eg.lr = try a.take(M * s.rank * 2);
    eg.lg = try a.take(M * s.vocab * 4);
    eg.cand = try a.take(M * max_cand * 8);
    eg.ex = try a.take(M * max_cand * 4);
    eg.tpm = try a.take(M * top1_parts * 4);
    eg.tpz = try a.take(M * top1_parts * 4);
    eg.tpi = try a.take(M * top1_parts * 4);
    eg.sp = try a.take(max_steps * M * 8);
    eg.srow = try a.take(max_steps * M * 8);
    eg.sbase = try a.take(M * 8);
    eg.slo = try a.take(M * 8);
    eg.out = try a.take((max_steps + 1) * M * 8);
    eg.conf = try a.take(max_steps * M * 4);
    eg.invalid = try a.take(4);
    eg.dbg = try a.take(max_steps * M * D * 2);
    try prompt.fill(e, eg.invalid, 0, 4);
    try e.s.synchronize();
    std.log.info("eagle head: {s}, taps {any}, window {d}, qk_norm {}, candidates {d}, fp8 {}, own gemv {s}, {d} MiB weights, {d} slots", .{ path, s.tapList(), s.window, s.qk_norm, eg.topk, fp8, if (fp8 or own_all) "all" else if (own_wo) "wo, w2" else "none", bytes >> 20, slots });
    return eg;
}
