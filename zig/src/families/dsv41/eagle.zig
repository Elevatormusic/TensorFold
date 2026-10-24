//! The TFE draft head (TF_DS_DRAFTER=eagle, TF_DS_EAGLE_HEAD=FILE): an EAGLE-style head trained on the target's hidden
//! states (the EAGLE study's eagle_train.py Head; the ideas are from the public EAGLE papers, no code from them), on rank
//! 0 only. Rank 0 sends rank 1 every draft in the verify frame, so rank 1 runs no head, and a pass has no collective.
//!
//! The head, for a row at position p with the target's taps at p (tap layers side by side, the final state last) and
//! the next token x_{p+1}: g = RMSNorm(fc(taps)); x = w_in [RMSNorm(embed(x_{p+1})) ; RMSNorm(g)]; one decoder layer
//! (heads x 128 queries, one key / value head, RoPE on 64 of 128 dims, attention over the stream's newest 129 positions,
//! then a SwiGLU MLP); logits = up(down(RMSNorm(o))). A stream's chain from its pending token at q0: the real rows below
//! q0 (their keys and values in the stream's ring), draft 1 from the row q0 - 1, and draft j > 1 from a row at position
//! q0 + j - 2 with g = the step before's o and the token draft j - 1 (its key and value go into the ring too; the next
//! pass's real rows write over them). tools/dsv41-zig/eagle/eagle_ref.py is this order of work in torch (the
//! self-test's reference), checked against the training code's chain.
//!
//! The g rows come with each round (absorbRound: the drafting streams' window rows, eager as the DSpark absorb) and each
//! prompt chunk that reaches the head (the last ring rows). A pass first runs the rows from the stream's last extended
//! position to q0 - 1 through the layer's input part (their keys and values), then `steps` single-row steps for all
//! streams at once. Every GEMM is a bf16 cuBLAS call (fp32 sums); the row kernels are zig/kernels/cuda/dsv41/eagle.cu.
const std = @import("std");
const cuda = @import("cuda");
const prompt = @import("prompt.zig");

const Engine = prompt.Engine;

/// A pass's streams at most (the pool's).
pub const max_streams = 16;
/// A stream's drafts a pass at most (a verify window holds the pending row and 5 drafts: lanes.window_rows).
pub const max_steps = 5;
/// The rows an extend or an absorb runs at once.
pub const chunk_rows = 160;
/// The taps a head reads at most.
pub const max_taps = 8;
/// The final hidden state's tap name (eagle_collect.py's FINAL): the target head's normed input.
pub const final_tap: u16 = 1000;
const ring_extra = 16;
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
        if (j.head_dim != 128 or j.rope_dim > j.head_dim or j.rope_dim % 2 != 0 or j.window + 1 > 512) return error.UnsupportedEagleHead;
        var s: Shape = .{ .dim = j.dim, .heads = j.heads, .kv_heads = j.kv_heads, .head_dim = j.head_dim, .inter = j.inter, .rank = j.rank, .vocab = j.vocab, .window = j.window, .rope_dim = j.rope_dim, .rope_theta = j.rope_theta, .eps = j.eps, .tap_norm = j.tap_norm };
        for (j.taps, 0..) |t, i| {
            if (t == final_tap and i + 1 != j.taps.len) return error.UnsupportedEagleHead;
            if (i > 0 and t <= j.taps[i - 1]) return error.UnsupportedEagleHead; // the engine's order: ascending, final last
            s.taps[i] = t;
        }
        s.n_taps = j.taps.len;
        return .{ .shape = s, .json = j, .data_at = 16 + n };
    }
};

/// A window of absorbed rows: `n` rows of positions start .. in `slot`, in input row order; keep false: a stream that
/// does not draft (its rows are written nowhere).
pub const Span = struct { slot: usize, start: usize, n: usize, keep: bool = true };

/// A stream of a pass: its slot, its pending token at position q0, its ids by position (seq[p] for p < q0).
pub const Req = struct { slot: usize, q0: usize, pending: i64, seq: []const i32 };

/// What a slot's rings hold: g rows of positions [g_lo, g_hi), keys and values of the real rows below `ext` from `lo`.
const Slot = struct { live: bool = false, lo: usize = 0, ext: usize = 0, g_lo: usize = 0, g_hi: usize = 0 };

const Fns = struct {
    rmsnorm: cuda.Function,
    rope: cuda.Function,
    attn: cuda.Function,
    silu_mul: cuda.Function,
    top1: cuda.Function,
};

pub const Eagle = struct {
    s: Shape,
    buf: cuda.DeviceBuffer, // the weights
    // the weights (bf16, row-major [out, in])
    fc_norm: u64,
    e_norm: u64,
    g_norm: u64,
    attn_norm: u64,
    mlp_norm: u64,
    out_norm: u64,
    fc: u64,
    w_in: u64,
    wq: u64,
    wkv: u64,
    wo: u64,
    w13: u64,
    w2: u64,
    down: u64,
    up: u64,
    tap_norms: u64 = 0,
    fb_norm: u64 = 0,
    mod: cuda.Module,
    f: Fns,
    slots: usize,
    ring: usize,
    // the rings: keys and values bf16 [slots * ring + 1, kv_heads * 128], g bf16 [slots * ring + 1, dim] (the last
    // row of each: a sink for the rows of streams that do not draft or get no drafts)
    kring: u64,
    vring: u64,
    gring: u64,
    st: [max_streams]Slot = @splat(.{}),
    // scratch (chunk_rows rows unless named)
    ids: u64, // i64
    rrow: u64, // i64: ring rows
    pos: u64, // i64
    tin: u64, // bf16 [rows, taps * dim]: an absorb's taps
    fin: u64, // bf16 [rows, dim]: the final state's rows of a prompt chunk
    ev: u64, // bf16 [rows, dim]: embedding rows
    gv: u64, // bf16 [rows, dim]: g rows
    cat: u64, // bf16 [rows, 2 dim]
    x: u64, // bf16 [rows, dim]
    h: u64, // bf16 [rows, dim]
    kv: u64, // bf16 [rows, 2 kv_heads 128]
    xs: [2]u64, // bf16 [max_streams, dim]: the steps' residual rows (alternate: the step before's is the feedback)
    q: u64, // bf16 [max_streams, heads 128]
    att: u64,
    gu: u64, // bf16 [max_streams, 2 inter]
    y: u64, // bf16 [max_streams, inter]
    lr: u64, // bf16 [max_streams, rank]
    lg: u64, // f32 [max_streams, vocab]
    sp: u64, // i64 [max_steps, max_streams]: each step's positions, ring rows, then [max_streams] bases and lows
    srow: u64,
    sbase: u64,
    slo: u64,
    out: u64, // i64 [max_steps + 1, max_streams]: the drafts by step
    conf: u64, // f32 [max_steps, max_streams]
    invalid: u64, // u32: a gather's index outside its table
    dbg: u64, // bf16 [max_steps, max_streams, dim]: each step's o (the self-test reads it)
    conf_a: f32 = 1.0,
    conf_b: f32 = 0.0,
    // the last pass on the host
    streams: usize = 0,
    steps: usize = 0,
    drafts: [max_streams][max_steps]i64 = undefined,
    confs: [max_streams][max_steps]f32 = undefined,
    // where the passes spend their time (ns, the host's clock around each pass)
    passes: u64 = 0,

    /// The head of `path` (its header read as `hdr`) on the device, the module, rings for `slots` streams and the
    /// scratch from the arena.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, e: *const Engine, a: *prompt.Arena, path: []const u8, hdr: Header, slots: usize) !*Eagle {
        const s = hdr.shape;
        const c = e.c;
        if (s.dim != c.hidden or s.vocab != c.vocab) return error.EagleShapeMismatch;
        if (slots == 0 or slots > max_streams) return error.BadStreams;
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
        eg.tap_norms = 0;
        eg.fb_norm = 0;
        if (std.c.getenv("TF_DS_EAGLE_CONF")) |v| {
            var it = std.mem.splitScalar(u8, std.mem.span(v), ',');
            eg.conf_a = try std.fmt.parseFloat(f32, it.next() orelse return error.BadEagleConf);
            eg.conf_b = try std.fmt.parseFloat(f32, it.next() orelse "0");
        }
        // the weights: one buffer, read in pieces
        const bytes: usize = @intCast(hdr.json.data_bytes);
        eg.buf = try cuda.DeviceBuffer.alloc(e.d, bytes);
        errdefer eg.buf.free();
        {
            var f = try std.Io.Dir.cwd().openFile(io, path, .{});
            defer f.close(io);
            const piece = try gpa.alloc(u8, 64 << 20);
            defer gpa.free(piece);
            var done: usize = 0;
            while (done < bytes) {
                const n = @min(piece.len, bytes - done);
                if (try f.readPositionalAll(io, piece[0..n], hdr.data_at + done) != n) return error.ShortEagleHead;
                try eg.buf.upload(done, piece[0..n]);
                done += n;
            }
        }
        const D = s.dim;
        const hq = s.heads * s.head_dim;
        const hkv = s.kv_heads * s.head_dim;
        const L = s.n_taps;
        const T = struct {
            fn at(h: Header, base: u64, name: []const u8, shape: []const u64) !u64 {
                const t = h.json.tensors.map.get(name) orelse return error.MissingEagleTensor;
                if (!std.mem.eql(u8, t.dtype, "bf16") or !std.mem.eql(u64, t.shape, shape)) {
                    std.log.err("eagle head: {s} has shape {any}, want {any}", .{ name, t.shape, shape });
                    return error.EagleShapeMismatch;
                }
                var n: u64 = 2;
                for (shape) |x| n *= x;
                if (t.offset + n > h.json.data_bytes) return error.ShortEagleHead;
                return base + t.offset;
            }
        };
        const b = eg.buf.ptr;
        const u = struct {
            fn v(x: usize) u64 {
                return @intCast(x);
            }
        }.v;
        eg.fc_norm = try T.at(hdr, b, "fc_norm", &.{u(D)});
        eg.e_norm = try T.at(hdr, b, "e_norm", &.{u(D)});
        eg.g_norm = try T.at(hdr, b, "g_norm", &.{u(D)});
        eg.attn_norm = try T.at(hdr, b, "attn_norm", &.{u(D)});
        eg.mlp_norm = try T.at(hdr, b, "mlp_norm", &.{u(D)});
        eg.out_norm = try T.at(hdr, b, "out_norm", &.{u(D)});
        eg.fc = try T.at(hdr, b, "fc.weight", &.{ u(D), u(L * D) });
        eg.w_in = try T.at(hdr, b, "w_in.weight", &.{ u(D), u(2 * D) });
        eg.wq = try T.at(hdr, b, "wq.weight", &.{ u(hq), u(D) });
        eg.wkv = try T.at(hdr, b, "wkv.weight", &.{ u(2 * hkv), u(D) });
        eg.wo = try T.at(hdr, b, "wo.weight", &.{ u(D), u(hq) });
        eg.w13 = try T.at(hdr, b, "w13.weight", &.{ u(2 * s.inter), u(D) });
        eg.w2 = try T.at(hdr, b, "w2.weight", &.{ u(D), u(s.inter) });
        eg.down = try T.at(hdr, b, "down.weight", &.{ u(s.rank), u(D) });
        eg.up = try T.at(hdr, b, "up.weight", &.{ u(s.vocab), u(s.rank) });
        if (s.tap_norm) {
            eg.tap_norms = try T.at(hdr, b, "tap_norms", &.{ u(L), u(D) });
            eg.fb_norm = try T.at(hdr, b, "fb_norm", &.{u(D)});
        }
        eg.mod = try cuda.Module.load(e.d, cuda.kernels.dsv41_eagle);
        errdefer eg.mod.unload();
        eg.f = .{
            .rmsnorm = try eg.mod.function("tf_eagle_rmsnorm_kernel"),
            .rope = try eg.mod.function("tf_eagle_rope_kernel"),
            .attn = try eg.mod.function("tf_eagle_attn_kernel"),
            .silu_mul = try eg.mod.function("tf_eagle_silu_mul_kernel"),
            .top1 = try eg.mod.function("tf_eagle_top1_kernel"),
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
        std.log.info("eagle head: {s}, taps {any}, {d} MiB weights, {d} slots", .{ path, s.tapList(), bytes >> 20, slots });
        return eg;
    }

    pub fn close(eg: *Eagle, gpa: std.mem.Allocator) void {
        eg.mod.unload();
        eg.buf.free();
        gpa.destroy(eg);
    }

    /// A new prompt in `slot`: its rings hold nothing yet.
    pub fn reset(eg: *Eagle, slot: usize) void {
        if (slot < max_streams) eg.st[slot] = .{};
    }

    // -- the row kernels ------------------------------------------------------------------------------------------

    fn rms(eg: *const Eagle, e: *const Engine, x: u64, ldx: usize, w: u64, out: u64, ldo: usize, rows: usize) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(c_longlong, @intCast(ldx)));
        a.add(w);
        a.add(out);
        a.add(@as(c_longlong, @intCast(ldo)));
        a.add(@as(c_int, @intCast(eg.s.dim)));
        a.add(eg.s.eps);
        try cuda.launch.launch(eg.f.rmsnorm, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 512 } }, e.s, &a);
    }

    fn rope(eg: *const Eagle, e: *const Engine, x: u64, ld: usize, heads: usize, pos: u64, rows: usize) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(c_longlong, @intCast(ld)));
        a.add(@as(c_int, @intCast(heads)));
        a.add(@as(c_int, @intCast(eg.s.head_dim)));
        a.add(@as(c_int, @intCast(eg.s.rope_dim)));
        a.add(pos);
        a.add(eg.s.rope_theta);
        try cuda.launch.launch(eg.f.rope, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, e.s, &a);
    }

    /// x [rows, k] bf16 (row stride ldx) @ w [n, k]^T into out (row stride ldo).
    fn mm(e: *const Engine, x: u64, ldx: usize, w: u64, out: u64, ldo: usize, f32_out: bool, rows: usize, k: usize, n: usize) !void {
        try e.blas.linearDense(x, ldx, w, out, ldo, f32_out, rows, k, n);
    }

    /// The keys and values of `rows` rows of h (normed): wkv, RoPE on the keys at `pos`, into the rings at ring rows
    /// `rrow` (i64 [rows], device).
    fn keysOut(eg: *const Eagle, e: *const Engine, h: u64, pos: u64, rrow: u64, rows: usize) !void {
        const hkv = eg.s.kv_heads * eg.s.head_dim;
        try mm(e, h, eg.s.dim, eg.wkv, eg.kv, 2 * hkv, false, rows, eg.s.dim, 2 * hkv);
        try eg.rope(e, eg.kv, 2 * hkv, eg.s.kv_heads, pos, rows);
        try e.ops.scatterRows(e.s, eg.kv, 2 * hkv * 2, rrow, eg.kring, hkv * 2, hkv * 2, rows);
        try e.ops.scatterRows(e.s, eg.kv + hkv * 2, 2 * hkv * 2, rrow, eg.vring, hkv * 2, hkv * 2, rows);
    }

    /// The layer's input of `rows` rows: x = w_in [RMSNorm(embed rows ev) ; RMSNorm(g rows)] (g at `g`, row stride
    /// dim) into `out`.
    fn input(eg: *const Eagle, e: *const Engine, g: u64, out: u64, rows: usize) !void {
        const D = eg.s.dim;
        try eg.rms(e, eg.ev, D, eg.e_norm, eg.cat, 2 * D, rows);
        try eg.rms(e, g, D, eg.g_norm, eg.cat + D * 2, 2 * D, rows);
        try mm(e, eg.cat, 2 * D, eg.w_in, out, D, false, rows, 2 * D, D);
    }

    // -- absorbs --------------------------------------------------------------------------------------------------

    /// The g rows of `spans` (their rows in order, taps bf16 [rows, taps * dim] at `taps`, row stride ldt elements):
    /// g = RMSNorm(fc(taps)) into each kept span's ring rows, then the slots' bookkeeping.
    pub fn absorb(eg: *Eagle, e: *const Engine, taps: u64, ldt: usize, spans: []const Span) !void {
        var total: usize = 0;
        for (spans) |sp| total += sp.n;
        if (total == 0) return;
        if (total > chunk_rows) return error.EagleAbsorbTooLarge;
        const D = eg.s.dim;
        const L = eg.s.n_taps;
        var rows: [chunk_rows]i64 = undefined;
        var r: usize = 0;
        for (spans) |sp| {
            if (sp.slot >= eg.slots) return error.BadSlot;
            // (more rows than the ring: two rows of one scatter go to one ring row, and either one can stay)
            if (sp.keep and sp.n > eg.ring) return error.EagleAbsorbTooLarge;
            for (0..sp.n) |j| {
                rows[r] = @intCast(if (sp.keep) sp.slot * eg.ring + (sp.start + j) % eg.ring else eg.slots * eg.ring);
                r += 1;
            }
        }
        try prompt.upload(e, eg.rrow, &rows, total * 8);
        var src = taps;
        var ld = ldt;
        if (eg.s.tap_norm) {
            for (0..L) |j| try eg.rms(e, taps + j * D * 2, ldt, eg.tap_norms + j * D * 2, eg.tin + j * D * 2, L * D, total);
            src = eg.tin;
            ld = L * D;
        }
        try mm(e, src, ld, eg.fc, eg.x, D, false, total, L * D, D);
        try eg.rms(e, eg.x, D, eg.fc_norm, eg.h, D, total);
        try e.ops.scatterRows(e.s, eg.h, D * 2, eg.rrow, eg.gring, D * 2, D * 2, total);
        for (spans) |sp| if (sp.keep) note(&eg.st[sp.slot], sp.start, sp.n, eg.ring);
    }

    /// A round's windows (lanes.zig verify): every row's layer taps at `taps` (row stride layer taps * dim) and, for a
    /// head with the final tap, the final states at `xc` (row stride dim), `rows` rows in all.
    pub fn absorbRound(eg: *Eagle, e: *const Engine, taps: u64, xc: u64, rows: usize, spans: []const Span) !void {
        const D = eg.s.dim;
        const L = eg.s.n_taps;
        if (!eg.s.final()) return eg.absorb(e, taps, L * D, spans);
        if (rows > chunk_rows) return error.EagleAbsorbTooLarge;
        const lt = L - 1;
        if (lt > 0) try e.ops.copyRows(e.s, taps, lt * D * 2, eg.tin, L * D * 2, lt * D * 2, rows);
        try e.ops.copyRows(e.s, xc, D * 2, eg.tin + lt * D * 2, L * D * 2, D * 2, rows);
        return eg.absorb(e, eg.tin, L * D, spans);
    }

    // -- the pass -------------------------------------------------------------------------------------------------

    /// Each stream's `steps` drafts and confidences (into drafts / confs): the real rows each one lacks, then the
    /// steps for all at once. A stream without the g row of q0 - 1 gets no drafts (confidence -30: survival ~0).
    pub fn pass(eg: *Eagle, e: *const Engine, reqs: []const Req, steps: usize) !void {
        const N = reqs.len;
        if (N == 0 or N > max_streams or steps == 0 or steps > max_steps) return error.BadDraftPass;
        const D = eg.s.dim;
        const W = eg.s.window;
        var ok: [max_streams]bool = @splat(false);
        var from: [max_streams]usize = undefined;
        // the extend's rows, every stream's in order, in chunks
        var c_ids: [chunk_rows]i64 = undefined;
        var c_row: [chunk_rows]i64 = undefined;
        var c_pos: [chunk_rows]i64 = undefined;
        var c_last: [max_streams]usize = undefined; // stream i's last row in the chunk (its x row)
        var c_who: [max_streams]usize = undefined;
        var nc: usize = 0;
        var nl: usize = 0;
        const cur: usize = 0;
        for (reqs, 0..) |rq, i| {
            if (rq.slot >= eg.slots or rq.q0 == 0) return error.BadSlot;
            const st = &eg.st[rq.slot];
            if (!st.live or rq.q0 > st.g_hi or rq.q0 <= st.g_lo) continue;
            var f = @min(st.ext, rq.q0 - 1);
            if (rq.q0 - f > W + 1) f = rq.q0 - (W + 1);
            if (f < st.g_lo) f = st.g_lo;
            if (f > st.ext) st.lo = @max(st.lo, f); // rows below f never got their keys: out of every window
            from[i] = f;
            ok[i] = true;
            for (f..rq.q0) |p| {
                if (nc == chunk_rows) {
                    try eg.extend(e, c_ids[0..nc], c_row[0..nc], c_pos[0..nc], c_last[0..nl], c_who[0..nl], cur);
                    nc = 0;
                    nl = 0;
                }
                const tok: i64 = if (p + 1 < rq.q0) rq.seq[p + 1] else rq.pending;
                c_ids[nc] = if (tok < 0) 0 else tok; // (an image position: no token embedding of its own)
                c_row[nc] = @intCast(rq.slot * eg.ring + p % eg.ring);
                c_pos[nc] = @intCast(p);
                nc += 1;
            }
            c_last[nl] = nc - 1;
            c_who[nl] = i;
            nl += 1;
        }
        if (nc > 0) try eg.extend(e, c_ids[0..nc], c_row[0..nc], c_pos[0..nc], c_last[0..nl], c_who[0..nl], cur);
        // the steps' tables: positions q0 - 1 + j - 1 (j from 1), ring rows, each stream's ring base and low
        var hp: [max_steps * max_streams]i64 = @splat(0);
        var hr: [max_steps * max_streams]i64 = @splat(0);
        var hb: [max_streams]i64 = @splat(0);
        var hl: [max_streams]i64 = @splat(0);
        for (reqs, 0..) |rq, i| {
            hb[i] = @intCast(rq.slot * eg.ring);
            hl[i] = @intCast(if (ok[i]) eg.st[rq.slot].lo else rq.q0 - 1);
            for (0..steps) |j| {
                const p = rq.q0 - 1 + j;
                hp[j * N + i] = @intCast(p);
                hr[j * N + i] = @intCast(if (ok[i]) rq.slot * eg.ring + p % eg.ring else eg.slots * eg.ring);
            }
        }
        try prompt.upload(e, eg.sp, &hp, steps * N * 8);
        try prompt.upload(e, eg.srow, &hr, steps * N * 8);
        try prompt.upload(e, eg.sbase, &hb, N * 8);
        try prompt.upload(e, eg.slo, &hl, N * 8);
        var x = eg.xs[cur];
        var prev = eg.xs[1 - cur];
        for (1..steps + 1) |j| {
            const pos = eg.sp + (j - 1) * N * 8;
            const rrow = eg.srow + (j - 1) * N * 8;
            if (j > 1) {
                std.mem.swap(u64, &x, &prev);
                try e.ops.gatherRows(e.s, e.w.embed, eg.s.vocab, eg.out + (j - 1) * N * 8, eg.ev, D * 2, N, eg.invalid);
                var fb = prev;
                if (eg.s.tap_norm) {
                    try eg.rms(e, prev, D, eg.fb_norm, eg.gv, D, N);
                    fb = eg.gv;
                }
                try eg.input(e, fb, x, N);
            }
            try eg.step(e, x, pos, rrow, N, j > 1);
            try e.ops.copyRows(e.s, x, D * 2, eg.dbg + (j - 1) * N * D * 2, D * 2, D * 2, N);
            try eg.rms(e, x, D, eg.out_norm, eg.h, D, N);
            try mm(e, eg.h, D, eg.down, eg.lr, eg.s.rank, false, N, D, eg.s.rank);
            try mm(e, eg.lr, eg.s.rank, eg.up, eg.lg, eg.s.vocab, true, N, eg.s.rank, eg.s.vocab);
            var a: cuda.Args = .{};
            a.add(eg.lg);
            a.add(@as(c_longlong, @intCast(eg.s.vocab)));
            a.add(@as(c_int, @intCast(eg.s.vocab)));
            a.add(eg.out + j * N * 8);
            a.add(eg.conf + (j - 1) * N * 4);
            a.add(eg.conf_a);
            a.add(eg.conf_b);
            try cuda.launch.launch(eg.f.top1, .{ .grid = .{ .x = @intCast(N) }, .block = .{ .x = 1024 } }, e.s, &a);
        }
        try e.s.synchronize();
        var ho: [(max_steps + 1) * max_streams]i64 = undefined;
        var hc: [max_steps * max_streams]f32 = undefined;
        try e.d.check(e.d.api.cuMemcpyDtoH_v2(&ho, eg.out, (steps + 1) * N * 8), "cuMemcpyDtoH");
        try e.d.check(e.d.api.cuMemcpyDtoH_v2(&hc, eg.conf, steps * N * 4), "cuMemcpyDtoH");
        for (reqs, 0..) |rq, i| {
            for (0..steps) |j| {
                eg.drafts[i][j] = if (ok[i]) ho[(j + 1) * N + i] else 0;
                eg.confs[i][j] = if (ok[i]) hc[j * N + i] else -30.0;
            }
            if (ok[i]) eg.st[rq.slot].ext = rq.q0;
        }
        eg.streams = N;
        eg.steps = steps;
        eg.passes += 1;
    }

    /// An extend chunk: the rows' layer inputs (token ids, g ring rows, positions), their keys and values into the
    /// rings, and each listed stream's last row's x into the steps' rows (xs[cur], row `who`).
    fn extend(eg: *Eagle, e: *const Engine, ids: []const i64, rows: []const i64, pos: []const i64, last: []const usize, who: []const usize, cur: usize) !void {
        const n = ids.len;
        const D = eg.s.dim;
        try prompt.upload(e, eg.ids, ids.ptr, n * 8);
        try prompt.upload(e, eg.rrow, rows.ptr, n * 8);
        try prompt.upload(e, eg.pos, pos.ptr, n * 8);
        try e.ops.gatherRows(e.s, e.w.embed, eg.s.vocab, eg.ids, eg.ev, D * 2, n, eg.invalid);
        try e.ops.gatherRows(e.s, eg.gring, eg.slots * eg.ring + 1, eg.rrow, eg.gv, D * 2, n, eg.invalid);
        try eg.input(e, eg.gv, eg.x, n);
        try eg.rms(e, eg.x, D, eg.attn_norm, eg.h, D, n);
        try eg.keysOut(e, eg.h, eg.pos, eg.rrow, n);
        for (last, who) |r, i| try e.ops.copyRows(e.s, eg.x + r * D * 2, D * 2, eg.xs[cur] + i * D * 2, D * 2, D * 2, 1);
    }

    /// One step of N rows on their residual rows x (in place: x becomes o): the attention (its keys and values first
    /// unless the extend wrote them: `keys`), then the MLP.
    fn step(eg: *Eagle, e: *const Engine, x: u64, pos: u64, rrow: u64, N: usize, keys: bool) !void {
        const s = eg.s;
        const D = s.dim;
        const hq = s.heads * s.head_dim;
        try eg.rms(e, x, D, eg.attn_norm, eg.h, D, N);
        try mm(e, eg.h, D, eg.wq, eg.q, hq, false, N, D, hq);
        try eg.rope(e, eg.q, hq, s.heads, pos, N);
        if (keys) try eg.keysOut(e, eg.h, pos, rrow, N);
        var a: cuda.Args = .{};
        a.add(eg.q);
        a.add(@as(c_longlong, @intCast(hq)));
        a.add(eg.kring);
        a.add(eg.vring);
        a.add(@as(c_int, @intCast(s.kv_heads)));
        a.add(eg.sbase);
        a.add(pos);
        a.add(eg.slo);
        a.add(@as(c_int, @intCast(eg.ring)));
        a.add(@as(c_int, @intCast(s.window)));
        a.add(@as(c_int, @intCast(s.heads)));
        a.add(1.0 / @sqrt(@as(f32, @floatFromInt(s.head_dim))));
        a.add(eg.att);
        a.add(@as(c_longlong, @intCast(hq)));
        try cuda.launch.launch(eg.f.attn, .{ .grid = .{ .x = @intCast(N), .y = @intCast(s.heads) }, .block = .{ .x = 128 } }, e.s, &a);
        try e.blas.linearDenseAdd(eg.att, hq, eg.wo, x, D, N, hq, D);
        try eg.rms(e, x, D, eg.mlp_norm, eg.h, D, N);
        try mm(e, eg.h, D, eg.w13, eg.gu, 2 * s.inter, false, N, D, 2 * s.inter);
        var b: cuda.Args = .{};
        b.add(eg.gu);
        b.add(@as(c_int, @intCast(s.inter)));
        b.add(eg.y);
        b.add(@as(c_longlong, @intCast(N * s.inter)));
        const count = N * s.inter;
        try cuda.launch.launch(eg.f.silu_mul, .{ .grid = .{ .x = @intCast(@min((count + 255) / 256, 65535)) }, .block = .{ .x = 256 } }, e.s, &b);
        try e.blas.linearDenseAdd(eg.y, s.inter, eg.w2, x, D, N, s.inter, D);
    }
};

/// A slot's bookkeeping after g rows of positions start .. start + n went into its ring: a gap (or a first window)
/// starts the slot over at `start`; rows rewritten below the extended position are extended again.
fn note(st: *Slot, start: usize, n: usize, ring: usize) void {
    if (!st.live or start > st.g_hi or start < st.g_lo) {
        st.* = .{ .live = true, .lo = start, .ext = start, .g_lo = start, .g_hi = start };
    } else if (start < st.ext) st.ext = start;
    st.g_hi = start + n;
    if (st.g_hi - st.g_lo > ring) st.g_lo = st.g_hi - ring;
}

test "a slot's rows: a first window, the next ones, a gap, a lag past the ring" {
    var st: Slot = .{};
    note(&st, 100, 28, 144); // a prompt's last 28 rows
    try std.testing.expectEqual(Slot{ .live = true, .lo = 100, .ext = 100, .g_lo = 100, .g_hi = 128 }, st);
    st.ext = 128; // a pass at q0 128
    note(&st, 128, 4, 144); // its round: the pending row and 3 drafts
    try std.testing.expectEqual(@as(usize, 128), st.ext);
    try std.testing.expectEqual(@as(usize, 132), st.g_hi);
    note(&st, 130, 6, 144); // 1 draft kept: the next window starts at 130, rows 130.. rewritten
    try std.testing.expectEqual(@as(usize, 128), st.ext); // (still 128: the pass there extends from 128)
    try std.testing.expectEqual(@as(usize, 136), st.g_hi);
    note(&st, 300, 2, 144); // a gap: starts over
    try std.testing.expectEqual(Slot{ .live = true, .lo = 300, .ext = 300, .g_lo = 300, .g_hi = 302 }, st);
    for (0..100) |k| note(&st, 302 + 2 * k, 2, 144);
    try std.testing.expectEqual(@as(usize, 502), st.g_hi);
    try std.testing.expectEqual(@as(usize, 502 - 144), st.g_lo);
}

// -- the self-test (TF_DS_EAGLE_CHECK=FILE: eagle_ref.py's vectors) -----------------------------------------------

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

/// The head against the reference's vectors in slot 0 (before serving; the slot is reset after): each case's rows
/// absorbed, one pass, each step's o, draft and top-1 probability against the file's. One JSON line a case on stderr
/// and a verdict; true when every step-1 o is within 3% (max abs diff over max abs) and every draft matches up to the
/// first step whose reference top-2 logit gap is under 0.05 (a rounding can take either token there).
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
