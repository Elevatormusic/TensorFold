//! The TFE draft head (TF_DS_DRAFTER=eagle) on rank 0, an EAGLE-style head: tools/dsv41-zig/eagle/README.md.
const std = @import("std");
const cuda = @import("cuda");
const prompt = @import("prompt.zig");

const file = @import("eagle_file.zig");

const Engine = prompt.Engine;
pub const Shape = file.Shape;
pub const Header = file.Header;
const Lin = file.Lin;
const Out = file.Out;
/// TF_DS_EAGLE_CHECK=FILE: the head against eagle_ref.py's vectors (eagle_check.zig).
pub const selfTest = @import("eagle_check.zig").selfTest;

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
/// The keys an attention row reads at most (eagle.cu's score buffer): a window of 511.
pub const max_keys = 512;
/// A two_stage head's candidates a row at most.
pub const max_cand = 512;
/// The rows of one own GEMV launch (eagle.cu gemv_rows); more rows run in launches of this many.
pub const gemv_rows = 16;
/// The parts of a low-rank row's top-1 (eagle.cu tf_eagle_top1_part_kernel).
pub const top1_parts = 64;
pub const ring_extra = 16;

/// Absorbed rows: `n` rows of positions start .. in `slot` (input row order); keep false: not drafting (no write).
pub const Span = struct { slot: usize, start: usize, n: usize, keep: bool = true };

/// A stream of a pass: its slot, its pending token at position q0, its ids by position (seq[p] for p < q0).
pub const Req = struct { slot: usize, q0: usize, pending: i64, seq: []const i32 };

/// What a slot's rings hold: g rows of positions [g_lo, g_hi), keys and values of the real rows below `ext` from `lo`.
const Slot = struct { live: bool = false, lo: usize = 0, ext: usize = 0, g_lo: usize = 0, g_hi: usize = 0 };

pub const Fns = struct {
    rmsnorm: cuda.Function,
    rope: cuda.Function,
    attn: cuda.Function,
    silu_mul: cuda.Function,
    top1: cuda.Function,
    topk: cuda.Function,
    rescore: cuda.Function,
    gemv_bf16: cuda.Function,
    gemv_fp8: cuda.Function,
    top1_part: cuda.Function,
    top1_join: cuda.Function,
};

pub const Eagle = struct {
    s: Shape,
    buf: cuda.DeviceBuffer, // the weights
    // the weights (row-major [out, in]; the norms bf16)
    fc_norm: u64,
    e_norm: u64,
    g_norm: u64,
    attn_norm: u64,
    mlp_norm: u64,
    out_norm: u64,
    fc: Lin,
    w_in: Lin,
    wq: Lin,
    wkv: Lin,
    wo: Lin,
    w13: Lin,
    w2: Lin,
    down: Lin,
    up: Lin,
    tap_norms: u64 = 0,
    fb_norm: u64 = 0,
    q_norm: u64 = 0,
    k_norm: u64 = 0,
    target: u64 = 0, // two_stage: the target head bf16 [vocab, dim]
    topk: usize = 0, // two_stage: the candidates a row (cand_k, or TF_DS_EAGLE_TOPK)
    mod: cuda.Module,
    f: Fns,
    slots: usize,
    ring: usize,
    // the rings: k, v bf16 [slots * ring + 1, kv_heads * 128], g bf16 [.., dim]; each one's last row is a sink row
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
    cand: u64, // i64 [max_streams, max_cand]: two_stage candidates
    ex: u64, // f32 [max_streams, max_cand]: their exact logits
    tpm: u64, // f32 [max_streams, top1_parts]: a low-rank top-1's parts (max, sum, id)
    tpz: u64,
    tpi: u64, // i32
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
    // TF_DS_EAGLE_GRAPH=1: the steps of a pass as one CUDA graph a (streams, steps)
    graphs: ?std.AutoHashMapUnmanaged(GraphKey, cuda.graph.Exec) = null,
    gpa: std.mem.Allocator = undefined,
    io: std.Io = undefined,
    // TF_DS_EAGLE_TAPDUMP: the file and its end
    dump: ?std.Io.File = null,
    dump_at: u64 = 0,

    const GraphKey = struct { streams: u32, steps: u32 };

    /// The head of `path` (eagle_file.zig).
    pub const open = file.open;

    pub fn close(eg: *Eagle, gpa: std.mem.Allocator) void {
        if (eg.graphs) |*g| {
            var it = g.valueIterator();
            while (it.next()) |x| x.deinit();
            g.deinit(gpa);
        }
        if (eg.dump) |f| f.close(eg.io);
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
        return eg.rmsOf(e, x, ldx, w, out, ldo, rows, eg.s.dim);
    }

    /// RMSNorm of `rows` rows of `d` features (rows ldx / ldo elements apart).
    fn rmsOf(eg: *const Eagle, e: *const Engine, x: u64, ldx: usize, w: u64, out: u64, ldo: usize, rows: usize, d: usize) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(c_longlong, @intCast(ldx)));
        a.add(w);
        a.add(out);
        a.add(@as(c_longlong, @intCast(ldo)));
        a.add(@as(c_int, @intCast(d)));
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

    /// x [rows, k] bf16 @ l^T into out (strides ldx, ldo) as `mode`: own GEMV (fp8; own bf16 to 16 rows), else cuBLAS.
    fn lin(eg: *const Eagle, e: *const Engine, x: u64, ldx: usize, l: Lin, out: u64, ldo: usize, mode: Out, rows: usize) !void {
        if (rows == 0) return;
        const eb: usize = if (l.fp8) 1 else 2;
        const fits = x % 16 == 0 and ldx % 8 == 0 and (l.k * eb) % 16 == 0 and l.w % 16 == 0;
        if (l.fp8 or (l.own and rows <= gemv_rows and fits)) {
            if (!fits) return error.EagleUnaligned;
            const chunks = l.k * eb / 16;
            var G: usize = 32;
            while (G > chunks) G /= 2;
            const per = 32 / G;
            const blocks = (l.n + 8 * per - 1) / (8 * per);
            const osz: usize = if (mode == .f32) 4 else 2;
            var r0: usize = 0;
            while (r0 < rows) : (r0 += gemv_rows) {
                var a: cuda.Args = .{};
                a.add(l.w);
                if (l.fp8) a.add(l.scale);
                a.add(@as(c_int, @intCast(l.n)));
                a.add(@as(c_int, @intCast(l.k)));
                a.add(x + r0 * ldx * 2);
                a.add(@as(c_longlong, @intCast(ldx)));
                a.add(@as(c_int, @intCast(@min(gemv_rows, rows - r0))));
                a.add(out + r0 * ldo * osz);
                a.add(@as(c_longlong, @intCast(ldo)));
                a.add(@backingInt(mode));
                a.add(@as(c_int, @intCast(G)));
                try cuda.launch.launch(if (l.fp8) eg.f.gemv_fp8 else eg.f.gemv_bf16, .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = 256 } }, e.s, &a);
            }
            return;
        }
        switch (mode) {
            .bf16, .f32 => try e.blas.linearDense(x, ldx, l.w, out, ldo, mode == .f32, rows, l.k, l.n),
            .add => try e.blas.linearDenseAdd(x, ldx, l.w, out, ldo, rows, l.k, l.n),
        }
    }

    /// Keys and values of `rows` normed rows h: wkv, k_norm (qk_norm), RoPE at `pos`, into ring rows `rrow` (device).
    fn keysOut(eg: *const Eagle, e: *const Engine, h: u64, pos: u64, rrow: u64, rows: usize) !void {
        const hd = eg.s.head_dim;
        const hkv = eg.s.kv_heads * hd;
        try eg.lin(e, h, eg.s.dim, eg.wkv, eg.kv, 2 * hkv, .bf16, rows);
        if (eg.s.qk_norm) for (0..eg.s.kv_heads) |kh| try eg.rmsOf(e, eg.kv + kh * hd * 2, 2 * hkv, eg.k_norm, eg.kv + kh * hd * 2, 2 * hkv, rows, hd);
        try eg.rope(e, eg.kv, 2 * hkv, eg.s.kv_heads, pos, rows);
        try e.ops.scatterRows(e.s, eg.kv, 2 * hkv * 2, rrow, eg.kring, hkv * 2, hkv * 2, rows);
        try e.ops.scatterRows(e.s, eg.kv + hkv * 2, 2 * hkv * 2, rrow, eg.vring, hkv * 2, hkv * 2, rows);
    }

    /// The layer input of `rows` rows: x = w_in [RMSNorm(embed rows ev) ; RMSNorm(g rows, stride dim)] into `out`.
    fn input(eg: *const Eagle, e: *const Engine, g: u64, out: u64, rows: usize) !void {
        const D = eg.s.dim;
        try eg.rms(e, eg.ev, D, eg.e_norm, eg.cat, 2 * D, rows);
        try eg.rms(e, g, D, eg.g_norm, eg.cat + D * 2, 2 * D, rows);
        try eg.lin(e, eg.cat, 2 * D, eg.w_in, out, D, .bf16, rows);
    }

    // -- absorbs --------------------------------------------------------------------------------------------------

    /// g = RMSNorm(fc(taps)) of `spans` (taps bf16 [rows, taps * dim], stride ldt) into each kept span's ring rows.
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
        try eg.lin(e, src, ld, eg.fc, eg.x, D, .bf16, total);
        try eg.rms(e, eg.x, D, eg.fc_norm, eg.h, D, total);
        try e.ops.scatterRows(e.s, eg.h, D * 2, eg.rrow, eg.gring, D * 2, D * 2, total);
        for (spans) |sp| if (sp.keep) note(&eg.st[sp.slot], sp.start, sp.n, eg.ring);
    }

    /// TF_DS_EAGLE_TAPDUMP: one record (u32 length, JSON header, bf16 tap rows); the format: eagle/README.md.
    pub fn tapDump(eg: *Eagle, e: *const Engine, kind: []const u8, slot: usize, start: usize, n: usize, cut: usize, src: u64, ld: usize) !void {
        const f = eg.dump orelse return;
        if (n == 0) return;
        const row = eg.s.n_taps * eg.s.dim * 2;
        try e.s.synchronize();
        const host = try eg.gpa.alloc(u8, n * row);
        defer eg.gpa.free(host);
        for (0..n) |r| try e.d.check(e.d.api.cuMemcpyDtoH_v2(host[r * row ..].ptr, src + r * ld * 2, row), "cuMemcpyDtoH");
        var hb: [256]u8 = undefined;
        const h = try std.fmt.bufPrint(&hb, "{{\"kind\": \"{s}\", \"slot\": {d}, \"start\": {d}, \"n\": {d}, \"cut\": {d}, \"taps\": {d}, \"dim\": {d}}}", .{ kind, slot, start, n, cut, eg.s.n_taps, eg.s.dim });
        var len: [4]u8 = undefined;
        std.mem.writeInt(u32, &len, @intCast(h.len), .little);
        try f.writePositionalAll(eg.io, &len, eg.dump_at);
        try f.writePositionalAll(eg.io, h, eg.dump_at + 4);
        try f.writePositionalAll(eg.io, host, eg.dump_at + 4 + h.len);
        eg.dump_at += 4 + h.len + host.len;
    }

    /// A round's windows (lanes.zig verify): layer taps at `taps` and, for a final-tap head, final states at `xc`.
    pub fn absorbRound(eg: *Eagle, e: *const Engine, taps: u64, xc: u64, rows: usize, spans: []const Span) !void {
        const D = eg.s.dim;
        const L = eg.s.n_taps;
        var src = taps;
        if (eg.s.final()) {
            if (rows > chunk_rows) return error.EagleAbsorbTooLarge;
            const lt = L - 1;
            if (lt > 0) try e.ops.copyRows(e.s, taps, lt * D * 2, eg.tin, L * D * 2, lt * D * 2, rows);
            try e.ops.copyRows(e.s, xc, D * 2, eg.tin + lt * D * 2, L * D * 2, D * 2, rows);
            src = eg.tin;
        }
        if (eg.dump != null) {
            var r: usize = 0;
            for (spans) |sp| {
                if (sp.keep) try eg.tapDump(e, "round", sp.slot, sp.start, sp.n, 0, src + r * L * D * 2, L * D);
                r += sp.n;
            }
        }
        return eg.absorb(e, src, L * D, spans);
    }

    // -- the pass -------------------------------------------------------------------------------------------------

    /// Each stream's `steps` drafts and confidences; a stream without the g row of q0 - 1 gets none (confidence -30).
    pub fn pass(eg: *Eagle, e: *const Engine, reqs: []const Req, steps: usize) !void {
        const N = reqs.len;
        if (N == 0 or N > max_streams or steps == 0 or steps > max_steps) return error.BadDraftPass;
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
        try eg.chain(e, N, steps, cur);
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

    /// A pass's steps for N streams: eager, or with TF_DS_EAGLE_GRAPH one CUDA graph a (N, steps) (fixed buffers).
    fn chain(eg: *Eagle, e: *const Engine, N: usize, steps: usize, cur: usize) !void {
        const g = if (eg.graphs) |*m| m else return eg.chainBody(e, N, steps, cur);
        const key: GraphKey = .{ .streams = @intCast(N), .steps = @intCast(steps) };
        if (g.getPtr(key)) |x| return x.launchOn(e.s);
        try cuda.graph.beginCapture(e.s, .thread_local);
        eg.chainBody(e, N, steps, cur) catch |err| {
            if (cuda.graph.endCapture(e.s)) |got| {
                var x = got;
                x.deinit();
            } else |_| {}
            return err;
        };
        var graph = try cuda.graph.endCapture(e.s);
        defer graph.deinit();
        var exec = try graph.instantiate();
        errdefer exec.deinit();
        try g.put(eg.gpa, key, exec);
        try exec.launchOn(e.s);
    }

    fn chainBody(eg: *Eagle, e: *const Engine, N: usize, steps: usize, cur: usize) !void {
        const D = eg.s.dim;
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
            try eg.lin(e, eg.h, D, eg.down, eg.lr, eg.s.rank, .bf16, N);
            try eg.lin(e, eg.lr, eg.s.rank, eg.up, eg.lg, eg.s.vocab, .f32, N);
            if (eg.topk > 0) {
                try eg.rescoreTop(e, N);
                var a: cuda.Args = .{};
                a.add(eg.ex);
                a.add(@as(c_longlong, max_cand));
                a.add(@as(c_int, @intCast(eg.topk)));
                a.add(eg.cand);
                a.add(@as(c_longlong, max_cand));
                a.add(eg.out + j * N * 8);
                a.add(eg.conf + (j - 1) * N * 4);
                a.add(eg.conf_a);
                a.add(eg.conf_b);
                try cuda.launch.launch(eg.f.top1, .{ .grid = .{ .x = @intCast(N) }, .block = .{ .x = 1024 } }, e.s, &a);
            } else {
                // the vocabulary's top-1 in top1_parts parts a row, then joined
                var a: cuda.Args = .{};
                a.add(eg.lg);
                a.add(@as(c_longlong, @intCast(eg.s.vocab)));
                a.add(@as(c_int, @intCast(eg.s.vocab)));
                a.add(eg.tpm);
                a.add(eg.tpz);
                a.add(eg.tpi);
                try cuda.launch.launch(eg.f.top1_part, .{ .grid = .{ .x = @intCast(N), .y = top1_parts }, .block = .{ .x = 256 } }, e.s, &a);
                var b: cuda.Args = .{};
                b.add(eg.tpm);
                b.add(eg.tpz);
                b.add(eg.tpi);
                b.add(@as(c_int, top1_parts));
                b.add(eg.out + j * N * 8);
                b.add(eg.conf + (j - 1) * N * 4);
                b.add(eg.conf_a);
                b.add(eg.conf_b);
                try cuda.launch.launch(eg.f.top1_join, .{ .grid = .{ .x = @intCast(N) }, .block = .{ .x = 32 } }, e.s, &b);
            }
        }
    }

    /// two_stage: each row's top `topk` low-rank logits (lg) as candidates, then their exact logits (ex) on rows h.
    fn rescoreTop(eg: *Eagle, e: *const Engine, N: usize) !void {
        var a: cuda.Args = .{};
        a.add(eg.lg);
        a.add(@as(c_longlong, @intCast(eg.s.vocab)));
        a.add(@as(c_int, @intCast(eg.s.vocab)));
        a.add(@as(c_int, @intCast(eg.topk)));
        a.add(eg.cand);
        a.add(@as(c_longlong, max_cand));
        try cuda.launch.launch(eg.f.topk, .{ .grid = .{ .x = @intCast(N) }, .block = .{ .x = 1024 } }, e.s, &a);
        var b: cuda.Args = .{};
        b.add(eg.target);
        b.add(@as(c_int, @intCast(eg.s.dim)));
        b.add(eg.h);
        b.add(@as(c_longlong, @intCast(eg.s.dim)));
        b.add(eg.cand);
        b.add(@as(c_longlong, max_cand));
        b.add(@as(c_int, @intCast(eg.topk)));
        b.add(eg.ex);
        b.add(@as(c_longlong, max_cand));
        try cuda.launch.launch(eg.f.rescore, .{ .grid = .{ .x = @intCast(N), .y = @intCast((eg.topk + 7) / 8) }, .block = .{ .x = 256 }, .shared = @intCast(eg.s.dim * 4) }, e.s, &b);
    }

    /// An extend chunk: the rows' layer inputs, their keys and values into the rings, last rows' x into xs[cur].
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

    /// One step of N residual rows x (x becomes o): attention (keys and values first if `keys`), then the MLP.
    fn step(eg: *Eagle, e: *const Engine, x: u64, pos: u64, rrow: u64, N: usize, keys: bool) !void {
        const s = eg.s;
        const D = s.dim;
        const hq = s.heads * s.head_dim;
        try eg.rms(e, x, D, eg.attn_norm, eg.h, D, N);
        try eg.lin(e, eg.h, D, eg.wq, eg.q, hq, .bf16, N);
        if (s.qk_norm) try eg.rmsOf(e, eg.q, s.head_dim, eg.q_norm, eg.q, s.head_dim, N * s.heads, s.head_dim);
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
        try eg.lin(e, eg.att, hq, eg.wo, x, D, .add, N);
        try eg.rms(e, x, D, eg.mlp_norm, eg.h, D, N);
        try eg.lin(e, eg.h, D, eg.w13, eg.gu, 2 * s.inter, .bf16, N);
        var b: cuda.Args = .{};
        b.add(eg.gu);
        b.add(@as(c_int, @intCast(s.inter)));
        b.add(eg.y);
        b.add(@as(c_longlong, @intCast(N * s.inter)));
        const count = N * s.inter;
        try cuda.launch.launch(eg.f.silu_mul, .{ .grid = .{ .x = @intCast(@min((count + 255) / 256, 65535)) }, .block = .{ .x = 256 } }, e.s, &b);
        try eg.lin(e, eg.y, s.inter, eg.w2, x, D, .add, N);
    }
};

/// A slot after g rows start .. start + n: a gap (or a first window) starts it over; rewritten rows extend again.
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
