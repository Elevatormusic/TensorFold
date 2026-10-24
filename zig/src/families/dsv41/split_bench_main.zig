//! tf-dsv41-split-bench AOT_DIR [N ...]: TF_DS_IDX_SPLIT (idx_split.zig) against the served keyed indexer (prompt.zig
//! indexer: index_score's keys and tile maxima, topk_select_pruned, in the served row blocks) on synthetic FP4 index
//! keys, queries and head weights, a chunk of R rows (TF_SPLIT_BENCH_ROWS, else 2048) ending at N compressed entries
//! (ratio 2: row r sees (2N - R + r + 1) / 2 of them). Each node's part of a 2- and a 4-node split runs on this one GPU (its columns'
//! scores, their k largest keys, moved to the row's columns), then the merge, and every output byte is compared with
//! the served selection's, on those rows and on rows seeing 1 to N columns. The keys and tile maxima the served scores
//! write past vis are checked against -inf's keys too (the merge keeps the served keys there word for word). Timed on
//! the whole chunk: served, each node's part and the merge; the all-gather between them is the lane's to time (NCCL).
//! A row block's vis goes to the kernels 16-byte aligned, as prompt.zig passes it since TF #4 (the view, else a copy);
//! TF_SPLIT_BENCH_VIEWS=1 passes the served path the views, as before (a short block at an odd row then finds no
//! captured _index_score variant unless N and its tile count are multiples of 16). One JSON line an N; exit 1 on any
//! difference.
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const ops = dsv41.ops;
const tri = dsv41.tri;
const tri_index = dsv41.tri_index;
const tri_attn = dsv41.tri_attn;
const idx_split = dsv41.idx_split;

const ih = 32; // index_n_heads
const id = 128; // index_head_dim
const k = 512; // index_topk
const max_rows = 2048; // a prompt chunk at most
/// The chunk's rows (TF_SPLIT_BENCH_ROWS, else a whole chunk).
var rows: usize = max_rows;
/// TF_SPLIT_BENCH_VIEWS=1: the served path's row blocks pass views of vis (prompt.zig before TF #4).
var views = false;
const key_elems: usize = 1 << 24; // prompt.zig's bound on a row block's keys
const pruned_rows: usize = 512; // prompt.zig's: topk_select_pruned's buffers' rows

const Bench = struct { o: *const ops.Ops, s: cuda.Stream };

/// topk_select's torch step (prompt.torchTopk without an Engine).
fn torchTopk(ctx: ?*anyopaque, _: tri.Tri, keys: u64, ks: usize, r: usize, n: usize, kk: usize, top: u64) anyerror!void {
    const b: *const Bench = @ptrCast(@alignCast(ctx.?));
    try b.o.topkI64(b.s, keys, ks, r, n, kk, top);
}

/// index_score's key of score x at column j (dsv41_ops.cu tf_ds_score_key): x's bits (-0 as +0) ordered as signed
/// integers, above the inverted column.
fn scoreKey(x: f32, j: usize) i64 {
    var bits: u32 = @bitCast(x);
    if (bits == 0x80000000) bits = 0;
    const sb: i32 = @bitCast(bits);
    const ordered: i32 = if (sb < 0) sb ^ 0x7FFFFFFF else sb;
    const hi: u64 = @as(u64, @as(u32, @bitCast(ordered))) << 32;
    return @bitCast(hi | @as(u64, 0xFFFFFFFF - @as(u32, @intCast(j))));
}

fn bf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @truncate((b +% 0x7FFF +% ((b >> 16) & 1)) >> 16);
}

/// The rows' visible columns: the lane's (a chunk ending at n, ratio 2) or every count from 1 to n.
const Pattern = enum { lane, wide };

fn visOf(p: Pattern, n: usize, r: usize) usize {
    return switch (p) {
        .lane => (2 * n - rows + r + 1) / 2,
        .wide => 1 + r * (n - 1) / (rows - 1),
    };
}

/// nvis's row of node g of a world's split: the 2-node split's two, then the 4-node split's four.
fn slot(world: usize, g: usize) usize {
    return (if (world == 2) @as(usize, 0) else 2) + g;
}

/// One N's device state.
const Run = struct {
    t: tri.Tri,
    o: *const ops.Ops,
    s: cuda.Stream,
    top: tri_index.TopK,
    n: usize,
    rb: usize,
    kx: tri_index.IndexK,
    q: u64,
    w: u64,
    vis: u64,
    nvis: u64,
    keys: u64,
    tmax: u64,
    bufs: tri_index.PrunedBufs,
    parts: u64,
    merged: u64,
    vblk: u64,

    /// The served keyed layer (prompt.zig indexer, keyed) -> out [rows, k].
    fn served(r: *const Run, out: u64) !void {
        var r0: usize = 0;
        while (r0 < rows) : (r0 += r.rb) {
            const m = @min(rows, r0 + r.rb) - r0;
            const vr = try r.visBlock(r0, m);
            try tri_index.indexScore(r.t, r.q + r0 * ih * id * 2, r.kx, r.w + r0 * ih * 2, vr, r.n, r.keys, null, true, r.tmax, false, m, ih, id);
            try tri_index.topkSelectPruned(r.t, r.keys, r.n, r.tmax, k, vr, out + r0 * k * 8, r.bufs, r.top, m, r.n);
        }
    }

    /// vis[r0..][0..m] as prompt.zig visBlock passes it: the view when 16-byte aligned, else a copy (with
    /// TF_SPLIT_BENCH_VIEWS=1 the view).
    fn visBlock(r: *const Run, r0: usize, m: usize) !u64 {
        const view = r.vis + r0 * 8;
        if (views or view % 16 == 0) return view;
        try r.o.copyRows(r.s, view, m * 8, r.vblk, m * 8, m * 8, 1);
        return r.vblk;
    }

    /// Node g of a world's part (prompt.zig splitKeyed before the all-gather) into its rows of parts.
    fn node(r: *const Run, world: usize, g: usize) !void {
        const sp = idx_split.span(r.n, world, g);
        try idx_split.nodeTop(r.t, r.o, r.s, r.q, r.kx, r.w, r.nvis + slot(world, g) * rows * 8, r.vblk, sp, r.keys, r.tmax, r.bufs, r.top, r.parts + g * rows * k * 8, rows, r.rb, k, ih, id);
    }

    fn merge(r: *const Run, world: usize, out: u64) !void {
        try idx_split.merge(r.t, r.o, r.s, r.parts, r.merged, world, rows, k, r.vis, out, r.top);
    }

    /// The split on one GPU: every node's part where the all-gather would land it, then the merge.
    fn split(r: *const Run, world: usize, out: u64) !void {
        for (0..world) |g| try r.node(world, g);
        try r.merge(world, out);
    }
};

/// The rows' vis and every node's (both splits) for a pattern.
fn setVis(gpa: std.mem.Allocator, p: Pattern, n: usize, vis: *cuda.DeviceBuffer, nvis: *cuda.DeviceBuffer) !void {
    var hv: [max_rows]i64 = undefined;
    for (hv[0..rows], 0..) |*x, r| x.* = @intCast(visOf(p, n, r));
    try vis.upload(0, std.mem.sliceAsBytes(hv[0..rows]));
    const hn = try gpa.alloc(i64, 6 * rows);
    defer gpa.free(hn);
    for ([_]usize{ 2, 4 }) |world| {
        for (0..world) |g| {
            const sp = idx_split.span(n, world, g);
            for (0..rows) |r| hn[slot(world, g) * rows + r] = idx_split.nodeVis(visOf(p, n, r), sp);
        }
    }
    try nvis.upload(0, std.mem.sliceAsBytes(hn));
}

/// Rows of a and b ([rows, k] int64 on the device) that differ, and the first.
const Diff = struct { rows: usize = 0, first: ?usize = null };

fn compare(gpa: std.mem.Allocator, a: *cuda.DeviceBuffer, b: *cuda.DeviceBuffer) !Diff {
    const h = try gpa.alloc(i64, 2 * rows * k);
    defer gpa.free(h);
    try a.download(0, std.mem.sliceAsBytes(h[0 .. rows * k]));
    try b.download(0, std.mem.sliceAsBytes(h[rows * k ..]));
    var d: Diff = .{};
    for (0..rows) |r| {
        if (!std.mem.eql(i64, h[r * k ..][0..k], h[rows * k + r * k ..][0..k])) {
            d.rows += 1;
            if (d.first == null) d.first = r;
        }
    }
    return d;
}

/// The served scores' keys and tile maxima past vis in a pattern's first row block (its first 16 rows): -inf's keys
/// at their columns (a tile's maximum: its first column's), or the first other.
const Dead = struct { keys: usize = 0, keys_other: usize = 0, tiles: usize = 0, tiles_other: usize = 0, first: ?[]const u8 = null };

fn deadKeys(a: std.mem.Allocator, gpa: std.mem.Allocator, r: *const Run, p: Pattern, keys: *cuda.DeviceBuffer, tmax: *cuda.DeviceBuffer, d: *Dead) !void {
    const m = @min(rows, r.rb);
    const mr = @min(m, 16);
    const nt: usize = tri.cdiv(r.n, tri_index.tile);
    try tri_index.indexScore(r.t, r.q, r.kx, r.w, r.vis, r.n, r.keys, null, true, r.tmax, false, m, ih, id);
    try r.s.synchronize();
    const hk = try gpa.alloc(i64, mr * r.n);
    defer gpa.free(hk);
    try keys.download(0, std.mem.sliceAsBytes(hk));
    const hm = try gpa.alloc(i64, mr * nt);
    defer gpa.free(hm);
    try tmax.download(0, std.mem.sliceAsBytes(hm));
    const ninf = -std.math.inf(f32);
    for (0..mr) |i| {
        const v = visOf(p, r.n, i);
        for (v..r.n) |j| {
            const key = hk[i * r.n + j];
            if (key == scoreKey(ninf, j)) {
                d.keys += 1;
            } else {
                d.keys_other += 1;
                if (d.first == null) d.first = try std.fmt.allocPrint(a, "{s} row {d} col {d} key 0x{x:0>16}", .{ @tagName(p), i, j, @as(u64, @bitCast(key)) });
            }
        }
        for (0..nt) |tt| {
            if (tt * tri_index.tile < v) continue;
            const x = hm[i * nt + tt];
            if (x == scoreKey(ninf, tt * tri_index.tile)) {
                d.tiles += 1;
            } else {
                d.tiles_other += 1;
                if (d.first == null) d.first = try std.fmt.allocPrint(a, "{s} row {d} tile {d} max 0x{x:0>16}", .{ @tagName(p), i, tt, @as(u64, @bitCast(x)) });
            }
        }
    }
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 2) {
        std.debug.print("usage: tf-dsv41-split-bench AOT_DIR [N ...]\n", .{});
        return 2;
    }
    if (std.c.getenv("TF_SPLIT_BENCH_ROWS")) |v| rows = try std.fmt.parseInt(usize, std.mem.span(v), 10);
    if (rows < 2 or rows > max_rows) {
        std.debug.print("TF_SPLIT_BENCH_ROWS: 2 .. {d}\n", .{max_rows});
        return 2;
    }
    if (std.c.getenv("TF_SPLIT_BENCH_VIEWS")) |v| views = std.mem.eql(u8, std.mem.span(v), "1");
    var sizes: std.ArrayList(usize) = .empty;
    for (args[2..]) |s| try sizes.append(a, try std.fmt.parseInt(usize, s, 10));
    if (sizes.items.len == 0) try sizes.appendSlice(a, &.{ 32768, 40000, 65536, 131072, 262144, 300001, 524288 });

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var set = try cuda.aot.Set.load(gpa, io, &driver, ctx.device, args[1]);
    defer set.deinit();
    const o = try ops.Ops.load(&driver, cuda.kernels.torch_pointwise, cuda.kernels.torch_movement, cuda.kernels.dsv41_ops);
    const t: tri.Tri = .{ .set = &set, .stream = stream };
    var bench: Bench = .{ .o = &o, .s = stream };
    var start = try cuda.Event.init(&driver, true);
    defer start.deinit();
    var end = try cuda.Event.init(&driver, true);
    defer end.deinit();
    var out_buf: [4096]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &out_buf);
    const wr = &out.interface;
    var prng = std.Random.DefaultPrng.init(0x1017_5b17);
    const rng = prng.random();
    var all_equal = true;

    // the queries (fp4_qd's bytes of normal bf16) and head weights, shared by every N
    var qx = try cuda.DeviceBuffer.alloc(&driver, max_rows * ih * id * 2);
    defer qx.free();
    var q = try cuda.DeviceBuffer.alloc(&driver, max_rows * ih * id * 2);
    defer q.free();
    var w = try cuda.DeviceBuffer.alloc(&driver, max_rows * ih * 2);
    defer w.free();
    {
        const hq = try gpa.alloc(u16, max_rows * ih * id);
        defer gpa.free(hq);
        for (hq) |*x| x.* = bf16(rng.floatNorm(f32));
        try qx.upload(0, std.mem.sliceAsBytes(hq));
        try tri_attn.fp4QdP2(t, qx.ptr, q.ptr, max_rows * ih * id);
        try stream.synchronize();
        const hw = try gpa.alloc(u16, max_rows * ih);
        defer gpa.free(hw);
        for (hw) |*x| x.* = bf16(rng.floatNorm(f32) * 0.05);
        try w.upload(0, std.mem.sliceAsBytes(hw));
    }
    var every = try cuda.DeviceBuffer.alloc(&driver, max_rows * 8);
    defer every.free();
    {
        var ev: [max_rows]i64 = undefined;
        @memset(&ev, tri_index.every_vis);
        try every.upload(0, std.mem.sliceAsBytes(ev[0..]));
    }
    var vis = try cuda.DeviceBuffer.alloc(&driver, max_rows * 8);
    defer vis.free();
    var nvis = try cuda.DeviceBuffer.alloc(&driver, 6 * max_rows * 8);
    defer nvis.free();
    var ktop = try cuda.DeviceBuffer.alloc(&driver, max_rows * k * 8);
    defer ktop.free();
    var tpos = try cuda.DeviceBuffer.alloc(&driver, pruned_rows * k * 8);
    defer tpos.free();
    var cand = try cuda.DeviceBuffer.alloc(&driver, pruned_rows * k * tri_index.tile * 8);
    defer cand.free();
    var parts = try cuda.DeviceBuffer.alloc(&driver, 4 * max_rows * k * 8);
    defer parts.free();
    var merged = try cuda.DeviceBuffer.alloc(&driver, 4 * max_rows * k * 8);
    defer merged.free();
    var vblk = try cuda.DeviceBuffer.alloc(&driver, max_rows * 8);
    defer vblk.free();
    var got: [2]cuda.DeviceBuffer = undefined; // served, split
    for (&got) |*g| g.* = try cuda.DeviceBuffer.alloc(&driver, max_rows * k * 8);
    defer {
        for (&got) |*g| g.free();
    }
    const top: tri_index.TopK = .{ .top = ktop.ptr, .ctx = &bench, .run = torchTopk };
    const bufs: tri_index.PrunedBufs = .{ .tpos = tpos.ptr, .cand = cand.ptr, .every = every.ptr };

    for (sizes.items) |n| {
        // prompt.zig's row blocks, and the split's bounds (every node at least 2 k columns, the gathered keys in key_cap)
        const rb: usize = @max(16, @min(rows, (@as(usize, 1) << 26) / (4 * n)));
        if (rb * n > key_elems or n < 2 * 4 * k or (tri_index.prunes(n, k) and rb > pruned_rows)) {
            try wr.print("{{\"n\": {d}, \"skipped\": true}}\n", .{n});
            try wr.flush();
            continue;
        }
        const nt: usize = tri.cdiv(n, tri_index.tile);
        var codes = try cuda.DeviceBuffer.alloc(&driver, n * id / 2);
        defer codes.free();
        var scales = try cuda.DeviceBuffer.alloc(&driver, n * id / 32);
        defer scales.free();
        var keys = try cuda.DeviceBuffer.alloc(&driver, rb * n * 8);
        defer keys.free();
        var tmax = try cuda.DeviceBuffer.alloc(&driver, (rb * nt + rows) * 8);
        defer tmax.free();
        {
            const hc = try gpa.alloc(u8, n * id / 2);
            defer gpa.free(hc);
            rng.bytes(hc);
            try codes.upload(0, hc);
            const hs = try gpa.alloc(u8, n * id / 32);
            defer gpa.free(hs);
            for (hs) |*x| x.* = 120 + rng.uintLessThan(u8, 8); // E8M0: 2^-7 .. 1
            try scales.upload(0, hs);
        }
        const run: Run = .{
            .t = t,
            .o = &o,
            .s = stream,
            .top = top,
            .n = n,
            .rb = rb,
            .kx = .{ .fp4 = .{ .codes = codes.ptr, .scales = scales.ptr } },
            .q = q.ptr,
            .w = w.ptr,
            .vis = vis.ptr,
            .nvis = nvis.ptr,
            .keys = keys.ptr,
            .tmax = tmax.ptr,
            .bufs = bufs,
            .parts = parts.ptr,
            .merged = merged.ptr,
            .vblk = vblk.ptr,
        };

        // every output byte, both splits, both patterns; the served keys past vis
        var diff: [2][2]Diff = undefined; // [pattern][2 nodes, 4 nodes]
        var dead: Dead = .{};
        for ([_]Pattern{ .wide, .lane }, 0..) |p, pi| {
            try setVis(gpa, p, n, &vis, &nvis);
            try deadKeys(a, gpa, &run, p, &keys, &tmax, &dead);
            for ([_]usize{ 2, 4 }, 0..) |world, wi| {
                for (&got, 0..) |*g, i| try g.fill8(@intCast(0x11 * (i + 1)), stream.handle);
                try run.served(got[0].ptr);
                try run.split(world, got[1].ptr);
                try stream.synchronize();
                diff[pi][wi] = try compare(gpa, &got[0], &got[1]);
                if (diff[pi][wi].rows != 0) all_equal = false;
            }
        }

        // the chunk timed on the lane's rows: served, each node's part of either split, the merges
        var ms: [9]f64 = undefined; // served, 2 nodes' parts, 4 nodes' parts, merge of 2, merge of 4
        for (0..9) |which| {
            const reps = 2;
            for (0..reps + 1) |rep| {
                if (rep == 1) try start.record(stream);
                switch (which) {
                    0 => try run.served(got[0].ptr),
                    1, 2 => try run.node(2, which - 1),
                    3, 4, 5, 6 => try run.node(4, which - 3),
                    7 => try run.merge(2, got[1].ptr),
                    else => try run.merge(4, got[1].ptr),
                }
            }
            try end.record(stream);
            try end.synchronize();
            ms[which] = @as(f64, try cuda.Event.elapsedMs(start, end)) / reps;
        }
        const slow2 = @max(ms[1], ms[2]);
        const slow4 = @max(@max(ms[3], ms[4]), @max(ms[5], ms[6]));

        try wr.print("{{\"n\": {d}, \"rows\": {d}, \"rb\": {d}, \"last_block_rows\": {d}, \"vis_views\": {}", .{ n, rows, rb, rows - (rows - 1) / rb * rb, views });
        for ([_][]const u8{ "wide", "lane" }, 0..) |name, pi| {
            for ([_]usize{ 2, 4 }, 0..) |world, wi| {
                const d = diff[pi][wi];
                try wr.print(", \"rows_differ_{s}_{d}\": {d}", .{ name, world, d.rows });
                if (d.first) |f| try wr.print(", \"first_differ_{s}_{d}\": {d}", .{ name, world, f });
            }
        }
        try wr.print(", \"dead_keys_minf\": {d}, \"dead_keys_other\": {d}, \"dead_tiles_minf\": {d}, \"dead_tiles_other\": {d}, \"first_dead_other\": ", .{ dead.keys, dead.keys_other, dead.tiles, dead.tiles_other });
        if (dead.first) |f| try wr.print("\"{s}\"", .{f}) else try wr.print("null", .{});
        try wr.print(", \"ms_chunk\": {{\"served\": {d:.2}, \"node2\": [{d:.2}, {d:.2}], \"node4\": [{d:.2}, {d:.2}, {d:.2}, {d:.2}], \"merge2\": {d:.2}, \"merge4\": {d:.2}, \"split2_less_gather\": {d:.2}, \"split4_less_gather\": {d:.2}}}}}\n", .{ ms[0], ms[1], ms[2], ms[3], ms[4], ms[5], ms[6], ms[7], ms[8], slow2 + ms[7], slow4 + ms[8] });
        try wr.flush();
    }
    return if (all_equal) 0 else 1;
}
