//! The checkpoint's tensors as the split takes them, by the loader's keys (plan.zig): whole, by input rows or by
//! output columns, in the dtypes the Python loader keeps, so each entry's bytes equal the lane's rank cache entry.
const std = @import("std");
const core = @import("core");
const st = core.safetensors;
const DType = @import("rank_cache.zig").DType;
const Config = @import("config.zig").Config;
const plan = @import("plan.zig");

/// One planned entry's layout once read.
pub const Slice = struct {
    dtype: DType,
    rank: u8,
    shape: [st.max_rank]usize,
    bytes: usize,
};

/// Every shard of a checkpoint folder, mapped read-only, with its tensors by name.
pub const Shards = struct {
    gpa: std.mem.Allocator,
    files: []st.File,
    names: std.StringHashMapUnmanaged(u32) = .empty,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Shards {
        var d = try std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
        defer d.close(io);
        var paths: std.ArrayList([]const u8) = .empty;
        defer {
            for (paths.items) |p| gpa.free(p);
            paths.deinit(gpa);
        }
        var it = d.iterate();
        while (try it.next(io)) |e| if (std.mem.endsWith(u8, e.name, ".safetensors")) {
            try paths.append(gpa, try std.fs.path.join(gpa, &.{ dir, e.name }));
        };
        std.mem.sort([]const u8, paths.items, {}, struct {
            fn lt(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lt);
        var sh: Shards = .{ .gpa = gpa, .files = try gpa.alloc(st.File, paths.items.len) };
        var n: usize = 0;
        errdefer {
            for (sh.files[0..n]) |*f| f.close(io);
            gpa.free(sh.files);
            sh.names.deinit(gpa);
        }
        for (paths.items) |p| {
            sh.files[n] = try st.File.open(gpa, io, p);
            n += 1;
            var names = sh.files[n - 1].names.iterator();
            while (names.next()) |kv| {
                const slot = try sh.names.getOrPut(gpa, kv.key_ptr.*);
                if (slot.found_existing) {
                    // the 1017 packs repeat the Engram q / k tables in two shards with the same dtype and shape: the
                    // later shard is used, as before. A name in two shards with another dtype or shape (for example
                    // a dtype-override file next to the pack) has no single source.
                    const was = sh.files[slot.value_ptr.*].get(kv.key_ptr.*).?;
                    const now = sh.files[n - 1].get(kv.key_ptr.*).?;
                    if (was.dtype != now.dtype or was.rank != now.rank or !std.mem.eql(usize, was.shape[0..was.rank], now.shape[0..now.rank])) {
                        std.log.warn("tensor {s} is in two shards with another dtype or shape: {s} and {s}", .{ kv.key_ptr.*, paths.items[slot.value_ptr.*], p });
                        return error.DuplicateTensor;
                    }
                }
                slot.value_ptr.* = @intCast(n - 1);
            }
        }
        return sh;
    }

    pub fn close(sh: *Shards, io: std.Io) void {
        for (sh.files) |*f| f.close(io);
        sh.gpa.free(sh.files);
        sh.names.deinit(sh.gpa);
        sh.* = undefined;
    }

    pub fn get(sh: *const Shards, name: []const u8) ?st.Tensor {
        const i = sh.names.get(name) orelse return null;
        return sh.files[i].get(name);
    }
};

/// A loader key split into its fields: "name|dtype" (plain) or "prefix|cols|rows|part" (an EXL3 part).
const Key = struct {
    name: []const u8,
    dtype: ?DType = null,
    exl3: bool = false,
    cols: ?[2]usize = null,
    rows: ?[2]usize = null,
    part: []const u8 = "",

    fn parse(key: []const u8) !Key {
        var f: [4][]const u8 = undefined;
        var n: usize = 0;
        var it = std.mem.splitScalar(u8, key, '|');
        while (it.next()) |x| : (n += 1) {
            if (n == 4) return error.BadKey;
            f[n] = x;
        }
        if (n == 2) {
            const d: ?DType = if (std.mem.eql(u8, f[1], "None")) null else if (std.mem.eql(u8, f[1], "torch.float32")) .f32 else if (std.mem.eql(u8, f[1], "torch.float16")) .f16 else return error.BadKey;
            return .{ .name = f[0], .dtype = d };
        }
        if (n != 4) return error.BadKey;
        return .{ .name = f[0], .exl3 = true, .cols = try range(f[1]), .rows = try range(f[2]), .part = f[3] };
    }

    /// "None" or Python's "(a, b)".
    fn range(text: []const u8) !?[2]usize {
        if (std.mem.eql(u8, text, "None")) return null;
        if (text.len < 5 or text[0] != '(' or text[text.len - 1] != ')') return error.BadKey;
        var it = std.mem.splitSequence(u8, text[1 .. text.len - 1], ", ");
        const a = try std.fmt.parseInt(usize, it.next() orelse return error.BadKey, 10);
        const b = try std.fmt.parseInt(usize, it.next() orelse return error.BadKey, 10);
        if (it.next() != null or b < a) return error.BadKey;
        return .{ a, b };
    }
};

fn dtypeOf(d: st.DType) !DType {
    return switch (d) {
        .bf16 => .bf16,
        .f16 => .f16,
        .f32 => .f32,
        .i16 => .i16,
        .i32 => .i32,
        .i64 => .i64,
        .u8 => .u8,
        else => error.UnsupportedDType,
    };
}

const Source = struct { t: st.Tensor, name: []const u8 };

/// The 1017 converter's name for one of the engine's MTP markov head names: markov_w1 is the engine's embed.weight,
/// markov_w2 its head.weight. Null for every other tensor. A pack written with the engine's own names resolves
/// before this is tried, so the September packs (and every earlier one) read exactly as they did.
fn converterName(name: []const u8, buf: []u8) ?[]const u8 {
    const markov = ".markov_head.";
    const weight = ".weight";
    const at = std.mem.lastIndexOf(u8, name, markov) orelse return null;
    if (!std.mem.endsWith(u8, name, weight) or name.len < at + markov.len + weight.len) return null;
    const which = name[at + markov.len .. name.len - weight.len];
    const conv = if (std.mem.eql(u8, which, "embed")) "markov_w1" else if (std.mem.eql(u8, which, "head")) "markov_w2" else return null;
    return std.fmt.bufPrint(buf, "{s}{s}{s}{s}", .{ name[0..at], markov, conv, weight }) catch null;
}

fn source(sh: *const Shards, k: Key, buf: []u8) !Source {
    const name = if (!k.exl3) k.name else blk: {
        const tail = if (std.mem.eql(u8, k.part, "tr")) "trellis" else if (std.mem.eql(u8, k.part, "suh")) "suh" else if (std.mem.eql(u8, k.part, "svh")) "svh" else return error.BadKey;
        break :blk try std.fmt.bufPrint(buf, "{s}.{s}", .{ k.name, tail });
    };
    if (sh.get(name)) |t| return .{ .t = t, .name = name };
    var conv_buf: [256]u8 = undefined;
    const conv = converterName(name, &conv_buf) orelse return error.MissingTensor;
    return .{ .t = sh.get(conv) orelse return error.MissingTensor, .name = conv };
}

/// The layout `key` reads into.
pub fn describe(sh: *const Shards, key: []const u8) !Slice {
    const k = try Key.parse(key);
    var buf: [256]u8 = undefined;
    const s = try source(sh, k, &buf);
    var out: Slice = .{ .dtype = try dtypeOf(s.t.dtype), .rank = s.t.rank, .shape = s.t.shape, .bytes = 0 };
    if (k.dtype) |d| out.dtype = d;
    if (k.exl3) {
        if (std.mem.eql(u8, k.part, "tr")) {
            if (k.rows) |r| out.shape[0] = (r[1] - r[0]) / 16;
            if (k.cols) |c| out.shape[1] = (c[1] - c[0]) / 16;
        } else if (std.mem.eql(u8, k.part, "suh")) {
            if (k.rows) |r| out.shape[0] = r[1] - r[0];
        } else if (k.cols) |c| out.shape[0] = c[1] - c[0];
    }
    var n: usize = out.dtype.size();
    for (out.shape[0..out.rank]) |d| n *= d;
    out.bytes = n;
    return out;
}

/// Where `key`'s source tensor lies in its shard (a reader that takes the whole tensor in one read, then readFrom).
pub const Span = struct { file: *const st.File, offset: usize, len: usize };

pub fn span(sh: *const Shards, key: []const u8) !Span {
    const k = try Key.parse(key);
    var buf: [256]u8 = undefined;
    const s = try source(sh, k, &buf);
    const f = &sh.files[sh.names.get(s.name).?];
    return .{ .file = f, .offset = @intFromPtr(s.t.bytes.ptr) - @intFromPtr(f.map.memory.ptr), .len = s.t.bytes.len };
}

/// `key`'s bytes into `out` (describe(key).bytes long).
pub fn read(sh: *const Shards, key: []const u8, out: []u8) !void {
    return readFrom(sh, key, null, out);
}

/// read, the source tensor's bytes taken from `bytes` (its span, read by the caller) instead of the mapped shard: a
/// column slice touches every row of its tensor, which through the map is a disk read a row.
pub fn readFrom(sh: *const Shards, key: []const u8, bytes: ?[]const u8, out: []u8) !void {
    const k = try Key.parse(key);
    var buf: [256]u8 = undefined;
    const s = try source(sh, k, &buf);
    var t = s.t;
    if (bytes) |b| {
        if (b.len != t.bytes.len) return error.BadSize;
        t.bytes = b;
    }
    const want = try describe(sh, key);
    if (out.len != want.bytes) return error.BadSize;
    if (!k.exl3) {
        const from = try dtypeOf(t.dtype);
        if (k.dtype == null or k.dtype.? == from) return @memcpy(out, t.bytes);
        if (from == .f16 and k.dtype.? == .f32) return widenHalf(t.bytes, out);
        // the 1017 quants hold the Engram q_weight / k_weight tables in bf16; the engine reads them as f32, and the
        // widening is exact (bf16 is an f32's top 16 bits), as the Python loader's .to(torch.float32) is.
        if (from == .bf16 and k.dtype.? == .f32) return widenBfloat(t.bytes, out);
        // the 2.8 bpw pack holds the router weights (ffn.gate.weight) in bf16; the engine reads them as f16, as the
        // Python loader's .to(torch.float16) does (round to nearest even, exact for every value f16 holds).
        if (from == .bf16 and k.dtype.? == .f16) return narrowBfloat(t.bytes, out);
        return error.UnsupportedConversion;
    }
    const es = (try dtypeOf(t.dtype)).size();
    if (std.mem.eql(u8, k.part, "tr")) {
        const row = t.dim(1) * t.dim(2) * es; // one K tile row: every N tile's words
        const k0 = if (k.rows) |r| r[0] / 16 else 0;
        if (k.cols) |c| { // output columns, of the given input rows when both are set (the 2D split's wo_b, down, Engram)
            const tile = t.dim(2) * es;
            const lo = c[0] / 16 * tile;
            const w = (c[1] - c[0]) / 16 * tile;
            const kt = if (k.rows) |r| (r[1] - r[0]) / 16 else t.dim(0);
            for (0..kt) |i| @memcpy(out[i * w ..][0..w], t.bytes[(k0 + i) * row + lo ..][0..w]);
            return;
        }
        if (k.rows != null) return @memcpy(out, t.bytes[k0 * row ..][0..out.len]);
        return @memcpy(out, t.bytes);
    }
    const sel: ?[2]usize = if (std.mem.eql(u8, k.part, "suh")) k.rows else k.cols;
    if (sel) |r| return @memcpy(out, t.bytes[r[0] * es ..][0..out.len]);
    @memcpy(out, t.bytes);
}

/// f16 -> f32, exact (the loader's .to(torch.float32) of an f16 tensor).
fn widenHalf(src: []const u8, out: []u8) void {
    const n = src.len / 2;
    for (0..n) |i| {
        const h: f16 = @bitCast(std.mem.readInt(u16, src[2 * i ..][0..2], .little));
        std.mem.writeInt(u32, out[4 * i ..][0..4], @bitCast(@as(f32, h)), .little);
    }
}

/// bf16 -> f32, exact (the same .to(torch.float32) of a bf16 tensor: bf16 is an f32's top 16 bits).
fn widenBfloat(src: []const u8, out: []u8) void {
    const n = src.len / 2;
    for (0..n) |i| {
        const bits = std.mem.readInt(u16, src[2 * i ..][0..2], .little);
        std.mem.writeInt(u32, out[4 * i ..][0..4], @as(u32, bits) << 16, .little);
    }
}

/// bf16 -> f16 for a tensor the kernel reads as fp16 (the MTP markov head; the 1017 checkpoint itself holds it in
/// bf16). Exact for every value f16 holds: bf16's 7 mantissa bits fit f16's 10, and the head's weights are far inside
/// f16's range; a value outside it saturates, as a cast does.
pub fn narrowBfloat(src: []const u8, out: []u8) void {
    const n = src.len / 2;
    for (0..n) |i| {
        const bits = std.mem.readInt(u16, src[2 * i ..][0..2], .little);
        const h: f16 = @floatCast(@as(f32, @bitCast(@as(u32, bits) << 16)));
        std.mem.writeInt(u16, out[2 * i ..][0..2], @bitCast(h), .little);
    }
}

test "loader keys parse as Python prints them" {
    const p = try Key.parse("layers.3.ffn.experts.17.w2|None|(1152, 2304)|tr");
    try std.testing.expect(p.exl3 and p.cols == null and p.rows.?[0] == 1152 and p.rows.?[1] == 2304);
    const q = try Key.parse("layers.0.ffn.gate.bias|torch.float32");
    try std.testing.expect(!q.exl3 and q.dtype.? == .f32);
    try std.testing.expectError(error.BadKey, Key.parse("a|b|c"));
    try std.testing.expectError(error.BadKey, Key.parse("a|(2, 1)|None|tr"));
}

test "the 1017 converter's markov names map to the engine's, and nothing else does" {
    var buf: [256]u8 = undefined;
    try std.testing.expectEqualStrings("mtp.2.markov_head.markov_w1.weight", converterName("mtp.2.markov_head.embed.weight", &buf).?);
    try std.testing.expectEqualStrings("mtp.2.markov_head.markov_w2.weight", converterName("mtp.2.markov_head.head.weight", &buf).?);
    try std.testing.expect(converterName("mtp.2.markov_head.markov_w1.weight", &buf) == null);
    try std.testing.expect(converterName("layers.1.engram.q_weight", &buf) == null);
    try std.testing.expect(converterName("embed.weight", &buf) == null);
    try std.testing.expect(converterName("mtp.2.markov_head.embed.scale", &buf) == null);
}

test "bf16 -> f16 keeps every value f16 holds" {
    const vals = [_]f32{ 0.5, -2.0, 0.0234375, 65504.0, 0.0, 1.0, -0.0 };
    var src: [vals.len * 2]u8 = undefined;
    for (vals, 0..) |v, i| {
        const bf: u16 = @truncate(@as(u32, @bitCast(v)) >> 16);
        std.mem.writeInt(u16, src[2 * i ..][0..2], bf, .little);
    }
    var out: [vals.len * 2]u8 = undefined;
    narrowBfloat(&src, &out);
    for (0..vals.len) |i| {
        // the value the bf16 actually holds (65504 is not bf16-representable: it rounds to 65280)
        const bf = std.mem.readInt(u16, src[2 * i ..][0..2], .little);
        const expected: f32 = @bitCast(@as(u32, bf) << 16);
        const h: f16 = @bitCast(std.mem.readInt(u16, out[2 * i ..][0..2], .little));
        try std.testing.expectEqual(expected, @as(f32, h));
    }
}

test "a 1017 pack: converter markov names and bf16 Engram read as the engine's keys" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    // the 1017 quant's header entries: markov_w1 / markov_w2 and the Engram q / k in bf16
    var body: [32]u8 = undefined;
    for (&body, 0..) |*x, i| x.* = @truncate(i * 7 + 3);
    const json = "{\"mtp.2.markov_head.markov_w1.weight\":{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[0,8]}," ++
        "\"mtp.2.markov_head.markov_w2.weight\":{\"dtype\":\"F16\",\"shape\":[2,2],\"data_offsets\":[8,16]}," ++
        "\"layers.1.engram.q_weight\":{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[16,24]}," ++
        "\"layers.1.engram.k_weight\":{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[24,32]}}";
    var file: std.ArrayList(u8) = .empty;
    defer file.deinit(a);
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, json.len, .little);
    try file.appendSlice(a, &len);
    try file.appendSlice(a, json);
    try file.appendSlice(a, &body);
    try tmp.dir.writeFile(io, .{ .sub_path = "model-00001-of-00001.safetensors", .data = file.items });
    const dir = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer a.free(dir);
    var sh = try Shards.open(a, io, dir);
    defer sh.close(io);
    // the engine's names find the converter's tensors, byte for byte
    const embed = try describe(&sh, "mtp.2.markov_head.embed.weight|None");
    try std.testing.expectEqual(DType.bf16, embed.dtype);
    try std.testing.expectEqual(@as(usize, 8), embed.bytes);
    var got: [8]u8 = undefined;
    try read(&sh, "mtp.2.markov_head.embed.weight|None", &got);
    try std.testing.expectEqualSlices(u8, body[0..8], &got);
    try read(&sh, "mtp.2.markov_head.head.weight|None", &got);
    try std.testing.expectEqualSlices(u8, body[8..16], &got);
    // the Engram q / k the pack holds in bf16 read as f32, exactly
    const q = try describe(&sh, "layers.1.engram.q_weight|torch.float32");
    try std.testing.expectEqual(DType.f32, q.dtype);
    try std.testing.expectEqual(@as(usize, 16), q.bytes);
    var wide: [16]u8 = undefined;
    try read(&sh, "layers.1.engram.q_weight|torch.float32", &wide);
    for (0..4) |i| {
        const bits = std.mem.readInt(u16, body[16 + 2 * i ..][0..2], .little);
        try std.testing.expectEqual(@as(u32, bits) << 16, std.mem.readInt(u32, wide[4 * i ..][0..4], .little));
    }
    var wide_k: [16]u8 = undefined;
    try read(&sh, "layers.1.engram.k_weight|torch.float32", &wide_k);
    for (0..4) |i| {
        const bits = std.mem.readInt(u16, body[24 + 2 * i ..][0..2], .little);
        try std.testing.expectEqual(@as(u32, bits) << 16, std.mem.readInt(u32, wide_k[4 * i ..][0..4], .little));
    }
}

test "a 2.8 bpw pack: bf16 router weights read as f16, f32 router bias as f32, a name in two shards fails" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    // the router weight in bf16 (0.5, -2.0, 0.0234375, 1.0) and the bias in f32 (0.25, -1.5)
    const vals = [_]f32{ 0.5, -2.0, 0.0234375, 1.0 };
    var body: [16]u8 = undefined;
    for (vals, 0..) |v, i| std.mem.writeInt(u16, body[2 * i ..][0..2], @truncate(@as(u32, @bitCast(v)) >> 16), .little);
    std.mem.writeInt(u32, body[8..12], @bitCast(@as(f32, 0.25)), .little);
    std.mem.writeInt(u32, body[12..16], @bitCast(@as(f32, -1.5)), .little);
    const json = "{\"layers.0.ffn.gate.weight\":{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[0,8]}," ++
        "\"layers.0.ffn.gate.bias\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[8,16]}}";
    var file: std.ArrayList(u8) = .empty;
    defer file.deinit(a);
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, json.len, .little);
    try file.appendSlice(a, &len);
    try file.appendSlice(a, json);
    try file.appendSlice(a, &body);
    try tmp.dir.writeFile(io, .{ .sub_path = "layer-00.safetensors", .data = file.items });
    const dir = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer a.free(dir);
    {
        var sh = try Shards.open(a, io, dir);
        defer sh.close(io);
        const w = try describe(&sh, "layers.0.ffn.gate.weight|torch.float16");
        try std.testing.expectEqual(DType.f16, w.dtype);
        try std.testing.expectEqual(@as(usize, 8), w.bytes);
        var half: [8]u8 = undefined;
        try read(&sh, "layers.0.ffn.gate.weight|torch.float16", &half);
        for (vals, 0..) |v, i| {
            const h: f16 = @bitCast(std.mem.readInt(u16, half[2 * i ..][0..2], .little));
            try std.testing.expectEqual(v, @as(f32, h));
        }
        var bias: [8]u8 = undefined;
        try read(&sh, "layers.0.ffn.gate.bias|torch.float32", &bias);
        try std.testing.expectEqualSlices(u8, body[8..16], &bias);
    }
    // the same names again in a second shard, the same dtypes and shapes (the 1017 packs' repeated Engram tables): the
    // later shard is used
    try tmp.dir.writeFile(io, .{ .sub_path = "layer-01.safetensors", .data = file.items });
    {
        var sh = try Shards.open(a, io, dir);
        defer sh.close(io);
        var bias: [8]u8 = undefined;
        try read(&sh, "layers.0.ffn.gate.bias|torch.float32", &bias);
        try std.testing.expectEqualSlices(u8, body[8..16], &bias);
    }
    // the router weight again in f16 (a dtype-override file next to the pack): no single source, so the open fails
    const json16 = "{\"layers.0.ffn.gate.weight\":{\"dtype\":\"F16\",\"shape\":[2,2],\"data_offsets\":[0,8]}}";
    var over: std.ArrayList(u8) = .empty;
    defer over.deinit(a);
    std.mem.writeInt(u64, &len, json16.len, .little);
    try over.appendSlice(a, &len);
    try over.appendSlice(a, json16);
    try over.appendSlice(a, body[0..8]);
    try tmp.dir.writeFile(io, .{ .sub_path = "zz-override.safetensors", .data = over.items });
    try std.testing.expectError(error.DuplicateTensor, Shards.open(a, io, dir));
}

test "a pack with the engine's own names reads them as before: the converter's name never wins" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var body: [24]u8 = undefined;
    for (&body, 0..) |*x, i| x.* = @truncate(i * 11 + 5);
    const json = "{\"mtp.2.markov_head.embed.weight\":{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[0,8]}," ++
        "\"mtp.2.markov_head.markov_w1.weight\":{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[8,16]}," ++
        "\"layers.1.engram.q_weight\":{\"dtype\":\"F32\",\"shape\":[1,2],\"data_offsets\":[16,24]}}";
    var file: std.ArrayList(u8) = .empty;
    defer file.deinit(a);
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, json.len, .little);
    try file.appendSlice(a, &len);
    try file.appendSlice(a, json);
    try file.appendSlice(a, &body);
    try tmp.dir.writeFile(io, .{ .sub_path = "model-00001-of-00001.safetensors", .data = file.items });
    const dir = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer a.free(dir);
    var sh = try Shards.open(a, io, dir);
    defer sh.close(io);
    var got: [8]u8 = undefined;
    try read(&sh, "mtp.2.markov_head.embed.weight|None", &got);
    try std.testing.expectEqualSlices(u8, body[0..8], &got);
    // an f32 table the engine reads as f32 is copied, not converted (the September packs' shape)
    var f32_got: [8]u8 = undefined;
    try read(&sh, "layers.1.engram.q_weight|torch.float32", &f32_got);
    try std.testing.expectEqualSlices(u8, body[16..24], &f32_got);
}

test "column, row and whole slices of a toy trellis and its scales" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    // a trellis [2, 4, 2] i16 (K 32, N 64, two words a tile), its suh [32] f16 and svh [64] f16, a bias in f16
    var body: [16 * 2 + 32 * 2 + 64 * 2 + 4 * 2]u8 = undefined;
    for (&body, 0..) |*x, i| x.* = @truncate(i);
    const json = "{\"w.trellis\":{\"dtype\":\"I16\",\"shape\":[2,4,2],\"data_offsets\":[0,32]}," ++
        "\"w.suh\":{\"dtype\":\"F16\",\"shape\":[32],\"data_offsets\":[32,96]}," ++
        "\"w.svh\":{\"dtype\":\"F16\",\"shape\":[64],\"data_offsets\":[96,224]}," ++
        "\"g.bias\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[224,232]}}";
    var file: std.ArrayList(u8) = .empty;
    defer file.deinit(a);
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, json.len, .little);
    try file.appendSlice(a, &len);
    try file.appendSlice(a, json);
    try file.appendSlice(a, &body);
    try tmp.dir.writeFile(io, .{ .sub_path = "model-00001-of-00001.safetensors", .data = file.items });
    const dir = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer a.free(dir);
    var sh = try Shards.open(a, io, dir);
    defer sh.close(io);
    // output columns 32..64 of the trellis: tiles 2 and 3 of each K row (bytes 8..16 and 24..32)
    var cols: [16]u8 = undefined;
    try std.testing.expectEqual(@as(usize, 16), (try describe(&sh, "w|(32, 64)|None|tr")).bytes);
    try read(&sh, "w|(32, 64)|None|tr", &cols);
    var expect: [16]u8 = undefined;
    @memcpy(expect[0..8], body[8..16]);
    @memcpy(expect[8..16], body[24..32]);
    try std.testing.expectEqualSlices(u8, &expect, &cols);
    var rows: [16]u8 = undefined;
    try read(&sh, "w|None|(16, 32)|tr", &rows);
    try std.testing.expectEqualSlices(u8, body[16..32], &rows);
    // both: K tile row 1, its tiles 2 and 3 (the 2D split's down and wo_b keys)
    var both: [8]u8 = undefined;
    try std.testing.expectEqual(@as(usize, 8), (try describe(&sh, "w|(32, 64)|(16, 32)|tr")).bytes);
    try read(&sh, "w|(32, 64)|(16, 32)|tr", &both);
    try std.testing.expectEqualSlices(u8, body[24..32], &both);
    var suh: [32]u8 = undefined;
    try read(&sh, "w|None|(16, 32)|suh", &suh);
    try std.testing.expectEqualSlices(u8, body[32 + 32 .. 96], &suh);
    var svh: [64]u8 = undefined;
    try read(&sh, "w|(32, 64)|None|svh", &svh);
    try std.testing.expectEqualSlices(u8, body[96 + 64 .. 224], &svh);
    // the same from a copy of the tensor's span, as tf-dsv41-rank-cache reads it
    const sp = try span(&sh, "w|(32, 64)|(16, 32)|tr");
    try std.testing.expectEqual(@as(usize, 8 + json.len), sp.offset);
    try std.testing.expectEqual(@as(usize, 32), sp.len);
    var copy: [32]u8 = undefined;
    @memcpy(&copy, file.items[sp.offset..][0..sp.len]);
    var both2: [8]u8 = undefined;
    try readFrom(&sh, "w|(32, 64)|(16, 32)|tr", &copy, &both2);
    try std.testing.expectEqualSlices(u8, &both, &both2);
    try std.testing.expectError(error.BadSize, readFrom(&sh, "w|(32, 64)|(16, 32)|tr", copy[0..16], &both2));
    var wide: [16]u8 = undefined;
    try read(&sh, "g.bias|torch.float32", &wide);
    const h0: f16 = @bitCast(std.mem.readInt(u16, body[224..226], .little));
    try std.testing.expectEqual(@as(f32, h0), @as(f32, @bitCast(std.mem.readInt(u32, wide[0..4], .little))));
}

const fixture_1017 = @embedFile("fixtures/1017-headers.json");

/// One pack of the fixture: its shards rebuilt from the real headers (a pattern in the body), then every check read
/// under the engine's key and compared with the pack's own tensor.
fn fixturePack(a: std.mem.Allocator, io: std.Io, pack: std.json.Value) !void {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    for (pack.object.get("files").?.array.items) |fv| {
        const f = fv.object;
        const name = f.get("name").?.string;
        const header = f.get("header").?.string;
        const body_bytes: usize = @intCast(f.get("body_bytes").?.integer);
        var file: std.ArrayList(u8) = .empty;
        defer file.deinit(a);
        var len: [8]u8 = undefined;
        std.mem.writeInt(u64, &len, header.len, .little);
        try file.appendSlice(a, &len);
        try file.appendSlice(a, header);
        const start = file.items.len;
        try file.resize(a, start + body_bytes);
        for (file.items[start..], 0..) |*x, i| x.* = @truncate(i * 37 + 11);
        try tmp.dir.writeFile(io, .{ .sub_path = name, .data = file.items });
    }
    const dir = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer a.free(dir);
    var sh = try Shards.open(a, io, dir);
    defer sh.close(io);
    for (pack.object.get("checks").?.array.items) |cv| {
        const c = cv.object;
        const key = c.get("key").?.string;
        const src_name = c.get("source").?.string;
        const want = c.get("want").?.string;
        const dtype_text = c.get("dtype").?.string;
        const want_dtype: DType = if (std.mem.eql(u8, dtype_text, "BF16")) .bf16 else if (std.mem.eql(u8, dtype_text, "F16")) .f16 else if (std.mem.eql(u8, dtype_text, "F32")) .f32 else return error.BadFixture;
        const src = sh.get(src_name) orelse return error.MissingTensor;
        const slice = try describe(&sh, key);
        try std.testing.expectEqual(want_dtype, slice.dtype);
        const got = try a.alloc(u8, slice.bytes);
        defer a.free(got);
        try read(&sh, key, got);
        if (std.mem.eql(u8, want, "same")) {
            try std.testing.expectEqualSlices(u8, src.bytes, got);
        } else if (std.mem.eql(u8, want, "f32")) {
            try std.testing.expectEqual(src.bytes.len * 2, got.len);
            for (0..src.bytes.len / 2) |i| {
                const bits = std.mem.readInt(u16, src.bytes[2 * i ..][0..2], .little);
                try std.testing.expectEqual(@as(u32, bits) << 16, std.mem.readInt(u32, got[4 * i ..][0..4], .little));
            }
        } else return error.BadFixture;
    }
}

test "1017 packs from their real headers: converter markov names and bf16 Engram read as the engine's keys" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    const parsed = try std.json.parseFromSlice(std.json.Value, a, fixture_1017, .{});
    defer parsed.deinit();
    for (parsed.value.object.get("packs").?.array.items) |pack| try fixturePack(a, io, pack);
}

test "a real pack reads under the engine's keys (TF_DSV41_PACK): 1017 markov names and Engram dtypes" {
    const dir = std.c.getenv("TF_DSV41_PACK") orelse return error.SkipZigTest;
    const a = std.testing.allocator;
    const io = std.testing.io;
    var sh = try Shards.open(a, io, std.mem.span(dir));
    defer sh.close(io);
    // the MTP markov head: the engine's names read the pack's own tensor or the 1017 converter's, byte for byte
    for ([_][]const u8{ "mtp.2.markov_head.embed.weight", "mtp.2.markov_head.head.weight" }) |name| {
        var kb: [128]u8 = undefined;
        const key = try std.fmt.bufPrint(&kb, "{s}|None", .{name});
        var buf: [256]u8 = undefined;
        const src = try source(&sh, try Key.parse(key), &buf);
        const got = try a.alloc(u8, src.t.bytes.len);
        defer a.free(got);
        try read(&sh, key, got);
        try std.testing.expectEqualSlices(u8, src.t.bytes, got);
    }
    // the Engram q / k: the engine reads f32; a bf16 table widens exactly, an f32 one is copied
    for ([_][]const u8{ "layers.1.engram.q_weight", "layers.1.engram.k_weight", "layers.14.engram.q_weight", "layers.14.engram.k_weight" }) |name| {
        var kb: [128]u8 = undefined;
        var nb: [128]u8 = undefined;
        const key = try std.fmt.bufPrint(&kb, "{s}|torch.float32", .{name});
        const raw_key = try std.fmt.bufPrint(&nb, "{s}|None", .{name});
        const src = sh.get(name) orelse return error.MissingTensor;
        const from = (try describe(&sh, raw_key)).dtype;
        const got = try a.alloc(u8, (try describe(&sh, key)).bytes);
        defer a.free(got);
        try read(&sh, key, got);
        if (from == .f32) {
            try std.testing.expectEqualSlices(u8, src.bytes, got);
        } else if (from == .bf16) {
            try std.testing.expectEqual(src.bytes.len * 2, got.len);
            for (0..src.bytes.len / 2) |i| {
                const bits = std.mem.readInt(u16, src.bytes[2 * i ..][0..2], .little);
                try std.testing.expectEqual(@as(u32, bits) << 16, std.mem.readInt(u32, got[4 * i ..][0..4], .little));
            }
        } else return error.UnsupportedConversion;
    }
    std.debug.print("TF_DSV41_PACK {s}: the markov head names and the Engram q/k read under the engine's keys\n", .{std.mem.span(dir)});
}

test "a pack's planned entries read to one digest (TF_DSV41_DIGEST_PACK): the loader's byte contract across builds" {
    const dir = std.c.getenv("TF_DSV41_DIGEST_PACK") orelse return error.SkipZigTest;
    const a = std.testing.allocator;
    const io = std.testing.io;
    // the plan's keys are many small allocations: one arena, freed with the test
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const cfg = try Config.read(arena.allocator(), io, std.mem.span(dir));
    var sh = try Shards.open(a, io, std.mem.span(dir));
    defer sh.close(io);
    const wants = try plan.wants(arena.allocator(), cfg, .{ .rank = 0, .world = 2 }, true);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(a);
    var n: usize = 0;
    var bytes: usize = 0;
    for (wants) |w| {
        const bar = std.mem.indexOfScalar(u8, w.key, '|') orelse w.key.len;
        const name = w.key[0..bar];
        // a representative set: the embedding and head, the DSpark tensors, the Engram q / k and layer 0's non-expert
        // tensors (the experts are the bulk of the pack and the same code reads them)
        const take = std.mem.startsWith(u8, name, "embed.") or std.mem.startsWith(u8, name, "norm.") or
            std.mem.eql(u8, name, "head") or std.mem.startsWith(u8, name, "mtp.") or
            std.mem.indexOf(u8, name, ".engram.") != null or
            (std.mem.startsWith(u8, name, "layers.0.") and std.mem.indexOf(u8, name, ".ffn.experts.") == null);
        if (!take) continue;
        h.update(w.key);
        h.update("\x00");
        const len = (try describe(&sh, w.key)).bytes;
        try buf.resize(a, len);
        try read(&sh, w.key, buf.items);
        h.update(buf.items);
        n += 1;
        bytes += len;
    }
    var out: [32]u8 = undefined;
    h.final(&out);
    std.debug.print("TF_DSV41_DIGEST_PACK {s}: {d} entries, {d} bytes, sha256 {x}\n", .{ std.mem.span(dir), n, bytes, out });
}
