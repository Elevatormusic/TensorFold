//! checkpoint.zig's 2.8 bpw pack test: bf16 router weights, an f32 router bias, and one name in two shards.

const std = @import("std");
const checkpoint = @import("checkpoint.zig");
const DType = @import("rank_cache.zig").DType;
const Shards = checkpoint.Shards;
const describe = checkpoint.describe;
const read = checkpoint.read;

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
    // the same names, dtypes and shapes in a second shard (the 1017 packs' repeated Engram tables): the later one wins
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
