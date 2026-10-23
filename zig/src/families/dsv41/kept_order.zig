//! The order of the kept prompt state primitives on the ranks (lanes.zig). Rank 0 issues its own copies first and
//! sends the frame only after them, so a follower never keeps or restores a state that rank 0 does not keep. A
//! follower that cannot do a restore that rank 0 did stops: if it continued, its state would differ from rank 0's.
const std = @import("std");

/// Saves kept state `sn` on every rank: room in `list` first, then this rank's copy (`x.copy`), then the frame to the
/// followers (`x.send`), then `sn` into `list`. If a step fails, `sn` is not in `list`; a failed copy sends no frame.
pub fn save(comptime T: type, gpa: std.mem.Allocator, list: *std.ArrayList(*T), sn: *T, x: anytype) !void {
    try list.ensureUnusedCapacity(gpa, 1);
    try x.copy();
    try x.send();
    list.appendAssumeCapacity(sn);
}

/// Restores a kept state on rank 0: its copies (`x.copy`) first, then the frame (`x.send`). A failed copy sends no frame.
pub fn restore(x: anytype) !void {
    try x.copy();
    try x.send();
}

/// Restores kept state `id` on a follower with `x.copy`. An error stops the follower (rank 0 did this restore).
pub fn follow(comptime B: type, kept: *const std.AutoHashMapUnmanaged(u64, B), id: u64, x: anytype) !void {
    const b = kept.get(id) orelse return error.UnknownSnapshot;
    try x.copy(b);
}

/// A test's copy and frame: the order of the calls, and the call that fails.
const Fake = struct {
    log: *std.ArrayList(u8),
    fail: u8 = 0, // 'c': the copy fails, 's': the frame fails

    fn copy(x: Fake) !void {
        try x.log.append(std.testing.allocator, 'c');
        if (x.fail == 'c') return error.CopyFailed;
    }

    fn send(x: Fake) !void {
        try x.log.append(std.testing.allocator, 's');
        if (x.fail == 's') return error.LinkFailed;
    }
};

test "a save whose copy fails sends no frame and keeps no state" {
    const a = std.testing.allocator;
    var log: std.ArrayList(u8) = .empty;
    defer log.deinit(a);
    var list: std.ArrayList(*u32) = .empty;
    defer list.deinit(a);
    var sn: u32 = 7;
    try std.testing.expectError(error.CopyFailed, save(u32, a, &list, &sn, Fake{ .log = &log, .fail = 'c' }));
    try std.testing.expectEqualStrings("c", log.items);
    try std.testing.expectEqual(@as(usize, 0), list.items.len);
}

test "a save whose frame fails keeps no state" {
    const a = std.testing.allocator;
    var log: std.ArrayList(u8) = .empty;
    defer log.deinit(a);
    var list: std.ArrayList(*u32) = .empty;
    defer list.deinit(a);
    var sn: u32 = 7;
    try std.testing.expectError(error.LinkFailed, save(u32, a, &list, &sn, Fake{ .log = &log, .fail = 's' }));
    try std.testing.expectEqual(@as(usize, 0), list.items.len);
}

test "a save copies, then sends, then keeps the state" {
    const a = std.testing.allocator;
    var log: std.ArrayList(u8) = .empty;
    defer log.deinit(a);
    var list: std.ArrayList(*u32) = .empty;
    defer list.deinit(a);
    var sn: u32 = 7;
    try save(u32, a, &list, &sn, Fake{ .log = &log });
    try std.testing.expectEqualStrings("cs", log.items);
    try std.testing.expectEqual(@as(usize, 1), list.items.len);
    try std.testing.expectEqual(&sn, list.items[0]);
}

test "a restore whose copy fails on rank 0 sends no frame" {
    const a = std.testing.allocator;
    var log: std.ArrayList(u8) = .empty;
    defer log.deinit(a);
    try std.testing.expectError(error.CopyFailed, restore(Fake{ .log = &log, .fail = 'c' }));
    try std.testing.expectEqualStrings("c", log.items);
}

/// A follower's copy that fails.
const FailCopy = struct {
    fn copy(_: FailCopy, _: u32) !void {
        return error.CopyFailed;
    }
};

test "a follower whose restore copy fails stops" {
    const a = std.testing.allocator;
    var kept: std.AutoHashMapUnmanaged(u64, u32) = .empty;
    defer kept.deinit(a);
    try kept.put(a, 3, 0);
    try std.testing.expectError(error.CopyFailed, follow(u32, &kept, 3, FailCopy{}));
}

test "a follower without the kept state stops" {
    const kept: std.AutoHashMapUnmanaged(u64, u32) = .empty;
    try std.testing.expectError(error.UnknownSnapshot, follow(u32, &kept, 3, FailCopy{}));
}
