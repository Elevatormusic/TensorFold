//! A watch on the ranks' TCP links (release fault F3). A rank waits for its peer's frames and RDMA writes without a
//! limit, so when the peer died while this rank was idle, or during a round whose waits got no error completion, the
//! rank stayed up: /health said 200 and every request hung. The watch reads nothing: it polls each link for the
//! peer's end (its process died: the kernel closed the socket) or a failed keepalive (its node is gone, link.zig).
//! Then it aborts the RDMA ring (every wait gives up, so a live round fails and its requests get the error) and, with
//! TF_DS_EXIT_ON_FAULT=1, ends the process with 75 after TF_DS_EXIT_GRACE, so a supervisor restarts both ranks.
const std = @import("std");
const rdma = @import("rdma.zig");
const posix = std.posix;
const Allocator = std.mem.Allocator;

/// POLLRDHUP (Linux): the peer closed its end.
const rdhup: i16 = 0x2000;
/// The peer closed its end, or the connection failed (a keepalive or a send got no answer).
const gone: i16 = rdhup | posix.POLL.HUP | posix.POLL.ERR;
const max_links = 3;
/// TP2's ring, world 4's three rings (ring2d.zig), or the neighbor gather's two (neighbor.zig).
const max_rings = 6;

pub const Watch = struct {
    fds: [max_links]posix.fd_t = @splat(-1),
    n: usize = 0,
    rings: [max_rings]*rdma.Ring = undefined,
    n_rings: usize = 0,
    rank: u32,
    running: std.atomic.Value(bool) = .init(true),
    orderly: std.atomic.Value(bool) = .init(false),
    /// Set when the watch found a peer gone (before it acts).
    lost: std.atomic.Value(bool) = .init(false),
    /// Held while the watch aborts the rings, so release() never frees them under it (a spin: it is taken at most twice).
    rings_busy: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    /// A watch thread on `fds` (rank 0's links to its followers, or a follower's one link to rank 0) that aborts
    /// `rings` (every RDMA ring this rank's GPU waits on) when a peer is gone.
    pub fn start(gpa: Allocator, fds: []const posix.fd_t, rings: []const *rdma.Ring, rank: u32) !*Watch {
        const w = try gpa.create(Watch);
        errdefer gpa.destroy(w);
        w.* = .{ .rank = rank, .n = @min(fds.len, max_links), .n_rings = @min(rings.len, max_rings) };
        @memcpy(w.fds[0..w.n], fds[0..w.n]);
        @memcpy(w.rings[0..w.n_rings], rings[0..w.n_rings]);
        w.thread = try std.Thread.spawn(.{}, loop, .{w});
        return w;
    }

    /// An orderly stop (rank 0 sends its done frame, or a follower takes it): a link that closes after it is no fault.
    pub fn stop(w: *Watch, gpa: Allocator) void {
        w.orderly.store(true, .release);
        w.running.store(false, .release);
        if (w.thread) |t| t.join();
        gpa.destroy(w);
    }

    /// A rank that ends on an error (its rings are freed next): aborts the rings and forgets them, so the watch never
    /// reads freed rings. It keeps the links watched and the exit 75 armed: an error is no orderly stop.
    pub fn release(w: *Watch, why: []const u8) void {
        w.lockRings();
        defer w.rings_busy.store(false, .release);
        for (w.rings[0..w.n_rings]) |r| r.abort(why);
        w.n_rings = 0;
    }

    fn lockRings(w: *Watch) void {
        while (w.rings_busy.swap(true, .acquire)) std.atomic.spinLoopHint();
    }

    fn quiet(w: *Watch) bool {
        return w.orderly.load(.acquire) or !w.running.load(.acquire);
    }

    /// Waits up to `ms`; true if an orderly stop came in that time.
    fn stopped(w: *Watch, ms: u64) bool {
        var left = ms;
        while (left > 0) : (left -|= 100) {
            if (w.quiet()) return true;
            const ts: std.c.timespec = .{ .sec = 0, .nsec = 100 * std.time.ns_per_ms };
            _ = std.c.nanosleep(&ts, null);
        }
        return w.quiet();
    }

    fn loop(w: *Watch) void {
        var pfds: [max_links]posix.pollfd = undefined;
        for (pfds[0..w.n], w.fds[0..w.n]) |*p, fd| p.* = .{ .fd = fd, .events = rdhup, .revents = 0 };
        while (!w.quiet()) {
            const k = posix.poll(pfds[0..w.n], 200) catch continue;
            if (k == 0) continue;
            for (pfds[0..w.n]) |p| if (p.revents & gone != 0) return w.peerGone();
        }
    }

    fn peerGone(w: *Watch) void {
        // rank 0's done frame and the close of its link come together: a second for this rank to take the stop
        if (w.stopped(1000)) return;
        w.lost.store(true, .release);
        {
            w.lockRings();
            defer w.rings_busy.store(false, .release);
            std.debug.print("[tensorfold] rank {d}: a peer rank's link closed (its process or node is gone); the waits of its {d} RDMA ring(s) are aborted\n", .{ w.rank, w.n_rings });
            for (w.rings[0..w.n_rings]) |r| r.abort("a peer rank is gone (its link closed)");
        }
        const grace_ms = exitGrace() orelse return;
        // a live round fails first and the lane host ends the process itself after the same grace
        if (w.stopped(grace_ms + 2000)) return;
        std.debug.print("[tensorfold] rank {d}: exit 75 (TF_DS_EXIT_ON_FAULT): a supervisor restarts both ranks\n", .{w.rank});
        // a stream may hold kernels that wait on the gone rank (NCCL's), which a normal exit would join
        std.c._exit(75);
    }
};

/// TF_DS_EXIT_ON_FAULT=1: the grace in ms (TF_DS_EXIT_GRACE seconds, default 5), as lane_host.faultExit reads it.
fn exitGrace() ?u64 {
    const on = std.c.getenv("TF_DS_EXIT_ON_FAULT") orelse return null;
    if (!std.mem.eql(u8, std.mem.span(on), "1")) return null;
    const g = std.c.getenv("TF_DS_EXIT_GRACE") orelse return 5000;
    const s = std.fmt.parseFloat(f64, std.mem.span(g)) catch return 5000;
    if (!(s >= 0) or s > 600) return 5000;
    return @intFromFloat(s * 1000);
}

const Link = @import("link.zig").Link;

fn testPort() u16 {
    return 29500 + @as(u16, @intCast(@mod(std.Io.Timestamp.now(std.testing.io, .awake).nanoseconds, 400)));
}

test "a peer that closes its link without a done frame is found within two seconds" {
    const port = testPort();
    const Peer = struct {
        fn run(p: u16) void {
            var l = Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, p, 10) catch return;
            l.send("one frame") catch {};
            const ts: std.c.timespec = .{ .sec = 0, .nsec = 300 * std.time.ns_per_ms };
            _ = std.c.nanosleep(&ts, null);
            l.close(); // the process "dies": its end closes
        }
    };
    const t = try std.Thread.spawn(.{}, Peer.run, .{port});
    var l = try Link.listen(.{ 127, 0, 0, 1 }, port);
    defer l.close();
    const w = try Watch.start(std.testing.allocator, &.{l.fd}, &.{}, 0);
    defer w.stop(std.testing.allocator);
    t.join();
    // a frame waiting unread is no end: the watch asks for the peer's close alone
    var waited: u32 = 0;
    while (!w.lost.load(.acquire) and waited < 40) : (waited += 1) {
        const ts: std.c.timespec = .{ .sec = 0, .nsec = 100 * std.time.ns_per_ms };
        _ = std.c.nanosleep(&ts, null);
    }
    try std.testing.expect(w.lost.load(.acquire));
}

test "a released watch forgets its rings and still finds the peer gone" {
    const port = testPort() + 800;
    const Peer = struct {
        fn run(p: u16) void {
            var l = Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, p, 10) catch return;
            const ts: std.c.timespec = .{ .sec = 0, .nsec = 300 * std.time.ns_per_ms };
            _ = std.c.nanosleep(&ts, null);
            l.close();
        }
    };
    const t = try std.Thread.spawn(.{}, Peer.run, .{port});
    var l = try Link.listen(.{ 127, 0, 0, 1 }, port);
    defer l.close();
    const w = try Watch.start(std.testing.allocator, &.{l.fd}, &.{}, 1);
    defer w.stop(std.testing.allocator);
    w.release("the test ends on an error");
    try std.testing.expectEqual(@as(usize, 0), w.n_rings);
    t.join();
    var waited: u32 = 0;
    while (!w.lost.load(.acquire) and waited < 40) : (waited += 1) {
        const ts: std.c.timespec = .{ .sec = 0, .nsec = 100 * std.time.ns_per_ms };
        _ = std.c.nanosleep(&ts, null);
    }
    // an error is no orderly stop: the watch still acts on the peer's end
    try std.testing.expect(w.lost.load(.acquire));
}

test "an orderly stop before the link closes is no fault" {
    const port = testPort() + 400;
    const Peer = struct {
        fn run(p: u16) void {
            var l = Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, p, 10) catch return;
            const ts: std.c.timespec = .{ .sec = 0, .nsec = 300 * std.time.ns_per_ms };
            _ = std.c.nanosleep(&ts, null);
            l.close();
        }
    };
    const t = try std.Thread.spawn(.{}, Peer.run, .{port});
    var l = try Link.listen(.{ 127, 0, 0, 1 }, port);
    defer l.close();
    const w = try Watch.start(std.testing.allocator, &.{l.fd}, &.{}, 1);
    w.orderly.store(true, .release); // the done frame came: the close that follows is expected
    t.join();
    const ts: std.c.timespec = .{ .sec = 1, .nsec = 500 * std.time.ns_per_ms };
    _ = std.c.nanosleep(&ts, null);
    try std.testing.expect(!w.lost.load(.acquire));
    w.stop(std.testing.allocator);
}
