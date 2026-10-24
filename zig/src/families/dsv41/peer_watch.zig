//! A watch on the ranks' TCP links (release fault F3). A rank waits for its peer's frames and RDMA writes without a
//! limit, so when the peer died while this rank was idle, or during a round whose waits got no error completion, the
//! rank stayed up: /health said 200 and every request hung. The watch reads nothing: it polls each link for the
//! peer's end (its process died: the kernel closed the socket) or a failed keepalive (its node is gone, link.zig).
//! Then it aborts the RDMA rings (every wait gives up, so a live round fails and its requests get the error) and, with
//! TF_DS_EXIT_ON_FAULT=1, ends the process with 75 after TF_DS_EXIT_GRACE, so a supervisor restarts both ranks.
//! Watch.end is the one teardown of a watch: every way a rank ends calls it before the rings are freed.
const std = @import("std");
const rdma = @import("rdma.zig");
const posix = std.posix;
const Allocator = std.mem.Allocator;

/// POLLRDHUP (Linux): the peer closed its end.
const rdhup: i16 = 0x2000;
/// The peer closed its end, or the connection failed (a keepalive or a send got no answer), or the descriptor is no
/// longer open (POLLNVAL: a poll on it would return at once, again and again).
const gone: i16 = rdhup | posix.POLL.HUP | posix.POLL.ERR | posix.POLL.NVAL;
const max_links = 3;
/// TP2's ring, world 4's three rings (ring2d.zig), or the neighbor gather's two (neighbor.zig).
const max_rings = 6;

/// How a rank's work with its peers ended, as its caller knows it.
pub const How = enum {
    /// rank 0 sent its done frame, or a follower took it: a link that closes after it is expected
    done,
    /// an error (a failed start, a failed send or receive, a failed round)
    failed,
};

/// What Watch.end found.
pub const Outcome = enum {
    /// the done frame: no fault
    orderly,
    /// an error while every peer link is still up: this rank's own fault (its caller reports it)
    failed,
    /// an error with a peer's link gone: with TF_DS_EXIT_ON_FAULT=1, end does not return (exit 75)
    peer_fault,
};

pub const Watch = struct {
    fds: [max_links]posix.fd_t = @splat(-1),
    n: usize = 0,
    rings: [max_rings]*rdma.Ring = undefined,
    n_rings: usize = 0,
    rank: u32,
    running: std.atomic.Value(bool) = .init(true),
    /// Set when the poll found a link's end (before the one-second wait for an orderly stop).
    seen: std.atomic.Value(bool) = .init(false),
    /// Set when the watch thread acted on a peer gone (it aborted the rings).
    lost: std.atomic.Value(bool) = .init(false),
    /// Held while a thread aborts the rings (a spin: it is taken at most twice).
    rings_busy: std.atomic.Value(bool) = .init(false),
    /// The exit 75 on a peer fault, after this grace in ms (faultGrace: TF_DS_EXIT_ON_FAULT=1); null: no exit.
    exit_grace: ?u64,
    thread: ?std.Thread = null,

    /// A watch thread on `fds` (rank 0's links to its followers, or a follower's one link to rank 0) that aborts
    /// `rings` (every RDMA ring this rank's GPU waits on) when a peer is gone. `exit_grace`: faultGrace() (tests: null).
    pub fn start(gpa: Allocator, fds: []const posix.fd_t, rings: []const *rdma.Ring, rank: u32, exit_grace: ?u64) !*Watch {
        const w = try gpa.create(Watch);
        errdefer gpa.destroy(w);
        w.* = .{ .rank = rank, .n = @min(fds.len, max_links), .n_rings = @min(rings.len, max_rings), .exit_grace = exit_grace };
        @memcpy(w.fds[0..w.n], fds[0..w.n]);
        @memcpy(w.rings[0..w.n_rings], rings[0..w.n_rings]);
        openLog();
        w.thread = try std.Thread.spawn(.{}, loop, .{w});
        return w;
    }

    /// True when a peer's link has ended (the watch thread saw it, or a poll that does not wait finds it now, before
    /// the thread sees it): rank 0 reads it before its done frame (a done frame after it is no orderly stop).
    pub fn faulted(w: *Watch) bool {
        return w.seen.load(.acquire) or w.linkGone();
    }

    /// The one teardown, for every way a rank ends (rank 0's Lanes.deinit, a follower's done frame or error, a start
    /// that fails after the watch started), always before the caller frees the rings (Model.close):
    /// 1. on an error, it aborts every RDMA ring, so no GPU wait of this rank waits on a gone peer;
    /// 2. it stops the watch thread and joins it, so no thread reads the rings after end returns;
    /// 3. it frees the watch (the caller frees the rings and links next);
    /// 4. on an error with a peer's link gone and TF_DS_EXIT_ON_FAULT=1, it ends the process with 75 (a normal exit
    ///    could wait on the gone rank in Model.close).
    pub fn end(w: *Watch, gpa: Allocator, how: How) Outcome {
        if (how == .failed) w.abortRings("this rank ends on an error");
        w.running.store(false, .release);
        if (w.thread) |t| t.join();
        const out: Outcome = switch (how) {
            .done => .orderly,
            .failed => if (w.faulted()) .peer_fault else .failed,
        };
        const rank = w.rank;
        const exit_on_fault = w.exit_grace != null;
        gpa.destroy(w);
        if (out == .peer_fault and exit_on_fault) {
            note("[tensorfold] rank {d}: a peer rank is gone; exit 75 (TF_DS_EXIT_ON_FAULT): a supervisor restarts both ranks\n", .{rank});
            std.c._exit(75);
        }
        return out;
    }

    fn abortRings(w: *Watch, why: []const u8) void {
        while (w.rings_busy.swap(true, .acquire)) std.atomic.spinLoopHint();
        defer w.rings_busy.store(false, .release);
        for (w.rings[0..w.n_rings]) |r| r.abort(why);
    }

    /// A peer's link has ended now (a poll that does not wait): the error that the caller has may have come from it
    /// before the watch thread saw it.
    fn linkGone(w: *Watch) bool {
        var pfds: [max_links]posix.pollfd = undefined;
        for (pfds[0..w.n], w.fds[0..w.n]) |*p, fd| p.* = .{ .fd = fd, .events = rdhup, .revents = 0 };
        const k = posix.poll(pfds[0..w.n], 0) catch return false;
        if (k == 0) return false;
        for (pfds[0..w.n]) |p| if (p.revents & gone != 0) return true;
        return false;
    }

    fn quiet(w: *Watch) bool {
        return !w.running.load(.acquire);
    }

    /// Waits up to `ms`; true if end came in that time.
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
            const k = posix.poll(pfds[0..w.n], 200) catch {
                sleepMs(200); // no spin on a poll that fails at once
                continue;
            };
            if (k == 0) continue;
            for (pfds[0..w.n]) |p| if (p.revents & gone != 0) {
                w.seen.store(true, .release);
                return w.peerGone();
            };
        }
    }

    /// A peer gone while this rank does not end (an idle rank, or a round whose waits got no error completion).
    fn peerGone(w: *Watch) void {
        // rank 0's done frame and the close of its link come together: a second for this rank to call end
        if (w.stopped(1000)) return;
        w.lost.store(true, .release);
        // the abort first: no log comes before it
        w.abortRings("a peer rank is gone (its link closed)");
        note("[tensorfold] rank {d}: a peer rank's link closed (its process or node is gone); the waits of its {d} RDMA ring(s) are aborted\n", .{ w.rank, w.n_rings });
        const grace_ms = w.exit_grace orelse return;
        // a live round fails first: its error goes out, then this rank calls end, or the lane host ends the process
        if (w.stopped(grace_ms + 2000)) return;
        note("[tensorfold] rank {d}: exit 75 (TF_DS_EXIT_ON_FAULT): a supervisor restarts both ranks\n", .{w.rank});
        // a stream may hold kernels that wait on the gone rank (NCCL's), which a normal exit would join
        std.c._exit(75);
    }
};

/// note's descriptor: stderr's file opened again with O_NONBLOCK (openLog); -1: none; -2: not opened yet.
var log_fd: std.atomic.Value(c_int) = .init(-2);

/// A new open file description of `fd` with O_NONBLOCK, so fd itself (shared with other threads) stays blocking.
fn nonblockingCopy(fd: c_int) c_int {
    var buf: [32]u8 = undefined;
    const path = std.fmt.bufPrintSentinel(&buf, "/proc/self/fd/{d}", .{fd}, 0) catch return -1;
    return std.c.open(path, .{ .ACCMODE = .WRONLY, .NONBLOCK = true, .APPEND = true, .CLOEXEC = true });
}

/// Opens note's descriptor once per process (Watch.start), before any fault path can need it.
fn openLog() void {
    if (log_fd.load(.acquire) != -2) return;
    const fd = nonblockingCopy(2);
    if (log_fd.cmpxchgStrong(-2, if (fd >= 0) fd else -1, .acq_rel, .acquire) != null and fd >= 0) _ = std.c.close(fd);
}

/// A log line to stderr that never blocks: a full pipe or terminal drops it; the abort and exit never wait on it.
fn note(comptime fmt: []const u8, args: anytype) void {
    var buf: [512]u8 = undefined;
    const line = std.fmt.bufPrint(&buf, fmt, args) catch return;
    const fd = log_fd.load(.acquire);
    if (fd >= 0) {
        _ = std.c.write(fd, line.ptr, line.len);
        return;
    }
    // a socket stderr cannot open through /proc: MSG_DONTWAIT; on any other descriptor send fails and the line drops
    _ = std.c.send(2, line.ptr, line.len, std.c.MSG.DONTWAIT | std.c.MSG.NOSIGNAL);
}

/// TF_DS_EXIT_ON_FAULT=1: the grace in ms (TF_DS_EXIT_GRACE seconds, default 5), as lane_host.faultExit reads it. The
/// exit 75 of a watch (Watch.start's exit_grace); null when the variable is not 1.
pub fn faultGrace() ?u64 {
    const on = std.c.getenv("TF_DS_EXIT_ON_FAULT") orelse return null;
    if (!std.mem.eql(u8, std.mem.span(on), "1")) return null;
    const g = std.c.getenv("TF_DS_EXIT_GRACE") orelse return 5000;
    const s = std.fmt.parseFloat(f64, std.mem.span(g)) catch return 5000;
    if (!(s >= 0) or s > 600) return 5000;
    return @intFromFloat(s * 1000);
}

const Link = @import("link.zig").Link;

// libc calls that the tests need and std.c does not declare
extern "c" fn pipe(fds: *[2]c_int) c_int;
extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

fn testPort() u16 {
    return 29500 + @as(u16, @intCast(@mod(std.Io.Timestamp.now(std.testing.io, .awake).nanoseconds, 400)));
}

fn sleepMs(ms: u64) void {
    const ts: std.c.timespec = .{ .sec = @intCast(ms / 1000), .nsec = @intCast((ms % 1000) * std.time.ns_per_ms) };
    _ = std.c.nanosleep(&ts, null);
}

/// The test's peer rank: it connects, then reads frames until a "done" frame or its link ends (keep), or closes its
/// end after `ms` (dies).
const Peer = struct {
    port: u16,
    die_ms: ?u64,

    fn run(p: Peer) void {
        var l = Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, p.port, 10) catch return;
        defer l.close(); // the process ends: its end closes
        if (p.die_ms) |ms| return sleepMs(ms);
        var buf: [64]u8 = undefined;
        while (true) {
            const f = l.recv(&buf) catch return;
            if (std.mem.eql(u8, f, "done")) return;
        }
    }
};

test "a start that fails ends the watch with no fault while the peer is up" {
    const port = testPort();
    const t = try std.Thread.spawn(.{}, Peer.run, .{Peer{ .port = port, .die_ms = null }});
    var l = try Link.listen(.{ 127, 0, 0, 1 }, port);
    const w = try Watch.start(std.testing.allocator, &.{l.fd}, &.{}, 0, null);
    // native.open fails after Lanes.init: end runs before Model.close frees the rings
    try std.testing.expectEqual(Outcome.failed, w.end(std.testing.allocator, .failed));
    l.close();
    t.join();
}

test "an idle follower whose rank 0 dies: the watch acts, then end finds the peer fault" {
    const port = testPort() + 400;
    const t = try std.Thread.spawn(.{}, Peer.run, .{Peer{ .port = port, .die_ms = 300 }});
    var l = try Link.listen(.{ 127, 0, 0, 1 }, port);
    defer l.close();
    const w = try Watch.start(std.testing.allocator, &.{l.fd}, &.{}, 1, null);
    t.join();
    // no receive runs: the watch alone finds the end (and aborts the rings) after its one-second wait
    var waited: u32 = 0;
    while (!w.lost.load(.acquire) and waited < 40) : (waited += 1) sleepMs(100);
    try std.testing.expect(w.lost.load(.acquire));
    try std.testing.expectEqual(Outcome.peer_fault, w.end(std.testing.allocator, .failed));
}

test "a sending leader whose follower dies gets PeerDown, and end finds the peer fault" {
    const port = testPort() + 800;
    const t = try std.Thread.spawn(.{}, Peer.run, .{Peer{ .port = port, .die_ms = 0 }});
    var l = try Link.listen(.{ 127, 0, 0, 1 }, port);
    defer l.close();
    const w = try Watch.start(std.testing.allocator, &.{l.fd}, &.{}, 0, null);
    t.join();
    // the first send after the close can still go out; a later one fails (no SIGPIPE: link.zig sends with NOSIGNAL)
    var err: ?anyerror = null;
    for (0..200) |_| {
        l.send("a round's frame") catch |e| {
            err = e;
            break;
        };
        sleepMs(10);
    }
    try std.testing.expectEqual(@as(?anyerror, error.PeerDown), err);
    try std.testing.expectEqual(Outcome.peer_fault, w.end(std.testing.allocator, .failed));
}

test "a follower that gets PeerDown calls end before the watch acts, and end still finds the peer fault" {
    const port = testPort() + 1200;
    const t = try std.Thread.spawn(.{}, Peer.run, .{Peer{ .port = port, .die_ms = 200 }});
    var l = try Link.listen(.{ 127, 0, 0, 1 }, port);
    defer l.close();
    const w = try Watch.start(std.testing.allocator, &.{l.fd}, &.{}, 1, null);
    var buf: [64]u8 = undefined;
    try std.testing.expectError(error.PeerDown, l.recv(&buf));
    // within the watch's one-second wait: the watch has not acted, end acts for it
    try std.testing.expect(!w.lost.load(.acquire));
    try std.testing.expectEqual(Outcome.peer_fault, w.end(std.testing.allocator, .failed));
    t.join();
}

test "a done frame, then the peer's close: end is orderly" {
    const port = testPort() + 1600;
    const t = try std.Thread.spawn(.{}, Peer.run, .{Peer{ .port = port, .die_ms = null }});
    var l = try Link.listen(.{ 127, 0, 0, 1 }, port);
    defer l.close();
    const w = try Watch.start(std.testing.allocator, &.{l.fd}, &.{}, 0, null);
    try std.testing.expect(!w.faulted());
    try l.send("done");
    t.join(); // the peer took done and closed its end: the watch may have seen it
    sleepMs(300);
    try std.testing.expect(!w.lost.load(.acquire));
    try std.testing.expectEqual(Outcome.orderly, w.end(std.testing.allocator, .done));
}

test "a descriptor that is no longer open counts as a link end (POLLNVAL), not as a poll to repeat" {
    var fds: [2]c_int = undefined;
    try std.testing.expectEqual(@as(c_int, 0), pipe(&fds));
    defer _ = std.c.close(fds[1]);
    const w = try Watch.start(std.testing.allocator, &.{fds[0]}, &.{}, 1, null);
    _ = std.c.close(fds[0]);
    var waited: u32 = 0;
    while (!w.lost.load(.acquire) and waited < 40) : (waited += 1) sleepMs(100);
    try std.testing.expect(w.lost.load(.acquire));
    try std.testing.expectEqual(Outcome.peer_fault, w.end(std.testing.allocator, .failed));
}

test "the exit 75 comes from Watch.start's grace alone: TF_DS_EXIT_ON_FAULT in the test's environment has no effect" {
    try std.testing.expectEqual(@as(c_int, 0), setenv("TF_DS_EXIT_ON_FAULT", "1", 1));
    defer _ = unsetenv("TF_DS_EXIT_ON_FAULT");
    try std.testing.expectEqual(@as(?u64, 5000), faultGrace());
    const port = testPort() + 2000;
    const t = try std.Thread.spawn(.{}, Peer.run, .{Peer{ .port = port, .die_ms = 0 }});
    var l = try Link.listen(.{ 127, 0, 0, 1 }, port);
    defer l.close();
    const w = try Watch.start(std.testing.allocator, &.{l.fd}, &.{}, 1, null);
    t.join();
    sleepMs(100);
    // a peer fault with no grace: end returns (the test runner goes on) instead of exit 75
    try std.testing.expectEqual(Outcome.peer_fault, w.end(std.testing.allocator, .failed));
}

test "note's descriptor never blocks on a full pipe, and the pipe's own descriptor stays blocking" {
    var fds: [2]c_int = undefined;
    try std.testing.expectEqual(@as(c_int, 0), pipe(&fds));
    defer _ = std.c.close(fds[0]);
    defer _ = std.c.close(fds[1]);
    const w = nonblockingCopy(fds[1]);
    try std.testing.expect(w >= 0);
    defer _ = std.c.close(w);
    const chunk: [4096]u8 = @splat('x');
    var full = false;
    // 4 MiB at most, more than a pipe holds: a write that would wait fails at once (EAGAIN)
    for (0..1024) |_| if (std.c.write(w, &chunk, chunk.len) < 0) {
        full = true;
        break;
    };
    try std.testing.expect(full);
    const flags: std.c.O = @bitCast(@as(u32, @intCast(std.c.fcntl(fds[1], std.c.F.GETFL))));
    try std.testing.expect(!flags.NONBLOCK);
}
