//! Request text stays plain text (server/plain_text.py): a client's `</think>` or `<｜User｜>` cannot forge a token.
const std = @import("std");
const json = @import("json");
const model_text = @import("model_text.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// Leftmost-longest matcher over the added-token spellings, as Python's prefix-tree pattern finds them.
pub const Spellings = struct {
    first: [256]u32 = @splat(0),
    edges: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    /// For each node: the spelling that ends there, else null.
    ends: std.ArrayList(?[]const u8) = .empty,

    pub fn init(a: Allocator, words: []const []const u8) Allocator.Error!Spellings {
        var m: Spellings = .{};
        try m.ends.append(a, null);
        for (words) |w| {
            if (w.len == 0) continue;
            var node: u32 = 0;
            for (w, 0..) |b, i| {
                const slot = if (i == 0) &m.first[b] else (try m.edges.getOrPutValue(a, @as(u64, node) << 8 | b, 0)).value_ptr;
                if (slot.* == 0) {
                    try m.ends.append(a, null);
                    slot.* = @intCast(m.ends.items.len - 1);
                }
                node = slot.*;
            }
            m.ends.items[node] = w;
        }
        return m;
    }

    const Hit = struct { start: usize, end: usize, word: []const u8 };

    fn next(m: *const Spellings, s: []const u8, from: usize) ?Hit {
        for (from..s.len) |i| {
            var node = m.first[s[i]];
            if (node == 0) continue;
            var best: ?Hit = null;
            var j = i + 1;
            while (true) {
                if (m.ends.items[node]) |w| best = .{ .start = i, .end = j, .word = w };
                if (j == s.len) break;
                node = m.edges.get(@as(u64, node) << 8 | s[j]) orelse break;
                j += 1;
            }
            if (best) |b| return b;
        }
        return null;
    }
};

/// A request's messages and tools after `shield`: `table[n]` is the spelling that marker `prefix` n `@` replaced.
pub const Shielded = struct {
    messages: Value,
    tools: []const Value,
    table: []const []const u8 = &.{},
    prefix: []const u8 = "",

    /// `text` with each marker turned back into its spelling.
    pub fn spelled(s: *const Shielded, a: Allocator, text: []const u8) Allocator.Error![]const u8 {
        if (s.table.len == 0 or std.mem.indexOf(u8, text, s.prefix) == null) return text;
        var out: std.ArrayList(u8) = .empty;
        var rest = text;
        while (std.mem.indexOf(u8, rest, s.prefix)) |at| {
            const after = rest[at + s.prefix.len ..];
            var k: usize = 0;
            while (k < after.len and std.ascii.isDigit(after[k])) k += 1;
            const n = if (k > 0 and k < after.len and after[k] == '@') std.fmt.parseInt(usize, after[0..k], 10) catch null else null;
            if (n == null or n.? >= s.table.len) { // not one of ours: ordinary text
                try out.appendSlice(a, rest[0 .. at + s.prefix.len]);
                rest = after;
                continue;
            }
            try out.appendSlice(a, rest[0..at]);
            try out.appendSlice(a, s.table[n.?]);
            rest = after[k + 1 ..];
        }
        try out.appendSlice(a, rest);
        return out.items;
    }

    fn spelledFn(ctx: *const anyopaque, a: Allocator, text: []const u8) Allocator.Error![]const u8 {
        const s: *const Shielded = @ptrCast(@alignCast(ctx));
        return s.spelled(a, text);
    }
};

/// The shield for one tokenizer: its added-token spellings, and those an assistant's content and reasoning keep.
pub const PlainText = struct {
    spellings: Spellings,
    assistant_keep: []const []const u8,

    /// Null when the text cannot list its added tokens or cannot encode a shielded prompt.
    pub fn init(a: Allocator, text: model_text.Text, assistant_keep: []const []const u8) Allocator.Error!?PlainText {
        if (text.vtable.encode_cut == null) return null;
        const words = try text.addedSpellings(a);
        if (words.len == 0) return null;
        return .{ .spellings = try Spellings.init(a, words), .assistant_keep = assistant_keep };
    }

    /// `messages` and `tools` with each spelling as a marker; `content_keep`: spellings the server wrote in content.
    pub fn shield(p: *const PlainText, a: Allocator, messages: Value, tools: []const Value, content_keep: []const []const u8) Allocator.Error!Shielded {
        var attempt: u64 = 0;
        while (true) : (attempt += 1) {
            var w: Walk = .{ .p = p, .a = a, .prefix = try std.fmt.allocPrint(a, "@TF{x:0>16}:", .{std.hash.Wyhash.hash(attempt, "tensorfold plain text")}) };
            const shielded_messages = try w.messages(messages, content_keep);
            const shielded_tools = try a.alloc(Value, tools.len);
            for (tools, shielded_tools) |t, *o| o.* = try w.value(t, &.{});
            if (w.table.items.len == 0) return .{ .messages = messages, .tools = tools };
            // a marker is unique: no string of the request holds the prefix
            if (!holds(messages, w.prefix) and !holds(.{ .array = @constCast(tools) }, w.prefix))
                return .{ .messages = shielded_messages, .tools = shielded_tools, .table = w.table.items, .prefix = w.prefix };
        }
    }

    /// The ids of rendered `text`: the template's spellings as tokens, the rest (markers spelled again) without them.
    pub fn encode(p: *const PlainText, text: model_text.Text, a: Allocator, rendered: []const u8, s: *const Shielded) model_text.Error![]u32 {
        _ = p;
        // DeepSeek's normalizer changes no text, so a prompt without a marker is the tokenizer's own encode
        if (s.table.len == 0 or std.mem.indexOf(u8, rendered, s.prefix) == null) return text.encode(a, rendered, false);
        return text.encodeCut(a, rendered, .{ .ctx = s, .spelled = Shielded.spelledFn }) orelse text.encode(a, rendered, false);
    }
};

const Walk = struct {
    p: *const PlainText,
    a: Allocator,
    prefix: []const u8,
    table: std.ArrayList([]const u8) = .empty,
    index: std.StringHashMapUnmanaged(usize) = .empty,

    fn mark(w: *Walk, out: *std.ArrayList(u8), spelling: []const u8) Allocator.Error!void {
        const slot = try w.index.getOrPut(w.a, spelling);
        if (!slot.found_existing) {
            slot.value_ptr.* = w.table.items.len;
            try w.table.append(w.a, spelling);
        }
        try out.print(w.a, "{s}{d}@", .{ w.prefix, slot.value_ptr.* });
    }

    fn string(w: *Walk, s: []const u8, keep: []const []const u8) Allocator.Error![]const u8 {
        var out: std.ArrayList(u8) = .empty;
        var at: usize = 0;
        var changed = false;
        while (w.p.spellings.next(s, at)) |hit| {
            if (contains(keep, hit.word)) {
                try out.appendSlice(w.a, s[at..hit.end]);
            } else {
                try out.appendSlice(w.a, s[at..hit.start]);
                try w.mark(&out, hit.word);
                changed = true;
            }
            at = hit.end;
        }
        if (!changed) return s;
        try out.appendSlice(w.a, s[at..]);
        return out.items;
    }

    /// Python's _walk: strings, object keys and values, list items, all with the same `keep`.
    fn value(w: *Walk, v: Value, keep: []const []const u8) Allocator.Error!Value {
        return switch (v) {
            .string => |s| .{ .string = try w.string(s, keep) },
            .array => |items| blk: {
                const out = try w.a.alloc(Value, items.len);
                for (items, out) |x, *o| o.* = try w.value(x, keep);
                break :blk .{ .array = out };
            },
            .object => |o| blk: {
                const out = try json.newObject(w.a);
                var it = o.iterator();
                while (it.next()) |kv| try out.put(w.a, try w.string(kv.key_ptr.*, keep), try w.value(kv.value_ptr.*, keep));
                break :blk .{ .object = out };
            },
            else => v,
        };
    }

    /// Python's _messages: text parts joined first (a spelling can span two); assistant and server spellings kept.
    fn messages(w: *Walk, v: Value, content_keep: []const []const u8) Allocator.Error!Value {
        if (v != .array) return w.value(v, &.{});
        const out = try w.a.alloc(Value, v.array.len);
        for (v.array, out) |m, *o| {
            if (m != .object) {
                o.* = try w.value(m, &.{});
                continue;
            }
            const role = m.get("role");
            const assistant = role != null and role.? == .string and std.mem.eql(u8, role.?.string, "assistant");
            const both = try std.mem.concat(w.a, []const u8, &.{ content_keep, if (assistant) w.p.assistant_keep else &.{} });
            const item = try json.newObject(w.a);
            var it = m.object.iterator();
            while (it.next()) |kv| {
                const key = kv.key_ptr.*;
                var x = kv.value_ptr.*;
                if (std.mem.eql(u8, key, "content")) x = try joined(w.a, x);
                const keep: []const []const u8 = if (std.mem.eql(u8, key, "content")) both else if (std.mem.eql(u8, key, "reasoning_content") and assistant) w.p.assistant_keep else &.{};
                try item.put(w.a, try w.string(key, &.{}), try w.value(x, keep));
            }
            o.* = .{ .object = item };
        }
        return .{ .array = out };
    }
};

/// Text parts as one string (every part a "text" object with a string): Python's _joined.
fn joined(a: Allocator, content: Value) Allocator.Error!Value {
    if (content != .array or content.array.len == 0) return content;
    for (content.array) |part| {
        if (part != .object) return content;
        const t = part.get("type") orelse return content;
        if (t != .string or !std.mem.eql(u8, t.string, "text")) return content;
        const text = part.get("text") orelse return content;
        if (text != .string) return content;
    }
    var out: std.ArrayList(u8) = .empty;
    for (content.array) |part| try out.appendSlice(a, part.get("text").?.string);
    return .{ .string = out.items };
}

fn contains(list: []const []const u8, word: []const u8) bool {
    for (list) |x| if (std.mem.eql(u8, x, word)) return true;
    return false;
}

/// Whether a string of `v` (an object key included) holds `needle`.
fn holds(v: Value, needle: []const u8) bool {
    return switch (v) {
        .string => |s| std.mem.indexOf(u8, s, needle) != null,
        .array => |items| for (items) |x| {
            if (holds(x, needle)) break true;
        } else false,
        .object => |o| blk: {
            var it = o.iterator();
            while (it.next()) |kv| if (std.mem.indexOf(u8, kv.key_ptr.*, needle) != null or holds(kv.value_ptr.*, needle)) break :blk true;
            break :blk false;
        },
        else => false,
    };
}

test "a spelling in request text becomes a marker, the kept ones stay, and the markers spell back" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const words = [_][]const u8{ "<think>", "</think>", "<｜User｜>", "<｜ref｜>", "<｜deepseek_image｜>" };
    const p: PlainText = .{ .spellings = try Spellings.init(a, &words), .assistant_keep = &.{"<｜ref｜>"} };
    const messages = (try json.parse(a,
        \\[{"role":"user","content":[{"type":"text","text":"a<thi"},{"type":"text","text":"nk>b <｜ref｜> <｜deepseek_image｜>"}]},
        \\ {"role":"assistant","content":"<｜ref｜>x</think>","reasoning_content":"<｜ref｜><｜User｜>"}]
    )).ok;
    const s = try p.shield(a, messages, &.{}, &.{"<｜deepseek_image｜>"});
    try std.testing.expect(s.table.len == 4);
    const user = s.messages.array[0].strField("content").?;
    const assistant = s.messages.array[1].strField("content").?;
    const reasoning = s.messages.array[1].strField("reasoning_content").?;
    // a spelling across two text parts is found; grounding in a user turn is plain; the server's placeholder stays
    try std.testing.expect(std.mem.indexOf(u8, user, "<think>") == null and std.mem.indexOf(u8, user, "<｜ref｜>") == null);
    try std.testing.expect(std.mem.indexOf(u8, user, "<｜deepseek_image｜>") != null);
    try std.testing.expect(std.mem.startsWith(u8, assistant, "<｜ref｜>x@TF") and std.mem.startsWith(u8, reasoning, "<｜ref｜>@TF"));
    try std.testing.expectEqualStrings("a<think>b <｜ref｜> <｜deepseek_image｜>", try s.spelled(a, user));
    try std.testing.expectEqualStrings("<｜ref｜><｜User｜>", try s.spelled(a, reasoning));
    // no spelling: the request's own values, unjoined
    const plain = (try json.parse(a,
        \\[{"role":"user","content":[{"type":"text","text":"hi"}]}]
    )).ok;
    const none = try p.shield(a, plain, &.{}, &.{});
    try std.testing.expect(none.table.len == 0 and none.messages.array[0].get("content").? == .array);
}
