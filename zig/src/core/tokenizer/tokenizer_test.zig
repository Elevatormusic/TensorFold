//! The tokenizer's cut encode and memo tests.

const std = @import("std");
const tokenizer = @import("tokenizer.zig");
const Allocator = std.mem.Allocator;
const Tokenizer = tokenizer.Tokenizer;
const Memo = tokenizer.Memo;
const parse = tokenizer.parse;

test "a cut encode: the text's own added tokens are tokens, the restored text between them is plain text" {
    const json =
        \\{"added_tokens":[{"id":10,"content":"<|end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
        \\  {"id":11,"content":"<t>","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false}],
        \\ "normalizer":{"type":"Sequence","normalizers":[]},
        \\ "pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":" ?\\S+|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},
        \\ "decoder":{"type":"ByteLevel"},
        \\ "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
        \\  "vocab":{"a":0,"b":1,"Ġ":2,"ab":3,"Ġa":4,"Ġab":5,"c":6,"<":7,"t":8,">":9},"merges":[["Ġ","a"],"a b","Ġa b"]}}
    ;
    const a = std.testing.allocator;
    var t = try parse(a, json);
    defer t.deinit();
    const Marks = struct {
        fn spelled(_: *const anyopaque, al: Allocator, text: []const u8) Allocator.Error![]const u8 {
            return std.mem.replaceOwned(u8, al, text, "@0@", "<t>");
        }
    };
    const restore: Tokenizer.Restore = .{ .ctx = &t, .spelled = Marks.spelled };
    const ids = try t.encodeCut(a, "ab<|end|>a@0@b<t>c", restore);
    defer a.free(ids);
    // added tokens take ids past the model's 10 (tokenizers' rule): "<|end|>" 10, "<t>" 11; the marker's "<t>" is plain
    try std.testing.expectEqualSlices(u32, &.{ 3, 10, 0, 7, 8, 9, 1, 11, 6 }, ids);
    // no marker and no added token: the same ids as encode, through the memo too
    t.memo = try Memo.create(a, 1 << 20);
    const plain = try t.encode(a, "ab abc");
    defer a.free(plain);
    const cut = try t.encodeCut(a, "ab abc", restore);
    defer a.free(cut);
    try std.testing.expectEqualSlices(u32, plain, cut);
}

test "with the memo, long segments encode as without it, and the second encode takes their kept ids" {
    const json =
        \\{"added_tokens":[{"id":7,"content":"<|end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
        \\ "pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":" ?\\S+|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},
        \\ "decoder":{"type":"ByteLevel"},
        \\ "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
        \\  "vocab":{"a":0,"b":1,"Ġ":2,"ab":3,"Ġa":4,"Ġab":5,"c":6},"merges":[["Ġ","a"],"a b","Ġa b"]}}
    ;
    const a = std.testing.allocator;
    var t = try parse(a, json);
    defer t.deinit();
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(a);
    for (0..300) |_| try text.appendSlice(a, "ab abc ");
    try text.appendSlice(a, "<|end|>");
    for (0..300) |_| try text.appendSlice(a, " ab cab");
    const plain = try t.encode(a, text.items);
    defer a.free(plain);
    t.memo = try Memo.create(a, 1 << 20);
    for (0..2) |_| {
        const kept = try t.encode(a, text.items);
        defer a.free(kept);
        try std.testing.expectEqualSlices(u32, plain, kept);
    }
    try std.testing.expectEqual(@as(u64, 2), t.memo.?.hits); // both segments, the second time
}

test "a kept segment's ids equal a fresh encode's wherever the segment sits" {
    const added =
        \\{"added_tokens":[{"id":20,"content":"<|end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
        \\ {"id":21,"content":"<|l|>","single_word":false,"lstrip":true,"rstrip":false,"normalized":false,"special":true},
        \\ {"id":22,"content":"<|r|>","single_word":false,"lstrip":false,"rstrip":true,"normalized":false,"special":true}],
    ;
    const kinds = [_][]const u8{
        added ++
            \\ "pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":" ?\\S+|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},
            \\ "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
            \\  "vocab":{"a":0,"b":1,"Ġ":2,"ab":3,"Ġa":4,"Ġab":5,"c":6,"x":7,"ï":8,"¿":9,"½":10},"merges":[["Ġ","a"],"a b","Ġa b"]}}
        ,
        added ++
            \\ "pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"first","split":true},
            \\ "model":{"type":"BPE","unk_token":"<unk>","fuse_unk":true,"byte_fallback":true,
            \\  "vocab":{"<unk>":0,"a":1,"b":2,"▁":3,"ab":4,"▁a":5,"▁ab":6,"c":7,"x":8,"<0xEF>":9,"<0xBF>":10,"<0xBD>":11},"merges":["a b","▁ a","▁a b"]}}
        ,
    };
    const a = std.testing.allocator;
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(a);
    for (0..200) |i| try body.appendSlice(a, if (i % 3 == 0) "ab abc " else "cab ab ");
    const b = body.items;
    // first or not, after lstrip and rstrip tokens, with a leading space, around a lossy byte, repeated in one prompt
    const texts = [_][]const []const u8{
        &.{b},               &.{ "<|end|>", b }, &.{ b, "<|end|>", b }, &.{ "x  <|l|>", b },
        &.{ "<|r|>   ", b }, &.{ " ", b },       &.{ b, "\xff", b },    &.{ "<|end|> ", b, "<|r|>", b },
    };
    for (kinds, 0..) |json, kind| {
        var t = try parse(a, json);
        defer t.deinit();
        var fresh: [texts.len][]u32 = undefined;
        for (texts, &fresh) |parts, *ids| {
            const s = try std.mem.concat(a, u8, parts);
            defer a.free(s);
            ids.* = try t.encode(a, s);
        }
        defer for (fresh) |ids| a.free(ids);
        // the Metaspace tokenizer prepends to the first segment alone, so a memo blind to `first` fails below
        if (kind == 1) try std.testing.expect(!std.mem.eql(u32, fresh[0], fresh[1][1..]));
        t.memo = try Memo.create(a, 1 << 20);
        for (0..2) |round| for (0..texts.len) |k| {
            const i = if (round == 0) k else texts.len - 1 - k;
            const s = try std.mem.concat(a, u8, texts[i]);
            defer a.free(s);
            const kept = try t.encode(a, s);
            defer a.free(kept);
            try std.testing.expectEqualSlices(u32, fresh[i], kept);
        };
        try std.testing.expect(t.memo.?.hits >= 10); // every long segment of the second round, at least
    }
}
