//! The 1017 chat template on this engine's template module: it compiles and renders. The template is
//! tools/dsv41-zig/serving/chat_template.jinja (the 1017 checkpoint ships none; this is ours, checked against DeepSeek's
//! reference renderings). Not part of `zig build test` (the template module is not a test root there); run it with:
//!
//!   TPL=tools/dsv41-zig/serving/chat_template.jinja \
//!     zig test -lc --dep template -Mroot=zig/tests/dsv41/template_check.zig \
//!       -Mtemplate=zig/src/core/template/template.zig
const std = @import("std");
const template = @import("template");

test "the 1017 chat template compiles and renders (TPL)" {
    const path = std.c.getenv("TPL") orelse return error.SkipZigTest;
    const a = std.testing.allocator;
    const src = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, std.mem.span(path), a, .limited(1 << 20));
    defer a.free(src);
    var diag = template.Diag{};
    defer if (diag.msg.len > 0) a.free(diag.msg);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, "{\"messages\":[{\"role\":\"system\",\"content\":\"You are helpful.\"},{\"role\":\"user\",\"content\":\"Hi\"},{\"role\":\"assistant\",\"content\":\"Hello\"},{\"role\":\"user\",\"content\":\"Bye\"}]}", .{});
    defer parsed.deinit();
    const out = template.renderChat(a, src, parsed.value.object.get("messages").?, .null, parsed.value, true, &diag) catch |err| {
        std.debug.print("TPL FAIL {t}: {s}\n", .{ err, diag.msg });
        return err;
    };
    defer a.free(out);
    // the header's system span and the two user turns are there, and the generation prompt opens the assistant turn
    try std.testing.expect(std.mem.indexOf(u8, out, "You are helpful.") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "Hi") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "Bye") != null);
    std.debug.print("TPL {s}: compiled and rendered {d} bytes\n", .{ std.mem.span(path), out.len });
}
