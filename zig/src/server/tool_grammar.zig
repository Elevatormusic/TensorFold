//! tool_choice "required" or a named function as a grammar: calls in the template's markup, arguments in their schema.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const prompt = @import("prompt.zig");
const tool_specs = @import("tool_specs.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// How the template writes a call: inside <tool_call>, Qwen3-Coder's XML (Nemotron, Qwen3.5) or a JSON object; or
/// DeepSeek-V4.1's DSML block (<｜DSML｜ calls> with an invoke a call and a parameter an argument).
pub const Style = enum { xml, json, dsml41 };

const dsml = "\u{ff5c}DSML\u{ff5c}";
const dsml41_open = "<" ++ dsml ++ " calls>";
const dsml41_close = "</" ++ dsml ++ " calls>";

const probe = "tfprobe_fn";
const key = "tfprobe_k";

/// Blanks around and between calls: the template's newlines, never an endless run.
const blanks = "[ \\t\\r\\n]{0,8}";

/// A DSML string argument's value: any text with no `<`, blanks, `/`, blanks, `｜` in it, so a look-alike of the end
/// tag (`</｜DSML｜  parameter>`) cannot run on inside the value and the exact end tag is the only way out.
/// The states of "no match yet": after text (s0), after `<` and blanks (s1), after `</` and blanks (s2).
const string_value = "([^<]|<([ \\t\\r\\n<]|/[ \\t\\r\\n]*<)*([^ \\t\\r\\n/<]|/[ \\t\\r\\n]*[^ \\t\\r\\n<\u{ff5c}]))*" ++
    "(<([ \\t\\r\\n<]|/[ \\t\\r\\n]*<)*(/[ \\t\\r\\n]*)?)?";

/// The most blank characters between an argument's JSON tokens (grammar.py BLANKS).
const value_blanks = 32;

/// The style of `rendered`, a turn that calls `probe` with {key: "v"}; null: markup this grammar does not write.
pub fn style(rendered: []const u8) ?Style {
    if (std.mem.lastIndexOf(u8, rendered, dsml41_open)) |d| {
        if (std.mem.indexOf(u8, rendered[d..], "<" ++ dsml ++ " invoke name=\"" ++ probe ++ "\">") != null) return .dsml41;
    }
    const at = std.mem.lastIndexOf(u8, rendered, "<tool_call>") orelse return null;
    const text = rendered[at..];
    if (std.mem.indexOf(u8, text, "<function=" ++ probe ++ ">") != null and std.mem.indexOf(u8, text, "<parameter=" ++ key ++ ">") != null) return .xml;
    if (std.mem.indexOf(u8, text, "\"" ++ probe ++ "\"") != null and std.mem.indexOf(u8, text, "\"arguments\"") != null) return .json;
    return null;
}

/// The grammar for a reply that must call one of `tools`: one call, or (`parallel`) one or more, then the end.
pub fn structure(srv: *Server, cx: *Cx, tools: []const Value, parallel: bool, thinking: bool) errors.Refused!api.Structure {
    const probe_turn = (try json.parse(cx.a, "[{\"role\": \"user\", \"content\": \"x\"}, {\"role\": \"assistant\", \"content\": \"\", \"tool_calls\": [{\"id\": \"call_0\", \"type\": \"function\", \"function\": {\"name\": \"" ++ probe ++ "\", \"arguments\": {\"" ++ key ++ "\": \"v\"}}}]}]")).ok;
    var scratch: Cx = .{ .a = cx.a };
    const rendered = if (prompt.renderIds(srv, &scratch, probe_turn, &.{}, false, null, false)) |ids| try srv.text.decode(cx.a, ids) else |_| "";
    const s = style(rendered) orelse return cx.refuse("tool_choice \"required\" or a named function needs a chat template that writes <tool_call> calls (XML parameters or a JSON object) or DSML calls, and this one does not: send \"auto\"");
    if (s == .dsml41 and !try dsmlNamesOk(cx.a, tools)) return cx.refuse("tool_choice \"required\" or a named function cannot write a tool or argument name with a double quote in a DSML call: send \"auto\"");
    const text = try build(cx.a, s, tools, parallel);
    return srv.compiledStructure(cx, .structural_tag, text, "tool_choice \"required\" or a named function", thinking);
}

/// The structural tag's JSON text (xgrammar's form).
pub fn build(a: Allocator, s: Style, tools: []const Value, parallel: bool) Allocator.Error![]const u8 {
    const calls = try a.alloc(Value, tools.len);
    for (tools, calls) |t, *c| c.* = try call(a, s, try tool_specs.toolName(a, t), try parameters(a, t));
    const one: Value = if (calls.len == 1) calls[0] else try node(a, "or", &.{.{ "elements", .{ .array = calls } }});
    const ws = try node(a, "regex", &.{.{ "pattern", .{ .string = blanks } }});
    var parts: std.ArrayList(Value) = .empty;
    try parts.append(a, ws);
    // DSML: every call of the reply sits inside one calls block
    if (s == .dsml41) try parts.appendSlice(a, &.{ try constant(a, dsml41_open), ws });
    try parts.append(a, one);
    if (parallel) try parts.append(a, try node(a, "star", &.{.{ "content", try node(a, "sequence", &.{.{ "elements", try list(a, &.{ ws, one }) }}) }}));
    if (s == .dsml41) try parts.appendSlice(a, &.{ ws, try constant(a, dsml41_close) });
    try parts.append(a, ws);
    const format = try node(a, "sequence", &.{.{ "elements", .{ .array = parts.items } }});
    return json.stringify(a, try node(a, "structural_tag", &.{.{ "format", format }}), .{ .ascii = false, .compact = true });
}

/// One call to `name`: <tool_call>, the name and its arguments in the template's style, </tool_call>.
fn call(a: Allocator, s: Style, name: []const u8, schema: Value) Allocator.Error!Value {
    const ws = try node(a, "regex", &.{.{ "pattern", .{ .string = blanks } }});
    const elements = switch (s) {
        .xml => try list(a, &.{
            try constant(a, "<tool_call>"),                                        ws,
            try constant(a, try std.fmt.allocPrint(a, "<function={s}>", .{name})), try arguments(a, schema, "qwen_xml"),
            try constant(a, "</function>"),                                        ws,
            try constant(a, "</tool_call>"),
        }),
        .json => blk: {
            const props = try json.newObject(a);
            const fixed = try json.newObject(a);
            try fixed.put(a, "const", .{ .string = name });
            try props.put(a, "name", .{ .object = fixed });
            try props.put(a, "arguments", schema);
            const whole = try json.newObject(a);
            try whole.put(a, "type", .{ .string = "object" });
            try whole.put(a, "properties", .{ .object = props });
            try whole.put(a, "required", try list(a, &.{ .{ .string = "name" }, .{ .string = "arguments" } }));
            break :blk try list(a, &.{ try constant(a, "<tool_call>"), ws, try arguments(a, .{ .object = whole }, "json"), ws, try constant(a, "</tool_call>") });
        },
        .dsml41 => try dsmlInvoke(a, name, schema),
    };
    return node(a, "sequence", &.{.{ "elements", elements }});
}

/// One DSML invoke: the name, each required argument in the schema's order, then each optional one at most once (in the
/// schema's order), the close. A plain string argument is the template's string="true" form (text up to the parameter's
/// close); any other is string="false" with its value as JSON in the property's schema, a string that has enum, const,
/// pattern or length limits too (tool_parse.zig dsml41Calls reads both). A schema that is more than direct
/// `properties` (a $ref, allOf, anyOf, oneOf, patternProperties, a schema for additionalProperties, or a required name
/// no property declares) goes whole to xgrammar's own DeepSeek-V4.1 parameter style.
fn dsmlInvoke(a: Allocator, name: []const u8, schema: Value) Allocator.Error!Value {
    const ws = try node(a, "regex", &.{.{ "pattern", .{ .string = blanks } }});
    var parts: std.ArrayList(Value) = .empty;
    try parts.append(a, try constant(a, try std.fmt.allocPrint(a, "<" ++ dsml ++ " invoke name=\"{s}\">", .{name})));
    const props: ?Value = if (schema.get("properties")) |pv| (if (pv == .object) pv else null) else null;
    if (needsFullSchema(schema, props)) {
        try parts.appendSlice(a, &.{ ws, try arguments(a, schema, "deepseek_v4_1_xml") });
    } else if (props) |ps| {
        var required: std.ArrayList([]const u8) = .empty;
        if (schema.get("required")) |r| if (r == .array) for (r.array) |x| if (x == .string) try required.append(a, x.string);
        // the required arguments in order, each once
        for (ps.object.keys(), ps.object.values()) |k, v| {
            if (!contains(required.items, k)) continue;
            try parts.appendSlice(a, &.{ ws, try dsmlParameter(a, k, v) });
        }
        // the optional ones: each may come once, after the required ones
        for (ps.object.keys(), ps.object.values()) |k, v| {
            if (contains(required.items, k)) continue;
            const one = try node(a, "sequence", &.{.{ "elements", try list(a, &.{ ws, try dsmlParameter(a, k, v) }) }});
            try parts.append(a, try node(a, "or", &.{.{ "elements", try list(a, &.{ one, try constant(a, "") }) }}));
        }
    }
    try parts.appendSlice(a, &.{ ws, try constant(a, "</" ++ dsml ++ " invoke>") });
    return .{ .array = parts.items };
}

/// Whether the call's arguments need more than the direct `properties` of the parameter schema.
fn needsFullSchema(schema: Value, props: ?Value) bool {
    for ([_][]const u8{ "$ref", "allOf", "anyOf", "oneOf", "patternProperties" }) |k| if (schema.get(k) != null) return true;
    if (schema.get("additionalProperties")) |ap| if (ap == .object) return true;
    if (schema.get("required")) |r| if (r == .array) for (r.array) |x| {
        if (x != .string) continue;
        if (props == null or props.?.get(x.string) == null) return true;
    };
    return false;
}

/// The schema keywords that describe a string without limiting it.
const plain_keywords = [_][]const u8{ "type", "description", "title", "default", "examples", "example", "$comment", "deprecated", "readOnly", "writeOnly" };

/// A string schema with nothing but its type and notes: its value is raw text. enum, const, pattern, minLength,
/// maxLength, format and the rest limit it, and only the JSON form carries those limits.
fn plainString(schema: Value) bool {
    const t = schema.get("type") orelse return false;
    if (t != .string or !std.mem.eql(u8, t.string, "string")) return false;
    for (schema.object.keys()) |k| if (!contains(&plain_keywords, k)) return false;
    return true;
}

fn dsmlParameter(a: Allocator, key_name: []const u8, schema: Value) Allocator.Error!Value {
    const is_string = plainString(schema);
    const begin = try std.fmt.allocPrint(a, "<" ++ dsml ++ " parameter name=\"{s}\" string=\"{s}\">", .{ key_name, if (is_string) "true" else "false" });
    const content = if (is_string) try node(a, "regex", &.{.{ "pattern", .{ .string = string_value } }}) else try arguments(a, schema, "json");
    return node(a, "tag", &.{ .{ "begin", .{ .string = begin } }, .{ "content", content }, .{ "end", .{ .string = "</" ++ dsml ++ " parameter>" } } });
}

/// DSML writes a name inside double quotes and dsml41Calls ends it at the first one: a name with a quote cannot be a call.
fn quoteFree(name: []const u8) bool {
    return std.mem.indexOfScalar(u8, name, '"') == null;
}

/// The schema a local `$ref` ("#/$defs/N" or "#/definitions/N") names in `root`, or null.
fn localRef(root: Value, ref: []const u8) ?Value {
    inline for (.{ "#/$defs/", "#/definitions/" }) |prefix| {
        if (std.mem.startsWith(u8, ref, prefix)) {
            const defs = root.get(prefix[2 .. prefix.len - 1]) orelse return null;
            return defs.get(ref[prefix.len..]);
        }
    }
    return null;
}

/// Whether the argument names the schema can write (its `properties` keys, also through allOf, anyOf, oneOf and the
/// local $refs it uses; a definition nothing refers to is never written) can be DSML attributes. `depth` stops a $ref cycle.
fn argNamesOk(root: Value, schema: Value, depth: usize) bool {
    if (depth > 16) return true;
    if (schema.get("properties")) |ps| if (ps == .object) for (ps.object.keys()) |k| if (!quoteFree(k)) return false;
    for ([_][]const u8{ "allOf", "anyOf", "oneOf" }) |kw| if (schema.get(kw)) |members| if (members == .array) for (members.array) |m| {
        if (m == .object and !argNamesOk(root, m, depth + 1)) return false;
    };
    if (schema.get("$ref")) |ref| if (ref == .string) if (localRef(root, ref.string)) |target| {
        if (target == .object and !argNamesOk(root, target, depth + 1)) return false;
    };
    return true;
}

/// Whether every offered tool name and argument name can be written in DSML attributes.
fn dsmlNamesOk(a: Allocator, tools: []const Value) Allocator.Error!bool {
    for (tools) |t| {
        if (!quoteFree(try tool_specs.toolName(a, t))) return false;
        const params = try parameters(a, t);
        if (!argNamesOk(params, params, 0)) return false;
    }
    return true;
}

fn contains(names: []const []const u8, name: []const u8) bool {
    for (names) |n| if (std.mem.eql(u8, n, name)) return true;
    return false;
}

/// The tool's parameter schema (OpenAI's function.parameters, a bare tool's parameters or input_schema).
fn parameters(a: Allocator, tool: Value) Allocator.Error!Value {
    const function = if (tool.get("function")) |f| (if (f == .object) f else tool) else tool;
    for ([_][]const u8{ "parameters", "input_schema" }) |k| if (function.get(k)) |p| if (p == .object) return p;
    const empty = try json.newObject(a);
    try empty.put(a, "type", .{ .string = "object" });
    try empty.put(a, "properties", .{ .object = try json.newObject(a) });
    return .{ .object = empty };
}

fn arguments(a: Allocator, schema: Value, style_name: []const u8) Allocator.Error!Value {
    return node(a, "json_schema", &.{ .{ "json_schema", schema }, .{ "style", .{ .string = style_name } }, .{ "max_whitespace_cnt", try json.intValue(a, value_blanks) } });
}

fn constant(a: Allocator, text: []const u8) Allocator.Error!Value {
    return node(a, "const_string", &.{.{ "value", .{ .string = text } }});
}

fn list(a: Allocator, items: []const Value) Allocator.Error!Value {
    return .{ .array = try a.dupe(Value, items) };
}

/// {"type": kind, fields...}.
fn node(a: Allocator, kind: []const u8, entries: []const struct { []const u8, Value }) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "type", .{ .string = kind });
    for (entries) |e| try o.put(a, e[0], e[1]);
    return .{ .object = o };
}

test "the probe call's markup picks the style" {
    try std.testing.expectEqual(Style.xml, style("<think></think><tool_call>\n<function=tfprobe_fn>\n<parameter=tfprobe_k>\nv\n</parameter>\n</function>\n</tool_call>\n").?);
    try std.testing.expectEqual(Style.json, style("<tool_call>\n{\"name\": \"tfprobe_fn\", \"arguments\": {\"tfprobe_k\": \"v\"}}\n</tool_call>").?);
    try std.testing.expectEqual(@as(?Style, null), style("<|tool_call>call:tfprobe_fn{tfprobe_k:<|\"|>v<|\"|>}<tool_call|>"));
    try std.testing.expectEqual(Style.dsml41, style("<\u{ff5c}Assistant\u{ff5c}></think>\n\n<\u{ff5c}DSML\u{ff5c} calls>\n<\u{ff5c}DSML\u{ff5c} invoke name=\"tfprobe_fn\">\n<\u{ff5c}DSML\u{ff5c} parameter name=\"tfprobe_k\" string=\"true\">v</\u{ff5c}DSML\u{ff5c} parameter>\n</\u{ff5c}DSML\u{ff5c} invoke>\n</\u{ff5c}DSML\u{ff5c} calls>").?);
}

test "a required call's grammar names every offered tool and its schema; one call without parallel" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}, {\"type\": \"function\", \"function\": {\"name\": \"now\"}}]")).ok;
    const many = try build(a, .xml, tools.array, true);
    try std.testing.expect(std.mem.indexOf(u8, many, "\"value\":\"<function=get_weather>\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, many, "\"value\":\"<function=now>\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, many, "\"style\":\"qwen_xml\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, many, "\"required\":[\"city\"]") != null);
    try std.testing.expect(std.mem.indexOf(u8, many, "\"type\":\"star\"") != null);
    const single = try build(a, .json, tools.array[0..1], false);
    try std.testing.expect(std.mem.indexOf(u8, single, "\"type\":\"star\"") == null and std.mem.indexOf(u8, single, "\"type\":\"or\"") == null);
    try std.testing.expect(std.mem.indexOf(u8, single, "\"const\":\"get_weather\"") != null);
}

test "a DSML required call: one calls block, the invoke's name, string and JSON parameters" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}, \"days\": {\"type\": \"integer\"}, \"unit\": {\"type\": \"string\"}}, \"required\": [\"city\", \"days\"]}}}]")).ok;
    const g = try build(a, .dsml41, tools.array, false);
    try std.testing.expect(std.mem.indexOf(u8, g, "\"value\":\"<\u{ff5c}DSML\u{ff5c} calls>\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, g, "\"value\":\"</\u{ff5c}DSML\u{ff5c} calls>\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, g, "invoke name=\\\"get_weather\\\">") != null);
    try std.testing.expect(std.mem.indexOf(u8, g, "parameter name=\\\"city\\\" string=\\\"true\\\">") != null);
    try std.testing.expect(std.mem.indexOf(u8, g, "parameter name=\\\"days\\\" string=\\\"false\\\">") != null);
    // a string value is the regex that keeps end-tag look-alikes out, not any text
    try std.testing.expect(std.mem.indexOf(u8, g, "\"type\":\"any_text\"") == null);
    try std.testing.expect(std.mem.indexOf(u8, g, "\"type\":\"regex\",\"pattern\":\"([^<]|<(") != null);
    // the optional unit may come once, after the required ones (an or with an empty choice, not a star)
    const unit = std.mem.indexOf(u8, g, "parameter name=\\\"unit\\\"").?;
    try std.testing.expect(std.mem.indexOf(u8, g, "parameter name=\\\"days\\\"").? < unit);
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, g, "\"type\":\"star\""));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, g, "{\"type\":\"const_string\",\"value\":\"\"}"));
}

test "a DSML string argument with enum, pattern or length limits is a JSON value in its schema" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\": \"function\", \"function\": {\"name\": \"w\", \"parameters\": {\"type\": \"object\", \"properties\": {\"unit\": {\"type\": \"string\", \"enum\": [\"celsius\", \"fahrenheit\"]}, \"city\": {\"type\": \"string\", \"description\": \"a city\"}, \"code\": {\"type\": \"string\", \"maxLength\": 3}}, \"required\": [\"unit\", \"city\", \"code\"]}}}]")).ok;
    const g = try build(a, .dsml41, tools.array, false);
    try std.testing.expect(std.mem.indexOf(u8, g, "parameter name=\\\"unit\\\" string=\\\"false\\\">") != null);
    try std.testing.expect(std.mem.indexOf(u8, g, "\"enum\":[\"celsius\",\"fahrenheit\"]") != null);
    try std.testing.expect(std.mem.indexOf(u8, g, "parameter name=\\\"code\\\" string=\\\"false\\\">") != null);
    try std.testing.expect(std.mem.indexOf(u8, g, "\"maxLength\":3") != null);
    // a string with only a note stays raw text
    try std.testing.expect(std.mem.indexOf(u8, g, "parameter name=\\\"city\\\" string=\\\"true\\\">") != null);
}

test "a DSML schema that is not direct properties goes whole to the DeepSeek-V4.1 parameter style" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\": \"function\", \"function\": {\"name\": \"r\", \"parameters\": {\"$ref\": \"#/$defs/args\", \"$defs\": {\"args\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}}}, {\"type\": \"function\", \"function\": {\"name\": \"m\", \"parameters\": {\"type\": \"object\", \"additionalProperties\": {\"type\": \"integer\"}}}}, {\"type\": \"function\", \"function\": {\"name\": \"d\", \"parameters\": {\"type\": \"object\", \"properties\": {\"x\": {\"type\": \"integer\"}}, \"required\": [\"x\"]}}}]")).ok;
    const g = try build(a, .dsml41, tools.array, false);
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, g, "\"style\":\"deepseek_v4_1_xml\""));
    // the direct-properties tool keeps the hand-written parameter
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, g, "parameter name=\\\"x\\\" string=\\\"false\\\">"));
}

test "a DSML name with a double quote is refused, not written as a broken attribute" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const fine = (try json.parse(a, "[{\"type\": \"function\", \"function\": {\"name\": \"w\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}}}}]")).ok;
    try std.testing.expect(try dsmlNamesOk(a, fine.array));
    const tool = (try json.parse(a, "[{\"type\": \"function\", \"function\": {\"name\": \"a\\\"b\"}}]")).ok;
    try std.testing.expect(!try dsmlNamesOk(a, tool.array));
    const prop = (try json.parse(a, "[{\"type\": \"function\", \"function\": {\"name\": \"w\", \"parameters\": {\"type\": \"object\", \"properties\": {\"a\\\"b\": {\"type\": \"string\"}}}}}]")).ok;
    try std.testing.expect(!try dsmlNamesOk(a, prop.array));
    const nested = (try json.parse(a, "[{\"type\": \"function\", \"function\": {\"name\": \"w\", \"parameters\": {\"allOf\": [{\"type\": \"object\", \"properties\": {\"a\\\"b\": {\"type\": \"string\"}}}]}}}]")).ok;
    try std.testing.expect(!try dsmlNamesOk(a, nested.array));
    // a definition nothing refers to is never written: it does not refuse the call; one a $ref uses does
    const unused = (try json.parse(a, "[{\"type\": \"function\", \"function\": {\"name\": \"w\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"$defs\": {\"x\": {\"type\": \"object\", \"properties\": {\"a\\\"b\": {\"type\": \"string\"}}}}}}}]")).ok;
    try std.testing.expect(try dsmlNamesOk(a, unused.array));
    const used = (try json.parse(a, "[{\"type\": \"function\", \"function\": {\"name\": \"w\", \"parameters\": {\"$ref\": \"#/$defs/x\", \"$defs\": {\"x\": {\"type\": \"object\", \"properties\": {\"a\\\"b\": {\"type\": \"string\"}}}}}}}]")).ok;
    try std.testing.expect(!try dsmlNamesOk(a, used.array));
}
