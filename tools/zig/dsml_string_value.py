"""A DSML string argument's grammar (zig/src/server/tool_grammar.zig string_value) against xgrammar's Python package.

The value may hold any text but no `<`, blanks, `/`, blanks, `｜`: an end-tag look-alike such as
`</｜DSML｜  parameter>` cannot run on inside the value, and the exact end tag is the only way out.
Run: python -I tools/zig/dsml_string_value.py (prints one line a case; exit 0 when every case is right).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string

BAR = "\uff5c"
DSML = BAR + "DSML" + BAR
BEGIN = "<" + DSML + ' parameter name="city" string="true">'
END = "</" + DSML + " parameter>"


def pattern() -> str:
    """string_value as the Zig source spells it: its string literals joined, Zig escapes undone."""
    src = (Path(__file__).resolve().parents[2] / "zig/src/server/tool_grammar.zig").read_text(encoding="utf-8")
    decl = re.search(r"const string_value = (.*?);\n", src, re.S)
    if decl is None:
        raise SystemExit("string_value not found in tool_grammar.zig")
    parts = re.findall(r'"((?:[^"\\]|\\.)*)"', decl.group(1))
    text = "".join(parts)
    text = re.sub(r"\\u\{([0-9a-fA-F]+)\}", lambda m: chr(int(m.group(1), 16)), text)
    return text.replace("\\\\", "\\")


def main() -> int:
    tag = {"type": "structural_tag", "format": {"type": "tag", "begin": BEGIN,
                                                "content": {"type": "regex", "pattern": pattern()}, "end": END}}
    grammar = xgr.Grammar.from_structural_tag(json.dumps(tag))
    good = ["Paris", "New York", "", "a < b", "x | y", "a<b>c", "see </div> here", "1 </ 2", "ends with <",
            "ends with </", "tabs\t< \n/ x", "line one\nline two", "<<</ a", "|" + DSML + "| text",
            "<" + DSML + " invoke", BAR + " bars " + BAR]
    bad = ["New York</" + DSML + "  parameter>", "x</ " + DSML + " parameter>", "x< /" + BAR + "y",
           "x<\t/\n" + BAR, "a</" + BAR + "b"]
    ok = True
    for value in good:
        got = _is_grammar_accept_string(grammar, BEGIN + value + END)
        ok &= got
        print(f"{'PASS' if got else 'FAIL'} accepts {value!r}")
    for value in bad:
        got = _is_grammar_accept_string(grammar, BEGIN + value + END)
        ok &= not got
        print(f"{'PASS' if not got else 'FAIL'} rejects {value!r}")
    print("DSML_STRING_VALUE_OK" if ok else "DSML_STRING_VALUE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
