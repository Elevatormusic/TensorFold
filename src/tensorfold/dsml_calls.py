"""Top-level tool-call blocks of a reply: DeepSeek-V4.1 DSML calls blocks and <tool_call> blocks.

A call counts only when its block is complete and well formed at the top level. Text in an open block is not a call.
Text in a closed block, for example a nested <tool_call>, is argument text. An unsure block is not a call.
"""

from __future__ import annotations

import json
import re
from typing import Any, NamedTuple

DSML = "｜DSML｜"
DSML_OPEN, DSML_CLOSE = f"<{DSML} calls>", f"</{DSML} calls>"
_INVOKE_OPEN, _INVOKE_CLOSE = f"<{DSML} invoke ", f"</{DSML} invoke>"
_PARAM_OPEN, _PARAM_CLOSE = f"<{DSML} parameter ", f"</{DSML} parameter>"
_INVOKE_RE = re.compile(rf'<{re.escape(DSML)} invoke name="([^"]+)">')
_PARAM_RE = re.compile(rf'<{re.escape(DSML)} parameter name="([^"]+)"(?: string="(true|false)")?>')
_SPACE_RE = re.compile(r"\s*")

CALL_OPEN, CALL_CLOSE = "<tool_call>", "</tool_call>"
# the exact tags, and the tags in any case (the call parser finds them in any case)
CALL_OPEN_RE, CALL_CLOSE_RE = re.compile(re.escape(CALL_OPEN)), re.compile(re.escape(CALL_CLOSE))
CALL_OPEN_ANY_RE = re.compile(re.escape(CALL_OPEN), re.IGNORECASE)
CALL_CLOSE_ANY_RE = re.compile(re.escape(CALL_CLOSE), re.IGNORECASE)


class Block(NamedTuple):
    kind: str       # "tag" for a <tool_call> block, "dsml" for a DSML calls block
    start: int
    end: int
    inner: str      # the text between the opening and the closing tag


def _close_of(text: str, pos: int, open_marker: str, close_marker: str) -> int:
    """The index of the close marker that ends the block whose opening tag ends at ``pos``, or -1.

    A block of the same kind in the text is skipped, so its close does not end this one. A reply cut inside a nested
    block therefore leaves the outer block open."""

    depth = 1
    opened, closed = text.find(open_marker, pos), text.find(close_marker, pos)
    while closed >= 0:
        if 0 <= opened < closed:
            depth += 1
            opened = text.find(open_marker, opened + len(open_marker))
            continue
        depth -= 1
        if depth == 0:
            return closed
        closed = text.find(close_marker, closed + len(close_marker))
    return -1


def dsml_block_end(text: str, start: int) -> int:
    """The index after the closing tag of the DSML calls block that opens at ``start``, or -1 when it is open."""

    close = _close_of(text, start + len(DSML_OPEN), DSML_OPEN, DSML_CLOSE)
    return close + len(DSML_CLOSE) if close >= 0 else -1


def top_level_blocks(text: str, *, any_case: bool = True) -> tuple[list[Block], int]:
    """(the closed blocks at the top level, in order; where the first open block begins, or -1).

    A block inside a closed block is part of it. A <tool_call> tag that never closes ends the search for <tool_call>
    blocks only: a closed DSML block after it still counts. A DSML block that never closes ends the search."""

    open_re, close_re = (CALL_OPEN_ANY_RE, CALL_CLOSE_ANY_RE) if any_case else (CALL_OPEN_RE, CALL_CLOSE_RE)
    blocks: list[Block] = []
    open_at, pos, tags = -1, 0, True
    tag = open_re.search(text)
    dsml = text.find(DSML_OPEN)
    while True:
        if tags and tag is not None and tag.start() < pos:
            tag = open_re.search(text, pos)
        if 0 <= dsml < pos:
            dsml = text.find(DSML_OPEN, pos)
        if tag is None and dsml < 0:
            return blocks, open_at
        if tag is not None and (dsml < 0 or tag.start() < dsml):
            closer = close_re.search(text, tag.end())
            if closer is None:
                open_at = tag.start() if open_at < 0 else open_at
                pos, tag, tags = tag.end(), None, False     # no later tag can close: look for DSML blocks only
                continue
            blocks.append(Block("tag", tag.start(), closer.end(), text[tag.end():closer.start()]))
            pos = closer.end()
        else:
            end = dsml_block_end(text, dsml)
            if end < 0:
                return blocks, dsml if open_at < 0 else open_at
            blocks.append(Block("dsml", dsml, end, text[dsml + len(DSML_OPEN):end - len(DSML_CLOSE)]))
            pos = end


def _parameters(body: str) -> dict[str, Any] | None:
    """The arguments of an invoke body, or None when the body is more than parameters and white space."""

    args: dict[str, Any] = {}
    at = _SPACE_RE.match(body, 0).end()
    while at < len(body):
        head = _PARAM_RE.match(body, at)
        close = _close_of(body, head.end(), _PARAM_OPEN, _PARAM_CLOSE) if head else -1
        if close < 0:
            return None
        key, is_string, value = head.group(1), head.group(2), body[head.end():close]
        args[key] = value
        if is_string == "false":
            try:
                args[key] = json.loads(value)
            except json.JSONDecodeError:
                pass
        at = _SPACE_RE.match(body, close + len(_PARAM_CLOSE)).end()
    return args


def parse_calls_block(inner: str) -> list[tuple[str, dict[str, Any]]] | None:
    """Every invoke of a DSML calls block as (name, arguments), or None when the block is not well formed.

    The block holds invokes and white space only, an invoke holds parameters and white space only, and it holds one
    invoke at least. A value is the text up to its closing tag: it can hold any markup."""

    calls: list[tuple[str, dict[str, Any]]] = []
    at = _SPACE_RE.match(inner, 0).end()
    while at < len(inner):
        head = _INVOKE_RE.match(inner, at)
        close = _close_of(inner, head.end(), _INVOKE_OPEN, _INVOKE_CLOSE) if head else -1
        args = _parameters(inner[head.end():close]) if close >= 0 else None
        if args is None:
            return None
        calls.append((head.group(1), args))
        at = _SPACE_RE.match(inner, close + len(_INVOKE_CLOSE)).end()
    return calls or None
