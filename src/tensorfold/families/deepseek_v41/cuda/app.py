"""OpenAI routes for the DeepSeek-V4.1 engine: TensorFold's CUDA App, with ignore_eos passed to the engine, image
input (--vision) laid out as DeepSeek's encoding lays it out, and every reasoning_effort name mapped to DeepSeek's
numeric effort."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

from tensorfold.cuda.server import App, PreparedRequest
from tensorfold.server.errors import RequestError


# reasoning_effort -> the "Reasoning Effort: N" (1-100) line DeepSeek's encoding puts before the first turn when
# thinking. DeepSeek names low / high / max (50 / 75 / 100); minimal, medium and xhigh sit between them, so every
# OpenAI tier reaches the model as its own value (none or off: thinking off). No effort sent: high, as the reference.
EFFORT_VALUES = {"minimal": 25, "low": 50, "medium": 62, "high": 75, "xhigh": 88, "max": 100}

# The guided_* fields that request_spec reads. A json_object request that also has one of them is a conflict.
GUIDED_FIELDS = ("guided_json", "guided_regex", "guided_choice", "guided_grammar")


class DsApp(App):
    reads_ignore_eos = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        template = self.template
        template.efforts = frozenset(EFFORT_VALUES)     # the server passes every name through unchanged
        # DeepSeek's encoding keeps a second leading system message (Codex's developer turn, say) as its own
        # <｜System｜> turn after the tools block: not merged into the first
        template.merge_leading = False
        render = template.render

        def render_effort(messages, *, enable_thinking: bool, extra: dict[str, Any] | None = None, **kw: Any) -> str:
            extra = dict(extra or {})
            if enable_thinking:
                name = extra.get("reasoning_effort") or "high"
                extra["reasoning_effort"] = EFFORT_VALUES.get(name, name)
            return render(messages, enable_thinking=enable_thinking, extra=extra, **kw)

        template.render = render_effort

    def _check_fields(self, body: dict[str, Any]) -> str | None:
        problem = super()._check_fields(body)
        if problem is None and isinstance(body, dict):
            try:
                self._response_format(body, "messages" in body)
            except RequestError as exc:
                return str(exc)
        return problem

    @staticmethod
    def _response_format(body: dict[str, Any], chat: bool) -> dict[str, Any]:
        """json_object as DeepSeek's encoding serves it: a "## Response Format" block in the system turn, no
        grammar. The engine has no constrained decoding, so a schema, regex, choice or grammar is HTTP 400. So is
        a json_object request that also has one of them: request_spec returns json_object first, and the other
        field must not stay in the body."""

        from tensorfold.engine.grammar import request_spec

        spec = request_spec(body)
        if spec is None:
            return body
        if spec.kind != "json" or not chat:
            raise RequestError("the DeepSeek-V4.1 engine has no constrained decoding: it serves response_format "
                               "json_object (as DeepSeek's API does), not json_schema, regex, choice or grammar")
        competing = DsApp._competing_controls(body)
        if competing:
            raise RequestError("the DeepSeek-V4.1 engine has no constrained decoding: it serves json_object only, "
                               f"so response_format json_object cannot go with {', '.join(competing)}")
        kwargs = body.get("chat_template_kwargs")
        if kwargs is not None and not isinstance(kwargs, dict):
            raise RequestError("chat_template_kwargs must be a JSON object or null")
        kwargs = {**(kwargs or {}), "response_format": {"type": "json_object"}}
        # both json_object fields go (response_format, structured_outputs.json_object): the block replaces them
        drop = ("response_format", "structured_outputs")
        return {**{k: v for k, v in body.items() if k not in drop}, "chat_template_kwargs": kwargs}

    @staticmethod
    def _competing_controls(body: dict[str, Any]) -> list[str]:
        """The set structured-output controls in the body other than json_object itself, by field name."""

        names = [name for name in GUIDED_FIELDS if body.get(name) is not None]
        so = body.get("structured_outputs")
        if isinstance(so, dict):
            # a key is set when it is not None and not False: the same test as request_spec
            names += [f"structured_outputs.{key}" for key, value in sorted(so.items())
                      if key != "json_object" and value is not None and value is not False]
        elif so is not None:
            names.append("structured_outputs")
        return names

    def run(self, body: dict[str, Any], chat: bool, emit: Callable[[dict[str, Any]], bool], *,
            prepared: PreparedRequest | None = None, cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        self.engine.request.stop_eos = not bool(body.get("ignore_eos", False))
        return super().run(body, chat, emit, prepared=prepared, cancelled=cancelled)

    def _prepare(self, body: dict[str, Any], chat: bool) -> PreparedRequest:
        """A chat with image_url parts (user turns and tool results): each image becomes DeepSeek's placeholder
        block, the prompt renders as text, and each placeholder token grows into the image's span."""

        body = self._response_format(body, chat)
        vision = getattr(self.engine, "vision", None)
        messages = body.get("messages") if chat else None
        if vision is None or not isinstance(messages, list) or not vision.has_images(messages):
            return super()._prepare(body, chat)
        try:
            text_messages, pictures = vision.split(messages)
        except (ValueError, OSError) as exc:
            raise RequestError(f"image input: {exc}") from exc
        prepared = super()._prepare({**body, "messages": text_messages}, chat)
        try:
            vp = vision.expand(prepared.prompt, pictures)
        except ValueError as exc:
            raise RequestError(f"image input: {exc}") from exc
        return dataclasses.replace(prepared, prompt=vp.token_ids, vision=vp,
                                   sampling=self.sampling_for(body, vp.token_ids))
