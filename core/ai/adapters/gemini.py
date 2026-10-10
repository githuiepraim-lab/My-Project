"""Google Gemini, through the existing core/gemini.py so its key lookup,
timeouts, Live-session rung and cooldown behaviour are reused rather than
duplicated. The router picks the exact model; this adapter asks for only that
model so fallback decisions stay in ONE place (the router)."""
from __future__ import annotations

from typing import Iterator

from ..types import AIError, Chunk, ModelSpec, Request, Response, Usage
from .base import Adapter


def _upper_types(schema):
    """Gemini's function schema wants OBJECT/STRING/...; JSON Schema is lower-case."""
    if isinstance(schema, dict):
        return {k: (v.upper() if k == "type" and isinstance(v, str) else _upper_types(v))
                for k, v in schema.items()}
    if isinstance(schema, list):
        return [_upper_types(v) for v in schema]
    return schema


class GeminiAdapter(Adapter):
    kind = "gemini"

    def _gemini(self):
        from core import gemini            # imported lazily: heavy SDK stays optional
        return gemini

    def _contents(self, req: Request):
        from google.genai import types as gt
        contents = []
        last_user = max((i for i, m in enumerate(req.messages) if m.get("role") == "user"),
                        default=-1)
        for i, m in enumerate(req.messages):
            role = m.get("role", "user")
            if role == "tool":
                part = gt.Part.from_function_response(
                    name=m.get("name", "tool"), response={"result": str(m.get("content", ""))})
                contents.append(gt.Content(role="user", parts=[part]))
            elif role == "assistant" and m.get("tool_calls"):
                parts = [gt.Part.from_function_call(name=t["name"], args=t.get("arguments", {}))
                         for t in m["tool_calls"]]
                contents.append(gt.Content(role="model", parts=parts))
            else:
                parts = [gt.Part.from_text(text=str(m.get("content", "")) or ".")]
                if i == last_user:
                    for im in req.images:
                        parts.append(gt.Part.from_bytes(data=im.data, mime_type=im.mime))
                contents.append(gt.Content(role="model" if role == "assistant" else "user",
                                           parts=parts))
        return contents

    def _config(self, req: Request):
        from google.genai import types as gt
        kw: dict = {}
        if req.system:
            kw["system_instruction"] = req.system
        if req.max_output_tokens:
            kw["max_output_tokens"] = req.max_output_tokens
        if req.temperature is not None:
            kw["temperature"] = req.temperature
        if req.json_mode:
            kw["response_mime_type"] = "application/json"
        if req.tools:
            decls = [{"name": t["name"], "description": t.get("description", ""),
                      "parameters": _upper_types(t.get("parameters")
                                                 or {"type": "object", "properties": {}})}
                     for t in req.tools]
            kw["tools"] = [gt.Tool(function_declarations=decls)]
            kw["automatic_function_calling"] = gt.AutomaticFunctionCallingConfig(disable=True)
        return gt.GenerateContentConfig(**kw)

    def complete(self, req: Request, model: ModelSpec) -> Response:
        g = self._gemini()
        timeout_ms = int(float(req.timeout or 25) * 1000)
        try:
            raw = g.call(self._contents(req), config=self._config(req), timeout_ms=timeout_ms,
                         key=self.api_key, models=(model.id,), cross_fallback=False)
        except ImportError as e:
            raise AIError("unsupported", f"google-genai is not installed: {e}") from None
        if raw is None:
            raise AIError("unavailable", f"{model.id} did not answer")
        calls = []
        for i, fc in enumerate(getattr(raw, "function_calls", None) or []):
            calls.append({"id": f"call_{i}", "name": getattr(fc, "name", ""),
                          "arguments": dict(getattr(fc, "args", None) or {})})
        um = getattr(raw, "usage_metadata", None)
        usage = Usage(input=int(getattr(um, "prompt_token_count", 0) or 0),
                      output=int(getattr(um, "candidates_token_count", 0) or 0),
                      cached=int(getattr(um, "cached_content_token_count", 0) or 0))
        try:
            text = (raw.text or "").strip()
        except Exception:               # a pure function-call reply has no text part
            text = ""
        return Response(text=text, tool_calls=calls, usage=usage)

    def stream(self, req: Request, model: ModelSpec) -> Iterator[Chunk]:
        # The ladder in core/gemini.py is request/response; keep one code path
        # and deliver the text as a single chunk instead of duplicating it.
        yield from super().stream(req, model)
