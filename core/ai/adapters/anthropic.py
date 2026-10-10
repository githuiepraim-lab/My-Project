"""Anthropic Messages API."""
from __future__ import annotations

import base64
import json
from typing import Iterator

from ..types import AIError, Chunk, ModelSpec, Request, Response, Usage
from .base import Adapter, parse_args

API_VERSION = "2023-06-01"


class AnthropicAdapter(Adapter):
    kind = "anthropic"

    @property
    def base(self) -> str:
        return str(self.cfg.get("base_url") or "https://api.anthropic.com").rstrip("/")

    def _headers(self) -> dict:
        return {"Content-Type": "application/json", "x-api-key": self.api_key,
                "anthropic-version": API_VERSION}

    def _messages(self, req: Request, model: ModelSpec) -> list[dict]:
        out: list[dict] = []

        def push(role: str, blocks: list[dict]) -> None:
            # The API requires strictly alternating roles; merge neighbours.
            if out and out[-1]["role"] == role:
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": blocks})

        last_user = max((i for i, m in enumerate(req.messages) if m.get("role") == "user"),
                        default=-1)
        for i, m in enumerate(req.messages):
            role = m.get("role", "user")
            if role == "tool":
                push("user", [{"type": "tool_result", "tool_use_id": m.get("tool_call_id", ""),
                               "content": str(m.get("content", ""))}])
            elif role == "assistant":
                blocks = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": str(m["content"])})
                for t in m.get("tool_calls") or []:
                    blocks.append({"type": "tool_use", "id": t["id"], "name": t["name"],
                                   "input": t.get("arguments", {})})
                if blocks:
                    push("assistant", blocks)
            else:
                blocks = []
                if i == last_user and req.images and "vision" in model.caps:
                    for im in req.images:
                        blocks.append({"type": "image", "source": {
                            "type": "base64", "media_type": im.mime,
                            "data": base64.b64encode(im.data).decode("ascii")}})
                blocks.append({"type": "text", "text": str(m.get("content", "")) or "."})
                push("user", blocks)
        return out

    def _payload(self, req: Request, model: ModelSpec, stream: bool) -> dict:
        p: dict = {"model": model.id, "max_tokens": req.max_output_tokens or 800,
                   "messages": self._messages(req, model), "stream": stream}
        if req.system:
            block = {"type": "text", "text": req.system}
            # Prompt caching: a long, stable system prompt is billed at a
            # fraction of the price on every repeat call inside the cache window.
            if self.cfg.get("prompt_cache", True) and len(req.system) > 3000:
                block["cache_control"] = {"type": "ephemeral"}
            p["system"] = [block]
        if req.temperature is not None:
            p["temperature"] = req.temperature
        if req.tools:
            p["tools"] = [{"name": t["name"], "description": t.get("description", ""),
                           "input_schema": t.get("parameters")
                           or {"type": "object", "properties": {}}} for t in req.tools]
        return p

    @staticmethod
    def _usage(u: dict | None) -> Usage:
        u = u or {}
        cached = int(u.get("cache_read_input_tokens") or 0)
        return Usage(input=int(u.get("input_tokens") or 0)
                     + cached + int(u.get("cache_creation_input_tokens") or 0),
                     output=int(u.get("output_tokens") or 0), cached=cached)

    def complete(self, req: Request, model: ModelSpec) -> Response:
        resp = self._post(f"{self.base}/v1/messages", self._payload(req, model, False),
                          self._headers(), req, 30)
        try:
            data = resp.json()
            blocks = data["content"]
        except (ValueError, KeyError, TypeError):
            raise AIError("invalid_response", "malformed message") from None
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        calls = [{"id": b.get("id", ""), "name": b.get("name", ""),
                  "arguments": b.get("input") if isinstance(b.get("input"), dict) else {}}
                 for b in blocks if b.get("type") == "tool_use"]
        return Response(text=text.strip(), tool_calls=calls, usage=self._usage(data.get("usage")),
                        finish_reason=str(data.get("stop_reason") or ""))

    def stream(self, req: Request, model: ModelSpec) -> Iterator[Chunk]:
        resp = self._post(f"{self.base}/v1/messages", self._payload(req, model, True),
                          self._headers(), req, 30, stream=True)
        tools: dict[int, dict] = {}
        usage = Usage()
        finish = ""
        try:
            for raw in resp.iter_lines():
                if req.cancel is not None and req.cancel.cancelled:
                    raise AIError("cancelled", "request cancelled")
                if not raw:
                    continue
                line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
                if not line.startswith("data:"):
                    continue
                try:
                    ev = json.loads(line[5:].strip())
                except ValueError:
                    continue
                t = ev.get("type")
                if t == "message_start":
                    usage = self._usage((ev.get("message") or {}).get("usage"))
                elif t == "content_block_start":
                    cb = ev.get("content_block") or {}
                    if cb.get("type") == "tool_use":
                        tools[ev.get("index", 0)] = {"id": cb.get("id", ""),
                                                     "name": cb.get("name", ""), "args": ""}
                elif t == "content_block_delta":
                    d = ev.get("delta") or {}
                    if d.get("type") == "text_delta" and d.get("text"):
                        yield Chunk(text=d["text"])
                    elif d.get("type") == "input_json_delta":
                        tools.setdefault(ev.get("index", 0),
                                         {"id": "", "name": "", "args": ""})["args"] += d.get("partial_json", "")
                elif t == "message_delta":
                    finish = (ev.get("delta") or {}).get("stop_reason") or finish
                    out = (ev.get("usage") or {}).get("output_tokens")
                    if out is not None:
                        usage.output = int(out)
                elif t == "error":
                    raise AIError("unavailable", str((ev.get("error") or {}).get("message", ""))[:200])
        except AIError:
            raise
        except Exception as e:
            from .base import classify_exception
            raise classify_exception(e) from None
        finally:
            resp.close()
        calls = [{"id": f["id"], "name": f["name"], "arguments": parse_args(f["args"])}
                 for _i, f in sorted(tools.items())]
        yield Chunk(done=True, tool_calls=calls, usage=usage, finish_reason=finish)

    def list_models(self) -> list[str]:
        try:
            r = self.session.get(f"{self.base}/v1/models", headers=self._headers(), timeout=10)
            r.raise_for_status()
            return sorted(m["id"] for m in r.json().get("data", []) if "id" in m)
        except Exception:
            return []
