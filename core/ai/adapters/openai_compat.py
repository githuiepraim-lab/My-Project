"""OpenAI-compatible Chat Completions: OpenAI, Groq, OpenRouter, LM Studio,
vLLM, llama.cpp server, Together, Mistral, DeepSeek and anything else that
speaks the same dialect — add one by giving it a base_url in settings."""
from __future__ import annotations

import base64
import json
from typing import Iterator

from ..types import AIError, Chunk, ModelSpec, Request, Response, Usage
from .base import Adapter, parse_args


class OpenAICompatAdapter(Adapter):
    kind = "openai_compat"

    @property
    def base(self) -> str:
        return str(self.cfg.get("base_url") or "https://api.openai.com/v1").rstrip("/")

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        if "openrouter" in self.base:
            h["X-Title"] = "Ephraim"
        return h

    # neutral -> wire ------------------------------------------------------
    def _messages(self, req: Request, model: ModelSpec) -> list[dict]:
        out = []
        if req.system:
            out.append({"role": "system", "content": req.system})
        msgs = req.messages
        last_user = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=-1)
        for i, m in enumerate(msgs):
            role = m.get("role", "user")
            if role == "tool":
                out.append({"role": "tool", "tool_call_id": m.get("tool_call_id", ""),
                            "content": str(m.get("content", ""))})
            elif role == "assistant" and m.get("tool_calls"):
                out.append({"role": "assistant", "content": m.get("content") or None,
                            "tool_calls": [{"id": t["id"], "type": "function",
                                            "function": {"name": t["name"],
                                                         "arguments": json.dumps(t.get("arguments", {}))}}
                                           for t in m["tool_calls"]]})
            elif i == last_user and req.images and "vision" in model.caps:
                parts = [{"type": "text", "text": str(m.get("content", ""))}]
                for im in req.images:
                    b64 = base64.b64encode(im.data).decode("ascii")
                    parts.append({"type": "image_url",
                                  "image_url": {"url": f"data:{im.mime};base64,{b64}"}})
                out.append({"role": "user", "content": parts})
            else:
                out.append({"role": role, "content": str(m.get("content", ""))})
        return out

    def _payload(self, req: Request, model: ModelSpec, stream: bool) -> dict:
        p: dict = {"model": model.id, "messages": self._messages(req, model), "stream": stream}
        mt = req.max_output_tokens or 800
        p[self.cfg.get("max_tokens_param", "max_tokens")] = mt
        if req.temperature is not None:
            p["temperature"] = req.temperature
        if req.json_mode:
            p["response_format"] = {"type": "json_object"}
        if req.tools:
            p["tools"] = [{"type": "function", "function": {
                "name": t["name"], "description": t.get("description", ""),
                "parameters": t.get("parameters") or {"type": "object", "properties": {}}}}
                for t in req.tools]
            p["tool_choice"] = "auto"
        if stream and self.cfg.get("stream_usage", False):
            p["stream_options"] = {"include_usage": True}
        return p

    @staticmethod
    def _usage(u: dict | None) -> Usage:
        u = u or {}
        det = u.get("prompt_tokens_details") or {}
        return Usage(input=int(u.get("prompt_tokens") or 0),
                     output=int(u.get("completion_tokens") or 0),
                     cached=int(det.get("cached_tokens") or 0))

    # calls ----------------------------------------------------------------
    def complete(self, req: Request, model: ModelSpec) -> Response:
        resp = self._post(f"{self.base}/chat/completions", self._payload(req, model, False),
                          self._headers(), req, 25)
        try:
            data = resp.json()
            choice = data["choices"][0]
            msg = choice.get("message") or {}
        except (ValueError, KeyError, IndexError, TypeError):
            raise AIError("invalid_response", "malformed completion") from None
        calls = [{"id": t.get("id", ""), "name": (t.get("function") or {}).get("name", ""),
                  "arguments": parse_args((t.get("function") or {}).get("arguments"))}
                 for t in (msg.get("tool_calls") or [])]
        return Response(text=(msg.get("content") or "").strip(), tool_calls=calls,
                        usage=self._usage(data.get("usage")),
                        finish_reason=str(choice.get("finish_reason") or ""))

    def stream(self, req: Request, model: ModelSpec) -> Iterator[Chunk]:
        resp = self._post(f"{self.base}/chat/completions", self._payload(req, model, True),
                          self._headers(), req, 25, stream=True)
        frags: dict[int, dict] = {}
        usage = None
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
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    ch = json.loads(body)
                except ValueError:
                    continue
                if ch.get("usage"):
                    usage = self._usage(ch["usage"])
                choices = ch.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    yield Chunk(text=delta["content"])
                for tc in delta.get("tool_calls") or []:
                    f = frags.setdefault(tc.get("index", 0),
                                         {"id": "", "name": "", "args": ""})
                    f["id"] = f["id"] or tc.get("id", "")
                    fn = tc.get("function") or {}
                    f["name"] += fn.get("name") or ""
                    f["args"] += fn.get("arguments") or ""
                finish = choices[0].get("finish_reason") or finish
        except AIError:
            raise
        except Exception as e:
            from .base import classify_exception
            raise classify_exception(e) from None
        finally:
            resp.close()
        calls = [{"id": f["id"], "name": f["name"], "arguments": parse_args(f["args"])}
                 for _i, f in sorted(frags.items())]
        yield Chunk(done=True, tool_calls=calls, usage=usage, finish_reason=finish)

    def list_models(self) -> list[str]:
        try:
            r = self.session.get(f"{self.base}/models", headers=self._headers(), timeout=10)
            r.raise_for_status()
            return sorted(m["id"] for m in r.json().get("data", []) if "id" in m)
        except Exception:
            return []
