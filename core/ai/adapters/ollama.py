"""Ollama native /api/chat (local, free, private)."""
from __future__ import annotations

import base64
import json
from typing import Iterator

from ..types import AIError, Chunk, ModelSpec, Request, Response, Usage
from .base import Adapter, classify_exception, parse_args


class OllamaAdapter(Adapter):
    kind = "ollama"

    @property
    def base(self) -> str:
        return str(self.cfg.get("base_url") or "http://localhost:11434").rstrip("/")

    def _payload(self, req: Request, model: ModelSpec, stream: bool) -> dict:
        msgs = []
        if req.system:
            msgs.append({"role": "system", "content": req.system})
        last_user = max((i for i, m in enumerate(req.messages) if m.get("role") == "user"),
                        default=-1)
        for i, m in enumerate(req.messages):
            role = m.get("role", "user")
            msg = {"role": role, "content": str(m.get("content", ""))}
            if role == "assistant" and m.get("tool_calls"):
                msg["tool_calls"] = [{"function": {"name": t["name"],
                                                   "arguments": t.get("arguments", {})}}
                                     for t in m["tool_calls"]]
            if i == last_user and req.images and "vision" in model.caps:
                msg["images"] = [base64.b64encode(im.data).decode("ascii") for im in req.images]
            msgs.append(msg)
        opts = {"num_predict": req.max_output_tokens or 800}
        if req.temperature is not None:
            opts["temperature"] = req.temperature
        p = {"model": model.id, "messages": msgs, "stream": stream,
             "keep_alive": self.cfg.get("keep_alive", "30m"), "options": opts}
        if req.json_mode:
            p["format"] = "json"
        if req.tools:
            p["tools"] = [{"type": "function", "function": {
                "name": t["name"], "description": t.get("description", ""),
                "parameters": t.get("parameters") or {"type": "object", "properties": {}}}}
                for t in req.tools]
        return p

    def _post_with_restart(self, payload, req, stream):
        try:
            return self._post(f"{self.base}/api/chat", payload, {}, req, 120, stream=stream)
        except AIError as e:
            if e.kind != "network":
                raise
            # Ollama is not running. Start it once (the existing helper), retry once.
            try:
                from core.llm_client import ensure_ollama_running
                if ensure_ollama_running():
                    return self._post(f"{self.base}/api/chat", payload, {}, req, 120, stream=stream)
            except AIError:
                raise
            except Exception:
                pass
            raise

    @staticmethod
    def _usage(d: dict) -> Usage:
        return Usage(input=int(d.get("prompt_eval_count") or 0),
                     output=int(d.get("eval_count") or 0))

    def complete(self, req: Request, model: ModelSpec) -> Response:
        resp = self._post_with_restart(self._payload(req, model, False), req, False)
        try:
            d = resp.json()
            msg = d["message"]
        except (ValueError, KeyError, TypeError):
            raise AIError("invalid_response", "malformed chat reply") from None
        calls = [{"id": f"call_{i}", "name": (t.get("function") or {}).get("name", ""),
                  "arguments": parse_args((t.get("function") or {}).get("arguments"))}
                 for i, t in enumerate(msg.get("tool_calls") or [])]
        return Response(text=(msg.get("content") or "").strip(), tool_calls=calls,
                        usage=self._usage(d), finish_reason=str(d.get("done_reason") or ""))

    def stream(self, req: Request, model: ModelSpec) -> Iterator[Chunk]:
        resp = self._post_with_restart(self._payload(req, model, True), req, True)
        calls: list[dict] = []
        usage = Usage()
        finish = ""
        try:
            for raw in resp.iter_lines():
                if req.cancel is not None and req.cancel.cancelled:
                    raise AIError("cancelled", "request cancelled")
                if not raw:
                    continue
                try:
                    ch = json.loads(raw)
                except ValueError:
                    continue
                msg = ch.get("message") or {}
                if msg.get("content"):
                    yield Chunk(text=msg["content"])
                for t in msg.get("tool_calls") or []:
                    calls.append({"id": f"call_{len(calls)}",
                                  "name": (t.get("function") or {}).get("name", ""),
                                  "arguments": parse_args((t.get("function") or {}).get("arguments"))})
                if ch.get("done"):
                    usage = self._usage(ch)
                    finish = str(ch.get("done_reason") or "")
                    break
        except AIError:
            raise
        except Exception as e:
            raise classify_exception(e) from None
        finally:
            resp.close()
        yield Chunk(done=True, tool_calls=calls, usage=usage, finish_reason=finish)

    def list_models(self) -> list[str]:
        try:
            r = self.session.get(f"{self.base}/api/tags", timeout=5)
            r.raise_for_status()
            return sorted(m.get("name", "") for m in r.json().get("models", []))
        except Exception:
            return []
