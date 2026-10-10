"""Adapter base class: HTTP plumbing shared by the REST adapters, and the
single place HTTP failures are turned into classified AIErrors."""
from __future__ import annotations

import json
from typing import Iterator

from ..redact import redact
from ..types import AIError, Chunk, ModelSpec, Request, Response, Usage

MAX_TEXT_CHARS = 200_000      # a reply larger than this is treated as broken


def classify_http(status: int, body: str = "", retry_after: str | None = None) -> AIError:
    low = (body or "").lower()
    ra = None
    try:
        ra = float(retry_after) if retry_after else None
    except ValueError:
        ra = None
    msg = redact(f"HTTP {status}: {(body or '')[:200]}")
    if status == 429 or "rate limit" in low or "quota" in low and status in (403, 429):
        return AIError("rate_limit", msg, status=status, retry_after=ra)
    if status in (401, 403):
        return AIError("auth", msg, status=status)
    if status == 404:
        return AIError("not_found", msg, status=status)
    if status in (408, 504):
        return AIError("timeout", msg, status=status)
    if status in (500, 502, 503, 529):
        return AIError("unavailable", msg, status=status, retry_after=ra)
    if 400 <= status < 500:
        return AIError("bad_request", msg, status=status)
    return AIError("unavailable", msg, status=status)


def classify_exception(e: Exception) -> AIError:
    import requests
    if isinstance(e, AIError):
        return e
    if isinstance(e, requests.exceptions.Timeout):
        return AIError("timeout", "request timed out")
    if isinstance(e, requests.exceptions.ConnectionError):
        return AIError("network", redact(f"connection failed: {e}")[:200])
    return AIError("unavailable", redact(f"{type(e).__name__}: {e}")[:200])


def parse_args(raw) -> dict:
    """Tool-call arguments arrive as a JSON string or a dict; anything else is
    treated as untrusted and dropped to {}."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            v = json.loads(raw)
            return v if isinstance(v, dict) else {}
        except ValueError:
            return {}
    return {}


def validate_response(resp: Response, offered_tools: set[str]) -> Response:
    """Provider output is untrusted input. Bound it, strip control characters
    and drop any tool call the model invented that we never offered."""
    text = resp.text if isinstance(resp.text, str) else str(resp.text or "")
    if len(text) > MAX_TEXT_CHARS:
        raise AIError("invalid_response", "reply exceeds size limit")
    resp.text = "".join(c for c in text if c == "\n" or c == "\t" or ord(c) >= 32)
    good = []
    for tc in resp.tool_calls or []:
        if (isinstance(tc, dict) and tc.get("name") in offered_tools
                and isinstance(tc.get("arguments", {}), dict)):
            good.append({"id": str(tc.get("id", ""))[:128], "name": tc["name"],
                         "arguments": tc.get("arguments", {})})
    resp.tool_calls = good
    if not resp.text.strip() and not resp.tool_calls:
        raise AIError("invalid_response", "empty reply")
    return resp


class Adapter:
    """One adapter per API dialect. Subclasses implement `complete` and, where
    the provider can, `stream`."""

    kind = ""

    def __init__(self, name: str, cfg: dict, api_key: str = "") -> None:
        self.name = name
        self.cfg = cfg
        self.api_key = api_key
        self._session = None

    # connection reuse: one Session (keep-alive pool) per adapter
    @property
    def session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": "Ephraim-AI/1"})
        return self._session

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def timeout(self, req: Request, default: float) -> tuple[float, float]:
        t = float(req.timeout or default)
        return (min(5.0, t), t)

    def complete(self, req: Request, model: ModelSpec) -> Response:
        raise NotImplementedError

    def stream(self, req: Request, model: ModelSpec) -> Iterator[Chunk]:
        """Default: one chunk from `complete`, for providers without streaming."""
        r = self.complete(req, model)
        if r.text:
            yield Chunk(text=r.text)
        yield Chunk(done=True, tool_calls=r.tool_calls, usage=r.usage,
                    finish_reason=r.finish_reason)

    def list_models(self) -> list[str]:
        return []

    def _post(self, url: str, payload: dict, headers: dict, req: Request,
              default_timeout: float, stream: bool = False):
        try:
            resp = self.session.post(url, json=payload, headers=headers,
                                     timeout=self.timeout(req, default_timeout),
                                     stream=stream)
        except Exception as e:
            raise classify_exception(e) from None
        if resp.status_code >= 400:
            try:
                body = resp.text
            finally:
                resp.close()
            raise classify_http(resp.status_code, body, resp.headers.get("retry-after"))
        return resp
