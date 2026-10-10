"""
Provider-neutral request / response types for the multi-AI layer.

Every adapter converts to and from these, so nothing above the adapters ever
needs to know that Anthropic wants `input_schema`, OpenAI wants `function`
wrappers and Gemini wants upper-case schema types.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

# ── capabilities ────────────────────────────────────────────────────────────
# A model advertises only what it genuinely supports. The router never assumes
# two models share a capability, and filters on these before ranking anything.
TEXT = "text"
VISION = "vision"      # image input
AUDIO = "audio"        # audio input
TOOLS = "tools"        # function / tool calling
JSON = "json"          # constrained JSON output
STREAM = "stream"      # incremental output
ALL_CAPS = frozenset({TEXT, VISION, AUDIO, TOOLS, JSON, STREAM})

# ── cost tiers ──────────────────────────────────────────────────────────────
LIGHT, STANDARD, STRONG = 0, 1, 2
TIER_NAMES = {LIGHT: "light", STANDARD: "standard", STRONG: "strong"}


class AIError(Exception):
    """A provider failure, classified so the router can react correctly.

    kind is one of: rate_limit, auth, timeout, unavailable, network,
    bad_request, not_found, invalid_response, unsupported, cancelled, budget.
    """

    def __init__(self, kind: str, message: str = "", *, status: int | None = None,
                 retry_after: float | None = None):
        super().__init__(message or kind)
        self.kind = kind
        self.status = status
        self.retry_after = retry_after

    @property
    def transient(self) -> bool:
        return self.kind in ("rate_limit", "timeout", "unavailable", "network")


class CancelToken:
    """Cooperative cancellation. Streaming adapters poll it between chunks and
    the router checks it between attempts; a request already in flight on a
    non-streaming socket is abandoned, not interrupted."""

    def __init__(self) -> None:
        self._ev = threading.Event()

    def cancel(self) -> None:
        self._ev.set()

    @property
    def cancelled(self) -> bool:
        return self._ev.is_set()

    def raise_if_cancelled(self) -> None:
        if self._ev.is_set():
            raise AIError("cancelled", "request cancelled")


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    id: str
    tier: int = STANDARD
    caps: frozenset = frozenset({TEXT, STREAM})
    price_in: float | None = None     # USD per 1M input tokens; None = unknown
    price_out: float | None = None    # USD per 1M output tokens
    context: int = 32_000
    max_output: int = 4_096
    local: bool = False               # runs on this machine: free, private

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.id}"

    def cost(self, tokens_in: int, tokens_out: int) -> float | None:
        if self.local:
            return 0.0
        if self.price_in is None or self.price_out is None:
            return None
        return (tokens_in * self.price_in + tokens_out * self.price_out) / 1_000_000


@dataclass
class Image:
    data: bytes
    mime: str = "image/png"


@dataclass
class Request:
    """One provider-neutral request.

    messages   [{"role": "user"|"assistant"|"tool", "content": str, ...}]
               A "tool" message carries tool_call_id + name; an assistant
               message may carry tool_calls=[{"id","name","arguments"}].
    tools      [{"name", "description", "parameters": <JSON Schema>}]
    """
    messages: list[dict] = field(default_factory=list)
    system: str = ""
    tools: list[dict] = field(default_factory=list)
    images: list[Image] = field(default_factory=list)
    json_mode: bool = False
    max_output_tokens: int | None = None
    temperature: float | None = None
    task: str = ""                       # hint: classify / chat / code / plan ...
    min_tier: int | None = None          # force a floor on model strength
    timeout: float | None = None
    cacheable: bool | None = None        # None = decided from the task
    private_context: str = ""            # memories / file text: gated per provider
    cancel: CancelToken | None = None
    meta: dict = field(default_factory=dict)

    def needs(self) -> set[str]:
        need = {TEXT}
        if self.images:
            need.add(VISION)
        if self.tools:
            need.add(TOOLS)
        if self.json_mode:
            need.add(JSON)
        return need


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cached: int = 0
    estimated: bool = False


@dataclass
class Response:
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)   # [{"id","name","arguments"}]
    usage: Usage = field(default_factory=Usage)
    provider: str = ""
    model: str = ""
    latency: float = 0.0
    cost: float | None = None
    cached: bool = False
    finish_reason: str = ""
    attempts: list[dict] = field(default_factory=list)     # what was tried, in order
    error: str = ""                                        # set only by ask_many

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}" if self.provider else ""


@dataclass
class Chunk:
    """One streamed piece. `text` for content; `done=True` carries the final
    tool calls and usage."""
    text: str = ""
    done: bool = False
    tool_calls: list[dict] = field(default_factory=list)
    usage: Usage | None = None
    finish_reason: str = ""
