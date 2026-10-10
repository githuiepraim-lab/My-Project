from .anthropic import AnthropicAdapter
from .base import Adapter
from .gemini import GeminiAdapter
from .ollama import OllamaAdapter
from .openai_compat import OpenAICompatAdapter

KINDS = {"gemini": GeminiAdapter, "anthropic": AnthropicAdapter,
         "openai_compat": OpenAICompatAdapter, "ollama": OllamaAdapter}


def make_adapter(name: str, cfg: dict, api_key: str = "") -> Adapter:
    return KINDS[cfg["kind"]](name, cfg, api_key)
