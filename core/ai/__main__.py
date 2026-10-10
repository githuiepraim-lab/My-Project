"""Command-line settings:  python -m core.ai <command>

  status                      providers, key source, health, usage
  enable|disable <provider>   turn a provider on or off
  key <provider>              store an API key (prompted; never echoed or logged)
  priority <provider> <n>     lower runs first
  add <name> <base_url> <model> [tier]   any OpenAI-compatible API
  strategy cost|priority|speed
  pin <provider/model>|auto   force a model, or go back to automatic routing
  models <provider>           ask the provider which models it offers
  test [prompt]               send one real request through the router
  usage                       tokens, latency and cost so far
"""
from __future__ import annotations

import getpass
import json
import sys

from . import config as C
from .hub import get_hub
from .redact import redact


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, args = argv[0], argv[1:]
    cfg = C.load()
    if cmd == "status":
        for n, p in cfg["providers"].items():
            src = C.key_source(n) or ("not needed" if p.get("requires_key", True) is False else "none")
            print(f"{n:11} {'ON ' if p.get('enabled') else 'off'}  priority={p.get('priority'):<4} "
                  f"key={src:14} models={','.join(m['id'] for m in p['models'])}")
        print("strategy:", cfg["settings"]["strategy"], "| mode:", cfg["settings"]["mode"])
    elif cmd in ("enable", "disable") and args:
        C.update_provider(args[0], enabled=(cmd == "enable"))
        print(f"{args[0]} {'enabled' if cmd == 'enable' else 'disabled'}")
    elif cmd == "key" and args:
        k = getpass.getpass(f"API key for {args[0]} (hidden): ").strip()
        print("stored in", C.set_api_key(args[0], k))
    elif cmd == "priority" and len(args) == 2:
        C.update_provider(args[0], priority=int(args[1]))
    elif cmd == "add" and len(args) >= 3:
        name, url, model = args[0].lower(), args[1], args[2]
        tier = int(args[3]) if len(args) > 3 else 1
        C.update_provider(name, kind="openai_compat", enabled=True, base_url=url, priority=60,
                          share_memory=False, stream_usage=False,
                          models=[{"id": model, "tier": tier, "caps": ["text", "stream", "tools"]}])
        print(f"added {name}. Now run:  python -m core.ai key {name}")
    elif cmd == "strategy" and args:
        C.update_settings(strategy=args[0])
    elif cmd == "pin" and args:
        if args[0] == "auto":
            C.update_settings(mode="auto", manual_model="")
        else:
            C.update_settings(mode="manual", manual_model=args[0])
    elif cmd == "models" and args:
        hub = get_hub()
        print("\n".join(hub._adapter(args[0]).list_models()) or "(provider returned no list)")
    elif cmd == "test":
        hub = get_hub()
        r = hub.ask(" ".join(args) or "Reply with the single word: ready")
        print(f"[{r.label}] {r.text}\n{r.usage.input} in / {r.usage.output} out, "
              f"{r.latency:.2f}s, cost={r.cost}")
    elif cmd == "usage":
        print(json.dumps(get_hub().usage.summary(), indent=2))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as e:
        print("error:", redact(e))
        sys.exit(1)
