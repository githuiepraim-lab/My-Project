"""
Integrations: connect the assistant to ANY web API without writing code.

An integration is a few lines of JSON (config/integrations.json): a base URL, how to
authenticate, and named actions. Nine free, key-less ones ship built in (weather,
Wikipedia, currency rates, crypto prices, dictionary, Hacker News, NASA picture of the
day, countries, books). Add your own by voice, with the CLI, or by editing the file —
including webhooks (Slack, Discord, IFTTT, Zapier, Make) so it can notify other tools.

    python -m core.integrations list
    python -m core.integrations call weather forecast latitude=-1.5 longitude=37.2
    python -m core.integrations test            # live check of every built-in (needs internet)
    python -m core.integrations key myapi       # store an API key (hidden prompt)

SAFETY. Reading (GET) is automatic. Anything that CHANGES something elsewhere
(POST/PUT/PATCH/DELETE) is refused unless a human approved it (on-screen or on Telegram).
Addresses on your own network (localhost, 192.168.x, cloud metadata) are blocked unless an
integration is explicitly marked local, so a poisoned web page cannot make the assistant
probe your machine or router. Keys live in the OS keyring / environment, never in the JSON,
and never in anything returned to the model.
"""
from __future__ import annotations

import copy
import ipaddress
import json
import re
import socket
import threading
from pathlib import Path
from urllib.parse import quote, urlparse

from core.ai import config as C
from core.ai.redact import redact

USER_FILE = C.CONFIG_DIR / "integrations.json"
MAX_BYTES = 1_000_000
MAX_OUT = 3500
_NAME = re.compile(r"^[a-z][a-z0-9_\-]{0,30}$")
_VAR = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)(?:\|([^}]*))?\}")
MUTATING = {"POST", "PUT", "PATCH", "DELETE"}


class NeedsApproval(Exception):
    """The call would change something outside this computer; a human must approve it first."""


class IntegrationError(Exception):
    pass


BUILTIN: dict = {
    "weather": {"description": "Weather forecast by coordinates (Open-Meteo, free, no key).",
                "base_url": "https://api.open-meteo.com", "auth": {"type": "none"}, "actions": {
                    "forecast": {"method": "GET", "path": "/v1/forecast", "doc": "latitude, longitude -> now + 3 days",
                                 "query": {"latitude": "{latitude}", "longitude": "{longitude}", "timezone": "auto",
                                           "forecast_days": "3", "current": "temperature_2m,wind_speed_10m,weather_code",
                                           "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum"}}}},
    "geocode": {"description": "Find coordinates of a place name (Open-Meteo, free).",
                "base_url": "https://geocoding-api.open-meteo.com", "auth": {"type": "none"}, "actions": {
                    "search": {"method": "GET", "path": "/v1/search", "doc": "name -> latitude/longitude",
                               "query": {"name": "{name}", "count": "{count|3}"}, "pick": "results"}}},
    "wikipedia": {"description": "Short summary of any Wikipedia article (free).",
                  "base_url": "https://en.wikipedia.org", "auth": {"type": "none"}, "actions": {
                      "summary": {"method": "GET", "path": "/api/rest_v1/page/summary/{title}", "doc": "title -> extract",
                                  "pick": "extract"}}},
    "fx": {"description": "Currency exchange rates from the European Central Bank (Frankfurter, free).",
           "base_url": "https://api.frankfurter.dev", "auth": {"type": "none"}, "actions": {
               "latest": {"method": "GET", "path": "/v1/latest", "doc": "base, symbols e.g. base=USD symbols=KES,EUR",
                          "query": {"base": "{base|USD}", "symbols": "{symbols|EUR}"}},
               "history": {"method": "GET", "path": "/v1/{date}", "doc": "date YYYY-MM-DD, base, symbols",
                           "query": {"base": "{base|USD}", "symbols": "{symbols|EUR}"}}}},
    "crypto": {"description": "Crypto prices (CoinGecko free tier).", "base_url": "https://api.coingecko.com",
               "auth": {"type": "none"}, "actions": {
                   "price": {"method": "GET", "path": "/api/v3/simple/price", "doc": "ids=bitcoin,ethereum vs=usd",
                             "query": {"ids": "{ids|bitcoin}", "vs_currencies": "{vs|usd}", "include_24hr_change": "true"}}}},
    "dictionary": {"description": "English word definitions (dictionaryapi.dev, free).",
                   "base_url": "https://api.dictionaryapi.dev", "auth": {"type": "none"}, "actions": {
                       "define": {"method": "GET", "path": "/api/v2/entries/en/{word}", "doc": "word -> meanings"}}},
    "hackernews": {"description": "Top Hacker News story ids / items (free).",
                   "base_url": "https://hacker-news.firebaseio.com", "auth": {"type": "none"}, "actions": {
                       "top": {"method": "GET", "path": "/v0/topstories.json", "doc": "ids of the top stories"},
                       "item": {"method": "GET", "path": "/v0/item/{id}.json", "doc": "id -> title, url, score"}}},
    "nasa": {"description": "NASA Astronomy Picture of the Day (DEMO_KEY, rate limited, free).",
             "base_url": "https://api.nasa.gov", "auth": {"type": "none"}, "actions": {
                 "apod": {"method": "GET", "path": "/planetary/apod", "doc": "today's picture + explanation",
                          "query": {"api_key": "DEMO_KEY"}}}},
    "countries": {"description": "Country facts (REST Countries, free).", "base_url": "https://restcountries.com",
                  "auth": {"type": "none"}, "actions": {
                      "info": {"method": "GET", "path": "/v3.1/name/{name}", "doc": "name -> capital, population, currencies",
                               "query": {"fields": "name,capital,population,currencies,languages,region"}}}},
    "books": {"description": "Search books (Open Library, free).", "base_url": "https://openlibrary.org",
              "auth": {"type": "none"}, "actions": {
                  "search": {"method": "GET", "path": "/search.json", "doc": "query -> titles, authors, year",
                             "query": {"q": "{query}", "limit": "5", "fields": "title,author_name,first_publish_year"}}}},
}

# Templates for services you point at your OWN URL/key. They ship disabled-by-need: they only
# work once you store the URL/key, and every send needs approval.
TEMPLATES: dict = {
    "slack": {"description": "Post to a Slack channel via an Incoming Webhook.", "url_secret": True, "auth": {"type": "none"},
              "actions": {"post": {"method": "POST", "path": "", "body": {"text": "{text}"}, "doc": "text"}}},
    "discord": {"description": "Post to a Discord channel via a webhook.", "url_secret": True, "auth": {"type": "none"},
                "actions": {"post": {"method": "POST", "path": "", "body": {"content": "{text}"}, "doc": "text"}}},
    "webhook": {"description": "Fire any webhook (IFTTT, Zapier, Make, n8n).", "url_secret": True, "auth": {"type": "none"},
                "actions": {"fire": {"method": "POST", "path": "", "body": {"value1": "{a|}", "value2": "{b|}", "value3": "{c|}"},
                                     "doc": "a, b, c"}}},
}


def _subst(template, params: dict, *, encode: bool):
    """Fill {name} / {name|default} in a string. Missing required values raise IntegrationError."""
    def rep(m):
        name, default = m.group(1), m.group(2)
        if name in params and str(params[name]) != "":
            v = str(params[name])
        elif default is not None:
            v = default
        else:
            raise IntegrationError(f"missing value '{name}'")
        return quote(v, safe=",") if encode else v
    return _VAR.sub(rep, template)


def _fill(obj, params):
    if isinstance(obj, str):
        return _subst(obj, params, encode=False)
    if isinstance(obj, dict):
        return {k: _fill(v, params) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fill(v, params) for v in obj]
    return obj


def _pick(data, path: str):
    for part in path.split("."):
        if isinstance(data, dict):
            data = data.get(part)
        elif isinstance(data, list) and part.isdigit():
            data = data[int(part)] if int(part) < len(data) else None
        else:
            return data
    return data


def public_host(host: str) -> bool:
    """True only if EVERY address the name resolves to is a public one."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        raise IntegrationError(f"cannot resolve {host}") from None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            return False
    return True


class Integrations:
    def __init__(self, user_file: Path | None = None, http=None, key_getter=None) -> None:
        self.user_file = Path(user_file) if user_file else USER_FILE
        self._http = http
        self._key = key_getter or C.get_api_key
        self._lock = threading.Lock()

    # ── registry ────────────────────────────────────────────────────────────
    def _user(self) -> dict:
        try:
            d = json.loads(self.user_file.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    def all(self) -> dict:
        out = copy.deepcopy(BUILTIN)
        out.update(copy.deepcopy(TEMPLATES))                 # usable once you store their URL (python -m core.integrations key slack_url)
        for n, v in self._user().items():
            if isinstance(v, dict) and _NAME.match(n) and isinstance(v.get("actions"), dict):
                out[n] = v
        return out

    def names(self) -> list[str]:
        return sorted(self.all())

    def describe(self, name: str) -> str:
        spec = self.all().get(name)
        if not spec:
            raise IntegrationError(f"no integration called '{name}'")
        acts = "; ".join(f"{a} ({s.get('method', 'GET')}): {s.get('doc', '')}" for a, s in spec["actions"].items())
        return f"{name} — {spec.get('description', '')} Actions: {acts}"

    def save(self, name: str, spec: dict) -> None:
        name = name.strip().lower()
        if not _NAME.match(name):
            raise IntegrationError("name must be lowercase letters, digits, - or _")
        self.validate(spec)
        with self._lock:
            d = self._user()
            d[name] = spec
            self.user_file.parent.mkdir(parents=True, exist_ok=True)
            self.user_file.write_text(json.dumps(d, indent=2), encoding="utf-8")

    def remove(self, name: str) -> bool:
        with self._lock:
            d = self._user()
            ok = d.pop(name, None) is not None
            if ok:
                self.user_file.write_text(json.dumps(d, indent=2), encoding="utf-8")
            return ok

    @staticmethod
    def validate(spec: dict) -> None:
        if not isinstance(spec, dict) or not isinstance(spec.get("actions"), dict) or not spec["actions"]:
            raise IntegrationError("an integration needs at least one action")
        if not spec.get("url_secret"):
            u = urlparse(str(spec.get("base_url", "")))
            if u.scheme not in ("http", "https") or not u.netloc:
                raise IntegrationError("base_url must be an http(s) address")
        for a, s in spec["actions"].items():
            if str(s.get("method", "GET")).upper() not in {"GET", *MUTATING}:
                raise IntegrationError(f"action {a}: method must be GET, POST, PUT, PATCH or DELETE")
        if spec.get("auth", {}).get("type", "none") not in ("none", "bearer", "header", "query"):
            raise IntegrationError("auth type must be none, bearer, header or query")

    # ── calling ─────────────────────────────────────────────────────────────
    def is_mutating(self, name: str, action: str) -> bool:
        spec = self.all().get(name) or {}
        return str(((spec.get("actions") or {}).get(action) or {}).get("method", "GET")).upper() in MUTATING

    def call(self, name: str, action: str, params: dict | None = None, *, approved: bool = False) -> str:
        params = {k: v for k, v in (params or {}).items() if isinstance(k, str)}
        spec = self.all().get(name)
        if not spec:
            raise IntegrationError(f"no integration called '{name}' (have: {', '.join(self.names())})")
        act = spec["actions"].get(action)
        if not act:
            raise IntegrationError(f"{name} has no action '{action}' (have: {', '.join(spec['actions'])})")
        method = str(act.get("method", "GET")).upper()
        if method in MUTATING and not approved:
            raise NeedsApproval(f"{name}.{action} would send data with {method}")
        secret_key = f"int_{name}"
        base = spec.get("base_url", "")
        if spec.get("url_secret"):
            base = self._key(f"{secret_key}_url")
            if not base:
                raise IntegrationError(f"store the {name} URL first:  python -m core.integrations key {name}_url")
        path = _subst(act.get("path", ""), params, encode=True)
        url = base.rstrip("/") + path
        query = {k: _subst(v, params, encode=False) for k, v in (act.get("query") or {}).items()}
        headers = {"Accept": "application/json", "User-Agent": "Ephraim-AI/1"}
        auth = spec.get("auth") or {"type": "none"}
        if auth["type"] != "none":
            key = self._key(auth.get("secret") or secret_key)
            if not key:
                raise IntegrationError(f"no API key stored for {name}:  python -m core.integrations key {name}")
            if auth["type"] == "bearer":
                headers["Authorization"] = f"Bearer {key}"
            elif auth["type"] == "header":
                headers[auth.get("name", "X-API-Key")] = key
            else:
                query[auth.get("name", "api_key")] = key
        body = _fill(act["body"], params) if act.get("body") is not None else None
        text = self._request(method, url, query, headers, body, bool(spec.get("local")))
        try:
            data = json.loads(text)
        except ValueError:
            return redact(text)[:MAX_OUT]
        if act.get("pick"):
            data = _pick(data, act["pick"])
        out = data if isinstance(data, str) else json.dumps(data, separators=(",", ":"), ensure_ascii=False)
        return redact(out if len(out) <= MAX_OUT else out[:MAX_OUT] + f"…[{len(out) - MAX_OUT} more chars]")

    def _request(self, method, url, query, headers, body, local: bool) -> str:
        import requests
        http = self._http or requests
        cur = url
        origin = urlparse(url).hostname
        for _hop in range(4):
            u = urlparse(cur)
            if u.scheme not in ("http", "https") or not u.hostname:
                raise IntegrationError("only http(s) addresses are allowed")
            # "local" means: this integration's own host may be on your network. A redirect
            # elsewhere must still be a PUBLIC address, or a local service could bounce the
            # assistant to the cloud-metadata address or your router.
            if not (local and u.hostname == origin) and not public_host(u.hostname):
                raise IntegrationError(f"{u.hostname} is a private/local address; blocked for safety "
                                       f"(mark the integration \"local\": true if you really mean it)")
            try:
                r = http.request(method, cur, params=query if _hop == 0 else None, headers=headers,
                                 json=body if method in MUTATING else None, timeout=20, stream=True, allow_redirects=False)
            except requests.exceptions.RequestException as e:
                raise IntegrationError(f"could not reach {u.hostname}: {type(e).__name__}") from None
            if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("Location"):
                cur = requests.compat.urljoin(cur, r.headers["Location"])        # re-checked at the top of the loop
                continue
            chunks, size = [], 0
            for ch in r.iter_content(65536):
                size += len(ch)
                if size > MAX_BYTES:
                    r.close()
                    raise IntegrationError("response is larger than 1 MB")
                chunks.append(ch)
            text = b"".join(chunks).decode(r.encoding or "utf-8", "replace")
            if r.status_code >= 400:
                raise IntegrationError(f"{u.hostname} answered HTTP {r.status_code}: {redact(text)[:160]}")
            return text
        raise IntegrationError("too many redirects")


def main(argv: list[str]) -> int:
    import getpass
    reg = Integrations()
    cmd, args = (argv[0] if argv else "list"), argv[1:]
    if cmd == "list":
        for n in reg.names():
            print(reg.describe(n))
    elif cmd == "call" and len(args) >= 2:
        params = dict(a.split("=", 1) for a in args[2:] if "=" in a)
        print(reg.call(args[0], args[1], params, approved="--yes" in args))
    elif cmd == "key" and args:
        print("stored in", C.set_api_key(f"int_{args[0]}", getpass.getpass(f"secret for {args[0]} (hidden): ").strip()))
    elif cmd == "remove" and args:
        print("removed" if reg.remove(args[0]) else "not found")
    elif cmd == "test":
        probes = {"weather": ("forecast", {"latitude": "-1.29", "longitude": "36.82"}), "geocode": ("search", {"name": "Nairobi"}),
                  "wikipedia": ("summary", {"title": "Python_(programming_language)"}), "fx": ("latest", {"base": "USD", "symbols": "KES"}),
                  "crypto": ("price", {}), "dictionary": ("define", {"word": "code"}), "hackernews": ("top", {}),
                  "nasa": ("apod", {}), "countries": ("info", {"name": "kenya"}), "books": ("search", {"query": "python"})}
        for n, (a, p) in probes.items():
            try:
                out = reg.call(n, a, p)
                print(f"OK    {n:11} {out[:60]!r}")
            except Exception as e:
                print(f"FAIL  {n:11} {e}")
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
