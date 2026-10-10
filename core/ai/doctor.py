"""
Connection doctor: says WHICH of the usual problems you have.

    python -m core.ai doctor

The app used to answer every failure with "a VPN may be required". That is one
possible cause among five, and the others (a mis-pasted key, DNS, a blocked
region, an HTTPS-intercepting proxy) need different fixes. This makes the
cheapest possible probes, in order, and stops at the first that fails.
"""
from __future__ import annotations

import re
import socket

HOST = "generativelanguage.googleapis.com"


def clean_key(raw) -> str:
    """What a pasted key should look like: no whitespace anywhere (a newline or
    space copied with it is the commonest cause of 'API key not valid'), no
    surrounding quotes, and no 'key=' prefix."""
    k = re.sub(r"\s+", "", str(raw or "")).strip("\"'`")
    return re.sub(r"^(?:api[_-]?key|key)=", "", k, flags=re.I)


def diagnose(key: str, *, dns=None, http_get=None, timeout: float = 8.0) -> tuple[str, str]:
    """Returns (status, plain-language message). status is one of: ok, no_key,
    bad_format, bad_key, region, quota, no_dns, ssl, no_network, slow, other."""
    key = clean_key(key)
    if not key:
        return "no_key", "No API key is saved. Open setup and paste your Gemini key."
    if not key.startswith("AIza") or not 30 <= len(key) <= 60:
        return "bad_format", ("That does not look like a Gemini API key (they start with 'AIza'). "
                              "Create one at aistudio.google.com/apikey and paste only the key.")
    dns = dns or socket.getaddrinfo
    try:
        dns(HOST, 443)
    except OSError:
        return "no_dns", ("This computer cannot find Google's servers (DNS lookup failed). Check "
                          "the internet connection, try another network, or turn a VPN on or off.")
    if http_get is None:
        import requests

        def http_get(url, headers, timeout):
            return requests.get(url, headers=headers, timeout=timeout)
    try:
        r = http_get(f"https://{HOST}/v1beta/models?pageSize=1", {"x-goog-api-key": key}, timeout)
    except Exception as e:
        name = type(e).__name__
        if "SSL" in name or "ssl" in str(e).lower():
            return "ssl", ("Something on your network is intercepting secure connections (a proxy, "
                           "antivirus 'web shield' or a school/work network). Try another network.")
        if "ReadTimeout" in name or "Timeout" in name and "Connect" not in name:
            return "slow", "Google answered too slowly. The connection is weak; try again or switch network."
        return "no_network", ("Google's servers cannot be reached from this network (it may block "
                              "them). Try another network, a mobile hotspot, or a VPN.")
    body = (getattr(r, "text", "") or "")[:400].lower()
    code = getattr(r, "status_code", 0)
    if code == 200:
        return "ok", ("Your key works and Google is reachable. If the live voice still fails, the "
                      "network is blocking the live (websocket) connection: try a hotspot or a VPN, "
                      "and pause any antivirus web protection.")
    if "location" in body and ("not supported" in body or "unsupported" in body):
        return "region", "Google does not offer this API from your current location. Use a VPN."
    if code == 429:
        return "quota", "The key is valid but its quota is used up. Wait, or use a different key."
    if code in (400, 401, 403):
        return "bad_key", ("Google rejected the key (invalid, expired or restricted). Create a new "
                           "one at aistudio.google.com/apikey and paste it again.")
    return "other", f"Google answered with HTTP {code}. Try again in a minute."
