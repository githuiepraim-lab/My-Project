"""
Connect JARVIS to your Telegram in under a minute — on iPhone or Android, with the official app.

    1. In Telegram, message @BotFather, send /newbot, and copy the token it gives you.
    2. Paste it here. JARVIS checks it and shows a short PAIRING CODE.
    3. Send that code to your new bot. JARVIS sees it, learns your chat id, and sends a test message.

Why a code: the bot is public. Without it, anyone who finds the bot and messages it first would
become "the owner" and could queue jobs and press your approval buttons. Only the message that
contains the one-time code, sent in a private chat, can claim it.

    python -m core.telegram_setup            guided setup in the terminal
"""
from __future__ import annotations

import re
import secrets
import time

API = "https://api.telegram.org/bot{token}/{method}"
_TOKEN = re.compile(r"^\d{6,12}:[A-Za-z0-9_-]{30,}$")
_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"          # no 0/O/1/I/L: nothing to misread on a phone screen


class SetupError(Exception):
    """A problem explained in plain words."""


def clean_token(raw: str) -> str:
    t = re.sub(r"\s+", "", str(raw or "")).strip("\"'`")
    t = re.sub(r"^(?:bot)?(?=\d)", "", t, flags=re.I)
    return re.sub(r"^https?://api\.telegram\.org/bot", "", t)


class TelegramSetup:
    def __init__(self, http=None, clock=time.monotonic, sleep=time.sleep) -> None:
        self._http, self._clock, self._sleep = http, clock, sleep

    # ── plumbing ────────────────────────────────────────────────────────────
    def _call(self, token: str, method: str, timeout: float = 20, **params) -> dict:
        import requests
        http = self._http or requests
        try:
            r = http.post(API.format(token=token, method=method), json=params, timeout=timeout)
        except Exception as e:
            raise SetupError("Could not reach Telegram. Check this computer's internet (Telegram can be blocked on some networks; try another network or a VPN).") from None
        try:
            data = r.json()
        except ValueError:
            raise SetupError(f"Telegram sent something unreadable (HTTP {r.status_code}).") from None
        if not data.get("ok"):
            code = data.get("error_code") or r.status_code
            if code in (401, 404):
                raise SetupError("Telegram rejected that token. Copy the WHOLE token from @BotFather (it looks like 123456789:ABCdef…), with no spaces.")
            if code == 409:
                raise SetupError("Another program is already listening to this bot (a webhook or another JARVIS). Stop it first.")
            raise SetupError(f"Telegram said: {str(data.get('description', 'error'))[:120]}")
        return data

    # ── the steps ───────────────────────────────────────────────────────────
    def verify(self, raw_token: str) -> dict:
        token = clean_token(raw_token)
        if not _TOKEN.match(token):
            raise SetupError("That does not look like a bot token. It should look like 123456789:ABCdef… (from @BotFather → /newbot).")
        me = self._call(token, "getMe")["result"]
        if not me.get("is_bot"):
            raise SetupError("That token does not belong to a bot.")
        return {"token": token, "id": me["id"], "username": me.get("username", ""), "name": me.get("first_name", "")}

    @staticmethod
    def new_code() -> str:
        return "".join(secrets.choice(_ALPHABET) for _ in range(6))

    def pair(self, token: str, code: str, timeout: float = 120, on_wait=None) -> dict:
        """Wait for a PRIVATE message containing `code`. Returns {"chat_id", "name"}. Stale updates and any
        webhook are cleared first; messages without the code (from anyone) are ignored."""
        self._call(token, "deleteWebhook", drop_pending_updates=True)
        want = re.sub(r"[^A-Z0-9]", "", code.upper())
        deadline, offset, ignored = self._clock() + timeout, 0, 0
        while self._clock() < deadline:
            left = max(1, int(deadline - self._clock()))
            data = self._poll(token, offset, min(10, left))
            for u in data.get("result", []):
                offset = max(offset, int(u["update_id"]) + 1)
                msg = u.get("message") or {}
                text = re.sub(r"[^A-Z0-9]", "", str(msg.get("text", "")).upper())
                chat = msg.get("chat") or {}
                if chat.get("type") == "private" and want and want in text:
                    who = " ".join(x for x in (chat.get("first_name"), chat.get("last_name")) if x) or chat.get("username", "you")
                    return {"chat_id": str(chat["id"]), "name": who}
                ignored += 1
            if on_wait:
                on_wait(max(0, int(deadline - self._clock())), ignored)
        raise SetupError(f"No message containing the code {code} arrived within {int(timeout)} seconds. "
                         f"Open your bot in Telegram, press Start, and send exactly: {code}")

    def _poll(self, token: str, offset: int, wait: int) -> dict:
        import requests
        http = self._http or requests
        try:
            r = http.post(API.format(token=token, method="getUpdates"), json={"offset": offset, "timeout": wait, "allowed_updates": ["message"]}, timeout=wait + 10)
            data = r.json()
        except Exception:
            self._sleep(2)
            return {"result": []}
        if not data.get("ok"):
            if (data.get("error_code") or r.status_code) == 409:
                raise SetupError("Another program is already listening to this bot (a webhook or another JARVIS). Stop it first.")
            self._sleep(2)
            return {"result": []}
        return data

    def send_test(self, token: str, chat_id: str) -> None:
        self._call(token, "sendMessage", chat_id=chat_id, text="✅ JARVIS is connected to this chat.\n\nI will message you here when jobs finish, reminders fire, "
                   "or something needs your approval. Send /help to see what you can ask me.")

    # ── storing ─────────────────────────────────────────────────────────────
    @staticmethod
    def save(token: str, chat_id: str, set_key=None, save_flag=None) -> None:
        if set_key is None:
            from core.ai import config as C
            set_key = C.set_api_key
        if save_flag is None:
            from memory.config_manager import _save_flag
            save_flag = _save_flag
        set_key("telegram", token)                          # the token is a secret: keyring/env/secrets.json, never the config
        save_flag("telegram_chat_id", str(chat_id))

    @staticmethod
    def disconnect(delete_key=None, save_flag=None) -> None:
        if delete_key is None:
            from core.ai import config as C
            delete_key = C.delete_api_key
        if save_flag is None:
            from memory.config_manager import _save_flag
            save_flag = _save_flag
        delete_key("telegram")
        save_flag("telegram_chat_id", "")

    @staticmethod
    def status() -> dict:
        from core.jobs import Notifier
        n = Notifier.from_config()
        return {"connected": n.ready, "chat_id": n.chat_id if n.ready else ""}


def main() -> int:
    import getpass
    s = TelegramSetup()
    try:
        me = s.verify(getpass.getpass("Paste your bot token (hidden): "))
        code = s.new_code()
        print(f"\n✓ Bot found: {me['name']} (@{me['username']})\n\nNow, in Telegram, open  https://t.me/{me['username']}  → Start → send this code:\n\n        {code}\n")
        got = s.pair(me["token"], code, on_wait=lambda left, ign: print(f"  waiting… {left}s left", end="\r"))
        print(f"\n✓ Paired with {got['name']}")
        s.save(me["token"], got["chat_id"])
        s.send_test(me["token"], got["chat_id"])
        print("✓ Saved. A test message was sent to your Telegram. Restart JARVIS to start using it.")
        return 0
    except SetupError as e:
        print(f"\n✗ {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
