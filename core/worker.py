"""
Headless job worker: runs your queue with NO window, so jobs keep running while your
laptop is off — as long as THIS runs on a machine that stays on (a spare PC, a
Raspberry Pi, a cheap always-on cloud VM).

    python -m core.worker                 run due jobs, forever
    python -m core.worker --telegram      ALSO accept tasks sent to your Telegram bot from your phone
    python -m core.worker --once          run what is due now, then exit (cron-friendly)

Needs the same config/ as the app (API keys for your AI providers, Telegram token and
chat id). It never opens a window and never touches the desktop.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")      # PDF reports use Qt's writer, no display needed


def main(argv: list[str]) -> int:
    from core.jobs import JobRunner, JobStore, Notifier, TelegramInbox, touch_worker_lock
    store, notifier = JobStore(), Notifier.from_config()
    log = lambda kind, job: print(f"[{time.strftime('%H:%M:%S')}] job #{job['id']} {kind}: {job['title'][:60]}", flush=True)
    runner = JobRunner(store, notifier=notifier, on_event=log)
    if "--once" in argv:
        store.recover()
        n = 0
        while (job := store.claim_due()) is not None:
            runner.run_job(job)
            n += 1
        print(f"ran {n} job(s)")
        return 0
    runner.start()
    print("Job worker running. Ctrl+C to stop." + ("  Telegram inbox ON." if "--telegram" in argv else ""), flush=True)
    inbox = TelegramInbox(store, notifier) if "--telegram" in argv else None
    if inbox and not notifier.ready:
        print("Telegram is not set up (token + telegram_chat_id). See docs/AI_UPGRADE.md.")
        return 2
    try:
        while True:
            touch_worker_lock()                        # tells a desktop app on the same machine to leave Telegram to us
            if inbox:
                try:
                    inbox.poll_once()
                except Exception as e:
                    print(f"[Telegram] {type(e).__name__}; retrying in 10s", flush=True)
                    time.sleep(10)
            else:
                time.sleep(30)
    except KeyboardInterrupt:
        runner.stop()
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
