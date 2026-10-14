import ast
import asyncio
import random
import time
import types
from pathlib import Path

import pytest
from core.audio_buffer import JitterBuffer, simulate


def test_starts_small_waits_just_enough_and_never_stalls_forever():
    jb = JitterBuffer(100)
    assert not jb.ready(0.04, 0.01) and jb.ready(0.10, 0.0)
    assert jb.ready(0.01, 0.0, turn_done=True)                         # a very short reply plays at once
    assert jb.ready(0.01, 0.6)                                         # waited long enough: play anyway


def test_grows_when_the_speaker_runs_dry_and_is_capped():
    jb = JitterBuffer(100, max_ms=900)
    sizes = []
    for _ in range(8):
        jb.underflow(); sizes.append(round(jb.need_s, 3))
    assert sizes[0] > 0.1 and sizes == sorted(sizes) and sizes[-1] == 0.9 and jb.underruns == 8
    assert jb.growths == 4                                             # stops "growing" once at the cap


def test_shrinks_slowly_after_a_clean_stretch_never_below_the_minimum():
    jb = JitterBuffer(400, min_ms=60, shrink_after_s=10)
    for _ in range(100):
        jb.clean(10.0)
    assert abs(jb.need_s - 0.06) < 1e-9
    jb.underflow()
    assert jb.need_s > 0.06


def net(kind, seconds, rnd):
    out, sent = [], 0.0
    while sent < seconds:
        t = {"good": sent * 0.8 + 0.12 + rnd.random() * 0.01,
             "choppy": sent * 0.9 + 0.1 + (rnd.random() < 0.12) * rnd.uniform(0.15, 0.6) + rnd.random() * 0.05}[kind]
        out.append((max(t, out[-1][0] if out else 0), 0.05)); sent += 0.05
    return out


def run(kind, make, replies=10, seeds=20):
    first = last = delay = n = 0
    for s in range(seeds):
        rnd, jb, per = random.Random(s), make(), []
        for _ in range(replies):
            r = simulate(net(kind, rnd.uniform(3, 8), rnd), jb)
            per.append(r["breaks"]); delay += r["start_delay"]; n += 1
        first += sum(per[:3]); last += sum(per[7:])
    return first / (3 * seeds), last / (3 * seeds), delay / n


def test_on_a_choppy_connection_it_learns_to_stop_the_breaks():
    old_first, old_last, _ = run("choppy", lambda: None)
    new_first, new_last, _ = run("choppy", JitterBuffer)
    assert old_last > 1.0                                              # the old player really breaks
    assert new_last < 0.2 * old_last                                   # once it has learned, almost never


def test_on_a_good_connection_it_costs_almost_no_delay():
    _, old_last, old_delay = run("good", lambda: None)
    _, new_last, new_delay = run("good", JitterBuffer)
    assert old_last == 0 and new_last == 0 and new_delay - old_delay < 0.08        # < 80 ms


# ── the real methods in main.py (main itself needs audio hardware libs) ─────────────────────
def _methods(*names):
    src = Path("main.py").read_text(encoding="utf-8")
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "JarvisLive")
    fns = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    ns = {"asyncio": asyncio, "time": time, "RECEIVE_SAMPLE_RATE": 24000, "CHANNELS": 1}
    exec(compile(ast.Module(fns, []), "main.py", "exec"), ns)
    return types.SimpleNamespace(**{n: ns[n] for n in names})


def make_self(jb):
    log = []
    q = asyncio.Queue()
    ev = asyncio.Event()
    s = types.SimpleNamespace(_jb=jb, _interrupted=False, audio_in_queue=q, _turn_done_event=ev, _jb_note_at=None,
                              ui=types.SimpleNamespace(write_log=log.append), log=log)
    return s


def test_prebuffer_collects_chunks_until_enough_audio_is_queued():
    m = _methods("_prebuffer")
    async def go():
        s = make_self(JitterBuffer(100))
        piece = b"\0" * 2400                                           # 50 ms at 24 kHz mono int16
        for _ in range(3):
            s.audio_in_queue.put_nowait(piece)
        out = await m._prebuffer(s, piece)
        return len(out)
    assert asyncio.run(go()) == 2400 * 2                               # 100 ms reached after 2 pieces; third stays queued


def test_prebuffer_never_holds_back_a_finished_or_interrupted_reply():
    m = _methods("_prebuffer")
    async def go():
        s = make_self(JitterBuffer(500))
        s._turn_done_event.set()
        a = await m._prebuffer(s, b"\0" * 480)
        s2 = make_self(JitterBuffer(500)); s2._interrupted = True
        b = await m._prebuffer(s2, b"\0" * 480)
        return len(a), len(b)
    assert asyncio.run(go()) == (480, 480)


def test_prebuffer_gives_up_waiting_instead_of_hanging():
    m = _methods("_prebuffer")
    async def go():
        s = make_self(JitterBuffer(400))
        t0 = time.monotonic()
        await m._prebuffer(s, b"\0" * 480)                             # nothing else ever arrives
        return time.monotonic() - t0
    assert asyncio.run(go()) < 1.5


def test_a_break_grows_the_buffer_and_tells_the_user_once():
    m = _methods("_note_playback")
    s = make_self(JitterBuffer(100))
    m._note_playback(s, True, 0.2); m._note_playback(s, True, 0.2)
    assert s._jb.underruns == 2 and s._jb.need_s > 0.2 and len(s.log) == 1 and "choppy" in s.log[0]
    m._note_playback(s, False, 0.2)
    assert s._jb.underruns == 2


def test_player_wiring_is_in_place():
    src = Path("main.py").read_text(encoding="utf-8")
    assert "chunk = await self._prebuffer(chunk)" in src and "_under = await asyncio.to_thread(stream.write" in src
    assert "bool(_under) and not _new_stretch" in src and "JitterBuffer" in src
