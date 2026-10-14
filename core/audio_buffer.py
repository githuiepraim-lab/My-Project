"""
Adaptive jitter buffer for the assistant's voice.

THE PROBLEM. The reply arrives over the network in small pieces. On a steady link each
piece comes before the previous one has finished playing; on a choppy one (mobile data, a
VPN, Wi-Fi at the edge of range) they arrive in bursts with gaps. The old player wrote each
piece to the speaker the moment it arrived, so a gap longer than the audio already queued
meant the speaker ran dry: a break in the middle of a sentence.

THE FIX. Hold back the start of each stretch of speech until a little audio is queued, then
play. How much is learned from the connection: it starts small (so a good link has no added
delay), grows when the speaker actually runs dry, and shrinks again after a clean stretch.
"""
from __future__ import annotations


class JitterBuffer:
    def __init__(self, initial_ms: int = 100, min_ms: int = 60, max_ms: int = 900,
                 shrink_after_s: float = 40.0) -> None:
        self.min_s, self.max_s = min_ms / 1000.0, max_ms / 1000.0
        self.need_s = max(self.min_s, min(initial_ms / 1000.0, self.max_s))
        self.shrink_after_s = shrink_after_s
        self._clean_s = 0.0
        self.underruns = 0
        self.growths = 0

    def ready(self, buffered_s: float, waited_s: float, turn_done: bool = False) -> bool:
        """Start playing? Yes when enough is queued, the reply is complete (it may be a short
        one), or we have waited long enough that waiting longer is worse than a gap."""
        return turn_done or buffered_s >= self.need_s or waited_s >= min(0.6, self.need_s * 2.5 + 0.05)

    def underflow(self) -> bool:
        """The speaker ran dry mid-speech. Returns True if the buffer grew (so the caller can say so)."""
        self.underruns += 1
        self._clean_s = 0.0
        new = min(self.max_s, self.need_s * 1.7 + 0.04)
        grew = new > self.need_s + 1e-9
        self.need_s = new
        self.growths += grew
        return grew

    def clean(self, seconds: float) -> None:
        """`seconds` of speech played without a break. After a while, give some delay back."""
        self._clean_s += seconds
        if self._clean_s >= self.shrink_after_s and self.need_s > self.min_s:
            self.need_s = max(self.min_s, self.need_s - 0.02)
            self._clean_s = 0.0

    def stats(self) -> str:
        return f"buffer {self.need_s * 1000:.0f} ms, {self.underruns} break(s), grew {self.growths}x"


def simulate(arrivals: list[tuple[float, float]], jb: JitterBuffer | None, step: float = 0.005) -> dict:
    """Play a recorded 'network' through a speaker model and count the breaks.

    arrivals = [(arrival_time_s, audio_seconds_in_piece), ...] for ONE reply, sorted by time.
    jb=None is the old behaviour (play the instant anything arrives). Returns breaks and the
    delay before the first sound. Used by the tests and bench to compare the two."""
    t, i = 0.0, 0
    queued = 0.0            # seconds of audio waiting
    playing = False
    breaks = 0
    in_break = False
    first_sound = None
    total = sum(a for _, a in arrivals)
    played = 0.0
    waited_since = None
    end_t = arrivals[-1][0] + total + 10
    while t < end_t and played < total - 1e-9:
        while i < len(arrivals) and arrivals[i][0] <= t:
            queued += arrivals[i][1]
            if waited_since is None:
                waited_since = arrivals[i][0]
            i += 1
        done = i >= len(arrivals)
        if not playing and queued > 0:
            if jb is None or jb.ready(queued, t - (waited_since or t), done):
                playing, first_sound = True, t
        if playing:
            if queued > 0:
                use = min(step, queued)
                queued -= use
                played += use
                in_break = False
            elif not done:
                if not in_break:
                    breaks += 1
                    if jb is not None:
                        jb.underflow()
                in_break = True
        t += step
    return {"breaks": breaks, "start_delay": first_sound or 0.0}
