"""
AI video maker: topic in, narrated video out — using free services, with fallbacks at each step.

    1. SCRIPT     your AI provider writes 3-8 scenes (narration + an image description)
    2. PICTURES   Pollinations (free, no key) draws each scene; if it is unreachable a clean
                  title card is drawn locally instead
    3. VOICE      edge-tts (free, online, many voices) -> else pyttsx3 (offline) -> else silent,
                  with the narration burned in as captions
    4. VIDEO      ffmpeg joins it all into an MP4 (your system ffmpeg, or the one bundled with
                  the imageio-ffmpeg package)

Honest scope: this makes narrated slideshow-style videos with AI-drawn pictures. It is not a
text-to-video model (those need a GPU or a paid service). Publishing is up to you: nothing is
uploaded anywhere.
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

OUT_DIR = Path.home() / "Documents" / "Ephraim Reports" / "videos"
W, H, FPS = 1280, 720, 25
IMAGE_URL = "https://image.pollinations.ai/prompt/{prompt}?width={w}&height={h}&nologo=true&seed={seed}"


@dataclass
class Scene:
    narration: str
    image_prompt: str


def ffmpeg_path() -> str | None:
    if (p := shutil.which("ffmpeg")):
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


_SCRIPT_SYSTEM = ("You write short narrated videos. Reply with ONLY JSON: "
                  '{"scenes": [{"narration": "1-2 sentences, spoken aloud", "image": "a vivid picture description, no text in the image"}]}. '
                  "Plain language, accurate, no filler, no claims you cannot stand behind. {n} scenes.")


def parse_script(text: str, max_scenes: int = 8) -> list[Scene]:
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b < a:
        raise ValueError("the AI did not return a script")
    data = json.loads(t[a:b + 1])
    scenes = []
    for s in (data.get("scenes") or [])[:max_scenes]:
        n, i = str(s.get("narration", "")).strip(), str(s.get("image", s.get("image_prompt", ""))).strip()
        if n:
            scenes.append(Scene(n[:400], (i or n)[:300]))
    if not scenes:
        raise ValueError("the script had no scenes")
    return scenes


def write_script(hub, topic: str, seconds: int = 45, style: str = "") -> list[Scene]:
    from core.ai.types import Request
    n = max(3, min(8, round(seconds / 8)))
    r = hub.ask(Request(messages=[{"role": "user", "content": f"Topic: {topic}\nTarget length: about {seconds} seconds.\nStyle: {style or 'clear and engaging'}"}],
                        system=_SCRIPT_SYSTEM.replace("{n}", str(n)), task="plan", min_tier=1, max_output_tokens=1500, timeout=90, cacheable=False))
    return parse_script(r.text, n)


def _qt() -> None:
    from core.reports import _ensure_qt_app
    _ensure_qt_app()


def _wrap(text: str, fm, width: int) -> list[str]:
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if fm.horizontalAdvance(t) > width and cur:
            lines.append(cur); cur = w
        else:
            cur = t
    return lines + ([cur] if cur else [])


def render_frame(src: bytes | None, caption: str, out: Path, index: int = 0, title: str = "") -> None:
    """The scene picture (or a local title card when there is none) with the caption on it."""
    _qt()
    from PyQt6.QtCore import QRectF, Qt
    from PyQt6.QtGui import QColor, QFont, QFontMetrics, QImage, QLinearGradient, QPainter
    img = QImage(W, H, QImage.Format.Format_RGB32)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    pic = QImage.fromData(src) if src else QImage()
    if not pic.isNull():
        pic = pic.scaled(W, H, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
        p.drawImage((W - pic.width()) // 2, (H - pic.height()) // 2, pic)
    else:                                                       # offline / failed: a calm gradient card
        hues = [(18, 52, 86), (52, 28, 82), (14, 74, 70), (86, 40, 36), (30, 36, 92)]
        c = hues[index % len(hues)]
        g = QLinearGradient(0, 0, W, H); g.setColorAt(0, QColor(*c)); g.setColorAt(1, QColor(8, 12, 24)); p.fillRect(0, 0, W, H, g)
        if title:
            f = QFont("Arial", 40, QFont.Weight.Bold); p.setFont(f); p.setPen(QColor(235, 245, 255, 235))
            p.drawText(QRectF(80, 120, W - 160, 300), int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap), title)
    f = QFont("Arial", 26, QFont.Weight.DemiBold); fm = QFontMetrics(f); p.setFont(f)
    lines = _wrap(caption, fm, W - 160)[:4]
    if lines:
        h = fm.height() * len(lines) + 30
        p.fillRect(0, H - h - 30, W, h + 30, QColor(0, 0, 0, 170))
        p.setPen(QColor(255, 255, 255))
        for i, ln in enumerate(lines):
            p.drawText(80, H - h - 30 + 22 + fm.ascent() + i * fm.height(), ln)
    p.end()
    img.save(str(out), "PNG")


def fetch_image(prompt: str, seed: int, get=None) -> bytes | None:
    """One picture from Pollinations (free, no key). None on any failure: the caller falls back."""
    try:
        import requests
        r = (get or requests.get)(IMAGE_URL.format(prompt=quote(prompt[:300]), w=W, h=H, seed=seed), timeout=60)
        if r.status_code == 200 and "image" in r.headers.get("content-type", "") and len(r.content) > 4000:
            return r.content
    except Exception:
        pass
    return None


def narrate(text: str, out: Path, voice: str = "en-US-AriaNeural") -> float | None:
    """Speech for `text` -> audio file; returns seconds, or None if no voice engine works."""
    try:
        import edge_tts
        mp3 = out.with_suffix(".mp3")
        asyncio.run(edge_tts.Communicate(text, voice).save(str(mp3)))
        if mp3.exists() and mp3.stat().st_size > 1000:
            return _duration(mp3)
    except Exception:
        pass
    try:
        import pyttsx3
        wav = out.with_suffix(".wav")
        e = pyttsx3.init(); e.save_to_file(text, str(wav)); e.runAndWait()
        if wav.exists() and wav.stat().st_size > 1000:
            return _duration(wav)
    except Exception:
        pass
    return None


def _duration(path: Path) -> float | None:
    ff = ffmpeg_path()
    if not ff:
        return None
    r = subprocess.run([ff, "-i", str(path)], capture_output=True, text=True, timeout=60)
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", r.stderr)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else None


def _run(cmd: list[str], timeout: int = 600) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + (r.stderr.strip().splitlines() or ["unknown error"])[-1][:200])


def assemble(frames: list[tuple[Path, Path | None, float]], out: Path) -> Path:
    """frames = [(image, audio or None, seconds)] -> one MP4."""
    ff = ffmpeg_path()
    if not ff:
        raise RuntimeError("ffmpeg is not available. Install it, or:  pip install imageio-ffmpeg")
    work = Path(tempfile.mkdtemp(prefix="vid-"))
    try:
        parts = []
        for i, (img, aud, secs) in enumerate(frames):
            seg = work / f"s{i}.mp4"
            cmd = [ff, "-y", "-loop", "1", "-framerate", str(FPS), "-t", f"{secs:.2f}", "-i", str(img)]
            cmd += ["-i", str(aud)] if aud else ["-f", "lavfi", "-t", f"{secs:.2f}", "-i", "anullsrc=r=44100:cl=stereo"]
            cmd += ["-vf", f"scale={W}:{H},fade=t=in:st=0:d=0.3,fade=t=out:st={max(0.1, secs - 0.3):.2f}:d=0.3,format=yuv420p",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-r", str(FPS), "-ar", "44100", "-ac", "2",
                    "-c:a", "aac", "-b:a", "128k", "-shortest", str(seg)]
            _run(cmd)
            parts.append(seg)
        lst = work / "list.txt"
        lst.write_text("\n".join(f"file '{p.as_posix()}'" for p in parts), encoding="utf-8")
        out.parent.mkdir(parents=True, exist_ok=True)
        _run([ff, "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(out)])
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def make_video(hub, topic: str, seconds: int = 45, style: str = "", out: Path | None = None, *, fetch=None,
               speak=None, progress=lambda msg: None) -> tuple[str, str]:
    """Returns (summary, path to the .mp4). fetch/speak default to the free services (looked up
    at call time, so they can be swapped)."""
    fetch, speak = fetch or fetch_image, speak or narrate
    topic = topic.strip()
    if not topic:
        raise ValueError("what should the video be about?")
    if ffmpeg_path() is None:
        raise RuntimeError("ffmpeg is not available (pip install imageio-ffmpeg)")
    seconds = max(15, min(int(seconds or 45), 180))
    progress("writing the script")
    scenes = write_script(hub, topic, seconds, style)
    work = Path(tempfile.mkdtemp(prefix="vidwork-"))
    notes = []
    try:
        frames = []
        pics = voiced = 0
        for i, sc in enumerate(scenes):
            progress(f"scene {i + 1} of {len(scenes)}")
            raw = fetch(sc.image_prompt, i + 1)
            pics += raw is not None
            png = work / f"f{i}.png"
            render_frame(raw, sc.narration, png, i, title=topic if raw is None else "")
            aud = None
            secs = None
            try:
                secs = speak(sc.narration, work / f"a{i}")
            except Exception:
                secs = None
            if secs:
                aud = next((p for p in (work / f"a{i}.mp3", work / f"a{i}.wav") if p.exists()), None)
                voiced += aud is not None
            frames.append((png, aud, max(3.0, (secs or len(sc.narration.split()) / 2.4) + 0.4)))
        stem = re.sub(r"[^A-Za-z0-9]+", "_", topic)[:40].strip("_") or "video"
        out = out or OUT_DIR / f"{stem}_{len(scenes)}scenes.mp4"
        progress("assembling the video")
        assemble(frames, out)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    total = sum(f[2] for f in frames)
    if pics < len(scenes):
        notes.append(f"{len(scenes) - pics} picture(s) used a title card (image service unreachable)")
    if voiced < len(scenes):
        notes.append("no narration voice was available, so it has captions only" if not voiced else f"{len(scenes) - voiced} scene(s) without a voice")
    return (f"Video ready: {out.name}, {len(scenes)} scenes, about {total:.0f} s." + (" Note: " + "; ".join(notes) + "." if notes else "")), str(out)
