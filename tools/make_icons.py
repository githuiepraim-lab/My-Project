"""Generate the phone app icons (run once; the PNGs are committed):  python tools/make_icons.py"""
from pathlib import Path
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "dashboard" / "static" / "icons"
BG, CYAN, DIM = (7, 9, 15), (0, 212, 255), (0, 120, 150)


def icon(size: int, safe: float = 1.0) -> Image.Image:
    """A glowing arc-reactor ring on the dashboard's dark background. `safe` < 1 keeps the art inside a
    maskable icon's safe zone (Android may crop the corners into a circle)."""
    s = size * 4                                              # draw big, shrink for smooth edges
    im = Image.new("RGB", (s, s), BG)
    d = ImageDraw.Draw(im)
    c, r = s / 2, s / 2 * 0.62 * safe
    for k, w in ((1.0, 0.10), (0.72, 0.045), (0.46, 0.03)):
        rr = r * k
        d.ellipse([c - rr, c - rr, c + rr, c + rr], outline=CYAN if k > 0.5 else DIM, width=int(s * w * safe))
    for a in range(0, 360, 60):                               # six spokes
        import math
        x, y = math.cos(math.radians(a)), math.sin(math.radians(a))
        d.line([c + x * r * 0.15, c + y * r * 0.15, c + x * r * 0.42, c + y * r * 0.42], fill=CYAN, width=int(s * 0.02 * safe))
    d.ellipse([c - r * 0.12, c - r * 0.12, c + r * 0.12, c + r * 0.12], fill=CYAN)
    return im.resize((size, size), Image.LANCZOS)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, size, safe in (("icon-192.png", 192, 1.0), ("icon-512.png", 512, 1.0), ("icon-maskable-512.png", 512, 0.78),
                             ("apple-touch-icon.png", 180, 0.9), ("favicon-32.png", 32, 1.0)):
        icon(size, safe).save(OUT / name, optimize=True)
        print("wrote", name)
