import io
import json
import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi"); pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from PIL import Image

STATIC = Path("dashboard/static")


@pytest.fixture
def client():
    from dashboard.server import DashboardServer
    srv = DashboardServer()
    return TestClient(srv._build_app()), srv


def test_manifest_has_everything_a_phone_needs_to_install_it(client):
    c, _ = client
    r = c.get("/manifest.webmanifest")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/manifest+json")
    m = r.json()
    assert m["display"] == "standalone" and m["start_url"] == "/" and m["scope"] == "/" and m["name"] and m["short_name"]
    assert re.fullmatch(r"#[0-9a-f]{6}", m["theme_color"]) and re.fullmatch(r"#[0-9a-f]{6}", m["background_color"])
    sizes = {(i["sizes"], i["purpose"]) for i in m["icons"]}
    assert ("192x192", "any") in sizes and ("512x512", "any") in sizes and ("512x512", "maskable") in sizes      # Android's install criteria


def test_every_icon_in_the_manifest_is_served_as_a_real_png_of_the_stated_size(client):
    c, _ = client
    for ic in c.get("/manifest.webmanifest").json()["icons"]:
        r = c.get(ic["src"])
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        w, h = Image.open(io.BytesIO(r.content)).size
        assert f"{w}x{h}" == ic["sizes"]
    r = c.get("/icons/apple-touch-icon.png")
    assert Image.open(io.BytesIO(r.content)).size == (180, 180)                   # iOS home-screen icon


def test_the_maskable_icon_keeps_its_art_inside_the_safe_zone(client):
    c, _ = client
    im = Image.open(io.BytesIO(c.get("/icons/icon-maskable-512.png").content)).convert("RGB")
    corner = im.getpixel((8, 8)); edge_mid = im.getpixel((256, 6))
    assert corner == edge_mid == (7, 9, 15)                                       # background only at the edges: nothing gets cropped away


def test_icon_route_cannot_be_used_to_read_other_files(client):
    c, _ = client
    for bad in ("../server.py", "..%2fserver.py", "app.html", "icon-192.png/../../server.py", "x.png%00", "..\\server.py"):
        assert c.get(f"/icons/{bad}").status_code in (404, 422), bad


def test_service_worker_never_caches_private_traffic(client):
    c, _ = client
    r = c.get("/sw.js")
    assert r.status_code == 200 and "javascript" in r.headers["content-type"] and r.headers["service-worker-allowed"] == "/"
    js = r.text
    assert '/api' in js and '"/ws"' in js and "method !== \"GET\"" in js              # API, websocket and non-GET go straight to the network
    assert "put(" not in js and "cache.add(e.request" not in js                          # nothing dynamic is ever written to a cache


def test_install_page_explains_both_platforms_and_telegram(client):
    c, _ = client
    r = c.get("/install")
    assert r.status_code == 200 and "Add to Home Screen" in r.text and "Install app" in r.text and "Telegram" in r.text
    assert "Safari" in r.text and "Chrome" in r.text and "same Wi-Fi" in r.text


@pytest.mark.parametrize("page", ["app.html", "login.html"])
def test_the_app_pages_carry_the_phone_install_tags(page):
    html = (STATIC / page).read_text(encoding="utf-8")
    for needle in ('rel="manifest"', 'name="theme-color"', 'apple-mobile-web-app-capable', 'rel="apple-touch-icon"', "viewport-fit=cover"):
        assert needle in html, (page, needle)


def test_installing_files_are_public_but_the_api_still_needs_a_token(client):
    c, srv = client
    for path in ("/manifest.webmanifest", "/sw.js", "/install", "/icons/icon-192.png", "/login"):
        assert c.get(path).status_code == 200, path
    assert c.get("/api/files").status_code == 401
    assert c.post("/api/upload", files={"file": ("x.txt", b"x")}).status_code == 401


def test_pages_and_manifest_are_consistent():
    m = json.loads((STATIC / "manifest.webmanifest").read_text())
    for ic in m["icons"]:
        assert (STATIC / ic["src"].lstrip("/")).is_file()
