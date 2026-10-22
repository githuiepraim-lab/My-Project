/* Minimal service worker: lets the phone treat JARVIS as an installable app and show the app shell
   instantly. It NEVER caches API calls, the WebSocket, uploads or anything under /api — those always
   go to the network, so nothing private is stored on the phone by this file. */
const SHELL = "jarvis-shell-v1";
const FILES = ["/login", "/icons/icon-192.png", "/manifest.webmanifest"];
self.addEventListener("install", e => { e.waitUntil(caches.open(SHELL).then(c => c.addAll(FILES)).then(() => self.skipWaiting())); });
self.addEventListener("activate", e => { e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== SHELL).map(k => caches.delete(k)))).then(() => self.clients.claim())); });
self.addEventListener("fetch", e => {
  const u = new URL(e.request.url);
  if (e.request.method !== "GET" || u.origin !== location.origin || u.pathname.startsWith("/api") || u.pathname === "/ws") return;   // network only
  e.respondWith(fetch(e.request).catch(() => caches.match(e.request).then(r => r || caches.match("/login"))));                    // network first, shell if offline
});
