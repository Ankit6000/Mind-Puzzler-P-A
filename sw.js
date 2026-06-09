const CACHE_NAME = "puzzle-lens-pages-v5";
const APP_SHELL = [
  "./",
  "./index.html",
  "./static/app.css",
  "./static/app.js",
  "./static/puzzle-core.js",
  "./static/trained-model.json",
  "./static/manifest.webmanifest",
  "./static/icons/icon-192.png",
  "./static/icons/icon-512.png",
  "./reference/11.png",
  "./reference/12.png",
  "./reference/13.png",
  "./reference/14.png",
  "./reference/21.png",
  "./reference/22.png",
  "./reference/23.png",
  "./reference/24.png",
  "./reference/31.png",
  "./reference/32.png",
  "./reference/33.png",
  "./reference/34.png",
  "./reference/41.png",
  "./reference/42.png",
  "./reference/43.png",
  "./reference/44.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(APP_SHELL))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  const url = new URL(request.url);
  if (request.method !== "GET" || url.origin !== self.location.origin) return;

  event.respondWith(
    caches.match(request).then((cached) => {
      if (cached) return cached;
      return fetch(request)
        .then((response) => {
          if (!response || response.status !== 200 || response.type !== "basic") return response;
          const copy = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(request, copy));
          return response;
        })
        .catch(() => caches.match("./index.html"));
    }),
  );
});
