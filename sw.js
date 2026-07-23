const CACHE="taiga-investment-os-v1";
self.addEventListener("install",event=>{
  event.waitUntil(caches.open(CACHE).then(cache=>cache.addAll(["./","./index.html","./manifest.webmanifest","./assets/icon.svg"])));
});
self.addEventListener("fetch",event=>{
  if(event.request.url.includes("data.json")||event.request.url.includes("history.json"))return;
  event.respondWith(fetch(event.request).catch(()=>caches.match(event.request)));
});
