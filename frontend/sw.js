// Receives images shared to the installed app (the manifest's share_target)
// and hands them to the page through the Cache API. Nothing else is cached.

const SHARE_CACHE = 'shared-image';

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);
    if (event.request.method !== 'POST' || !url.pathname.endsWith('/share-target')) {
        return;
    }
    event.respondWith((async () => {
        const home = url.pathname.slice(0, -'share-target'.length);
        try {
            const file = (await event.request.formData()).get('image');
            if (file) {
                const cache = await caches.open(SHARE_CACHE);
                await cache.put('/shared-image', new Response(file, { headers: { 'Content-Type': file.type } }));
                return Response.redirect(home + '?shared=1', 303);
            }
        } catch {
            // fall through to the plain page
        }
        return Response.redirect(home, 303);
    })());
});
