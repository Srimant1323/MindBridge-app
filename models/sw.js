/**
 * MindBridge Service Worker v18
 * - Network-first for HTML, JS and CSS (new deploys show up immediately)
 * - Cache fallback for offline use of the app shell
 * - Model files (Hugging Face) and the transformers library are NOT handled here;
 *   Transformers.js keeps its own browser cache.
 */
const CACHE_NAME = 'mindbridge-v18';
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/css/style.css',
  '/css/premium_ui.css',
  '/css/clinical_features.css',
  '/js/app.js',
  '/js/demographics.js',
  '/js/dashboard.js',
  '/js/tracker.js',
  '/js/interventions.js',
  '/js/nlp.js',
  '/assets/lexicon.js',
  '/js/audio.js',
  '/js/facial_affect.js',
  '/js/cbt_agent.js',
  '/js/clinical_summary.js',
  '/js/explainer.js',
  '/js/emotion.js',
  '/js/phq9.js',
  '/js/journal.js',
  '/js/camera.js',
  '/js/reflection.js',
  '/js/wrap.js',
  '/js/streak.js'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache =>
      // cache each file separately so one missing file does not block the rest
      Promise.all(STATIC_ASSETS.map(url =>
        cache.add(url).catch(err => console.warn('[SW] Not cached:', url, err))
      ))
    )
  );
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', event => {
  const req = event.request;
  if (req.method !== 'GET') return;
  if (!req.url.startsWith('http')) return;

  // Large model files, the transformers library and the OCR engine: leave to the browser
  if (req.url.includes('huggingface.co') ||
      req.url.includes('cdn.jsdelivr.net') ||
      req.url.includes('tesseract')) return;

  // Same-origin only from here
  if (new URL(req.url).origin !== self.location.origin) return;

  // Network first (HTML, JS, CSS, everything else); fall back to cache when offline
  event.respondWith(
    fetch(req)
      .then(response => {
        if (response && response.status === 200) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(req, clone));
        }
        return response;
      })
      .catch(() =>
        caches.match(req).then(cached => cached || (req.mode === 'navigate' ? caches.match('/index.html') : undefined))
      )
  );
});
