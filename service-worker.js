/* MySwingMatch Service Worker
 * 전략 요약 (stale 캐시 사고 방지에 중점):
 *  - HTML / API : network-first  → 항상 최신 먼저, 네트워크 실패 시에만 캐시 폴백
 *  - 정적 자산(아이콘·이미지·manifest) : cache-first → 빠른 로딩
 *  - 새 버전 배포 시 CACHE_VERSION만 올리면 옛 캐시 자동 삭제
 *  - 결제/인증/분석 등 동적 요청은 캐시하지 않음
 */

const CACHE_VERSION = 'msm-v1';           // ← 코드 바꿀 때마다 v2, v3...로 올리면 옛 캐시 자동 정리
const STATIC_CACHE  = CACHE_VERSION + '-static';
const RUNTIME_CACHE = CACHE_VERSION + '-runtime';

// 설치 시 미리 받아둘 핵심 정적 파일 (있으면 오프라인에서도 기본 표시)
const PRECACHE_URLS = [
  '/',
  '/index.html',
  '/manifest.webmanifest',
  '/icons/icon-192.png',
  '/icons/icon-512.png',
  '/icons/apple-touch-icon.png',
];

// 캐시에 절대 넣지 않을 경로 (결제·인증·분석·API·외부 도메인)
function isNeverCache(url) {
  return (
    url.pathname.startsWith('/api/') ||
    url.pathname.includes('/auth') ||
    url.hostname.includes('supabase') ||
    url.hostname.includes('railway.app') ||
    url.hostname.includes('portone') ||
    url.hostname.includes('inicis') ||
    url.hostname.includes('google') ||
    url.hostname.includes('kakao')
  );
}

// 정적 자산 판별 (cache-first 대상)
function isStaticAsset(url) {
  return /\.(png|jpg|jpeg|gif|svg|webp|ico|woff2?|ttf|css)$/i.test(url.pathname) ||
         url.pathname === '/manifest.webmanifest';
}

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => cache.addAll(PRECACHE_URLS).catch(() => {}))
      .then(() => self.skipWaiting())   // 새 SW 즉시 활성화 대기
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((k) => !k.startsWith(CACHE_VERSION))  // 현재 버전 외 전부 삭제
          .map((k) => caches.delete(k))
      )
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;

  // GET 외(POST 등 결제·로그인)는 그냥 통과 — 절대 캐시 안 함
  if (req.method !== 'GET') return;

  let url;
  try { url = new URL(req.url); } catch (e) { return; }

  // 외부/민감 요청은 SW가 손대지 않고 네트워크로 직행
  if (url.origin !== self.location.origin || isNeverCache(url)) return;

  // 정적 자산 → cache-first
  if (isStaticAsset(url)) {
    event.respondWith(
      caches.match(req).then((cached) => {
        if (cached) return cached;
        return fetch(req).then((res) => {
          if (res && res.status === 200) {
            const copy = res.clone();
            caches.open(STATIC_CACHE).then((c) => c.put(req, copy));
          }
          return res;
        });
      })
    );
    return;
  }

  // 그 외(HTML 등) → network-first, 실패 시 캐시 폴백
  event.respondWith(
    fetch(req)
      .then((res) => {
        if (res && res.status === 200) {
          const copy = res.clone();
          caches.open(RUNTIME_CACHE).then((c) => c.put(req, copy));
        }
        return res;
      })
      .catch(() =>
        caches.match(req).then((cached) => cached || caches.match('/index.html'))
      )
  );
});
