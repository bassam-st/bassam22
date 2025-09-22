// Service Worker لبسام الذكي PWA
const CACHE_NAME = 'bassam-smart-v1.0';
const STATIC_CACHE_NAME = 'bassam-static-v1.0';

// الملفات المهمة التي نريد تخزينها
const STATIC_FILES = [
  '/',
  '/manifest.json',
  '/service-worker.js'
];

// أحداث التثبيت
self.addEventListener('install', (event) => {
  console.log('🔧 Service Worker: Installing...');
  
  event.waitUntil(
    caches.open(STATIC_CACHE_NAME)
      .then((cache) => {
        console.log('📦 Service Worker: Caching static files');
        return cache.addAll(STATIC_FILES);
      })
      .then(() => {
        console.log('✅ Service Worker: Installation complete');
        return self.skipWaiting();
      })
      .catch((error) => {
        console.error('❌ Service Worker: Installation failed', error);
      })
  );
});

// تفعيل Service Worker
self.addEventListener('activate', (event) => {
  console.log('🚀 Service Worker: Activating...');
  
  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames
            .filter((cacheName) => {
              // حذف المخازن القديمة
              return cacheName !== CACHE_NAME && 
                     cacheName !== STATIC_CACHE_NAME;
            })
            .map((cacheName) => {
              console.log('🗑️ Service Worker: Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            })
        );
      })
      .then(() => {
        console.log('✅ Service Worker: Activation complete');
        return self.clients.claim();
      })
  );
});

// استراتيجية Network First للمحتوى الديناميكي
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  
  // تجاهل الطلبات الخارجية
  if (url.origin !== location.origin) {
    return;
  }
  
  event.respondWith(
    handleRequest(event.request)
  );
});

async function handleRequest(request) {
  const url = new URL(request.url);
  
  try {
    // للملفات الثابتة: Cache First
    if (STATIC_FILES.includes(url.pathname)) {
      return await cacheFirst(request);
    }
    
    // للصفحة الرئيسية والطلبات الديناميكية: Network First
    if (url.pathname === '/' || request.method === 'POST') {
      return await networkFirst(request);
    }
    
    // للباقي: Network First مع fallback
    return await networkFirst(request);
    
  } catch (error) {
    console.error('❌ Service Worker: Request failed', error);
    
    // في حالة فشل الشبكة، حاول من المخزن
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // إذا لم توجد نسخة مخزنة، أرجع صفحة أساسية
    if (request.destination === 'document') {
      return await caches.match('/');
    }
    
    // للطلبات الأخرى، أرجع response فارغ
    return new Response('خدمة غير متوفرة حالياً', {
      status: 503,
      statusText: 'Service Unavailable',
      headers: { 'Content-Type': 'text/plain; charset=utf-8' }
    });
  }
}

// استراتيجية Cache First للملفات الثابتة
async function cacheFirst(request) {
  const cachedResponse = await caches.match(request);
  
  if (cachedResponse) {
    // تحديث المخزن في الخلفية
    fetch(request)
      .then(response => {
        if (response.ok) {
          const responseClone = response.clone();
          caches.open(STATIC_CACHE_NAME)
            .then(cache => cache.put(request, responseClone));
        }
      })
      .catch(() => {}); // تجاهل أخطاء التحديث
    
    return cachedResponse;
  }
  
  // إذا لم توجد في المخزن، جلب من الشبكة
  const networkResponse = await fetch(request);
  
  if (networkResponse.ok) {
    const responseClone = networkResponse.clone();
    const cache = await caches.open(STATIC_CACHE_NAME);
    await cache.put(request, responseClone);
  }
  
  return networkResponse;
}

// استراتيجية Network First للمحتوى الديناميكي
async function networkFirst(request) {
  try {
    const networkResponse = await fetch(request);
    
    // تخزين الاستجابات الناجحة
    if (networkResponse.ok && request.method === 'GET') {
      const responseClone = networkResponse.clone();
      const cache = await caches.open(CACHE_NAME);
      
      // تخزين مؤقت لمدة محدودة
      await cache.put(request, responseClone);
      
      // تنظيف المخزن إذا أصبح كبيراً
      cleanupCache();
    }
    
    return networkResponse;
    
  } catch (error) {
    console.log('🔄 Service Worker: Network failed, trying cache');
    
    // البحث في المخزن
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    throw error;
  }
}

// تنظيف المخزن المؤقت
async function cleanupCache() {
  try {
    const cache = await caches.open(CACHE_NAME);
    const requests = await cache.keys();
    
    // إذا كان المخزن كبيراً، احذف أقدم 10 عناصر
    if (requests.length > 50) {
      const oldRequests = requests.slice(0, 10);
      await Promise.all(
        oldRequests.map(request => cache.delete(request))
      );
    }
  } catch (error) {
    console.error('❌ Service Worker: Cache cleanup failed', error);
  }
}

// معالجة الرسائل من الصفحة الرئيسية
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
  
  if (event.data && event.data.type === 'CACHE_URLS') {
    const urls = event.data.urls;
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urls))
      .catch(error => console.error('❌ Failed to cache URLs:', error));
  }
});

// إشعار عند توفر تحديث جديد
self.addEventListener('message', (event) => {
  if (event.data === 'CHECK_UPDATE') {
    // إرسال إشعار للصفحة الرئيسية
    self.clients.matchAll().then(clients => {
      clients.forEach(client => {
        client.postMessage({
          type: 'UPDATE_AVAILABLE',
          message: 'يتوفر تحديث جديد لبسام الذكي'
        });
      });
    });
  }
});

console.log('🎉 Service Worker: بسام الذكي PWA محمل بنجاح!');
