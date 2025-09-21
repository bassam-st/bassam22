# --- بحث عربي موثوق عبر DuckDuckGo مع رجوع احتياطي ---
from ddgs import DDGS   # استبدل duckduckgo_search القديمة

def search_ar(q: str, maxn: int = 8):
    """
    يجبر البحث على العربية بإضافة 'بالعربية' وتحديد المنطقة العربية.
    وفي حال فشل/لا توجد نتائج، يعمل رجوع احتياطي عام.
    يرجع قائمة: [{title, href, body}, ...]
    """
    results = []
    query = f"{q} بالعربية"

    # محاولة 1: منطقة السعودية/العربية + فلترة معتدلة + آخر سنة
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(
                query,
                region="sa-ar",           # تركيز عربي
                safesearch="moderate",
                timelimit="y",            # آخر سنة
                max_results=maxn
            ):
                results.append({
                    "title": r.get("title", ""),
                    "href": r.get("href", ""),
                    "body": r.get("body", "")
                })
    except Exception as e:
        print("DDG primary failed:", e)

    # محاولة 2: إن لم نجد شيئًا، بحث عام بدون region
    if not results:
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=maxn):
                    results.append({
                        "title": r.get("title", ""),
                        "href": r.get("href", ""),
                        "body": r.get("body", "")
                    })
        except Exception as e:
            print("DDG fallback failed:", e)

    return results
