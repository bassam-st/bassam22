from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from ddgs import DDGS

app = FastAPI()

# ---------- HTML القالب الأساسي ----------
def page_html(query: str = "", results_html: str = "") -> str:
    return f"""<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>الذكي Bassam — بحث عربي + تلخيص + أسعار + صور + PDF</title>
<style>
  body {{ font-family: system-ui, Tahoma, Arial; background:#f7f7fb; color:#222; }}
  .wrap {{ max-width: 900px; margin: 28px auto; padding: 0 12px; }}
  h1 {{ font-size: 22px; margin-bottom: 16px; }}
  form {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
  input[type=text]{{ flex: 1 1 420px; padding:12px 14px; border:1px solid #ccc; border-radius:10px; }}
  button {{ padding:12px 18px; border:0; background:#006eff; color:#fff; border-radius:10px; cursor:pointer; }}
  .modes {{ margin:8px 0 2px; display:flex; gap:18px; align-items:center; }}
  .card {{ background:#fff; border:1px solid #eee; border-radius:14px; padding:16px; margin-top:16px; }}
  .item {{ border-bottom:1px solid #eee; padding:12px 0; }}
  .item:last-child {{ border-bottom:0; }}
  .item a {{ color:#0b63d1; text-decoration:none; }}
  .muted {{ color:#666; font-size: 13px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>الذكي Bassam — بحث عربي + تلخيص + أسعار + صور + PDF</h1>

  <form action="/" method="post" enctype="application/x-www-form-urlencoded">
    <input type="text" name="q" placeholder="اكتب سؤالك هنا..." value="{query}" autofocus />
    <button type="submit">بحث</button>
    <div class="modes muted">
      <label><input type="radio" name="mode" value="search" checked /> بحث عربي</label>
      <!-- يمكن إضافة أوضاع لاحقًا مثل تلخيص/أسعار/صور -->
    </div>
  </form>

  <div class="card" id="result">
    {results_html if results_html else '<span class="muted">اكتب سؤالك ثم اضغط بحث.</span>'}
  </div>

  <p class="muted" style="margin-top:18px">يعمل بـ FastAPI — الذكي Bassam ©</p>
</div>
</body>
</html>
"""

# ---------- دالة البحث العربي عبر DDGS ----------
def search_ar(q: str, maxn: int = 10):
    """
    نجبر البحث على العربية بإضافة كلمة 'بالعربية' ونستخدم region عربي.
    نرجع قائمة قواميس: [{title, href, body}, ...]
    """
    results = []
    query = f"{q.strip()} بالعربية" if q else ""

    if not query:
        return results

    # محاولة 1: تركيز عربي + آخر سنة
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(
                query,
                region="sa-ar",
                safesearch="moderate",
                timelimit="y",
                max_results=maxn
            ):
                results.append({
                    "title": r.get("title", "") or "",
                    "href": r.get("href", "") or "",
                    "body": r.get("body", "") or "",
                })
    except Exception as e:
        print("DDGS primary failed:", e)

    # محاولة 2: لو ما طلع شيء — بحث عام
    if not results:
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=maxn):
                    results.append({
                        "title": r.get("title", "") or "",
                        "href": r.get("href", "") or "",
                        "body": r.get("body", "") or "",
                    })
        except Exception as e:
            print("DDGS fallback failed:", e)

    return results


# ---------- GET: عرض الصفحة ----------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # صفحة نظيفة بدون نتائج
    return HTMLResponse(page_html())


# ---------- POST: تنفيذ البحث وعرض النتائج ----------
@app.post("/", response_class=HTMLResponse)
async def search(request: Request, q: str = Form(...), mode: str = Form("search")):
    q = (q or "").strip()
    if not q:
        # رجوع لنفس الصفحة برسالة بسيطة
        html = page_html("", '<span class="muted">الرجاء كتابة سؤال ثم الضغط على بحث.</span>')
        return HTMLResponse(html)

    # حالياً نستخدم وضع "بحث عربي" فقط
    hits = search_ar(q, maxn=10)

    if not hits:
        results_html = '<div class="muted">لم أعثر على نتائج. جرّب صياغة أخرى، أو أضف كلمة «بالعربية».</div>'
    else:
        # نبني HTML للنتائج
        items = []
        for h in hits:
            title = (h.get("title") or "").strip() or "(بدون عنوان)"
            link = (h.get("href") or "").strip()
            body = (h.get("body") or "").strip()
            items.append(f"""
              <div class="item">
                <div><a href="{link}" target="_blank" rel="noopener">{title}</a></div>
                <div class="muted">{body}</div>
              </div>
            """)
        results_html = f"<div><b>نتائج عن:</b> {q}</div>" + "".join(items)

    return HTMLResponse(page_html(q, results_html))


# ---------- مسار بسيط لفحص الصحة ----------
@app.get("/health", response_class=HTMLResponse)
async def health():
    return HTMLResponse("OK")
