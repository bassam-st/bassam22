# main.py — Bassam الذكي: بحث عربي + تلخيص + أسعار + صور + PDF
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
import requests, re, html, time
from diskcache import Cache
from fpdf import FPDF

# ---------------------------------------------
# 🔹 تفعيل FastAPI
# ---------------------------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------------------------------------------
# 🔹 كاش للتسريع
# ---------------------------------------------
cache = Cache("./cache")

# ---------------------------------------------
# 🔹 بحث عربي موثوق عبر DuckDuckGo مع رجوع احتياطي
# ---------------------------------------------
def search_ar(q: str, maxn: int = 8):
    results = []
    query = f"{q} بالعربية"

    # محاولة 1: بحث مخصص للمنطقة العربية
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
                    "title": r.get("title",""),
                    "href":  r.get("href",""),
                    "body":  r.get("body","")
                })
    except Exception as e:
        print("DDG primary failed:", e)

    # محاولة 2: بحث عام إذا لم توجد نتائج
    if not results:
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=maxn):
                    results.append({
                        "title": r.get("title",""),
                        "href":  r.get("href",""),
                        "body":  r.get("body","")
                    })
        except Exception as e:
            print("DDG fallback failed:", e)

    return results

# ---------------------------------------------
# 🔹 مسارات افتراضية للتجربة
# ---------------------------------------------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------------------------------------------
# 🔹 البحث في الواجهة
# ---------------------------------------------
@app.post("/", response_class=HTMLResponse)
async def search(request: Request, q: str = Form(...), mode: str = Form("summarize")):
    result_text = ""
    hits = search_ar(q, maxn=8)

    if not hits:
        result_text = "لم أعثر على نتائج. جرّب صياغة أخرى أو أضف كلمة 'بالعربية'."
    else:
        # إذا كان الوضع تلخيص
        if mode == "summarize":
            joined = "\n".join([h["body"] for h in hits if h["body"]])
            if not joined.strip():
                result_text = "لم أجد محتوى للتلخيص."
            else:
                # ترجمة للتاكد انها بالعربي
                try:
                    result_text = GoogleTranslator(source="auto", target="ar").translate(joined[:1000])
                except Exception:
                    result_text = joined[:500]

        # أسعار المتاجر (مستقبلاً ممكن تطويرها أكثر)
        elif mode == "prices":
            lines = []
            for h in hits:
                if h["title"]:
                    lines.append(f"- {h['title']} → {h['href']}")
            result_text = "\n".join(lines[:10]) or "لا توجد أسعار."

        # صور
        elif mode == "images":
            result_text = "ميزة الصور ستُضاف لاحقاً."

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result_text,
        "query": q
    })
