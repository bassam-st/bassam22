# main.py — Bassam الذكي (بحث عربي + تلخيص + أسعار + صور + PDF)

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ddgs import DDGS   # ✅ التصحيح هنا (بدل duckduckgo_search)

from readability import Document
from bs4 import BeautifulSoup
from diskcache import Cache
from fpdf import FPDF
import httpx, re, os

# ----- إعدادات FastAPI -----
app = FastAPI(title="Bassam الذكي", version="2.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

cache = Cache(directory=".cache")

# ----- دالة البحث -----
def search_ar(q: str, maxn: int = 8):
    """
    بحث عربي باستخدام ddgs مع fallback.
    يرجع قائمة نتائج: [{title, href, body}, ...]
    """
    results = []
    query = f"{q} بالعربية"

    try:
        with DDGS() as ddgs:
            for r in ddgs.text(
                query,
                region="sa-ar",       # تركيز على المحتوى العربي
                safesearch="moderate",
                timelimit="y",        # آخر سنة
                max_results=maxn
            ):
                results.append({
                    "title": r.get("title", ""),
                    "href": r.get("href", ""),
                    "body": r.get("body", "")
                })
    except Exception as e:
        print("DDGS primary failed:", e)

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
            print("DDGS fallback failed:", e)

    return results

# ----- الصفحة الرئيسية -----
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ----- API: بحث -----
@app.post("/search", response_class=HTMLResponse)
async def do_search(request: Request, q: str = Form(...), mode: str = Form("summarize")):
    if not q.strip():
        return templates.TemplateResponse("index.html", {"request": request, "error": "الرجاء إدخال سؤال"})

    results = search_ar(q)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "query": q, "results": results, "mode": mode}
    )

# ----- Health check -----
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
