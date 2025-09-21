# main.py — Bassam الذكي (نسخة مُحدَّثة تدعم ddgs وتنسيق TemplateResponse الجديد)
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from ddgs import DDGS  # <— الاسم الجديد لمكتبة DuckDuckGo
import httpx
from bs4 import BeautifulSoup
from datetime import datetime

app = FastAPI(title="Bassam الذكي", version="1.2")

# ربط static + القوالب
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ---------- بحث عربي موثوق عبر DuckDuckGo مع رجوع احتياطي ----------
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
                region="sa-ar",          # تركيز عربي
                safesearch="moderate",
                timelimit="y",           # آخر سنة
                max_results=maxn
            ):
                results.append({
                    "title": r.get("title", "") or "",
                    "href":  r.get("href", "") or "",
                    "body":  r.get("body", "") or ""
                })
    except Exception as e:
        print("DDG primary failed:", e)

    # محاولة 2: إن لم نجد شيئًا، بحث عام بدون region
    if not results:
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=maxn):
                    results.append({
                        "title": r
