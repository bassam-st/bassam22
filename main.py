# main.py — Bassam: بحث عربي + تلخيص + أسعار + صور + PDF
from fastapi import FastAPI, Request, Form, Query, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from html import escape
import re, html, time
import requests
from urllib.parse import urlparse
from diskcache import Cache
from bs4 import BeautifulSoup
from readability import Document
from fpdf import FPDF

# محاولة استيراد DuckDuckGo
try:
    from duckduckgo_search import ddg
except Exception:
    ddg = None

app = FastAPI(title="Bassam App", version="1.1")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ربط static + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# كاش بسيط على القرص
cache = Cache(".cache")

# هيدر طلبات
HDRS = {
    "User-Agent": "Mozilla/5.0 (compatible; BassamBot/1.4)",
    "Accept-Language": "ar,en;q=0.7"
}

# نطاقات عربية مفضّلة لتعزيز ترتيبها
PREFERRED_AR = {
    "ar.wikipedia.org", "ar.m.wikipedia.org", "mawdoo3.com",
    "almrsal.com", "sasapost.com", "bbcarabic.com", "aljazeera.net",
    "arabic.cnn.com", "ar.wikihow.com"
}

# ===== أدوات لغة بسيطة =====
AR_RE = re.compile(r"[اأإآء-ي]")

def is_arabic(s: str, min_ar=6) -> bool:
    """كشف بسيط: هل بالنص أحرف عربية كافية؟"""
    return len(AR_RE.findall(s or "")) >= min_ar

def tokenize(s: str):
    s = re.sub(r"[^\w\s\u0600-\u06FF]+", " ", (s or "").lower())
    return [t for t in s.split() if t]

def summarize_from_text(text: str, query: str, max_sentences=3):
    """تلخيص بسيط عبر اختيار الجُمل الأقرب لعبارات السؤال."""
    sentences = re.split(r'(?<=[\.\!\?\؟])\s+|\n+', text or "")
    q_terms = set(tokenize(query))
    scored = []
    for s in sentences:
        s2 = s.strip()
        if len(s2) < 25:
            continue
        # لا تشترط عربية بقسوة، فقط أعطِ أولوية للجُمل العربية
        base = 1 + (1 if is_arabic(s2, 4) else 0)
        inter = q_terms & set(tokenize(s2))
        score = base + len(inter)
        if score > 1:
            scored.append((score, s2))
    scored.sort(key=lambda x: x[0], reverse=True)
    return " ".join([s for _, s in scored[:max_sentences]])

def domain_of(url: str):
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

# ===== جلب الصفحات واستخراج نص قابل للتلخيص =====
def fetch(url: str, timeout=8) -> str:
    r = requests.get(url, headers=HDRS, timeout=timeout)
    r.raise_for_status()
    # تجنّب الصفحات الهائلة
    if len(r.text) > 2_000_000:
        return r.text[:2_000_000]
    return r.text

def extract_readable_text(html_text: str) -> str:
    try:
        doc = Document(html_text)
        content_html = doc.summary()
        soup = BeautifulSoup(content_html, "html.parser")
        text = soup.get_text(separator="\n")
        return html.unescape(text)
    except Exception:
        soup = BeautifulSoup(html_text or "", "html.parser")
        return soup.get_text(separator="\n")

# ===== ترتيب النتائج: فضّل العربي والمواقع الموثوقة =====
def priority_key(item):
    d = domain_of(item.get("href") or item.get("url") or "")
    base = 2.0
    if d in PREFERRED_AR:
        base -= 0.6
    # أعطِ أولوية لأي عنوان/وصف عربي
    title = (item.get("title") or "") + " " + (item.get("body") or "")
    if is_arabic(title, 6):
        base -= 0.5
    return base

# ===== الصفحة الرئيسية =====
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "q": "", "mode": "summary",
         "result_panel": "", "answer_text": ""}
    )

# ===== نموذج البحث (POST) =====
@app.post("/", response_class=HTMLResponse)
def run(request: Request, question: str = Form(...), mode: str = Form("summary")):
    q = (question or "").strip()
    if not q:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "q": "", "mode": mode,
             "result_panel": "", "answer_text": ""}
        )
    panel, answer = handle_summary(q)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "q": q, "mode": "summary",
         "result_panel": panel, "answer_text": answer}
    )

# ===== منطق البحث + التلخيص (دائمًا بالعربي قدر الإمكان) =====
def ddg_variants(q: str):
    """عدة استعلامات لضمان نتائج حتى لو فشل واحد."""
    queries = [
        q + " بالعربية",
        q + " ar",
        q,
    ]
    regions = ["xa-ar", "sa-ar", "ma-ar", "eg-ar", "wt-wt"]
    seen = set()
    out = []
    if not ddg:
        return out
    for query in queries:
        for reg in regions:
            try:
                res = ddg(query, region=reg, safesearch="Moderate", max_results=12) or []
                for r in res:
                    key = r.get("href") or r.get("url")
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    out.append(r)
            except Exception:
                continue
            # إن وجدنا عددًا جيدًا نكتفي
            if len(out) >= 18:
                return out
    return out

def handle_summary(q: str):
    cache_key = "sum2:" + q
    cached = cache.get(cache_key)
    if cached:
        return cached, ""

    results = ddg_variants(q)
    if not results:
        panel = '<div class="card">لم أعثر على نتائج. جرّب صياغة أخرى أو أضف كلمة "بالعربية".</div>'
        cache.set(cache_key, panel, expire=300)
        return panel, "لا توجد بيانات."

    # رتّب النتائج بأولوية العربي
    results = sorted(results, key=priority_key)

    source_cards = []
    combined_chunks = []

    for r in results:
        href = r.get("href") or r.get("url")
        title = r.get("title") or href
        snippet = r.get("body") or ""
        if not href:
            continue

        # كاش لكل رابط
        ckey = "url:" + href
        text = cache.get(ckey)
        if text is None:
            try:
                raw = fetch(href, timeout=8)
                text = extract_readable_text(raw)
                cache.set(ckey, text, expire=60 * 60 * 12)
            except Exception:
                text = ""

        # إن فشل الاستخراج، استخدم مقتطف DuckDuckGo
        useful_text = text if text and len(text) > 150 else snippet

        # لو ما في شيء مفيد، تجاهل
        if not useful_text or len(useful_text) < 60:
            continue

        # أعطِ تلخيصًا بسيطًا
        summ = summarize_from_text(useful_text, q, max_sentences=3)
        if not summ:
            # كمل بالمقتطف مباشرة
            summ = " ".join((useful_text.strip().split())[:60]) + "…"

        combined_chunks.append(summ)

        # بطاقة مصدر
        domain = domain_of(href)
        source_cards.append(
            f'<div class="card" style="margin-top:10px">'
            f'<strong>{escape(title)}</strong>'
            f'<div class="summary" style="margin-top:8px">{escape(summ)}</div>'
            f'<div style="margin-top:8px"><a target="_blank" href="{escape(href)}">فتح المصدر</a></div>'
            f'</div>'
        )

        if len(source_cards) >= 4:
            break

    if not combined_chunks:
        panel = '<div class="card">لم أعثر على نصوص مناسبة. جرّب صياغة أخرى.</div>'
        cache.set(cache_key, panel, expire=300)
        return panel, "لا توجد بيانات."

    final_answer = " ".join(combined_chunks)
    panel = (
        f'<div style="margin-top:18px">'
        f'<h3>سؤالك:</h3><div class="card">{escape(q)}</div>'
        f'<h3 style="margin-top:12px">الملخص (بالعربية):</h3>'
        f'<div class="summary">{escape(final_answer)}</div>'
        f'<h3 style="margin-top:12px">المصادر:</h3>'
        f'{"".join(source_cards)}'
        f'</div>'
    )

    cache.set(cache_key, panel, expire=60 * 60)
    return panel, final_answer

# ===== تصدير PDF =====
@app.get("/export_pdf")
def export_pdf(q: str, mode: str = "summary"):
    panel_html = cache.get("sum2:" + q)
    text_for_pdf = "لا توجد بيانات."
    if panel_html:
        soup = BeautifulSoup(panel_html, "html.parser")
        summ_div = soup.find("div", {"class": "summary"})
        if summ_div:
            text_for_pdf = summ_div.get_text(" ", strip=True)

    title = f"ملخص البحث: {q}"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.multi_cell(0, 10, title)
    pdf.ln(4)
    pdf.set_font("Arial", size=12)
    for line in (text_for_pdf or "").split("\n"):
        pdf.multi_cell(0, 8, line)

    pdf_bytes = pdf.output(dest="S").encode("latin1", "ignore")
    headers = {
        "Content-Disposition": 'attachment; filename="bassam_ai_summary.pdf"',
        "Content-Type": "application/pdf",
    }
    return Response(content=pdf_bytes, headers=headers)

# ===== صحة =====
@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/health")
def health():
    return {"ok": True}
