# main.py — Bassam الذكي: تلخيص + أسعار + صور + PDF
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from duckduckgo_search import DDGS
import re
from fpdf import FPDF

app = FastAPI(title="Bassam الذكي", version="1.0")

# ربط static + القوالب
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ========= أدوات مساعدة =========
MARKET_SITES = [
    "aliexpress.com", "amazon.com", "amazon.sa", "amazon.ae", "amazon.eg",
    "noon.com", "ebay.com", "alibaba.com", "temu.com", "made-in-china.com",
]

def ddg_text(query: str, max_results: int = 12):
    items = []
    with DDGS() as dd:
        for r in dd.text(keywords=query, region="xa-ar", safesearch="moderate", max_results=max_results):
            items.append(r)  # dict: title, href, body
    return items

def ddg_images(query: str, max_results: int = 20):
    items = []
    with DDGS() as dd:
        for r in dd.images(keywords=query, region="xa-ar", safesearch="off", max_results=max_results):
            items.append(r)  # dict: image, title, url
    return items

PRICE_RE = re.compile(r"(?i)(US?\$|USD|EUR|GBP|AED|SAR|EGP|QAR|KWD|OMR|د\.إ|ر\.س|ج\.م|د\.ك|ر\.ق|ر\.ع)\s*[\d\.,]+")
AR_NUM = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# ========= الصفحات =========
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "q": "", "mode": "summary", "answer_text": "", "result_panel": ""}
    )

@app.post("/", response_class=HTMLResponse)
async def run(request: Request, question: str = Form(...), mode: str = Form("summary")):
    q = (question or "").strip()
    if not q:
        return templates.TemplateResponse("index.html", {"request": request, "q": "", "mode": mode, "answer_text": "", "result_panel": ""})

    if mode == "prices":
        panel, answer = handle_prices(q)
    elif mode == "images":
        panel, answer = handle_images(q)
    else:
        panel, answer = handle_summary(q)

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "q": q, "mode": mode, "answer_text": answer, "result_panel": panel}
    )

# لقبول HEAD على /
@app.head("/")
async def head_root():
    return Response(status_code=204)

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

from datetime import datetime
@app.get("/time")
async def time_now():
    return {"time": datetime.utcnow().isoformat()}

# ========= منطق التلخيص =========
def handle_summary(q: str):
    # نحاول بالعربية أولاً ثم نرجع إنجليزي لو قليل النتائج
    results = ddg_text(q + " بالعربية", max_results=12)
    if not results:
        results = ddg_text(q, max_results=8)

    picked = []
    for r in results:
        title = r.get("title") or ""
        body = (r.get("body") or "").strip()
        href = r.get("href") or ""
        if not body:
            continue
        # نأخذ 2-3 جمل قصيرة
        snippet = " ".join(body.split()[:40])
        picked.append((title, snippet, href))
        if len(picked) >= 3:
            break

    if not picked:
        panel = '<div class="card" style="margin-top:12px;">لم أعثر على نصوص مناسبة. جرّب صياغة أخرى.</div>'
        return panel, "لا توجد بيانات."

    answer_text = "ملخص:\n" + "\n".join([f"- {t} — {s}" for t, s, _ in picked])

    cards = []
    for (t, s, u) in picked:
        cards.append(
            f'<div class="card" style="margin-top:10px">'
            f'<strong>{t}</strong>'
            f'<div class="summary" style="margin-top:8px">{s}</div>'
            f'<div style="margin-top:8px"><a target="_blank" href="{u}">فتح المصدر</a></div>'
            f'</div>'
        )
    panel = f'<div style="margin-top:18px"><h3>الملخص (من النتائج):</h3>{"".join(cards)}</div>'
    return panel, answer_text

# ========= منطق الأسعار =========
def handle_prices(q: str):
    site_filter = " OR ".join([f"site:{s}" for s in MARKET_SITES])
    results = ddg_text(f"{q} {site_filter}", max_results=20)

    seen = set()
    cards, lines = [], []
    for r in results:
        url = r.get("href") or ""
        if not url or url in seen:
            continue
        seen.add(url)
        title = r.get("title") or url
        snippet = (r.get("body") or "").translate(AR_NUM)
        m = PRICE_RE.search(snippet or "")
        price = m.group(0) if m else ""
        price_html = f"<div><strong>السعر:</strong> {price}</div>" if price else "<div>السعر غير واضح — افتح الرابط للتأكد.</div>"
        cards.append(
            f'<div class="card" style="margin-top:10px">'
            f'<strong>{title}</strong>{price_html}'
            f'<div style="margin-top:8px"><a target="_blank" href="{url}">فتح المصدر</a></div>'
            f'</div>'
        )
        lines.append(f"- {title} | {price or '—'} | {url}")
        if len(cards) >= 8:
            break

    if not cards:
        panel = '<div class="card" style="margin-top:12px;">لم أجد نتائج أسعار مناسبة. جرّب اسم موديل أدق أو متجر معين.</div>'
        return panel, "لا توجد بيانات."

    panel = f'<div style="margin-top:18px;"><h3>نتائج أسعار عن: {q}</h3>{"".join(cards)}</div>'
    answer = "نتائج أسعار:\n" + "\n".join(lines)
    return panel, answer

# ========= منطق الصور =========
def handle_images(q: str):
    items = ddg_images(q, max_results=16)
    if not items:
        panel = '<div class="card" style="margin-top:12px;">لم أجد صورًا مناسبة.</div>'
        return panel, ""

    cards = []
    for it in items[:16]:
        img = it.get("image")
        src = it.get("url") or img
        if img:
            cards.append(f'<div class="imgcard"><a href="{src}" target="_blank"><img src="{img}" alt=""/></a></div>')
    panel = f'<div style="margin-top:18px;"><h3>نتائج صور عن: {q}</h3><div class="imggrid">{"".join(cards)}</div></div>'
    return panel, ""

# ========= تصدير PDF بسيط =========
@app.get("/export_pdf")
def export_pdf(q: str, mode: str = "summary"):
    # نعيد تشغيل المنطق لنفس السؤال لإخراج نص
    if mode == "prices":
        _, text = handle_prices(q)
        title = f"بحث أسعار: {q}"
    elif mode == "images":
        _, text = handle_images(q)
        title = f"نتائج صور: {q}"
        if not text:
            text = "يرجى فتح الموقع لمعاينة الصور والروابط."
    else:
        _, text = handle_summary(q)
        title = f"ملخص البحث: {q}"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.multi_cell(0, 10, title)
    pdf.ln(4)
    pdf.set_font("Arial", size=12)
    for line in (text or "").split("\n"):
        pdf.multi_cell(0, 8, line)

    data = pdf.output(dest="S").encode("latin1", "ignore")
    headers = {
        "Content-Disposition": 'attachment; filename="bassam_ai.pdf"',
        "Content-Type": "application/pdf",
    }
    return Response(content=data, headers=headers)
