# main.py — بحث عربي + ترجمة تلقائية للإنجليزي + تلخيص + أسعار + صور + PDF
from fastapi import FastAPI, Form, Request, Response, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

import re, requests, time, html
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from readability import Document
from diskcache import Cache
from fpdf import FPDF

# بحث DuckDuckGo
from duckduckgo_search import ddg
try:
    from duckduckgo_search import DDGS  # للصور
except Exception:
    DDGS = None

# الترجمة التلقائية
try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

app = FastAPI(title="Bassam App", version="1.2")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ربط static + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
cache = Cache(".cache")

# ========= إعدادات بسيطة =========
HDRS = {"User-Agent": "Mozilla/5.0 (compatible; BassamBot/1.4)", "Accept-Language": "ar,en;q=0.8"}
PREFERRED_AR_DOMAINS = {"ar.wikipedia.org","ar.m.wikipedia.org","mawdoo3.com","almrsal.com","sasapost.com",
                        "arabic.cnn.com","bbcarabic.com","aljazeera.net","ar.wikihow.com"}
MARKET_SITES = ["alibaba.com","1688.com","aliexpress.com","amazon.com","amazon.ae","amazon.sa","amazon.eg",
                "noon.com","jumia.com","jumia.com.eg","ebay.com","made-in-china.com","temu.com","souq.com"]

# ========= أدوات اللغة =========
AR_RE = re.compile(r"[اأإآء-ي]")
def is_arabic(text: str, min_ar=8) -> bool:
    return len(AR_RE.findall(text or "")) >= min_ar

def translate_to_ar(text: str) -> str:
    """ترجمة تلقائية للغة العربية إذا كانت المكتبة متوفرة وإلا يرجع النص كما هو."""
    text = (text or "").strip()
    if not text:
        return text
    if is_arabic(text, 4):
        return text
    try:
        if GoogleTranslator:
            return GoogleTranslator(source="auto", target="ar").translate(text)
    except Exception:
        pass
    return text

STOP = set("""من في على إلى عن أن إن بأن كان تكون يكون التي الذي الذين هذا هذه ذلك هناك ثم حيث كما اذا إذا أو و يا ما مع قد لم لن بين لدى عند بعد قبل دون غير حتى كل أي كيف لماذا متى هل الى ال""".split())
def tokenize(s: str):
    s = re.sub(r"[^\w\s\u0600-\u06FF]+", " ", (s or "").lower())
    return [t for t in s.split() if t and t not in STOP]

def score_sentences(text: str, query: str):
    sentences = re.split(r'(?<=[\.\!\?\؟])\s+|\n+', text or "")
    q_terms = set(tokenize(query))
    out = []
    for s in sentences:
        s2 = s.strip()
        if len(s2) < 25:
            continue
        score = len(q_terms & set(tokenize(s2)))
        if score > 0:
            out.append((score + (len(s2)>=80), s2))
    out.sort(reverse=True, key=lambda x: x[0])
    return [s for _, s in out[:8]]

def summarize_from_text(text: str, query: str, max_sentences=3):
    sents = score_sentences(text, query)
    txt = " ".join(sents[:max_sentences]) if sents else ""
    # إن كان الملخص غير عربي → نترجمه
    return translate_to_ar(txt)

def domain_of(url: str):
    try:
        return urlparse(url).netloc.lower()
    except:
        return url

# ========= جلب الصفحات =========
def fetch(url: str, timeout=7):
    r = requests.get(url, headers=HDRS, timeout=timeout)
    r.raise_for_status()
    return r.text

def fetch_and_extract(url: str, timeout=7):
    try:
        raw = fetch(url, timeout=timeout)
        doc = Document(raw)
        content_html = doc.summary()
        text = BeautifulSoup(content_html, "html.parser").get_text(separator="\n")
        return html.unescape(text), raw
    except Exception:
        return "", ""

# ========= أولوية النتائج =========
def priority_key(item, mode="summary"):
    d = domain_of(item.get("href") or item.get("link") or item.get("url") or "")
    base = 2
    if d in PREFERRED_AR_DOMAINS: base -= 1
    if mode == "prices" and any(d.endswith(ms) or d==ms for ms in MARKET_SITES): base -= 0.5
    return base

# ========= الصفحات =========
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def run(request: Request, question: str = Form(...), mode: str = Form("summary")):
    q = (question or "").strip()
    if not q:
        return templates.TemplateResponse("index.html",
            {"request": request, "result_panel": "", "q": "", "mode": mode, "answer_text": ""})

    if mode == "prices":
        panel, answer_text = await handle_prices(q, return_plain=True)
    elif mode == "images":
        panel, answer_text = await handle_images(q)
    else:
        panel, answer_text = await handle_summary(q, return_plain=True)

    return templates.TemplateResponse("index.html",
        {"request": request, "result_panel": panel, "q": q, "mode": mode, "answer_text": (answer_text or "")})

@app.post("/feedback")
async def feedback(domain: str = Form(...), delta: int = Form(...)):
    return JSONResponse({"ok": True, "domain": domain, "score": 0})

# ========= وضع التلخيص (مع ترجمة تلقائية) =========
async def handle_summary(q: str, return_plain=False):
    cache_key = "sum:" + q
    cached = cache.get(cache_key)
    if cached and not return_plain:
        return cached, ""

    # نجبر البحث على الميل العربي أولًا
    results = ddg(q + " بالعربية", region="xa-ar", safesearch="Moderate", max_results=14) or []
    if not results:
        results = ddg(q, region="xa-ar", safesearch="Moderate", max_results=14) or []

    source_cards, chunks = [], []
    for r in sorted(results, key=lambda it: priority_key(it, "summary")):
        href = r.get("href") or r.get("link") or r.get("url")
        title = r.get("title") or ""
        if not href:
            continue
        page_text, _ = fetch_and_extract(href)
        if not page_text:
            continue
        summ = summarize_from_text(page_text, q, max_sentences=3)
        if not summ:
            continue
        # عنوان المصدر أيضًا نترجمه إن كان إنجليزي
        title_ar = translate_to_ar(title) if title else ""
        chunks.append(summ)
        source_cards.append(make_summary_card(title_ar or title, href, summ))
        if len(source_cards) >= 3:
            break

    # لو مافي شيء عربي، نأخذ نتائج إنجليزية ونترجمها
    if not chunks:
        results_en = ddg(q, region="wt-wt", safesearch="Moderate", max_results=8) or []
        for r in results_en:
            href = r.get("href") or r.get("link") or r.get("url")
            title = r.get("title") or ""
            if not href:
                continue
            txt, _ = fetch_and_extract(href)
            if not txt:
                continue
            # نأخذ أول 60 كلمة مثلًا ثم نترجم
            preview = " ".join((txt.strip().split())[:60])
            summ = translate_to_ar(preview)
            if not summ:
                continue
            chunks.append(summ)
            source_cards.append(make_summary_card(translate_to_ar(title) or href, href, summ))
            if len(source_cards) >= 3:
                break

    if not chunks:
        panel = '<div class="card" style="margin-top:12px;">لم أعثر على نتائج. جرّب صياغة أخرى أو أضف كلمة "بالعربية".</div>'
        cache.set(cache_key, panel, expire=60*5)
        return (panel, "") if return_plain else (panel, None)

    final_answer = " ".join(chunks)
    panel = (
        f'<div style="margin-top:18px;">'
        f'<h3>سؤالك:</h3><div class="card">{html.escape(q)}</div>'
        f'<h3 style="margin-top:12px;">الملخّص:</h3><div class="summary">{html.escape(final_answer)}</div>'
        f'<h3 style="margin-top:12px;">المصادر:</h3>'
        f'{"".join(source_cards)}'
        f'</div>'
    )
    cache.set(cache_key, panel, expire=60*60)
    return (panel, final_answer) if return_plain else (panel, None)

def make_summary_card(title, url, summ):
    return (
        f'<div class="card" style="margin-top:10px;"><strong>{html.escape(title)}</strong>'
        f'<div class="summary" style="margin-top:8px;">{html.escape(summ)}</div>'
        f'<div style="margin-top:8px;"><a target="_blank" href="{html.escape(url)}">فتح المصدر</a></div>'
        f'</div>'
    )

# ========= وضع الأسعار (بدون تغيير كبير) =========
PRICE_RE = re.compile(r"(?i)(US?\s*\$|USD|EUR|GBP|AED|SAR|EGP|QAR|KWD|OMR|د\.إ|ر\.س|ج\.م|د\.ك|ر\.ق|ر\.ع)\s*[\d\.,]+")
AR_NUM = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
def extract_price_from_html(html_text: str):
    if not html_text: return ""
    text = BeautifulSoup(html_text, "html.parser").get_text(separator=" ").translate(AR_NUM)
    m = PRICE_RE.search(text)
    return m.group(0).strip() if m else ""

async def handle_prices(q: str, return_plain=False):
    cache_key = "price:" + q
    cached = cache.get(cache_key)
    if cached and not return_plain:
        return cached, ""

    sites_filter = " OR ".join([f"site:{s}" for s in MARKET_SITES])
    results = ddg(f'{q} {sites_filter}', region="xa-ar", safesearch="Off", max_results=20) or []
    if not results:
        results = ddg(q + " " + sites_filter, region="wt-wt", safesearch="Off", max_results=20) or []

    cards, seen, lines = [], set(), []
    for r in sorted(results, key=lambda it: priority_key(it, "prices")):
        url = r.get("href") or r.get("link") or r.get("url")
        title = r.get("title") or ""
        snippet = r.get("body") or ""
        if not url or url in seen:
            continue
        seen.add(url)
        price = ""
        try:
            ckey = "purl:" + url
            html_page = cache.get(ckey)
            if html_page is None:
                html_page = fetch(url, timeout=7)
                if html_page and len(html_page) < 1_500_000:
                    cache.set(ckey, html_page, expire=60*60*6)
            price = extract_price_from_html(html_page or "")
        except Exception:
            price = ""
        title_ar = translate_to_ar(title)
        snippet_ar = translate_to_ar(snippet)
        cards.append(make_price_card(title_ar or title, url, price, snippet_ar))
        lines.append(f"- {title_ar or title} | {price or '—'} | {url}")
        if len(cards) >= 8:
            break

    if not cards:
        panel = '<div class="card" style="margin-top:12px;">لم أجد نتائج مناسبة. جرّب اسمًا أدق للمنتج (الموديل/الطراز).</div>'
        cache.set(cache_key, panel, expire=60*5)
        return (panel, "") if return_plain else (panel, None)

    answer_text = "نتائج أسعار:\n" + "\n".join(lines)
    panel = f'<div style="margin-top:18px;"><h3>بحث أسعار عن: {html.escape(q)}</h3>{"".join(cards)}</div>'
    cache.set(cache_key, panel, expire=60*30)
    return (panel, answer_text) if return_plain else (panel, None)

def make_price_card(title, url, price, snippet):
    price_html = f"<div><strong>السعر:</strong> {html.escape(price)}</div>" if price else "<div>السعر غير واضح – افتح المصدر للتحقق.</div>"
    sn = f'<div class="note" style="margin-top:6px;">{html.escape((snippet or "")[:180])}</div>' if snippet else ""
    return (f'<div class="card" style="margin-top:10px;"><strong>{html.escape(title)}</strong>'
            f'{price_html}'
            f'<div style="margin-top:8px;"><a target="_blank" href="{html.escape(url)}">فتح المصدر</a></div>'
            f'{sn}</div>')

# ========= وضع الصور =========
async def handle_images(q: str):
    key = "img:" + q
    cached = cache.get(key)
    if cached:
        return cached, ""
    items = []
    try:
        if DDGS:
            with DDGS() as dd:
                for it in dd.images(keywords=q, region="xa-ar", safesearch="Off", max_results=20):
                    items.append({"title": it.get("title") or "", "image": it.get("image"), "source": it.get("url")})
        else:
            results = ddg(q + " صور", region="xa-ar", safesearch="Off", max_results=20) or []
            for r in results:
                items.append({"title": r.get("title") or "", "image": None, "source": r.get("href") or r.get("url")})
    except Exception:
        items = []

    if not items:
        panel = '<div class="card" style="margin-top:12px;">لم أجد صورًا مناسبة. حاول تفاصيل أكثر أو كلمة "صور".</div>'
        cache.set(key, (panel, ""), expire=60*10)
        return panel, ""

    cards = []
    for it in items[:16]:
        img = it.get("image")
        src = it.get("source")
        title = translate_to_ar(it.get("title") or "")
        if img:
            cards.append(f'<div class="imgcard"><a href="{html.escape(src or img)}" target="_blank"><img src="{html.escape(img)}" alt=""/></a></div>')
        else:
            cards.append(f'<div class="card"><a href="{html.escape(src)}" target="_blank">{html.escape(title or "فتح المصدر")}</a></div>')
    panel = f'<div style="margin-top:18px;"><h3>نتائج صور عن: {html.escape(q)}</h3><div class="imggrid">{"".join(cards)}</div></div>'
    cache.set(key, (panel, ""), expire=60*20)
    return panel, ""

# ========= تصدير PDF =========
@app.get("/export_pdf")
def export_pdf(q: str, mode: str = "summary"):
    if mode == "prices":
        panel_html = cache.get("price:" + q)
        text_for_pdf = "—"
        if panel_html:
            soup = BeautifulSoup(panel_html, "html.parser")
            lines = []
            for c in soup.select(".card"):
                title = c.find("strong").get_text(" ", strip=True) if c.find("strong") else ""
                price_el = c.find(text=re.compile("السعر:"))
                price = price_el.parent.get_text(" ", strip=True).replace("السعر:","").strip() if price_el else "—"
                link = c.find("a")["href"] if c.find("a") else ""
                lines.append(f"- {title} | {price} | {link}")
            text_for_pdf = "نتائج أسعار:\n" + "\n".join(lines[:20])
        title = f"بحث أسعار: {q}"
    elif mode == "images":
        title = f"نتائج صور: {q}"
        text_for_pdf = "يرجى فتح الروابط من المتصفح لمعاينة الصور."
    else:
        panel_html = cache.get("sum:" + q)
        if not panel_html:
            text_for_pdf = "لا توجد بيانات."
        else:
            soup = BeautifulSoup(panel_html, "html.parser")
            summary_div = soup.find("div", {"class": "summary"})
            text_for_pdf = summary_div.get_text(" ", strip=True) if summary_div else "لا توجد بيانات."
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
        "Content-Disposition": f'attachment; filename="bassam_ai_{mode}.pdf"',
        "Content-Type": "application/pdf",
    }
    return Response(content=pdf_bytes, headers=headers)
