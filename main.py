# main.py — بحث عربي مجاني + تلخيص ذكي + أسعار المتاجر + صور + تقييم + PDF + نسخ + وضع ليلي
from fastapi import FastAPI, Form, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from duckduckgo_search import ddg
try:
    from duckduckgo_search import DDGS  # للصور (إن توفرت)
except Exception:
    DDGS = None

from readability import Document
from bs4 import BeautifulSoup
from diskcache import Cache
from urllib.parse import urlparse, urlencode
from fpdf import FPDF
import requests, re, html, time

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
cache = Cache(".cache")

# ---------------- إعدادات ----------------
PREFERRED_AR_DOMAINS = {
    "ar.wikipedia.org", "ar.m.wikipedia.org",
    "mawdoo3.com", "almrsal.com", "sasapost.com",
    "arabic.cnn.com", "bbcarabic.com", "aljazeera.net",
    "ar.wikihow.com", "moe.gov.sa", "yemen.gov.ye", "moh.gov.sa"
}

MARKET_SITES = [
    "alibaba.com", "1688.com", "aliexpress.com",
    "amazon.com", "amazon.ae", "amazon.sa", "amazon.eg",
    "noon.com", "jumia.com", "jumia.com.eg",
    "ebay.com", "made-in-china.com", "temu.com", "souq.com"
]

HDRS = {
    "User-Agent": "Mozilla/5.0 (compatible; BassamBot/1.3)",
    "Accept-Language": "ar,en;q=0.8"
}

# -------- أدوات اللغة والملخص --------
AR_RE = re.compile(r"[اأإآء-ي]")
def is_arabic(text: str, min_ar_chars: int = 12) -> bool:
    return len(AR_RE.findall(text or "")) >= min_ar_chars

STOP = set("""من في على إلى عن أن إن بأن كان تكون يكون التي الذي الذين هذا هذه ذلك هناك ثم حيث كما اذا إذا أو و يا ما مع قد لم لن بين لدى لدى، عند بعد قبل دون غير حتى كل أي كيف لماذا متى هل الى ال""".split())

def tokenize(s: str):
    s = re.sub(r"[^\w\s\u0600-\u06FF]+", " ", s.lower())
    toks = [t for t in s.split() if t and t not in STOP]
    return toks

def score_sentences(text: str, query: str):
    sentences = re.split(r'(?<=[\.\!\?\؟])\s+|\n+', text or "")
    q_terms = set(tokenize(query))
    scored = []
    for s in sentences:
        s2 = s.strip()
        if len(s2) < 25 or not is_arabic(s2, 8):
            continue
        terms = set(tokenize(s2))
        inter = q_terms & terms
        score = len(inter) + (len(s2) >= 80)
        if score > 0:
            scored.append((score, s2))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [s for _, s in scored[:8]]

def summarize_from_text(text: str, query: str, max_sentences=3):
    sents = score_sentences(text, query)
    return " ".join(sents[:max_sentences]) if sents else ""

def domain_of(url: str):
    try:
        return urlparse(url).netloc.lower()
    except:
        return url

# -------- نقاط النطاقات (تعلم ذاتي بسيط) --------
def get_scores():
    return cache.get("domain_scores", {}) or {}

def save_scores(scores):
    cache.set("domain_scores", scores, expire=0)

def bump_score(domain: str, delta: int):
    if not domain:
        return
    scores = get_scores()
    scores[domain] = scores.get(domain, 0) + delta
    save_scores(scores)

# -------- جلب الصفحات --------
def fetch(url: str, timeout=6):
    r = requests.get(url, headers=HDRS, timeout=timeout)
    r.raise_for_status()
    return r.text

def fetch_and_extract(url: str, timeout=6):
    try:
        html_text = fetch(url, timeout=timeout)
        doc = Document(html_text)
        content_html = doc.summary()
        soup = BeautifulSoup(content_html, "html.parser")
        text = soup.get_text(separator="\n")
        return html.unescape(text), html_text
    except Exception:
        return "", ""

# -------- استخراج الأسعار --------
PRICE_RE = re.compile(r"(?i)(US?\s*\$|USD|EUR|GBP|AED|SAR|EGP|QAR|KWD|OMR|د\.إ|ر\.س|ج\.م|د\.ك|ر\.ق|ر\.ع)\s*[\d\.,]+")
AR_NUM = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def extract_price_from_html(html_text: str):
    if not html_text:
        return ""
    text = BeautifulSoup(html_text, "html.parser").get_text(separator=" ")
    text = text.translate(AR_NUM)
    m = PRICE_RE.search(text)
    return m.group(0).strip() if m else ""

def try_get_price(url: str):
    try:
        h = fetch(url, timeout=6)
        price = extract_price_from_html(h)
        if price:
            return price
        soup = BeautifulSoup(h, "html.parser")
        meta_price = soup.find(attrs={"itemprop": "price"}) or soup.find("meta", {"property":"product:price:amount"})
        if meta_price:
            val = meta_price.get("content") or meta_price.text
            if val and re.search(r"[\d\.,]", val):
                return val.strip()
        time.sleep(0.2)
        h2 = fetch(url, timeout=6)
        return extract_price_from_html(h2)
    except Exception:
        return ""

# ---------------- أولوية ذكية ----------------
def priority_key(item, mode="summary"):
    scores = get_scores()
    d = domain_of(item.get("href") or item.get("link") or item.get("url") or "")
    base = 2
    if d in PREFERRED_AR_DOMAINS: base -= 1
    if mode == "prices" and any(d.endswith(ms) or d==ms for ms in MARKET_SITES): base -= 0.5
    base -= 0.05 * scores.get(d, 0)
    return base

# ---------------- المسارات ----------------
@app.get("/", response_class=HTMLResponse)
async def page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result_panel": "", "q": "", "mode": "summary", "answer_text": ""})

@app.post("/", response_class=HTMLResponse)
async def run(request: Request, question: str = Form(...), mode: str = Form("summary")):
    q = (question or "").strip()
    if not q:
        return templates.TemplateResponse("index.html", {"request": request, "result_panel": "", "q": "", "mode": mode, "answer_text": ""})

    if mode == "prices":
        panel, answer_text = await handle_prices(q, return_plain=True)
    elif mode == "images":
        panel, answer_text = await handle_images(q)
    else:
        panel, answer_text = await handle_summary(q, return_plain=True)

    return templates.TemplateResponse("index.html", {"request": request, "result_panel": panel, "q": q, "mode": mode, "answer_text": (answer_text or "")})

@app.post("/feedback")
async def feedback(domain: str = Form(...), delta: int = Form(...)):
    bump_score(domain, int(delta))
    return JSONResponse({"ok": True, "domain": domain, "score": get_scores().get(domain, 0)})

# -------- وضع: بحث & تلخيص عربي --------
async def handle_summary(q: str, return_plain=False):
    cache_key = "sum:" + q
    cached = cache.get(cache_key)
    if cached and not return_plain:
        return cached, ""

    results = ddg(q + " بالعربية", region="xa-ar", safesearch="Moderate", max_results=12) or []
    if not results:
        results = ddg(q + " بالعربية", region="sa-ar", safesearch="Moderate", max_results=12) or []
    if not results:
        results = ddg(q, region="xa-ar", safesearch="Moderate", max_results=12) or []

    source_cards, combined_chunks = [], []
    for r in sorted(results, key=lambda it: priority_key(it, "summary")):
        href = r.get("href") or r.get("link") or r.get("url")
        title = r.get("title") or ""
        if not href:
            continue
        d = domain_of(href)

        ckey = "url:" + href
        val = cache.get(ckey)
        if val is None:
            txt, raw = fetch_and_extract(href)
            if txt and len(txt) > 200:
                cache.set(ckey, (txt, raw), expire=60*60*24)
            val = (txt, raw)
        page_text, _ = val

        if not page_text or not is_arabic(page_text):
            continue

        summ = summarize_from_text(page_text, q, max_sentences=3)
        if not summ:
            continue

        combined_chunks.append(summ)
        source_cards.append(make_summary_card(title, href, summ, d))
        if len(source_cards) >= 3:
            break

    # Fallback إنجليزي بسيط
    if not combined_chunks:
        results_en = ddg(q, region="wt-wt", safesearch="Moderate", max_results=8) or []
        for r in results_en:
            href = r.get("href") or r.get("link") or r.get("url")
            title = r.get("title") or ""
            if not href:
                continue
            txt, _ = fetch_and_extract(href)
            if not txt:
                continue
            summ = " ".join((txt.strip().split())[:50])
            if not summ:
                continue
            combined_chunks.append(summ)
            source_cards.append(make_summary_card(title or href, href, summ, domain_of(href)))
            if len(source_cards) >= 3:
                break

    if not combined_chunks:
        panel = '<div class="card" style="margin-top:12px;">لم أعثر على محتوى كافٍ. غيّر صياغة السؤال أو أضف كلمة "بالعربية".</div>'
        cache.set(cache_key, panel, expire=60*5)
        return (panel, "") if return_plain else (panel, None)

    final_answer = " ".join(combined_chunks)
    panel = (
        f'<div style="margin-top:18px;">'
        f'<h3>سؤالك:</h3><div class="card">{html.escape(q)}</div>'
        f'<h3 style="margin-top:12px;">الملخّص (من المصادر):</h3><div class="summary">{html.escape(final_answer)}</div>'
        f'<h3 style="margin-top:12px;">المصادر:</h3>'
        f'{"".join(source_cards)}'
        f'</div>'
    )
    cache.set(cache_key, panel, expire=60*60)
    return (panel, final_answer) if return_plain else (panel, None)

def make_summary_card(title, url, summ, domain):
    return (
        f'<div class="card" style="margin-top:10px;"><strong>{html.escape(title)}</strong>'
        f'<div class="summary" style="margin-top:8px;">{html.escape(summ)}</div>'
        f'<div style="margin-top:8px;"><a target="_blank" href="{html.escape(url)}">فتح المصدر</a></div>'
        f'{feedback_buttons(domain)}'
        f'</div>'
    )

# -------- وضع: بحث أسعار المتاجر --------
async def handle_prices(q: str, return_plain=False):
    cache_key = "price:" + q
    cached = cache.get(cache_key)
    if cached and not return_plain:
        return cached, ""

    sites_filter = " OR ".join([f"site:{s}" for s in MARKET_SITES])
    results = ddg(f'{q} {sites_filter}', region="xa-ar", safesearch="Off", max_results=20) or []
    if not results:
        results = ddg(q + " " + sites_filter, region="wt-wt", safesearch="Off", max_results=20) or []

    cards, seen = [], set()
    lines_for_pdf = []
    for r in sorted(results, key=lambda it: priority_key(it, "prices")):
        url = r.get("href") or r.get("link") or r.get("url")
        title = r.get("title") or ""
        snippet = r.get("body") or ""
        if not url or url in seen:
            continue
        seen.add(url)
        d = domain_of(url)

        price = ""
        try:
            ckey = "purl:" + url
            html_page = cache.get(ckey)
            if html_page is None:
                html_page = fetch(url, timeout=6)
                if html_page and len(html_page) < 1_500_000:
                    cache.set(ckey, html_page, expire=60*60*6)
            price = extract_price_from_html(html_page or "")
            if not price and d.endswith("aliexpress.com"):
                soup = BeautifulSoup(html_page or "", "html.parser")
                meta_price = soup.find(attrs={"itemprop": "price"})
                if meta_price:
                    price = (meta_price.get("content") or meta_price.text or "").strip()
        except Exception:
            price = ""

        cards.append(make_price_card(title, url, price, snippet, d))
        lines_for_pdf.append(f"- {title} | {price or '—'} | {url}")
        if len(cards) >= 8:
            break

    if not cards:
        panel = '<div class="card" style="margin-top:12px;">لم أجد نتائج مناسبة في المتاجر. جرّب اسمًا أدق للمنتج (الموديل/الطراز) أو أضف site:aliexpress.com.</div>'
        cache.set(cache_key, panel, expire=60*5)
        return (panel, "") if return_plain else (panel, None)

    answer_text = "نتائج أسعار:\n" + "\n".join(lines_for_pdf)
    panel = f'<div style="margin-top:18px;"><h3>بحث أسعار عن: {html.escape(q)}</h3>{"".join(cards)}</div>'
    cache.set(cache_key, panel, expire=60*30)
    return (panel, answer_text) if return_plain else (panel, None)

def make_price_card(title, url, price, snippet, domain):
    price_html = f"<div><strong>السعر:</strong> {html.escape(price)}</div>" if price else "<div>السعر غير واضح – افتح المصدر للتحقق.</div>"
    sn = f'<div class="note" style="margin-top:6px;">{html.escape((snippet or "")[:180])}</div>' if snippet else ""
    return (
        f'<div class="card" style="margin-top:10px;"><strong>{html.escape(title)}</strong>'
        f'{price_html}'
        f'<div style="margin-top:8px;"><a target="_blank" href="{html.escape(url)}">فتح المصدر</a></div>'
        f'{sn}'
        f'{feedback_buttons(domain)}'
        f'</div>'
    )

# -------- وضع: بحث الصور --------
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
        title = it.get("title") or ""
        if img:
            cards.append(f'<div class="imgcard"><a href="{html.escape(src or img)}" target="_blank"><img src="{html.escape(img)}" alt=""/></a></div>')
        else:
            cards.append(f'<div class="card"><a href="{html.escape(src)}" target="_blank">{html.escape(title or "فتح المصدر")}</a></div>')

    panel = f'<div style="margin-top:18px;"><h3>نتائج صور عن: {html.escape(q)}</h3><div class="imggrid">{"".join(cards)}</div></div>'
    cache.set(key, (panel, ""), expire=60*20)
    return panel, ""

# -------- تصدير PDF --------
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

def feedback_buttons(domain: str):
    d = html.escape(domain or "")
    return f'''
      <div class="fb">
        <button class="btn-mini" onclick="sendFeedback('{d}', 1)">👍 مفيد</button>
        <button class="btn-mini" onclick="sendFeedback('{d}', -1)">👎 غير دقيق</button>
      </div>
    '''

@app.get("/health")
def health():
    return {"ok": True}
