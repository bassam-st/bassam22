# main.py — Bassam الذكي (بحث عربي + تلخيص + أسعار + صور + PDF) مع إجبار العربية بالترجمة
from fastapi import FastAPI, Form, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from html import escape
from bs4 import BeautifulSoup
from readability import Document
from diskcache import Cache
from urllib.parse import urlparse
from fpdf import FPDF
from deep_translator import GoogleTranslator

import requests, re, html, time

# ---- بحث DuckDuckGo
from duckduckgo_search import ddg

# تطبيق FastAPI
app = FastAPI(title="Bassam الذكي", version="1.3")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ربط static + templates + كاش
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
cache = Cache(".cache")

# صحة
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# ---------------- إعدادات ----------------
PREFERRED_AR_DOMAINS = {
    "ar.wikipedia.org", "ar.m.wikipedia.org", "mawdoo3.com",
    "almrsal.com", "sasapost.com", "arabic.cnn.com", "bbcarabic.com",
    "aljazeera.net", "ar.wikihow.com"
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

def to_ar(text: str) -> str:
    """ترجمة تلقائية إلى العربية إن لم يكن النص عربياً."""
    try:
        if not text:
            return ""
        if is_arabic(text, 6):
            return text
        return GoogleTranslator(source="auto", target="ar").translate(text)
    except Exception:
        return text or ""

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
        if len(s2) < 25:
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
    joined = " ".join(sents[:max_sentences]) if sents else ""
    return to_ar(joined)  # إجبار العربية

def domain_of(url: str):
    try:
        return urlparse(url).netloc.lower()
    except:
        return url

# -------- تعلم ذاتي بسيط للنطاقات --------
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
PRICE_RE = re.compile(r"(?i)(US?\s*\$|USD|EUR|GBP|AED|SAR|EGP|QAR|KWD|OMR|د\.إ|ر\.س|
