# main.py — Bassam الذكي: تلخيص + أسعار + صور + رياضيات + مشاعر + مطوّر (مجاني)

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import re, json
import requests
from html import escape

app = FastAPI(title="Bassam الذكي", version="3.0")

# ====== مجلدات ======
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ====== محاولات استيراد اختيارية ======
# ترجمة عربية تلقائية (مهم)
try:
    from deep_translator import GoogleTranslator
    def to_ar(text: str) -> str:
        if not text: return ""
        if re.search(r"[\u0600-\u06FF]", text):  # فيه حروف عربية؟
            return text
        try: return GoogleTranslator(source='auto', target='ar').translate(text)
        except Exception: return text
except Exception:
    def to_ar(text: str) -> str: return text

# بحث DuckDuckGo (اختياري لأسعار/صور)
try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None

# رياضيات قوي (اختياري)
try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations, implicit_multiplication_application
    )
    TRANS = (standard_transformations + (implicit_multiplication_application,))
    def _symp(s: str):
        return parse_expr(s, transformations=TRANS, evaluate=True)
except Exception:
    sp = None
    def _symp(s: str): raise RuntimeError("sympy غير مثبت")

# تحويل وحدات (اختياري)
try:
    from pint import UnitRegistry
    ureg = UnitRegistry()
except Exception:
    ureg = None

# مشاعر (اختياري)
try:
    from nrclex import NRCLex
except Exception:
    NRCLex = None

# ====== أدوات عامة ======
def wiki_summary_ar(query: str) -> str:
    """ملخص من ويكيبيديا بالعربي ثم الإنجليزي مع ترجمة"""
    q = (query or "").strip()
    if not q: return "اكتب سؤالك لكي أبحث عن خلاصة مناسبة."
    # عربي أولاً
    try:
        r = requests.get(
            "https://ar.wikipedia.org/w/api.php",
            params={"action":"opensearch","search":q,"limit":1,"namespace":0,"format":"json"},
            timeout=10,
        )
        data = r.json()
        if data and len(data) >= 2 and data[1]:
            title = data[1][0]
            s = requests.get(
                f"https://ar.wikipedia.org/api/rest_v1/page/summary/{title}",
                timeout=10
            ).json()
            if s.get("extract"): return s["extract"]
    except Exception:
        pass
    # إنجليزي ثم ترجمة
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action":"opensearch","search":q,"limit":1,"namespace":0,"format":"json"},
            timeout=10,
        )
        data = r.json()
        if data and len(data) >= 2 and data[1]:
            title = data[1][0]
            s = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
                timeout=10
            ).json()
            if s.get("extract"): return to_ar(s["extract"])
    except Exception:
        pass
    return "لم أعثر على نتائج. جرِّب صياغة أخرى أو أضِف كلمة «بالعربية»."

# ====== واجهة رئيسية ======
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "q": "", "mode": "summary", "answer_text": "", "result_panel": ""}
    )

# ====== تلخيص/أسعار/صور عبر نفس النموذج ======
@app.post("/", response_class=HTMLResponse)
async def main_post(request: Request, question: str = Form(...), mode: str = Form("summary")):
    q = (question or "").strip()
    answer_text, panel = "", ""

    if mode == "summary":
        answer_text = to_ar(wiki_summary_ar(q))

    elif mode == "prices":
        panel = await prices_panel(q)

    elif mode == "images":
        panel = await images_panel(q)

    else:
        panel = '<div class="card">اختر تبويبًا صحيحًا.</div>'

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "q": q, "mode": mode, "answer_text": answer_text, "result_panel": panel}
    )

# ====== أسعار ======
async def prices_panel(q: str) -> str:
    """إن توفّر DDGS نجلب نتائج، وإلا نعطي روابط بحث جاهزة بالمواقع"""
    q_disp = escape(q)
    if DDGS:
        cards = []
        site_filters = [
            "site:amazon.sa", "site:amazon.ae", "site:noon.com",
            "site:jumia.com", "site:temu.com", "site:aliexpress.com",
        ]
        query = f"{q} " + " OR ".join(site_filters)
        try:
            with DDGS() as dd:
                results = list(dd.text(query, region="xa-ar", safesearch="off", max_results=18))
            seen = set()
            for r in results:
                url = r.get("href") or r.get("url")
                title = r.get("title") or url
                if not url or url in seen: continue
                seen.add(url)
                cards.append(f"""
                  <div class="card" style="margin-top:10px">
                    <strong>{escape(to_ar(title))}</strong>
                    <div style="margin-top:6px"><a target="_blank" href="{escape(url)}">فتح المصدر</a></div>
                  </div>
                """)
                if len(cards) >= 8: break
            if cards:
                return f'<div style="margin-top:12px"><h3>بحث أسعار عن: {q_disp}</h3>{"".join(cards)}</div>'
        except Exception:
            pass
    # Fallback: روابط بحث جاهزة
    links = [
        ("Amazon SA", f"https://www.amazon.sa/s?k={q}"),
        ("Amazon AE", f"https://www.amazon.ae/s?k={q}"),
        ("Noon",     f"https://www.noon.com/uae-ar/search?q={q}"),
        ("Jumia",    f"https://www.jumia.com.eg/catalog/?q={q}"),
        ("Temu",     f"https://www.temu.com/search_result.html?search_key={q}"),
        ("AliExpress", f"https://www.aliexpress.com/wholesale?SearchText={q}"),
    ]
    items = "".join([f'<li><a target="_blank" href="{escape(u)}">{escape(name)}</a></li>' for name,u in links])
    return f"""
      <div class="card" style="margin-top:12px">
        <strong>روابط سريعة للمتاجر (افتح للتحقق من السعر):</strong>
        <ul style="margin-top:6px">{items}</ul>
      </div>
    """

# ====== صور ======
async def images_panel(q: str) -> str:
    q_disp = escape(q)
    cards = []
    if DDGS:
        try:
            with DDGS() as dd:
                for it in dd.images(keywords=q, region="xa-ar", safesearch="off", max_results=16):
                    img = it.get("image")
                    src = it.get("url") or img
                    title = to_ar(it.get("title") or "")
                    if img:
                        cards.append(f'<div class="imgcard"><a href="{escape(src)}" target="_blank"><img src="{escape(img)}" alt=""/></a></div>')
            if cards:
                return f'<div style="margin-top:12px"><h3>نتائج صور عن: {q_disp}</h3><div class="imggrid">{"".join(cards)}</div></div>'
        except Exception:
            pass
    # Fallback: رابط بحث صور
    return f"""
      <div class="card" style="margin-top:12px">
        لم أستطع جلب الصور تلقائيًا. افتح بحث الصور:
        <div style="margin-top:8px">
          <a target="_blank" href="https://duckduckgo.com/?q={q_disp}&iax=images&ia=images">بحث صور DuckDuckGo</a>
        </div>
      </div>
    """

# ====== وضع المطوّر (أكواد) ======
DEV_SNIPPETS = [
    {
        "tags": ["python","requests","api","http"],
        "title": "بايثون: طلب GET بسيط مع معالجة أخطاء",
        "body": """# requests: GET + timeout + فحص الحالة
import requests

def fetch_json(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()  # أو r.text
    except requests.exceptions.RequestException as e:
        print("HTTP error:", e)
        return None

data = fetch_json("https://api.github.com/repos/psf/requests")
print(data and data.get("stargazers_count"))
"""
    },
    {
        "tags": ["python","beautifulsoup","scraping","html"],
        "title": "بايثون: سحب عنوان الصفحة وروابطها بـ BeautifulSoup",
        "body": """import requests
from bs4 import BeautifulSoup

url = "https://example.com"
html = requests.get(url, timeout=10).text
soup = BeautifulSoup(html, "html.parser")

print("العنوان:", soup.title and soup.title.get_text())
for a in soup.select("a[href]")[:20]:
    print(a.get_text(strip=True), "->", a["href"])
"""
    },
    {
        "tags": ["python","fastapi","web"],
        "title": "هيكل FastAPI سريع مع نقطة صحة",
        "body": """from fastapi import FastAPI

app = FastAPI()

@app.get("/healthz")
def healthz():
    return {"ok": True}

# شغل محلياً:
# uvicorn main:app --reload --port 8000
"""
    },
    {
        "tags": ["python","asyncio"],
        "title": "بايثون: تنفيذ مهام غير متزامنة asyncio.gather",
        "body": """import asyncio
import httpx

async def fetch(client, url):
    r = await client.get(url, timeout=10)
    return url, r.status_code

async def main():
    urls = ["https://example.com", "https://httpbin.org/get"]
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[fetch(client, u) for u in urls])
    for u, s in results:
        print(u, "=>", s)

asyncio.run(main())
"""
    },
    {
        "tags": ["python","sqlite","sql","security"],
        "title": "SQLite: استعلام آمن بالـ parameters (تجنّب الحقن)",
        "body": """import sqlite3

con = sqlite3.connect("app.db")
cur = con.cursor()

cur.execute("CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT)")
con.commit()

name = "ali'); DROP TABLE users; --"
cur.execute("INSERT INTO users(name) VALUES (?)", (name,))
con.commit()

cur.execute("SELECT id, name FROM users WHERE name LIKE ?", ("%ali%",))
print(cur.fetchall())
con.close()
"""
    },
    {
        "tags": ["network","tcp","socket","python"],
        "title": "TCP: خادم/عميل بسيطان في بايثون",
        "body": """# server.py
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("0.0.0.0", 9000))
    s.listen(1)
    print("TCP server on :9000")
    conn, addr = s.accept()
    with conn:
        print("client:", addr)
        conn.sendall(b"hello from server")
        data = conn.recv(1024)
        print("client says:", data)

# client.py
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as c:
    c.connect(("127.0.0.1", 9000))
    print(c.recv(1024))
    c.sendall(b"hi!")
"""
    },
    {
        "tags": ["network","udp","socket","python"],
        "title": "UDP: مرسل/مستقبل بسيطان",
        "body": """# recv.py
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("0.0.0.0", 9001))
print("UDP listen :9001")
print("got:", s.recvfrom(2048))

# send.py
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.sendto(b"ping", ("127.0.0.1", 9001))
"""
    },
    {
        "tags": ["network","linux","cli"],
        "title": "أوامر تشخيص شبكة سريعة (لينكس)",
        "body": """# Ping, Trace, DNS, واجهات
ping -c 4 8.8.8.8
traceroute 8.8.8.8
dig example.com +short
ip a
ss -lntp
"""
    },
    {
        "tags": ["http","curl","cli"],
        "title": "cURL: أمثلة شائعة",
        "body": """# GET
curl -i https://httpbin.org/get
# POST form
curl -i -X POST -d "name=bassam&x=1" https://httpbin.org/post
# JSON
curl -i -X POST -H "Content-Type: application/json" -d '{"q":"hello"}' https://httpbin.org/post
"""
    },
]

def _score(q: str, tags: list[str], title: str, body: str) -> float:
    ql = q.lower()
    score = 0.0
    for t in tags:
        if t in ql: score += 2.0
    for w in re.findall(r"[a-zA-Z\u0621-\u064A]+", ql):
        if w in title.lower(): score += 1.0
        if w in body.lower():  score += 0.2
    return score

def smart_dev_answer(question: str) -> str:
    q = (question or "").strip()
    if not q:
        return '<div class="card">اكتب سؤالك التقني وسيتم اقتراح أكواد جاهزة + شرح موجز.</div>'
    intro = "🔧 إجابة مطوّر مختصرة وواضحة — أقرب وصفات/أكواد لسؤالك:"
    ranked = sorted(DEV_SNIPPETS, key=lambda s: _score(q, s["tags"], s["title"], s["body"]), reverse=True)[:3]
    if not ranked or _score(q, ranked[0]["tags"], ranked[0]["title"], ranked[0]["body"]) <= 0:
        return '<div class="card">لم أعثر على كود مناسب مباشرة. جرّب كلمات تقنية أدق.</div>'
    blocks = []
    for s in ranked:
        blocks.append(f"""
        <div class="card" style="margin-top:10px">
          <strong>{escape(s["title"])}</strong>
          <pre>{escape(s["body"])}</pre>
          <div class="fb">
            <button class="btn-mini" onclick="navigator.clipboard.writeText(this.parentNode.previousElementSibling.innerText)">نسخ الكود</button>
          </div>
        </div>
        """)
    return f"<div class='card'>{intro}</div>" + "".join(blocks)

@app.post("/dev", response_class=HTMLResponse)
async def dev_mode(request: Request, question: str = Form(...), mode: str = Form("dev")):
    panel = smart_dev_answer(question)
    return templates.TemplateResponse("index.html", {"request": request, "q": question, "mode": "dev", "answer_text": "", "result_panel": panel})

# ====== رياضيات ======
def _is_system(text: str) -> bool:
    return ("," in text and "=" in text) or ("\n" in text and "=" in text)

def _detect_task(q: str) -> str:
    ql = q.strip().lower()
    if any(k in ql for k in ["بسّط","بسط","simplify"]): return "simplify"
    if any(k in ql for k in ["فك","وسّع","expand"]):    return "expand"
    if any(k in ql for k in ["اشتق","deriv"]):          return "diff"
    if any(k in ql for k in ["كامل","تكامل","integr"]): return "integrate"
    if any(k in ql for k in ["حد","limit"]):            return "limit"
    if any(k in ql for k in ["حل","solve"]):            return "solve"
    return "solve" if "=" in q else "simplify"

def _latex(expr) -> str:
    try: return sp.latex(expr)
    except Exception: return str(expr)

def math_steps_single(expr_text: str, task: str):
    x,y,z = sp.symbols('x y z')
    steps = []
    expr = _symp(expr_text)
    if task == "simplify":
        steps.append("تبسيط التعبير.")
        res = sp.simplify(expr)
    elif task == "expand":
        steps.append("توسيع/تفكيك الحدود.")
        res = sp.expand(expr)
    elif task == "diff":
        var = list(expr.free_symbols)[0] if expr.free_symbols else x
        steps.append(f"اشتقاق بالنسبة إلى {var}.")
        res = sp.diff(expr, var)
    elif task == "integrate":
        var = list(expr.free_symbols)[0] if expr.free_symbols else x
        steps.append(f"تكامل غير محدد بالنسبة إلى {var}.")
        res = sp.integrate(expr, var)
    elif task == "limit":
        var = list(expr.free_symbols)[0] if expr.free_symbols else x
        steps.append(f"حساب النهاية عندما {var} → 0 (يمكن تحديدها: limit(expr, x, a)).")
        res = sp.limit(expr, var, 0)
    elif task == "solve":
        if isinstance(expr, sp.Equality):
            steps.append("حل معادلة.")
            res = sp.solve(sp.Eq(expr.lhs, expr.rhs), list(expr.free_symbols))
        else:
            if expr.free_symbols:
                var = list(expr.free_symbols)[0]
                steps.append("حل جذور متعددة حدود بالنسبة للمتغير الرئيسي.")
                res = sp.solve(sp.Eq(expr, 0), var)
            else:
                steps.append("لا يوجد متغيرات للحل.")
                res = expr
    else:
        steps.append("عملية غير معروفة، تم استخدام تبسيط.")
        res = sp.simplify(expr)
    return sp.spretty(res), steps

def math_steps_system(text: str):
    parts = [t.strip() for t in re.split(r'[,\n]+', text) if t.strip()]
    eqs = []; syms = set()
    for p in parts:
        if "=" in p:
            L, R = p.split("=",1)
            L = _symp(L); R = _symp(R)
            eqs.append(sp.Eq(L, R))
            syms |= L.free_symbols | R.free_symbols
    if not eqs: return "لم يتم التعرف على معادلات صالحة.", []
    vars_sorted = sorted(list(syms), key=lambda s: s.name)
    sol = sp.solve(eqs, vars_sorted, dict=True)
    steps = [f"حل نظام مكوّن من {len(eqs)} معادلات و{len(vars_sorted)} متغير(ات)."]
    return sp.spretty(sol), steps

def convert_units(expr: str):
    if not ureg: return ""
    t = expr.lower().replace("الى","to").replace("إلى","to")
    if " to " not in t: return ""
    val, to_unit = t.split(" to ", 1)
    q = ureg(val.strip())
    res = q.to(to_unit.strip())
    return f"{res.magnitude:g} {res.units}"

@app.post("/math", response_class=HTMLResponse)
async def math_api(request: Request, expression: str = Form(...)):
    q = (expression or "").strip()
    # تحويل وحدات؟
    try:
        cu = convert_units(q)
        if cu:
            panel = f'<div class="card" style="margin-top:12px"><strong>تحويل وحدات:</strong><div class="summary">{escape(cu)}</div></div>'
            return templates.TemplateResponse("index.html", {"request": request, "q": q, "mode": "math", "answer_text": "", "result_panel": panel})
    except Exception:
        pass
    if not sp:
        panel = '<div class="card" style="margin-top:12px">الرياضيات المتقدمة تتطلب تثبيت sympy. أضفها إلى requirements.txt ثم انشر.</div>'
        return templates.TemplateResponse("index.html", {"request": request, "q": q, "mode": "math", "answer_text": "", "result_panel": panel})
    try:
        if _is_system(q):
            out, steps = math_steps_system(q)
        else:
            task = _detect_task(q)
            out, steps = math_steps_single(q, task)
        html_steps = "".join([f"<li>{escape(s)}</li>" for s in steps])
        panel = f"""
        <div class="card" style="margin-top:12px">
          <strong>النتيجة:</strong>
          <pre>{escape(out)}</pre>
          <div style="margin-top:8px"><strong>الخطوات:</strong><ul>{html_steps}</ul></div>
        </div>
        """
        return templates.TemplateResponse("index.html", {"request": request, "q": q, "mode": "math", "answer_text": "", "result_panel": panel})
    except Exception as e:
        panel = f'<div class="card" style="margin-top:12px">تعذّر فهم المسألة: {escape(str(e))}</div>'
        return templates.TemplateResponse("index.html", {"request": request, "q": q, "mode": "math", "answer_text": "", "result_panel": panel})

# ====== مشاعر ======
@app.post("/emotion", response_class=JSONResponse)
async def emotion_api(text: str = Form(...)):
    txt = (text or "").strip()
    if not txt:
        return {"emotion": "محايد", "scores": {}}
    # NRCLex إنجليزي، نترجم أولاً
    eng = txt
    if to_ar("hello") != "hello":  # يعني الترجمة متوفرة
        try:
            eng = GoogleTranslator(source="auto", target="en").translate(txt)
        except Exception:
            pass
    if not NRCLex:
        return {"emotion": "تتطلب nrclex (اختياري)", "scores": {}}
    emo = NRCLex(eng)
    scores = emo.raw_emotion_scores or {}
    if not scores:
        return {"emotion": "محايد", "scores": {}}
    top = max(scores.items(), key=lambda kv: kv[1])[0]
    ar_map = {
        "joy": "فرح", "sadness": "حزن", "anger": "غضب", "fear": "خوف",
        "surprise": "دهشة", "anticipation": "توقّع", "trust": "ثقة",
        "disgust": "اشمئزاز"
    }
    return {"emotion": ar_map.get(top, top), "scores": scores}

# ====== صحة ======
@app.get("/healthz")
def healthz(): return {"ok": True}
@app.get("/health")
def health(): return {"ok": True}
