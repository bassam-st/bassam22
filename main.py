# -*- coding: utf-8 -*-
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from duckduckgo_search import DDGS
from sympy import symbols, sympify, simplify, diff, integrate, sqrt, sin, cos, tan
import sympy as sp
import re

app = FastAPI(title="Bassam الذكي — مصغّر")

# المجلدات
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -------- أدوات مساعدة --------

def normalize_math(expr: str) -> str:
    """تطبيع/تنظيف تعبير رياضي ليقبله sympy."""
    t = (expr or "").strip()

    # احذف "y=" أو "f(x)=" أو أي متغير مفرد يساوي
    t = re.sub(r'^\s*[yf]\s*\(\s*x\s*\)\s*=\s*', '', t, flags=re.I)
    t = re.sub(r'^\s*[a-zA-Z]\s*=\s*', '', t)

    # إذا كان المستخدم كتب بالعربية "مشتق: ..." أو "تكامل: ..." خذ ما بعد النقطتين
    m = re.search(r'[,:؛]\s*(.+)$', t)
    t = m.group(1) if m else t

    # استبدالات LaTeX الشائعة
    t = (t.replace('\\cdot', '*')
           .replace('\\sin', 'sin').replace('\\cos', 'cos').replace('\\tan', 'tan')
           .replace('\\sqrt', 'sqrt')
           .replace('^', '**'))

    # أرقام عربية إلى إنجليزية (كـ احتياط)
    arabic_digits = '٠١٢٣٤٥٦٧٨٩'
    for i, d in enumerate(arabic_digits):
        t = t.replace(d, str(i))

    # مسافات زائدة
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def detect_math_task(q: str) -> str:
    """استنتاج نوع المهمة من النص العربي: مشتق/تكامل/تبسيط/تقييم."""
    text = q.lower()
    if any(w in text for w in ['مشتق', 'اشتق', 'اشتقاق', 'derivative', 'diff']):
        return 'diff'
    if any(w in text for w in ['تكامل', 'integral', 'integrate']):
        return 'int'
    if any(w in text for w in ['بسّط', 'تبسيط', 'simplify']):
        return 'simp'
    # إن لم يذكر نوع المهمة نحاول التبسيط كافتراضي
    return 'auto'


def solve_math(q: str) -> str:
    """حل رياضيات أساسي (مشتق/تكامل/تبسيط/تقييم) وإرجاع HTML عربي بسيط."""
    try:
        task = detect_math_task(q)
        expr_txt = normalize_math(q)

        # متغيرات شائعة
        x, y, t = symbols('x y t')

        expr = sympify(expr_txt, dict(sin=sin, cos=cos, tan=tan, sqrt=sqrt))

        if task == 'diff':
            res = diff(expr, x)
            return f"مشتق التعبير بالنسبة إلى x:<br><b>{sp.latex(res)}</b>"
        elif task == 'int':
            res = integrate(expr, x)
            return f"تكامل غير محدد بالنسبة إلى x:<br><b>{sp.latex(res)}</b> + C"
        elif task == 'simp':
            res = simplify(expr)
            return f"تبسيط التعبير:<br><b>{sp.latex(res)}</b>"
        else:
            # محاولة تقييم أو تبسيط
            res = simplify(expr)
            return f"نتيجة التبسيط/التقييم:<br><b>{sp.latex(res)}</b>"

    except Exception as e:
        return ( "تعذّر فهم التعبير الرياضي. جرّب أمثلة مثل:<br>"
                 "<code>مشتق: x**3 + 2*sin(x)</code> أو <code>تكامل: cos(x)</code> أو <code>تبسيط: (x^2-1)/(x-1)</code><br>"
                 f"<small>الخطأ: {str(e)}</small>" )


def web_search_ar(query: str, maxn: int = 5) -> str:
    """بحث عربي سريع مع تلخيص مختصر كرابط + نبذة (HTML)."""
    items = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=maxn):
                # r يحوي عادة: title, href, body
                title = r.get("title") or r.get("source") or "نتيجة"
                href  = r.get("href") or r.get("link") or "#"
                body  = r.get("body") or r.get("snippet") or ""
                items.append((title, href, body))

        if not items:
            return "لم أعثر على نتائج. جرّب صياغة أخرى."

        html = ["<ol style='margin:0 0 0 18px;padding:0'>"]
        for (title, href, body) in items:
            html.append(
                f"<li style='margin:8px 0'>"
                f"<a href='{href}' target='_blank' rel='noopener'>{title}</a>"
                f"<div style='color:#555;font-size:14px;margin-top:4px'>{body}</div>"
                f"</li>"
            )
        html.append("</ol>")
        return "\n".join(html)

    except Exception as e:
        return f"حدث خطأ أثناء البحث: {str(e)}"


def lightweight_ai_answer(q: str) -> str:
    """
    ردّ عربي بسيط (بدون استهلاك نماذج مدفوعة).
    - يلتقط الكلمات المفتاحية ويقدّم توضيحًا صغيرًا.
    - إن كان السؤال قصيرًا جدًا أو غير واضح، يطلب مزيدًا من التوضيح.
    """
    q = (q or "").strip()
    if len(q) < 2:
        return "أحتاج توضيحًا أكثر. اكتب سؤالك بجملة كاملة."

    # أمثلة ردود بسيطة جدًا مبنية على كلمات مفتاحية
    m = q.lower()
    if 'الصين' in m:
        return "تفسير سريع: الصين دولة في شرق آسيا عاصمتها بكين، وتُعد ثاني أكبر اقتصاد عالميًا."
    if 'رياضيات' in m or 'مشتق' in m or 'تكامل' in m:
        return "إن كان سؤالك رياضيًا، اختر وضع «رياضيات» واكتب: مشتق: التعبير أو تكامل: التعبير."
    if any(w in m for w in ['من هو', 'من هي', 'ما هو', 'ما هي']):
        return "سؤال تعريفي: أعطني اسم الشيء المطلوب بوضوح وسأعطيك نبذة سريعة."
    # رد عام مقتضب
    return "تفسير سريع: " + q[:120] + ("..." if len(q) > 120 else "")

# -------- الواجهات --------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": "", "mode": "ذكاء", "q": ""}
    )

@app.post("/", response_class=HTMLResponse)
async def solve(request: Request, q: str = Form(""), mode: str = Form("ذكاء")):
    q = (q or "").strip()
    result = ""

    if mode == "بحث عربي":
        result = web_search_ar(q)
    elif mode == "رياضيات":
        result = solve_math(q)
    else:  # "ذكاء"
        result = lightweight_ai_answer(q)

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result, "mode": mode, "q": q}
    )

# نقطة تشغيل Uvicorn: main:app
