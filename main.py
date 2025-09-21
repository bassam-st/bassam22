from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# رياضيات
from sympy import symbols, sympify, diff, integrate, simplify
# بحث (الحزمة الجديدة)
from duckduckgo_search import DDGS

app = FastAPI()

# static + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --------- بحث عربي مبسّط ---------
def ar_search(q: str, maxn: int = 6):
    items = []
    query = f"{q} بالعربية"
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, region="sa-ar", safesearch="moderate", timelimit="y", max_results=maxn):
                items.append({"title": r.get("title",""), "href": r.get("href",""), "body": r.get("body","")})
    except Exception:
        pass
    if not items:
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=maxn):
                    items.append({"title": r.get("title",""), "href": r.get("href",""), "body": r.get("body","")})
        except Exception:
            pass
    return items

# --------- حلول رياضيات أساسية ---------
def solve_math(q: str) -> str:
    """
    أمثلة:
    - مشتق x**3
    - تكامل sin(x)
    - بسّط (x+x)
    """
    x, y, z = symbols("x y z")
    s = q.strip().replace("**", "^").replace("^", "**")

    # تحديد العملية
    op = None
    if any(w in s for w in ["مشتق", "المشتقة", "اشتق", "deriv"]):
        op = "diff"
    elif any(w in s for w in ["تكامل", "integral", "integrate"]):
        op = "integrate"
    elif any(w in s for w in ["بسّط", "بسط", "simplify"]):
        op = "simplify"

    # أخذ التعبير بعد الكلمة المفتاحية إن وجدت
    for key in ["مشتق", "المشتقة", "اشتق", "تكامل", "بسّط", "بسط"]:
        if key in s:
            s = s.split(key,1)[-1].strip(":،. ").strip()
            break
    if not s:
        s = q

    try:
        expr = sympify(s)
        if op == "diff":
            return f"المشتقة بالنسبة لـ x: {diff(expr, x)}"
        elif op == "integrate":
            return f"التكامل بالنسبة لـ x: {integrate(expr, x)}"
        elif op == "simplify":
            return f"التبسيط: {simplify(expr)}"
        else:
            return f"تفسير سريع:\n• تبسيط: {simplify(expr)}\n• مشتقة (x): {diff(expr, x)}"
    except Exception as e:
        return f"تعذر فهم التعبير الرياضي. جرّب مثل: 'مشتق x**3' أو 'تكامل sin(x)'.\nالخطأ: {e}"

# --------- الواجهات ---------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html",
                                      {"request": request, "result": "", "mode": "ذكاء", "q": ""})

@app.post("/", response_class=HTMLResponse)
async def ask(request: Request, q: str = Form(...), mode: str = Form("ذكاء")):
    q = (q or "").strip()
    if not q:
        result = "اكتب سؤالك أولًا."
    else:
        if mode == "رياضيات":
            result = solve_math(q)
        elif mode == "بحث عربي":
            items = ar_search(q, maxn=6)
            if not items:
                result = "لم أعثر على نتائج. جرّب صياغة أخرى."
            else:
                result = "\n".join(
                    f"<p><strong>{it['title']}</strong><br>"
                    f"<a href='{it['href']}' target='_blank'>{it['href']}</a><br>{it['body']}</p>"
                    for it in items
                )
        else:
            # ذكاء بسيط: تحية/توجيه، وإلا جرّب رياضيات ثم بحث
            if any(w in q for w in ["السلام", "مرحبا", "هلا"]):
                result = "هلا بك! اسأل رياضيات (مثال: مشتق x**3) أو اختر 'بحث عربي'."
            else:
                trial = solve_math(q)
                if "تعذر" not in trial:
                    result = trial
                else:
                    items = ar_search(q, maxn=4)
                    if items:
                        lines = "\n".join(f"<p><strong>{i['title']}</strong><br><a href='{i['href']}' target='_blank'>{i['href']}</a><br>{i['body']}</p>" for i in items)
                        result = "لم أفهم السؤال كرياضيات — هذه نتائج بحث قد تساعدك:<br>" + lines
                    else:
                        result = "أحتاج توضيحًا أكثر، أو جرّب وضع 'بحث عربي'."

    return templates.TemplateResponse("index.html",
                                      {"request": request, "result": result, "mode": mode, "q": q})
