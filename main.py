# -*- coding: utf-8 -*-
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import re

# ====== Sympy (رياضيات رمزية) ======
from sympy import (
    symbols, Eq, sympify, sin, cos, tan, log, sqrt, pi, E,
    diff, integrate, simplify, solveset, S
)

app = FastAPI()

# استخدم مجلد العمل الحالي كقالب (لدعم index.html الموجود في الجذر)
templates = Jinja2Templates(directory=".")

# رموز افتراضية للاستخدام إن لم يحدد المستخدم المتغير
x, y, z = symbols('x y z')

# دوال/ثوابت نسمح بها داخل sympify
SAFE_LOCALS = {
    "x": x, "y": y, "z": z,
    "sin": sin, "cos": cos, "tan": tan,
    "log": log, "sqrt": sqrt, "pi": pi, "E": E
}


def _pick_var(q: str):
    """
    يحاول استنتاج المتغير المطلوب التفاضل/التكامل بالنسبة له.
    أمثلة: 'بالنسبة ل x' ، 'بالنسبة إلى س'
    الافتراضي: x
    """
    # ابحث عن صيغة عربية أو إنجليزية
    m = re.search(r"بالنسبة\s+(?:ل|إلى)\s*([a-zA-Zسصقفكلمنهوي])", q)
    if m:
        v = m.group(1)
        # إذا كتب س بالعربية اعتبرها x
        if v in ["س", "x", "X"]:
            return x
        if v in ["ص", "y", "Y"]:
            return y
        if v in ["ز", "z", "Z"]:
            return z
    # ابحث إن ذكر المستخدم مباشرة حرف متغير شائع
    for v in ["x", "y", "z", "س", "ص", "ز"]:
        if re.search(rf"\b{v}\b", q):
            return {"x": x, "y": y, "z": z, "س": x, "ص": y, "ز": z}[v]
    return x


def solve_math(q: str) -> str | None:
    """
    يحاول فهم السؤال كمسألة رياضيّة ويعيد نتيجة عربية.
    يعيد None إن لم يتعرف على أنها مسألة رياضية.
    """
    txt = q.strip()
    low = txt.lower()

    # الكلمات المفتاحية
    is_diff = any(k in txt for k in ["مشتق", "تفاضل", "اشتق"])
    is_intg = "تكامل" in txt
    is_simplify = any(k in txt for k in ["يبسط", "تبسيط", "بسّط"])
    is_solve = ("حل" in txt and "معادلة" in txt) or ("=" in txt)

    # استخرج التعبير بعد الكلمة المفتاحية إن وُجد
    def after(word):
        i = txt.find(word)
        if i == -1:
            return None
        return txt[i+len(word):].strip()

    try:
        if is_diff:
            expr_str = after("مشتق") or after("تفاضل") or after("اشتق")
            if not expr_str:
                expr_str = txt
            var = _pick_var(txt)
            expr = sympify(expr_str, locals=SAFE_LOCALS)
            res = diff(expr, var)
            return f"المشتقة بالنسبة إلى {var}:\n{res}"

        if is_intg:
            expr_str = after("تكامل") or txt
            var = _pick_var(txt)
            expr = sympify(expr_str, locals=SAFE_LOCALS)
            res = integrate(expr, var)
            return f"التكامل بالنسبة إلى {var}:\n{res}"

        if is_simplify:
            expr_str = after("يبسط") or after("تبسيط") or txt
            expr = sympify(expr_str, locals=SAFE_LOCALS)
            res = simplify(expr)
            return f"تبسيط التعبير:\n{res}"

        if is_solve:
            # صيغة معادلة: إما ذكر 'حل معادلة' أو وجود علامة '='
            if "=" in txt:
                left, right = txt.split("=", 1)
                left = sympify(left, locals=SAFE_LOCALS)
                right = sympify(right, locals=SAFE_LOCALS)
                var = _pick_var(txt)
                sol = solveset(Eq(left, right), var, domain=S.Complexes)
                return f"حل المعادلة بالنسبة إلى {var}:\n{sol}"
            else:
                # مثلاً: "حل معادلة x**2 - 4"
                expr_str = after("حل") or txt
                expr = sympify(expr_str, locals=SAFE_LOCALS)
                var = _pick_var(txt)
                sol = solveset(Eq(expr, 0), var, domain=S.Complexes)
                return f"الجذور (حل {expr_str}=0) بالنسبة إلى {var}:\n{sol}"

        # إن لم تُذكر كلمات خاصة: جرّب تقييم عددي مباشر
        if any(ch in low for ch in list("0123456789xyzسصز+-*/^().")):
            expr = sympify(txt, locals=SAFE_LOCALS)
            num = expr.evalf()
            return f"القيمة العددية:\n{num}"

    except Exception as e:
        return f"لم أفهم صيغة المسألة جيدًا. جرّب مثلاً:\n" \
               f"- مشتق sin(x)*cos(x)\n" \
               f"- تكامل x**2 بالنسبة ل x\n" \
               f"- حل معادلة x**2 = 9\n" \
               f"- بسّط (x**2 - 1)/(x-1)\n\n" \
               f"تفاصيل فنية: {e}"

    # ليس مسألة رياضيّة على الأرجح
    return None


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # يعرض الصفحة فقط
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": "", "mode": "رياضيات"}
    )


@app.post("/", response_class=HTMLResponse)
async def handle_form(
    request: Request,
    q: str = Form(...),
    mode: str = Form("تلخيص ذكي")  # الحقل موجود في صفحتك، لكن سنركّز على الرياضيات هنا
):
    # 1) جرّب كرياضيات أولاً
    math_ans = solve_math(q)
    if math_ans:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": math_ans, "mode": "رياضيات", "q": q}
        )

    # 2) إن لم تكن رياضيات، أعِد رسالة إرشادية (بدون بحث ويب)
    msg = (
        "اسألني مسألة رياضيّة وسأحلّها رمزيًا.\n"
        "أمثلة:\n"
        "- مشتق sin(x)*cos(x)\n"
        "- تكامل (x**2 + 3*x) بالنسبة ل x\n"
        "- حل معادلة x**2 = 9\n"
        "- بسّط (x**2 - 1)/(x-1)\n"
        "- 2 + 3*4 + sqrt(16)"
    )
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": msg, "mode": "إرشاد", "q": q}
    )
