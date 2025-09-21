# -*- coding: utf-8 -*-
# Bassam – Math AI (Arabic) using FastAPI + SymPy

from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import re

# SymPy
from sympy import (
    symbols, sympify, simplify, diff, integrate, Eq, S, solveset, sin, cos, tan,
    sqrt, pi, E
)

app = FastAPI(title="Bassam – Math AI (Arabic)")

# للسماح بالطلبات من أي متصفح
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# الصفحة الرئيسية: تعيد index.html الموجود بجذر المشروع
@app.get("/")
def home():
    return FileResponse("index.html")

# أدوات مساعدة
_AR_DIFF      = re.compile(r"^\s*تفاضل[:\s]+(.+?)(?:\s+(?:بالنسبة\s*إ?لى|عن)\s+([a-zA-Z]))?\s*$")
_AR_INTEG     = re.compile(r"^\s*تكامل[:\s]+(.+?)(?:\s+(?:بالنسبة\s*إ?لى|عن)\s+([a-zA-Z]))?\s*$")
_AR_SIMPLIFY  = re.compile(r"^\s*تبسيط[:\s]+(.+?)\s*$")
_AR_SOLVE     = re.compile(r"^\s*حل[:\s]+(.+?)\s*$")

# سمِح ببعض الرموز/الدوال الآمنة
def _allowed_locals():
    x, y, z, t, a, b, c = symbols("x y z t a b c")
    return {
        "x": x, "y": y, "z": z, "t": t, "a": a, "b": b, "c": c,
        "sin": sin, "cos": cos, "tan": tan, "sqrt": sqrt,
        "pi": pi, "e": E
    }

def _symp(expr_text: str):
    # استبدال ^ بـ ** ودعم الفاصلة العربية
    expr_text = expr_text.replace("^", "**").replace("،", ",")
    return sympify(expr_text, locals=_allowed_locals())

def solve_math_ar(query: str) -> str:
    q = query.strip()

    # 1) أوامر عربية صريحة
    m = _AR_DIFF.match(q)
    if m:
        expr = _symp(m.group(1))
        var  = symbols(m.group(2)) if m.group(2) else symbols("x")
        res  = diff(expr, var)
        return f"تفاضل {expr} بالنسبة إلى {var} = {res}"

    m = _AR_INTEG.match(q)
    if m:
        expr = _symp(m.group(1))
        var  = symbols(m.group(2)) if m.group(2) else symbols("x")
        res  = integrate(expr, var)
        return f"تكامل {expr} بالنسبة إلى {var} = {res}"

    m = _AR_SIMPLIFY.match(q)
    if m:
        expr = _symp(m.group(1))
        res  = simplify(expr)
        return f"تبسيط {expr} = {res}"

    m = _AR_SOLVE.match(q)
    if m:
        text = m.group(1)
        # إذا كان فيه مساواة نستعملها، وإلا نفترض = 0
        if "=" in text:
            left, right = text.split("=", 1)
            left_s, right_s = _symp(left), _symp(right)
            eq = Eq(left_s, right_s)
        else:
            eq = Eq(_symp(text), 0)

        # نحاول تحديد المتغيّر تلقائياً (الأكثر شيوعاً)
        for v in ["x", "y", "z", "t", "a", "b", "c"]:
            var = _allowed_locals()[v]
            sol = solveset(eq, var, domain=S.Complexes)
            if sol is not S.EmptySet:
                return f"حل {eq} بالنسبة إلى {var}: {sol}"
        # fallback عام
        return f"الحل: {solveset(eq, domain=S.Complexes)}"

    # 2) صيغة SymPy مباشرة: diff(), integrate(), solve(), simplify()
    try:
        expr = _symp(q)
        # لو كانت نتيجة عددية بسيطة، اعرضها أيضاً بشكل عشري إذا أمكن
        try:
            return f"النتيجة: {expr} ≈ {expr.evalf()}"
        except Exception:
            return f"النتيجة: {expr}"
    except Exception as e:
        return f"تعذر فهم الصيغة. جرّب أمثلة مثل:\n" \
               f"تفاضل: x^2 بالنسبة إلى x\n" \
               f"تكامل: sin(x) بالنسبة إلى x\n" \
               f"حل: x^2 - 4 = 0\n" \
               f"تبسيط: (x^2 + 2*x + 1)\n" \
               f"أو استخدم صيغة SymPy مثل: diff(x^3, x)"

# واجهة API تتلقى q من النموذج وتعيد JSON
@app.post("/")
async def api_solve(q: str = Form(...)):
    result = solve_math_ar(q)
    return JSONResponse({"result": result})
