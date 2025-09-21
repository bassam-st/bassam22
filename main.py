# main.py — Bassam الذكي: بحث عربي + حل رياضيات + توليد PDF
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from io import BytesIO
from datetime import datetime
import re

# ddgs (بديل duckduckgo_search)
try:
    from ddgs import DDGS
except Exception:
    DDGS = None

# PDF
from fpdf import FPDF

# SymPy للرياضيات
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)
TRANS = standard_transformations + (implicit_multiplication_application,)

app = FastAPI(title="Bassam الذكي")
templates = Jinja2Templates(directory=".")

# ================= أدوات الرياضيات =================
_AR_DIGITS = str.maketrans("٠١٢٢٣٤٥٦٧٨٩".replace("٢","2"), "0123456789")  # احتياط للتشكيل

def _normalize_math_text(s: str) -> str:
    s = (s or "").strip()
    s = s.translate(_AR_DIGITS)
    s = s.replace("π", "pi")
    return s

def _looks_like_math(q: str) -> bool:
    ql = q.strip().lower()
    return (
        ql.startswith("احسب") or
        any(k in ql for k in ["sin","cos","tan","log","ln","sqrt","تكامل","اشتقاق","حل","limit","حد","معادلة"]) or
        bool(re.search(r"[=+\-*/^()]", ql))
    )

def _parse_sympy(expr_text: str):
    return parse_expr(expr_text, transformations=TRANS, evaluate=True)

def _deg_to_rad_if_needed(text: str, expr):
    if "°" in text:
        return parse_expr(re.sub(r"(\d+)\s*°", r"(\1*pi/180)", text), transformations=TRANS, evaluate=True)
    return expr

def solve_math(q: str) -> tuple[str, list]:
    steps = []
    text = _normalize_math_text(q)
    text = re.sub(r"^\s*احسب(وا)?\s*[:：]?\s*", "", text, flags=re.IGNORECASE)

    # نظام معادلات؟
    if ("," in text and "=" in text) or ("\n" in text and "=" in text):
        parts = [p.strip() for p in re.split(r"[,\n]+", text) if p.strip()]
        eqs = []; syms = set()
        for p in parts:
            if "=" in p:
                L, R = p.split("=", 1)
                L = _parse_sympy(L); R = _parse_sympy(R)
                eqs.append(sp.Eq(L, R))
                syms |= L.free_symbols | R.free_symbols
        if not eqs:
            return "لم يتم التعرف على معادلات صالحة.", []
        vars_sorted = sorted(list(syms), key=lambda s: s.name)
        steps.append(f"حل نظام مكوّن من {len(eqs)} معادلات.")
        sol = sp.solve(eqs, vars_sorted, dict=True)
        return sp.spretty(sol), steps

    # معادلة واحدة أو تعبير
    expr = _parse_sympy(text)
    expr = _deg_to_rad_if_needed(text, expr)

    if isinstance(expr, sp.Equality):
        steps.append("حل معادلة واحدة.")
        res = sp.solve(sp.Eq(expr.lhs, expr.rhs), list(expr.free_symbols))
        return sp.spretty(res), steps

    # تقييم عددي إن أمكن
    if not expr.free_symbols:
        steps.append("تقييم عددي مباشر.")
        return sp.spretty(sp.N(expr)), steps

    # كلمات مفتاحية
    ql = q.lower()
    if any(k in ql for k in ["اشتق", "deriv", "مشتقة"]):
        var = list(expr.free_symbols)[0]
        steps.append(f"اشتقاق بالنسبة إلى {var}.")
        return sp.spretty(sp.diff(expr, var)), steps
    if any(k in ql for k in ["تكامل", "integr"]):
        var = list(expr.free_symbols)[0]
        steps.append(f"تكامل غير محدد بالنسبة إلى {var}.")
        return sp.spretty(sp.integrate(expr, var)), steps
    if any(k in ql for k in ["حد", "limit"]):
        var = list(expr.free_symbols)[0]
        steps.append(f"حساب نهاية عندما {
