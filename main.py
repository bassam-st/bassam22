# main.py — تطبيق Bassam الذكي
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import httpx

app = FastAPI(title="Bassam الذكي", version="1.0")

# ربط مجلد static + القوالب
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# الصفحة الرئيسية: تعرض index.html
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# نفس الصفحة لكن تستقبل POST (من نموذج البحث)
@app.post("/", response_class=HTMLResponse)
async def search(request: Request, question: str = Form(...), mode: str = Form("summary")):
    # هنا تقدر تضيف المنطق لاحقاً (بحث، أسعار، صور…)
    answer_text = f"🔎 هذا مثال: بحثت عن — {question} (الوضع: {mode})"
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "q": question,
            "mode": mode,
            "answer_text": answer_text,
            "result_panel": "",
        },
    )


# صفحة فحص الصحة (تساعد Render)
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


# صفحة API للتجربة (اختياري)
@app.get("/time")
async def time_now():
    import datetime
    return {"time": datetime.datetime.utcnow().isoformat()}
