# main.py — تطبيق Bassam الذكي 🚀
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Bassam الذكي", version="1.0")

# ربط مجلد static + القوالب
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# الصفحة الرئيسية: تعرض index.html
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# نموذج البحث (POST)
@app.post("/", response_class=HTMLResponse)
async def search(request: Request, question: str = Form(...), mode: str = Form("summary")):
    # مؤقتًا: نعيد السؤال + نوع البحث (إلى أن نضيف المنطق لاحقًا)
    answer_text = f"🔍 سؤالك: {question}\n⚙️ النمط: {mode}"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "q": question,
            "mode": mode,
            "answer_text": answer_text,
            "result_panel": ""
        }
    )

# مسار فحص الصحة
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

# مسار الوقت الحالي
from datetime import datetime
@app.get("/time")
async def time_now():
    return {"time": datetime.utcnow().isoformat()}

# لقبول طلب HEAD (حتى لا يظهر خطأ 405)
from fastapi import Response
@app.head("/")
async def home_head():
    return Response(status_code=204)
