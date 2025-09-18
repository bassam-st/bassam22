# main.py
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from html import escape

app = FastAPI(title="Bassam App", version="1.0")

# CORS (اختياري لكنه مفيد)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ربط مجلد static + قوالب Jinja
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# الصفحة الرئيسية: تعرض templates/index.html
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# مسار البحث البسيط (ينفّذ الفورم الموجود في index.html)
@app.get("/search", response_class=HTMLResponse)
def search(request: Request, q: str = Query("", description="كلمة البحث")):
    result_html = f"""
    <html lang="ar" dir="rtl">
    <head><meta charset="utf-8"><title>نتيجة البحث</title></head>
    <body style="font-family: system-ui; padding:20px">
      <h2>🔎 نتيجة البحث:</h2>
      <p>بحثت عن: <b>{escape(q)}</b></p>
      <p><a href="/">⬅ العودة</a></p>
    </body>
    </html>
    """
    return HTMLResponse(result_html)
