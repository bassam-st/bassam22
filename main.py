
# main.py
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone

app = FastAPI(title="Bassam App", version="1.0")

# للسماح بطلبات من المتصفح إذا احتجتها
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
      <head><meta charset="utf-8"><title>Bassam App</title></head>
      <body style="font-family: system-ui; padding:20px">
        <h2>تم تشغيل تطبيق FastAPI 🚀</h2>
        <p>روابط للتجربة السريعة:</p>
        <ul>
          <li><a href="/healthz">/healthz</a> — فحص الصحة</li>
          <li><a href="/time">/time</a> — الوقت الآن (UTC)</li>
          <li><a href="/echo?q=hello">/echo?q=hello</a> — إرجاع ما ترسله له</li>
        </ul>
      </body>
    </html>
    """

@app.get("/time")
def time_now():
    now = datetime.now(timezone.utc).isoformat()
    return {"utc": now}

@app.get("/echo")
def echo(q: str = Query(..., description="النص المراد إرجاعه")):
    return {"you_sent": q}
