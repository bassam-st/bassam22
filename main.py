# main.py — تطبيق Bassam الذكي (نسخة ثابتة)
# يعمل بـ FastAPI + قوالب Jinja2 ويبحث بالعربية عبر DuckDuckGo

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from duckduckgo_search import DDGS
import traceback

# -------------------------------------------------
# إعداد التطبيق والقوالب
# لو كان index.html في جذر المستودع (وليس داخل مجلد templates)
# نضبط Jinja2 ليقرأ من الجذر "."
# إن كان عندك مجلد "templates"، غيّر السطر إلى: Jinja2Templates(directory="templates")
# -------------------------------------------------
app = FastAPI(title="Bassam الذكي", version="1.0.0")
templates = Jinja2Templates(directory=".")

# -------------------------------------------------
# دالة بحث عربي مع رجوع احتياطي
# -------------------------------------------------
def search_ar(q: str, maxn: int = 8):
    """
    يجبر البحث على العربية بإضافة 'بالعربية' وتحديد المنطقة العربية.
    وفي حال فشل/لا توجد نتائج، يعمل رجوع احتياطي عام.
    يرجع قائمة قوامها قواميس: {title, href, body}
    """
    results = []
    query = f"{q} بالعربية"

    # محاولة 1: منطقة عربية + فلترة معتدلة + آخر سنة
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(
                query,
                region="sa-ar",         # تركيز عربي
                safesearch="moderate",
                timelimit="y",          # آخر سنة
                max_results=maxn
            ):
                results.append({
                    "title": r.get("title", ""),
                    "href":  r.get("href", ""),
                    "body":  r.get("body", "")
                })
    except Exception:
        # فقط تسجيل داخلي؛ لا نكسر التطبيق
        print("DDG primary failed:\n", traceback.format_exc())

    # محاولة 2: إن لم نجد شيئًا، بحث عام بدون region/توقيت
    if not results:
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=maxn):
                    results.append({
                        "title": r.get("title", ""),
                        "href":  r.get("href", ""),
                        "body":  r.get("body", "")
                    })
        except Exception:
            print("DDG fallback failed:\n", traceback.format_exc())

    return results

# -------------------------------------------------
# الصفحة الرئيسية
# -------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # نعرض index.html كما هو. إن كان القالب يستخدم متغيرات، نمرّر قيم افتراضية.
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "answer_text": "",   # مكان النتيجة (إن كان القالب يعرضه)
            "raw_json": ""       # للاستخدام الاختياري
        }
    )

# -------------------------------------------------
# استقبال البحث من النموذج (Form POST "/")
# الحقول المتوقعة: q (النص)، mode (اختياري)
# -------------------------------------------------
@app.post("/", response_class=HTMLResponse)
async def search(request: Request, q: str = Form(...), mode: str = Form("summary")):
    q = (q or "").strip()
    if not q:
        # نعيد نفس الصفحة مع رسالة ودية
        msg = "الرجاء كتابة سؤالك أولًا."
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "answer_text": msg, "raw_json": ""}
        )

    try:
        results = search_ar(q, maxn=8)

        if not results:
            answer_html = "لم أعثر على نتائج. جرِّب صياغة أخرى أو أضف كلمة «بالعربية»."
        else:
            # نبني HTML بسيط للنتيجة
            lines = []
            lines.append("<div style='line-height:1.7'>")
            lines.append(f"<p><b>سؤالك:</b> {q}</p>")
            lines.append("<ol>")
            for r in results:
                title = r.get("title", "").replace("<", "&lt;").replace(">", "&gt;")
                href  = r.get("href", "#")
                body  = r.get("body", "")
                lines.append(
                    f"<li style='margin-bottom:8px'><a href='{href}' target='_blank' rel='noopener'>{title}</a>"
                    + (f"<div style='font-size:13px;opacity:.8'>{body}</div>" if body else "")
                    + "</li>"
                )
            lines.append("</ol>")
            lines.append("</div>")
            answer_html = "\n".join(lines)

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "answer_text": answer_html, "raw_json": ""}
        )

    except Exception:
        # في حال أي خطأ غير متوقع: نعرض رسالة ودية ولا نسقط السيرفر
        err_msg = "حدث خطأ غير متوقع أثناء المعالجة. حاول لاحقًا أو غيّر صياغة سؤالك."
        print("Unhandled error:\n", traceback.format_exc())
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "answer_text": err_msg, "raw_json": ""}
        )

# -------------------------------------------------
# صحة وتشخيص
# -------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/time")
def time_now():
    import datetime
    return {"time": datetime.datetime.utcnow().isoformat()}

# ملاحظة:
# إذا كنت تستخدم Render مع Procfile فيه:
# web: uvicorn main:app --host 0.0.0.0 --port 10000
# فتأكّد أن اسم الملف main.py والمتغير app كما هو هنا.
