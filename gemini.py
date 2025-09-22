import json
import logging
import os
import html

from google import genai
from google.genai import types


# IMPORTANT: KEEP THIS COMMENT
# Follow these instructions when using this blueprint:
# - Note that the newest Gemini model series is "gemini-2.5-flash" or gemini-2.5-pro"
#   - do not change this unless explicitly requested by the user

# This API key is from Gemini Developer API Key, not vertex AI API Key
def get_gemini_client():
    """الحصول على عميل Gemini (يعيد None إذا لم يوجد API key)"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def answer_with_ai(question: str):
    """
    الإجابة على الأسئلة باستخدام Gemini AI بطريقة ذكية وباللغة العربية.
    يعيد dict مع 'text' و 'html' أو None إذا لم يكن API key متوفراً.
    """
    try:
        client = get_gemini_client()
        if not client:
            return None
            
        # بناء prompt ذكي باللغة العربية
        system_prompt = """أنت "بسام الذكي" - مساعد ذكي عربي متخصص في:
1. الإجابة على الأسئلة بوضوح ودقة باللغة العربية
2. شرح المفاهيم العلمية والرياضية بطريقة مبسطة
3. تقديم معلومات موثوقة ومفيدة
4. المساعدة في حل المشاكل وتقديم النصائح

قواعد مهمة:
- أجب باللغة العربية دائماً
- اجعل إجابتك واضحة ومفصلة
- استخدم أمثلة عملية عند الحاجة
- إذا لم تعرف الإجابة، قل ذلك بصراحة
- تجنب المعلومات الخاطئة أو المضللة"""

        prompt = f"{system_prompt}\n\nالسؤال: {question}\n\nالإجابة:"
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1000,
            )
        )

        if response.text:
            answer_text = response.text.strip()
            
            # تنسيق HTML للإجابة
            safe_question = html.escape(question)
            safe_answer = html.escape(answer_text)
            
            # تحويل النقاط والأرقام إلى تنسيق أفضل
            formatted_answer = safe_answer.replace('\n', '<br>')
            formatted_answer = formatted_answer.replace('- ', '<br>• ')
            formatted_answer = formatted_answer.replace('. ', '.<br><br>')
            
            html_response = f'''
            <div class="card">
                <h4>🤖 إجابة بسام الذكي</h4>
                <h5 style="color:#666;">❓ السؤال: {safe_question}</h5>
                <hr>
                <div style="background:#f9f9f9;padding:15px;border-radius:8px;line-height:1.6;">
                    {formatted_answer}
                </div>
                <small style="color:#999;margin-top:10px;display:block;">
                    💡 تم توليد هذه الإجابة بواسطة Gemini AI
                </small>
            </div>
            '''
            
            return {
                "text": answer_text, 
                "html": html_response
            }
        else:
            return None
            
    except Exception as e:
        logging.error(f"خطأ في Gemini AI: {e}")
        error_html = f'''
        <div class="card" style="border-left:4px solid #ff6b6b;">
            <h4>❌ خطأ في نظام الذكاء الاصطناعي</h4>
            <p>عذراً، حدث خطأ أثناء معالجة سؤالك. يرجى المحاولة مرة أخرى.</p>
            <small style="color:#666;">تفاصيل الخطأ: {html.escape(str(e))}</small>
        </div>
        '''
        return {
            "text": f"خطأ في AI: {str(e)}", 
            "html": error_html
        }


def smart_math_help(question: str):
    """
    مساعدة ذكية في الرياضيات باستخدام Gemini
    """
    try:
        client = get_gemini_client()
        if not client:
            return None
            
        system_prompt = """أنت مدرس رياضيات خبير. اشرح الحلول الرياضية خطوة بخطوة باللغة العربية.

قواعد مهمة:
- اشرح كل خطوة بوضوح
- استخدم الأرقام والرموز الرياضية
- قدم أمثلة مشابهة إذا أمكن
- تأكد من دقة الحسابات"""

        prompt = f"{system_prompt}\n\nالسؤال الرياضي: {question}\n\nالحل التفصيلي:"
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if response.text:
            answer_text = response.text.strip()
            safe_answer = html.escape(answer_text)
            formatted_answer = safe_answer.replace('\n', '<br>')
            
            html_response = f'''
            <div class="card">
                <h4>🧮 مساعد الرياضيات الذكي</h4>
                <hr>
                <div style="background:#f0f8ff;padding:15px;border-radius:8px;line-height:1.8;">
                    {formatted_answer}
                </div>
            </div>
            '''
            
            return {
                "text": answer_text, 
                "html": html_response
            }
        else:
            return None
            
    except Exception as e:
        logging.error(f"خطأ في مساعد الرياضيات: {e}")
        return None


def is_gemini_available() -> bool:
    """فحص توفر Gemini API"""
    return os.environ.get("GEMINI_API_KEY") is not None
