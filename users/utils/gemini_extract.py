import base64
import json
import time
import traceback
import hashlib

import google.generativeai as genai
from PIL import Image
import io

# ──────────────────────────────────────────────────────────────────────
# 🔑 API KEY POOL
# ──────────────────────────────────────────────────────────────────────
API_KEYS = [
    "AIzaSyDmZ22YZofRf3NeYsJSIswrDXXCyTcmcRU",
    "AIzaSyCghyHFi8DWSy026L1jB_OEmd-QWcSezCY",
    "AIzaSyALNXMUxpVnDQ9-jVlVo02rXjLC0hwCSy0",
]

genai.configure(api_key=API_KEYS[0])

# Fixed: Valid Gemini Model Names
MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-2.0-flash-lite-preview-02-05"
]

_result_cache = {}
_key_index = 0

def _rotate_key():
    global _key_index
    _key_index = (_key_index + 1) % len(API_KEYS)
    genai.configure(api_key=API_KEYS[_key_index])
    print(f"DEBUG: Rotated to API key index {_key_index}")

def _call_gemini(model_name, img_b64, prompt, max_retries=2):
    keys_attempted = 0
    total_keys = len(API_KEYS)
    
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
                            {"text": prompt},
                        ],
                    }
                ],
                generation_config={"response_mime_type": "application/json"},
            )
            if not response or not response.text: raise ValueError("Empty response")
            return json.loads(response.text)
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                if keys_attempted < total_keys:
                    _rotate_key()
                    keys_attempted += 1
                    continue
            raise e

def extract_cheque_info(image_path):
    try:
        from PIL import ImageOps
        with Image.open(image_path) as img:
            try: img = ImageOps.exif_transpose(img)
            except: pass
            img.thumbnail((1200, 1200))
            if img.mode != "RGB": img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            img_bytes = buf.getvalue()

        img_hash = hashlib.md5(img_bytes).hexdigest()
        if img_hash in _result_cache: return _result_cache[img_hash]

        img_b64 = base64.b64encode(img_bytes).decode()
        prompt = """
        Extract bank cheque details with 100% accuracy.
        Return ONLY JSON:
        {
          "is_cheque": true,
          "prediction": "VALID",
          "message": "Summary",
          "details": {
            "account_number": "...",
            "ifsc_code": "...",
            "cheque_number": "...",
            "payee_name": "...",
            "amount_words": "...",
            "amount_number": "...",
            "signature_present": "Yes/No",
            "signature_remarks": "..."
          }
        }
        """

        last_error = ""
        for model_name in MODELS:
            try:
                result = _call_gemini(model_name, img_b64, prompt)
                if "prediction" not in result:
                    result["prediction"] = "VALID" if result.get("is_cheque") else "INVALID"
                _result_cache[img_hash] = result
                return result
            except Exception as e:
                last_error = str(e)
                continue

        # Real Error instead of mock
        raise Exception(f"AI Quota Exhausted: {last_error}")

    except Exception as e:
        return {
            "is_cheque": False,
            "prediction": "INVALID",
            "message": f"Error: {str(e)}",
            "details": {
                "account_number": "N/A", "ifsc_code": "N/A", "cheque_number": "N/A",
                "payee_name": "N/A", "amount_words": "N/A", "amount_number": "N/A",
                "signature_present": "N/A", "signature_remarks": f"Fail: {str(e)}"
            }
        }
