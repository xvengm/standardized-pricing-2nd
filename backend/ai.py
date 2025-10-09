# backend/ai.py
import os
import json
import logging
from typing import List, Dict, Any, Optional

log = logging.getLogger("pricing.ai")

# -------- AI config (kept here, not in database.py) --------
GEMINI_TXT    = os.getenv("GEMINI_MODEL_TEXT",   "gemini-1.5-flash")
GEMINI_VISION = os.getenv("GEMINI_MODEL_VISION", "gemini-1.5-flash")

# Hard safety limits so uploads can’t hang the server
AI_TIMEOUT     = int(os.getenv("AI_TIMEOUT_SECONDS", "30"))  # per call timeout
AI_MAX_CHARS   = int(os.getenv("AI_MAX_CHARS", "8000"))      # truncate large text
AI_ENABLED_FOR_CSV = os.getenv("USE_AI_FOR_CSV", "false").lower() == "true"

# -------- Gemini client (optional) --------
GEMINI_AVAILABLE = False
_client = None
try:
    from google import genai
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        _client = genai.Client(api_key=api_key)
        GEMINI_AVAILABLE = True
except Exception as e:
    log.warning("Gemini not available: %s", e)
    GEMINI_AVAILABLE = False
    _client = None


def _trim_text(text: str) -> str:
    if not text:
        return ""
    return text if len(text) <= AI_MAX_CHARS else text[:AI_MAX_CHARS]


def _safe_json_rows(s: str) -> List[Dict[str, Any]]:
    """
    Accepts a model response string and tries to extract a JSON object with a top-level
    'rows' array. Handles code-fences and minor formatting issues.
    """
    if not s:
        return []
    # Strip code fences if any
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        idx = s.find("{")
        if idx >= 0:
            s = s[idx:]
    try:
        data = json.loads(s)
        if isinstance(data, dict) and isinstance(data.get("rows"), list):
            return data["rows"]
        if isinstance(data, list):
            return data
    except Exception:
        # Attempt to locate the first/last braces
        try:
            start = s.find("{")
            end   = s.rfind("}")
            if start >= 0 and end > start:
                data = json.loads(s[start:end+1])
                if isinstance(data, dict) and isinstance(data.get("rows"), list):
                    return data["rows"]
                if isinstance(data, list):
                    return data
        except Exception:
            pass
    return []


def extract_rows_from_text(text: str, hint: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Use Gemini text model to extract pricing rows from raw text (CSV pasted text, PDF text fallback, etc.)
    Return list of dicts with schema:
      commodity, brand?, price_text, unit_text?, state, city, market?, date_uploaded?
    """
    if not (GEMINI_AVAILABLE and text):
        return []
    try:
        model = _client.models
        prompt = f"""
Extract product price rows as JSON with this exact shape:

rows: [
  {{
    "commodity": "...",
    "brand": "...(optional)",
    "price_text": "...",               # keep symbols/codes (e.g., ₦, NGN, USD)
    "unit_text": "...(optional)",      # e.g., "50kg", "1 crate"
    "state": "...",
    "city": "...",
    "market": "...(optional)",
    "date_uploaded": "...(optional, YYYY-MM-DD)"
  }},
  ...
]

RULES:
- Return ONLY JSON (no prose).
- {("Notes: " + hint) if hint else ""}
"""
        trimmed = _trim_text(text)
        resp = model.generate_content(
            model=GEMINI_TXT,
            contents=[{"role": "user", "parts": [{"text": prompt}, {"text": trimmed}]}],
            request_options={"timeout": AI_TIMEOUT},
        )
        return _safe_json_rows(getattr(resp, "text", ""))
    except Exception as e:
        log.error("Gemini text extraction failed: %s", e)
        return []


def extract_rows_from_file(mime: str, data: bytes, hint: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Use Gemini vision to extract rows from PDF/Image. Returns same schema as above.
    """
    if not (GEMINI_AVAILABLE and data and mime):
        return []
    try:
        prompt = f"""
Extract rows as JSON (same schema as text extraction).
- Return ONLY JSON.
- If table present, read it robustly even if messy or rotated.
- {("Notes: " + hint) if hint else ""}
"""
        resp = _client.models.generate_content(
            model=GEMINI_VISION,
            contents=[{"role": "user", "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime, "data": data}}
            ]}],
            request_options={"timeout": AI_TIMEOUT},
        )
        return _safe_json_rows(getattr(resp, "text", ""))
    except Exception as e:
        log.error("Gemini vision extraction failed: %s", e)
        return []
