# backend/routers/admin.py
from typing import List, Dict, Any, Optional, Tuple
import io, csv, re
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from pydantic import BaseModel
from PIL import Image
import pdfplumber

from backend.database import get_session
from backend.services.ingest import standardize_and_store
from backend.ai import GEMINI_AVAILABLE, extract_rows_from_text, extract_rows_from_file

router = APIRouter()

# ------------------------ Helpers: CSV header mapping ------------------------

_SYNONYMS = {
    "commodity":  ["commodity","item","product","name","goods","article"],
    "brand":      ["brand","make","model"],
    "price_text": ["price_text","price","amount","cost","value","price(ngn)","price_ngn","price_usd","price(usd)"],
    "unit_text":  ["unit_text","unit","qty","quantity","size","weight","pack_size"],
    "unit_value": ["unit_value","uvalue","qty_value"],
    "unit_name":  ["unit_name","uname","uom","measure"],
    "state":      ["state","region","province"],
    "city":       ["city","town","lga","locality"],
    "market":     ["market","market_name","marketplace"],
    "date_uploaded": ["date_uploaded","date","created_at","dt","timestamp"],
    "currency":   ["currency","curr","fx","ngn","usd","eur","ghs","kes","zar"],
    "location":   ["location","place","state_city","city_state","where"],
}

_LOC_SPLIT = re.compile(r"\s*[,/|\-–]\s*")

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.strip().lower()) if isinstance(s, str) else ""

def _best_key(colname: str) -> Optional[str]:
    cn = _norm(colname)
    for target, opts in _SYNONYMS.items():
        for o in opts:
            if cn == _norm(o):
                return target
    # fuzzy prefix/contain match
    for target, opts in _SYNONYMS.items():
        for o in opts:
            if _norm(o) and (cn.startswith(_norm(o)) or _norm(o) in cn):
                return target
    return None

def _sniff_reader(txt: str) -> csv.DictReader:
    sample = txt[:2000]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;|\t")
        return csv.DictReader(io.StringIO(txt), dialect=dialect)
    except Exception:
        return csv.DictReader(io.StringIO(txt))  # default comma

def _compose_unit_text(rec: Dict[str, Any]) -> str:
    uv, un = rec.get("unit_value"), rec.get("unit_name")
    if uv and un:
        return f"{uv} {un}".strip()
    return rec.get("unit_text") or ""

def _price_to_text(price_val: Any, currency: Optional[str]) -> str:
    if price_val is None or price_val == "":
        return ""
    s = str(price_val).strip()
    # remove thousands separators if numeric-like
    m = re.fullmatch(r"\d[\d,.\s]*", s)
    if m:
        s = s.replace(",", "").replace(" ", "")
    if currency:
        # keep currency word/symbol if provided
        cur = currency.strip()
        return f"{cur} {s}" if cur and not s.startswith(cur) else s
    return s

def _split_location(location: str) -> Tuple[str, str, str]:
    if not location:
        return "", "", ""
    parts = [p for p in _LOC_SPLIT.split(location) if p]
    if len(parts) == 1:
        return parts[0], "", ""
    if len(parts) == 2:
        return parts[0], parts[1], ""
    # 3 or more: state, city, market (rest joined)
    return parts[0], parts[1], " - ".join(parts[2:])

def _rows_from_csv_bytes(b: bytes) -> List[Dict[str, Any]]:
    # decode robustly
    try:
        txt = b.decode("utf-8-sig")
    except UnicodeDecodeError:
        txt = b.decode("latin1", "ignore")

    rdr = _sniff_reader(txt)
    if not rdr.fieldnames:
        return []

    # build column map
    colmap: Dict[str, str] = {}
    for raw in rdr.fieldnames:
        k = _best_key(raw or "")
        if k and k not in colmap:
            colmap[k] = raw

    rows: List[Dict[str, Any]] = []
    for rec in rdr:
        rec = rec or {}
        # unified dict (lower keys)
        low = {(_norm(k)): (v.strip() if isinstance(v, str) else v) for k, v in rec.items()}

        def get_val(key: str) -> Any:
            rawkey = colmap.get(key)
            if rawkey is not None:
                return rec.get(rawkey)
            # fallback: direct by synonyms
            for cand in _SYNONYMS.get(key, []):
                v = rec.get(cand)
                if v not in (None, ""):
                    return v
            return None

        # location handling
        state = (get_val("state") or "") or ""
        city = (get_val("city") or "") or ""
        market = (get_val("market") or "") or ""
        if not (state and city):
            loc = get_val("location") or ""
            s, c, m = _split_location(str(loc))
            state = state or s
            city = city or c
            market = market or m or market

        # unit handling
        unit_text = get_val("unit_text") or ""
        if not unit_text:
            unit_text = _compose_unit_text({
                "unit_value": get_val("unit_value"),
                "unit_name": get_val("unit_name"),
                "unit_text": unit_text
            })

        # price handling
        price_text = get_val("price_text")
        if price_text in (None, ""):
            # look for numeric price columns named "price" / "amount" etc.
            for k in ["price","amount","cost","value","price_ngn","price_usd"]:
                v = rec.get(k)
                if v not in (None, ""):
                    price_text = v
                    break
        price_text = _price_to_text(price_text, get_val("currency"))
        
        num_ngn = rec.get("price_ngn")
        if num_ngn not in (None, ""):
            pt = str(num_ngn).strip()
            # Hard-stop: store exactly what CSV says, as NGN
            price_text = pt
            # And force currency to NGN so no FX happens downstream
            currency = "NGN"

        # minimal validity
        commodity = (get_val("commodity") or "").strip()
        state = (state or "").strip()
        city = (city or "").strip()
        if not (commodity and price_text and state and city):
            # skip incomplete
            continue

        rows.append({
            "commodity": commodity,
            "brand": (get_val("brand") or "").strip(),
            "price_text": price_text,
            "unit_text": unit_text,
            "state": state,
            "city": city,
            "market": (market or "").strip(),
            "date_uploaded": (get_val("date_uploaded") or "").strip(),
            "currency": currency,
        })
    return rows

# ------------------------ Preview (optional) ------------------------

class UploadSummary(BaseModel):
    accepted: int
    rejected: int
    errors: int
    items: List[Dict[str, Any]]

@router.post("/parse-csv", response_model=UploadSummary)
async def admin_parse_csv(file: UploadFile = File(...), use_ai: bool = Query(False)):
    """
    Optional preview of CSV using the same robust parser as upload-csv.
    If use_ai=True and no rows are found, we’ll try Gemini text extraction on the raw text.
    """
    b = await file.read()
    rows = _rows_from_csv_bytes(b)

    if not rows and use_ai and GEMINI_AVAILABLE:
        try:
            # let AI infer rows from raw text
            txt = b.decode("utf-8-sig", "ignore")
            rows = extract_rows_from_text(txt, hint="CSV-like content") or []
        except Exception:
            rows = []

    if not rows:
        raise HTTPException(400, "No valid rows found in CSV.")
    return UploadSummary(accepted=len(rows), rejected=0, errors=0, items=rows)

# ------------------------ Direct CSV commit (no preview required) ------------------------

@router.post("/upload-csv", response_model=UploadSummary)
async def admin_upload_csv(file: UploadFile = File(...)):
    """
    Commit CSV straight into the system. Uses robust header mapping.
    """
    b = await file.read()
    rows = _rows_from_csv_bytes(b)
    if not rows:
        raise HTTPException(400, "No valid rows found in CSV.")

    accepted = rejected = errors = 0
    items: List[Dict[str, Any]] = []
    with get_session() as sess:
        for r in rows:
            try:
                res = standardize_and_store(sess, r, source="csv")
                if res["status"] == "accepted":
                    accepted += 1
                    items.append(res["standardized"])
                else:
                    rejected += 1
            except Exception:
                errors += 1
    return UploadSummary(accepted=accepted, rejected=rejected, errors=errors, items=items)

# ------------------------ Manual (immediate) ------------------------

@router.post("/manual", response_model=UploadSummary)
def admin_manual(
    commodity: str = Form(...),
    brand: Optional[str] = Form(None),
    unit_value: float = Form(1.0),
    unit_name: str = Form("unit"),
    price_text: str = Form(...),
    state: str = Form(...),
    city: str = Form(...),
    market: Optional[str] = Form(None),
    date_uploaded: Optional[str] = Form(None),
):
    unit_text = f"{unit_value} {unit_name}".strip()
    row = {
        "commodity": commodity.strip(),
        "brand": (brand or "").strip(),
        "price_text": price_text.strip(),
        "unit_text": unit_text,
        "state": state.strip(),
        "city": city.strip(),
        "market": (market or "").strip(),
        "date_uploaded": (date_uploaded or "").strip(),
    }
    with get_session() as sess:
        res = standardize_and_store(sess, row, source="manual")
        items = [res["standardized"]] if res["status"] == "accepted" else []
        return UploadSummary(accepted=len(items), rejected=1 - len(items), errors=0, items=items)

# ------------------------ PDF & Image: preview then commit ------------------------

def _rows_from_pdf_bytes(b: bytes, use_ai: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if use_ai and GEMINI_AVAILABLE:
        rows = extract_rows_from_file("application/pdf", b, "tabular or messy PDF of prices") or []
    if not rows:
        try:
            with pdfplumber.open(io.BytesIO(b)) as pdf:
                for pg in pdf.pages:
                    for tbl in (pg.extract_tables() or []):
                        if not tbl or not tbl[0]:
                            continue
                        headers = [(h or "").strip().lower().replace(" ", "_") for h in tbl[0]]
                        for r in tbl[1:]:
                            rec = {headers[i]: (r[i] or "").strip() if i < len(r) else "" for i in range(len(headers))}
                            commodity = rec.get("commodity") or rec.get("item") or rec.get("product") or rec.get("name") or ""
                            price_text = rec.get("price_text") or rec.get("price") or rec.get("amount") or ""
                            state = rec.get("state") or ""
                            city = rec.get("city") or ""
                            if not (commodity and price_text and state and city):
                                continue
                            rows.append({
                                "commodity": commodity,
                                "brand": rec.get("brand",""),
                                "price_text": str(price_text),
                                "unit_text": rec.get("unit_text",""),
                                "state": state, "city": city,
                                "market": rec.get("market",""),
                                "date_uploaded": rec.get("date_uploaded",""),
                            })
        except Exception:
            pass
    return rows

def _rows_from_image_bytes(b: bytes, use_ai: bool) -> List[Dict[str, Any]]:
    if not (use_ai and GEMINI_AVAILABLE):
        return []
    try:
        img = Image.open(io.BytesIO(b)).convert("RGB")
        buf = io.BytesIO(); img.save(buf, format="PNG"); b = buf.getvalue()
    except Exception:
        pass
    return extract_rows_from_file("image/png", b, "poster/receipt/board with prices and locations") or []

class CommitPayload(BaseModel):
    rows: List[Dict[str, Any]]
    source: Optional[str] = "preview"

@router.post("/parse-pdf", response_model=UploadSummary)
async def admin_parse_pdf(file: UploadFile = File(...), use_ai: bool = Query(True)):
    b = await file.read()
    rows = _rows_from_pdf_bytes(b, use_ai)
    if not rows:
        raise HTTPException(400, "No rows found in PDF.")
    return UploadSummary(accepted=len(rows), rejected=0, errors=0, items=rows)

@router.post("/parse-image", response_model=UploadSummary)
async def admin_parse_image(file: UploadFile = File(...), use_ai: bool = Query(True)):
    b = await file.read()
    rows = _rows_from_image_bytes(b, use_ai)
    if not rows:
        raise HTTPException(400, "No rows found in image.")
    return UploadSummary(accepted=len(rows), rejected=0, errors=0, items=rows)

@router.post("/commit", response_model=UploadSummary)
def admin_commit(payload: CommitPayload):
    if not payload.rows:
        return UploadSummary(accepted=0, rejected=0, errors=0, items=[])
    accepted = rejected = errors = 0
    items: List[Dict[str, Any]] = []
    with get_session() as sess:
        for r in payload.rows:
            try:
                res = standardize_and_store(sess, r, source=payload.source or "preview")
                if res["status"] == "accepted":
                    accepted += 1
                    items.append(res["standardized"])
                else:
                    rejected += 1
            except Exception:
                errors += 1
    return UploadSummary(accepted=accepted, rejected=rejected, errors=errors, items=items)
