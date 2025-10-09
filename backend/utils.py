# utils.py
import re
import numpy as np
from typing import Optional, Tuple
from sqlmodel import Session

# currency tables
CURRENCY_ALIASES = {"₦":"NGN","ngn":"NGN","naira":"NGN","n":"NGN", "$":"USD","usd":"USD","dollar":"USD"}
CURRENCY_TO_NGN = {"NGN":1.0, "USD":1600.0}

UNIT_CANON = {"kg","g","tonne","bag","crate","liter","litre","l","ml","unit"}

def parse_price_and_currency(raw: str) -> Tuple[float,str]:
    if not raw:
        raise ValueError("empty price")
    m = re.search(r"\d{1,3}(?:[,\s]?\d{3})*(?:\.\d+)?", raw)
    if not m:
        raise ValueError(f"no numeric value in '{raw}'")
    value = float(m.group(0).replace(",","").replace(" ",""))
    r = raw.lower()
    curr = "NGN"
    for k in ["usd","$","ngn","naira","₦"]:
        if k in r:
            curr = CURRENCY_ALIASES.get(k, "USD" if k=="$" else k.upper())
            break
    return value, curr

def to_ngn(value: float, currency: str) -> float:
    fx = CURRENCY_TO_NGN.get(currency.upper())
    if fx is None:
        raise ValueError(f"unsupported currency {currency}")
    return round(value*fx, 2)

def clean_unit(unit_text: Optional[str]) -> Tuple[float,str]:
    if not unit_text:
        return 1.0, "unit"
    t = unit_text.strip().lower()
    m = re.search(r"(?:\d+(?:\.\d+)?)", t)
    qty = float(m.group(0)) if m else 1.0
    u = "unit"
    for cand in UNIT_CANON:
        if re.search(rf"\b{re.escape(cand)}\b", t):
            u = "liter" if cand in {"litre","l"} else cand
            break
    if u == "g":  return round(qty/1000.0, 3), "kg"
    if u == "ml": return round(qty/1000.0, 3), "liter"
    return qty, u

def classify_category(commodity: str) -> str:
    """
    Map many real-world spellings/brands/cases to one lowercase canonical name.
    Anything not matched returns 'other' so the caller can reject it.
    """
    s = (commodity or "").strip().lower()
    if not s:
        return "other"

    MAP = {
        # Food staples
        "rice": [
            "rice", "mama", "mama gold", "royal stallion", "stallion", "caprice",
            "long grain", "parboiled", "basmati", "jasmine", "ofada"
        ],
        "beans": [
            "beans", "oloyin", "brown beans", "white beans",
            "black eye", "black-eye", "blackeye"
        ],
        "garri": [
            "garri", "gari", "garry", "ijebu garri", "yellow garri", "white garri"
        ],
        "palm oil": [
            "palm oil", "palmoil", "palm-oil", "red oil"
        ],
        "ground nut oil": [
            # main term
            "ground nut oil",
    
            # close variations
            "granot oil", "granot-oil", "granote oil", "granote-oil",
            "groundnut oil", "groundnut-oil", "groundnutoil",
            "peanut oil", "peanut-oil", "peanutoil",
    
            # generic references people use
            "cooking oil", "frying oil", "edible oil", "kitchen oil",
            "veg oil", "veg. oil", "vegetable oil", "vegetable-oil", "vegetableoil",
            "salad oil", "table oil"
        ],

        "sugar": [
            "sugar", "granulated sugar", "refined sugar", "white sugar", "brown sugar"
        ],
        "flour": [
            "flour", "wheat flour", "baking flour", "all-purpose flour", "floor"  # common typo
        ],

        # Energy
        "petrol": [
            "petrol", "pms", "fuel", "gasoline", "premium motor spirit"
        ],
        "cooking gas": [
            "cooking gas", "lpg", "liquefied petroleum gas", "butane", "propane"
        ],

        # Others
        "cement": [
            "cement", "dangote", "lafarge", "bua", "elephant", "ashaka"
        ],
        "yam": [
            "yam", "yam tuber", "tuber of yam"
        ],
        "egg": [
            "egg", "eggs", "crate of egg", "crate of eggs", "egg crate"
        ],
    }

    for canon, keys in MAP.items():
        for k in keys:
            if k in s:
                return canon

    return "other"


def iqr_outlier(v: float, series):
    series = list(series)
    if len(series) < 5:
        return False
    q1, q3 = np.percentile(series, [25,75]); iqr = q3-q1
    low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
    return not (low <= v <= high)

def normalize_market(market: Optional[str]) -> Optional[str]:
    if market is None:
        return None
    m = market.strip()
    return None if m == "" else m

# --- Compatibility helper for SQLAlchemy/SQLModel result shapes
def fetch_scalar_list(session: Session, stmt):
    res = session.exec(stmt)
    # Some versions return ScalarResult (already scalarized)
    if hasattr(res, "all") and not hasattr(res, "scalars"):
        return list(res.all())
    # Others return Result -> need .scalars()
    if hasattr(res, "scalars"):
        return list(res.scalars().all())
    try:
        return list(res.all())
    except Exception:
        return list(res)
