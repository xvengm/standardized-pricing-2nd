# budgeting.py
"""
AI Budget Planner microservice for Nigeria Standard Prices.

- Endpoint: POST /ai/budget-plan
- Reads current prices from your *existing* API (backed by your DB), so results
  always reflect the database.
- Uses Gemini 1.5 Flash (if GOOGLE_API_KEY is set) to propose quantities within
  the user's budget; otherwise falls back to a deterministic greedy allocator.

Front-end compatibility:
Returns:
{
  "plan": [
    {"item": "...", "quantity": 2, "unit": "bag", "estimated_price": 72000}
  ],
  "summary": {
    "initial_budget": 250000,
    "total_cost": 216000,
    "remaining_balance": 34000,
    "notes": "..."
  }
}

Run:
  uvicorn budgeting:app --host 0.0.0.0 --port 8001

Env:
  PRICE_API_BASE   (default: http://127.0.0.1:8000)
  GOOGLE_API_KEY   (optional: enables Gemini 1.5 Flash)
  BUDGET_LOC_JSON  (optional: JSON list of location dicts to try first)
"""

from __future__ import annotations

import os
import re
import json
import math
import asyncio
from typing import List, Dict, Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conlist

# --------------------------- Config ---------------------------

PRICE_API_BASE = os.getenv("PRICE_API_BASE", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_LOCATIONS = [
    {"state": "Lagos",  "city": "Ikeja",         "market": ""},
    {"state": "Rivers", "city": "Port Harcourt", "market": "Oil Mill Market"},
    {"state": "FCT",    "city": "Wuse",          "market": "Wuse Market"},
    {"state": "Oyo",    "city": "Ibadan",        "market": ""},
]
try:
    if os.getenv("BUDGET_LOC_JSON"):
        DEFAULT_LOCATIONS = json.loads(os.getenv("BUDGET_LOC_JSON"))
except Exception:
    pass

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # optional

# --------------------------- Models ---------------------------

class BudgetRequest(BaseModel):
    budget: float = Field(..., gt=0, description="Total budget in NGN")
    commodities: List[str] = Field(..., min_items=1, description="List of commodity names")
    # Optional hint: if user passes state/city/market, try those first
    state: Optional[str] = None
    city: Optional[str] = None
    market: Optional[str] = None

class PlanItem(BaseModel):
    item: str
    quantity: float
    unit: str
    estimated_price: int

class PlanSummary(BaseModel):
    initial_budget: int
    total_cost: int
    remaining_balance: int
    notes: Optional[str] = ""

class BudgetPlanResponse(BaseModel):
    plan: List[PlanItem]
    summary: PlanSummary

# --------------------------- App ---------------------------

app = FastAPI(title="AI Budget Planner", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------- Utilities ---------------------------

def naira_i(n: Optional[float]) -> int:
    try:
        return int(round(float(n)))
    except Exception:
        return 0

def unique_keep_order(seq: List[str]) -> List[str]:
    seen = set(); out = []
    for s in seq:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

def normalize_commodity(name: str) -> str:
    # Keep it simple; align with your API's capitalization rules
    if not name:
        return name
    return name.strip()

# ---------------------- Price Catalog Fetch ----------------------

async def fetch_latest_for_loc(client: httpx.AsyncClient, commodity: str, loc: Dict[str,str]) -> Optional[Dict[str, Any]]:
    """Try /latest-price for a single location."""
    params = {
        "commodity": normalize_commodity(commodity),
        "state": loc.get("state", ""),
        "city": loc.get("city", ""),
        "market": loc.get("market", ""),
        "brand": "",
        "sku": "",
    }
    url = f"{PRICE_API_BASE}/latest-price"
    try:
        r = await client.get(url, params=params, timeout=15)
        if not r.is_success:
            return None
        j = r.json()
        if not j or j.get("price_ngn") is None:
            return None
        # Normalize fields
        unit_name = (j.get("unit") or {}).get("unit") or j.get("unit_name") or "unit"
        unit_value = (j.get("unit") or {}).get("value") or j.get("unit_value") or 1
        return {
            "commodity": j.get("commodity") or commodity,
            "unit_name": str(unit_name),
            "unit_value": float(unit_value),
            "unit_price": float(j["price_ngn"]),
            "state": (j.get("location") or {}).get("state") or loc.get("state"),
            "city": (j.get("location") or {}).get("city") or loc.get("city"),
            "market": (j.get("location") or {}).get("market") or loc.get("market") or "",
            "effective_date": j.get("effective_date") or j.get("as_of"),
        }
    except Exception:
        return None

async def get_price_row(client: httpx.AsyncClient, commodity: str, preferred: Optional[Dict[str,str]] = None) -> Optional[Dict[str,Any]]:
    """Try preferred location first, then configured fallbacks."""
    tries = []
    if preferred and (preferred.get("state") and preferred.get("city")):
        tries.append(preferred)
    tries.extend(DEFAULT_LOCATIONS)

    seen = set()
    for loc in tries:
        key = (loc.get("state"), loc.get("city"), loc.get("market", ""))
        if key in seen:
            continue
        seen.add(key)
        row = await fetch_latest_for_loc(client, commodity, loc)
        if row:
            return row
    return None

async def build_price_book(commodities: List[str], preferred: Optional[Dict[str,str]]) -> Dict[str, Dict[str,Any]]:
    """Map commodity -> {unit_name, unit_value, unit_price, source location}."""
    async with httpx.AsyncClient() as client:
        rows = await asyncio.gather(*[get_price_row(client, c, preferred) for c in commodities])
    book: Dict[str, Dict[str,Any]] = {}
    for c, r in zip(commodities, rows):
        if r:
            book[c] = r
    return book

# ---------------------- Heuristic Allocator ----------------------

def greedy_allocate(budget: int, book: Dict[str, Dict[str,Any]]) -> List[PlanItem]:
    """
    Simple, robust baseline: buy at least 1 unit of each affordable item,
    then keep adding one unit of the cheapest available item until budget is exhausted.
    """
    items = []
    remaining = budget

    # canonical list so UI order is stable
    names = list(book.keys())

    # First pass: 1 unit each if possible
    for name in names:
        unit_price = naira_i(book[name]["unit_price"])
        if unit_price <= 0 or unit_price > remaining:
            continue
        items.append(PlanItem(
            item=name,
            quantity=1,
            unit=book[name]["unit_name"],
            estimated_price=unit_price
        ))
        remaining -= unit_price

    # Build a price list and repeatedly add cheapest possible
    price_list = [(name, naira_i(book[name]["unit_price"])) for name in names if naira_i(book[name]["unit_price"]) > 0]
    price_list.sort(key=lambda x: x[1])

    while price_list and remaining >= price_list[0][1]:
        for name, price in price_list:
            if price <= remaining:
                # find existing line
                for it in items:
                    if it.item == name:
                        it.quantity += 1
                        it.estimated_price += price
                        remaining -= price
                        break
                else:
                    items.append(PlanItem(
                        item=name,
                        quantity=1,
                        unit=book[name]["unit_name"],
                        estimated_price=price
                    ))
                    remaining -= price
            if remaining < price_list[0][1]:
                break

    return items

# ---------------------- Gemini Allocator (optional) ----------------------

def gemini_available() -> bool:
    return bool(GOOGLE_API_KEY)

def _extract_json_block(text: str) -> Optional[dict]:
    """Extract first top-level JSON object from a string."""
    if not text:
        return None
    # Try fenced block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

async def gemini_allocate(budget: int, book: Dict[str, Dict[str,Any]]) -> Optional[List[PlanItem]]:
    """
    Ask Gemini 1.5 Flash for a JSON plan. Returns None if model or parsing fails.
    """
    if not gemini_available():
        return None

    try:
        # Lazy import so the file works without the package when Gemini is disabled
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=GOOGLE_API_KEY)

        # Prepare compact price book for the prompt
        price_rows = []
        for name, row in book.items():
            price_rows.append({
                "item": name,
                "unit": row["unit_name"],
                "unit_value": row["unit_value"],
                "unit_price": naira_i(row["unit_price"]),
            })

        system_msg = (
            "You are a budgeting assistant for Nigerian market prices. "
            "Given a total budget (NGN) and a list of items with unit prices, "
            "return a JSON object with an array 'plan'. Each element must have: "
            "item (string), quantity (integer >= 0), unit (string), estimated_price (integer NGN for that item). "
            "Do not exceed the budget. Try to include at least one unit of each affordable item. "
            "Reply ONLY with JSON."
        )
        user_payload = {
            "budget": budget,
            "price_book": price_rows
        }

        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = await asyncio.to_thread(
            model.generate_content,
            [{"role": "system", "parts": [system_msg]},
             {"role": "user",   "parts": [json.dumps(user_payload, ensure_ascii=False)]}],
            generation_config={"temperature": 0.2}
        )

        text = ""
        try:
            text = resp.text  # type: ignore
        except Exception:
            # fallback: concatenate candidates
            if hasattr(resp, "candidates") and resp.candidates:  # type: ignore
                text = " ".join(getattr(p, "content", "") for p in resp.candidates)  # type: ignore

        data = _extract_json_block(text)
        if not data or "plan" not in data or not isinstance(data["plan"], list):
            return None

        out: List[PlanItem] = []
        total = 0
        for row in data["plan"]:
            name = str(row.get("item") or "").strip()
            qty = float(row.get("quantity") or 0)
            unit = str(row.get("unit") or book.get(name, {}).get("unit_name") or "unit")
            # Recompute price defensively from our unit_price
            unit_price = naira_i((book.get(name) or {}).get("unit_price"))
            est = naira_i(qty * unit_price)
            if name and qty > 0 and unit_price > 0:
                out.append(PlanItem(item=name, quantity=qty, unit=unit, estimated_price=est))
                total += est

        if total <= budget and out:
            return out
        return None
    except Exception:
        return None

# --------------------------- API Routes ---------------------------

@app.get("/")
async def root():
    return {
        "service": "budgeting-ai",
        "gemini_available": gemini_available(),
        "price_api_base": PRICE_API_BASE,
        "locations": DEFAULT_LOCATIONS,
    }

@app.post("/ai/budget-plan", response_model=BudgetPlanResponse)
async def budget_plan(req: BudgetRequest):
    # Build canonical commodity list
    commodities = unique_keep_order([normalize_commodity(c) for c in req.commodities if str(c).strip()])
    if not commodities:
        raise HTTPException(status_code=400, detail="No commodities provided.")

    # Prefer user-passed location first
    preferred = None
    if req.state and req.city:
        preferred = {"state": req.state, "city": req.city, "market": req.market or ""}

    # 1) Price book (pulls from your DB-backed API)
    book = await build_price_book(commodities, preferred)
    if not book:
        raise HTTPException(status_code=404, detail="No prices found for the requested items.")

    # Remove items with no valid unit_price
    book = {k: v for k, v in book.items() if v.get("unit_price", 0) > 0}
    if not book:
        raise HTTPException(status_code=404, detail="No priced items available.")

    budget_i = naira_i(req.budget)

    # 2) Try Gemini (if configured)
    plan_items: Optional[List[PlanItem]] = await gemini_allocate(budget_i, book)

    # 3) Fallback to greedy
    if not plan_items:
        plan_items = greedy_allocate(budget_i, book)

    # 4) Summaries
    total_cost = sum(naira_i(p.estimated_price) for p in plan_items)
    remaining = max(0, budget_i - total_cost)

    # Helpful note for UI
    missing = [c for c in commodities if c not in book]
    notes = []
    if gemini_available():
        notes.append("Optimized with Gemini 1.5 Flash" if plan_items else "Gemini used (fallback applied)")
    else:
        notes.append("AI not configured — heuristic allocation used")
    if missing:
        notes.append(f"No live price for: {', '.join(missing)}")

    return BudgetPlanResponse(
        plan=plan_items,
        summary=PlanSummary(
            initial_budget=budget_i,
            total_cost=total_cost,
            remaining_balance=remaining,
            notes="; ".join(notes)
        )
    )

# --------------------------- Main ---------------------------

if __name__ == "__main__":
    # Allows: python budgeting.py
    import uvicorn
    uvicorn.run("budgeting:app", host="0.0.0.0", port=int(os.getenv("PORT", "8001")), reload=False)
