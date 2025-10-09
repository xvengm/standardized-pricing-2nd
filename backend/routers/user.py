# routers/user.py
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException
from sqlmodel import select
from sqlalchemy import func, distinct
import numpy as np
from datetime import datetime, timezone, timedelta

from backend.database import get_session
from backend.models import MainPrice, AggregateDaily, Locations
from backend.schemas import PriceResponse
from backend.utils import normalize_market
from backend.ai import GEMINI_AVAILABLE

router = APIRouter()

@router.get("/commodities")
def get_commodities(
    state: Optional[str] = None,
    city: Optional[str] = None,
    market: Optional[str] = None,
):
    """
    Get list of available (unique) commodities.
    Optional filters make it location-specific without breaking old callers.
    """
    with get_session() as sess:
        stmt = (
            select(
                MainPrice.commodity_lc,                # for deduping
                func.min(MainPrice.commodity).label("disp_name")  # nice display text
            )
            .group_by(MainPrice.commodity_lc)
        )

        if state:
            stmt = stmt.where(MainPrice.state == state.strip())
        if city:
            stmt = stmt.where(MainPrice.city == city.strip())
        if market is not None:
            market_q = normalize_market(market)
            stmt = stmt.where(
                (MainPrice.market == market_q)
                if market_q is not None
                else MainPrice.market.is_(None)
            )

        # stable, human-friendly ordering
        stmt = stmt.order_by(func.min(MainPrice.commodity))

        rows = sess.exec(stmt).all()

        # keep the old shape so the frontend doesn’t break
        return [{"name": (disp or lc.title()), "unit": "kg"} for lc, disp in rows]


@router.get("/brands")
def get_brands(commodity: str):
    """Get available brands for a commodity"""
    with get_session() as sess:
        brands = sess.exec(
            select(distinct(MainPrice.brand)).where(
                MainPrice.commodity_lc == commodity.lower(),
                MainPrice.brand.is_not(None)
            ).order_by(MainPrice.brand)
        ).all()
        return {"brands": [b for b in brands if b]}

@router.get("/units")
def get_units(commodity: str):
    """Get available units for a commodity"""
    with get_session() as sess:
        stmt = (
            select(MainPrice.unit_value, MainPrice.unit_name)
            .where(
                MainPrice.commodity_lc == commodity.lower(),
                MainPrice.unit_value.is_not(None),
                MainPrice.unit_name.is_not(None),
            )
            .distinct()
            .order_by(MainPrice.unit_value, MainPrice.unit_name)  # optional but nice
        )
        rows = sess.exec(stmt).all()

    return {"units": [{"unit_value": v, "unit_name": n} for (v, n) in rows]}


@router.get("/latest-price")
def get_latest_price(
    commodity: str, 
    state: str, 
    city: str, 
    market: Optional[str] = None, 
    brand: Optional[str] = None,
    sku: Optional[str] = None
):
    """Get latest price for specific commodity and location"""
    commodity_q = commodity.strip().lower()
    brand_q = (brand or "").strip().lower() or None
    state_q = state.strip()
    city_q = city.strip()
    market_q = normalize_market(market)

    with get_session() as sess:
        stmt = select(MainPrice).where(
            MainPrice.commodity_lc == commodity_q,
            MainPrice.state == state_q,
            MainPrice.city == city_q,
            (MainPrice.market == market_q) if market_q is not None else MainPrice.market.is_(None)
        )
        if brand_q:
            stmt = stmt.where(MainPrice.brand_lc == brand_q)
        
        row = sess.exec(stmt.order_by(MainPrice.processed_at.desc()).limit(1)).first()

        if not row:
            return {"price_ngn": None, "commodity": commodity, "message": "No data found"}
        
        return {
            "price_ngn": row.price_ngn,
            "commodity": row.commodity,
            "brand": row.brand,
            "location": {"state": row.state, "city": row.city, "market": row.market or ""},
            "unit": {"value": row.unit_value, "unit": row.unit_name},
            "effective_date": row.effective_date.isoformat(),
            "processed_at": row.processed_at.isoformat()
        }

@router.get("/prices")
def get_historical_prices(
    commodity: str,
    state: Optional[str] = None,
    city: Optional[str] = None,
    market: Optional[str] = None,
    limit: int = 100,
    order: str = "desc"
):
    """Get historical prices with filtering"""
    commodity_q = commodity.strip().lower()
    
    with get_session() as sess:
        stmt = select(MainPrice).where(MainPrice.commodity_lc == commodity_q)
        
        if state:
            stmt = stmt.where(MainPrice.state == state.strip())
        if city:
            stmt = stmt.where(MainPrice.city == city.strip())
        if market:
            market_q = normalize_market(market)
            stmt = stmt.where(
                (MainPrice.market == market_q) if market_q is not None else MainPrice.market.is_(None)
            )
        
        if order == "desc":
            stmt = stmt.order_by(MainPrice.effective_date.desc())
        else:
            stmt = stmt.order_by(MainPrice.effective_date.asc())
            
        stmt = stmt.limit(limit)
        
        rows = sess.exec(stmt).all()
        
        items = []
        for row in rows:
            items.append({
                "price_ngn": row.price_ngn,
                "commodity": row.commodity,
                "brand": row.brand,
                "location": {"state": row.state, "city": row.city, "market": row.market or ""},
                "unit": {"value": row.unit_value, "unit": row.unit_name},
                "effective_date": row.effective_date.isoformat(),
                "date": row.effective_date.isoformat(),  # alias for compatibility
                "processed_at": row.processed_at.isoformat()
            })
        
        return {"items": items}

@router.get("/average-price")
def average_price(commodity: str, state: str, city: str, market: Optional[str] = None, day: Optional[str] = None):
    commodity_q = (commodity or "").strip().lower()
    state_q = (state or "").strip()
    city_q = (city or "").strip()
    market_q = normalize_market(market)
    from datetime import datetime, timezone
    d = day or datetime.now(timezone.utc).date().isoformat()
    key = f"{commodity}|{state}|{city}|{market_q or ''}|{d}"

    with get_session() as sess:
        agg = sess.get(AggregateDaily, key)
        if agg:
            return {"commodity": commodity, "state": state, "city": city, "market": market_q, "date": d,
                    "median_price_ngn": agg.median_price_ngn, "n": agg.n}

        vals = sess.exec(
            select(MainPrice.price_ngn).where(
                MainPrice.commodity_lc == commodity_q,
                MainPrice.state == state_q, MainPrice.city == city_q,
                (MainPrice.market == market_q) if market_q is not None else MainPrice.market.is_(None)
            )
        )
        # Make robust across SQLAlchemy versions
        if hasattr(vals, "scalars"):
            values = list(vals.scalars().all())
        else:
            values = list(vals.all())

        if not values:
            return {"commodity": commodity, "state": state, "city": city, "market": market_q, "median_price_ngn": None, "n": 0}
        return {"commodity": commodity, "state": state, "city": city, "market": market_q,
                "median_price_ngn": float(np.median(sorted(values))), "n": len(values)}

@router.get("/locations")
def locations():
    out = {}
    with get_session() as sess:
        rows = sess.exec(select(Locations)).all()
        for r in rows:
            out.setdefault(r.state, {})
            out[r.state].setdefault(r.city, [])
            if r.market and r.market not in out[r.state][r.city]:
                out[r.state][r.city].append(r.market)
    return out

@router.post("/budget")
async def generate_budget(payload: Dict[str, Any]):
    """AI-powered budget generation"""
    budget = payload.get("budget", 0)
    state = payload.get("state", "")
    city = payload.get("city", "")
    market = payload.get("market", "")
    goods = payload.get("goods", [])
    
    if not budget or budget <= 0:
        raise HTTPException(400, "Invalid budget amount")
    if not goods:
        raise HTTPException(400, "No goods selected")
    
    market_q = normalize_market(market) if market else None
    
    with get_session() as sess:
        items = []
        total_cost = 0
        
        for good in goods:
            # Get latest price for this good in the specified location
            stmt = select(MainPrice).where(
                MainPrice.commodity_lc == good.lower(),
                MainPrice.state == state,
                MainPrice.city == city,
                (MainPrice.market == market_q) if market_q is not None else MainPrice.market.is_(None)
            ).order_by(MainPrice.processed_at.desc()).limit(1)
            
            row = sess.exec(stmt).first()
            
            if row:
                # Calculate optimal quantity within budget
                remaining_budget = budget - total_cost
                max_affordable = remaining_budget / row.price_ngn
                
                if max_affordable >= row.unit_value:
                    # Can afford at least one unit
                    quantity = min(max_affordable, row.unit_value * 3)  # Max 3 units
                    item_cost = quantity * (row.price_ngn / row.unit_value)
                    
                    items.append({
                        "commodity": row.commodity,
                        "quantity": round(quantity, 2),
                        "unit": row.unit_name,
                        "price_per_unit": round(row.price_ngn / row.unit_value, 2),
                        "total": round(item_cost, 2)
                    })
                    total_cost += item_cost
        
        remaining = budget - total_cost
        
        # Generate AI-powered note if available
        note = None
        if GEMINI_AVAILABLE and remaining > 0:
            try:
                import google.generativeai as genai
                model = genai.GenerativeModel("gemini-1.5-flash")
                prompt = f"""
                Budget: ₦{budget:,.2f}
                Spent: ₦{total_cost:,.2f}
                Remaining: ₦{remaining:,.2f}
                Items: {[item['commodity'] for item in items]}
                
                Provide a brief shopping tip or suggestion for the remaining budget in Nigerian context.
                Keep it under 50 words.
                """
                response = model.generate_content(prompt)
                note = response.text.strip()
            except:
                pass
        
        if not note and remaining > 1000:
            note = f"Consider adding vegetables or seasonings with your remaining ₦{remaining:,.2f}"
        
        return {
            "items": items,
            "total_cost": round(total_cost, 2),
            "remaining": round(remaining, 2),
            "note": note
        }
