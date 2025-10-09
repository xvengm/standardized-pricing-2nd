# services/ingest.py
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np
from sqlmodel import Session, select
from sqlalchemy import func

from backend.models import StagingPrice, MainPrice, AggregateDaily
from backend.validators import validate_or_upsert_location
from backend.utils import (parse_price_and_currency, to_ngn, clean_unit, classify_category,
                           iqr_outlier, normalize_market, fetch_scalar_list)

def standardize_and_store(sess: Session, row: Dict[str, Any], source: str) -> Dict[str, Any]:
    # --- Extract & basic required checks
    commodity = (row.get("commodity") or "").strip()
    brand = (row.get("brand") or "").strip() or None
    unit_text = (row.get("unit_text") or "").strip() or None
    state = (row.get("state") or "").strip()
    city = (row.get("city") or "").strip()
    market = normalize_market(row.get("market"))
    date_up = row.get("date_uploaded")
    try:
        date_uploaded = datetime.fromisoformat(date_up).date() if date_up else datetime.now(timezone.utc).date()
    except Exception:
        date_uploaded = datetime.now(timezone.utc).date()

    if not (commodity and state and city):
        return {"status": "rejected", "reason": "missing commodity/state/city", "row": row}

    # --- Location presence (auto-upsert)
    validate_or_upsert_location(sess, state, city, market)

    # --- Price parsing / currency
    try:
        forced_currency = (row.get("currency") or "").strip() or None
        if forced_currency:
            price_value = float(str(row.get("price_text")).replace(",", "").replace(" ", ""))
            currency = forced_currency
        else:
            price_value, currency = parse_price_and_currency(row.get("price_text") or "")

    except Exception as e:
        return {"status": "rejected", "reason": f"bad price: {e}", "row": row}

    # --- Outlier check vs historical (same commodity/location)
    hist_vals = fetch_scalar_list(sess, select(StagingPrice.price_value).where(
        func.lower(StagingPrice.commodity) == commodity.lower(),
        StagingPrice.state == state,
        StagingPrice.city == city,
        (StagingPrice.market == market) if market is not None else StagingPrice.market.is_(None)
    ))
    outlier = iqr_outlier(price_value, hist_vals)

    # --- Units
    uval, uname = clean_unit(unit_text)

    # --- Write staging
    staging = StagingPrice(
        commodity=commodity, brand=brand, raw_price=row.get("price_text") or "",
        price_value=price_value, currency_original=currency, unit_text=unit_text,
        unit_value=uval, unit_name=uname, state=state, city=city, market=market,
        date_uploaded=date_uploaded, source_type=source,
        valid=(not outlier), is_outlier=outlier,
        note=None if not outlier else "IQR outlier",
        commodity_lc=commodity.lower(), brand_lc=(brand.lower() if brand else None),
    )
    sess.add(staging); sess.commit(); sess.refresh(staging)

    if outlier:
        return {"status":"rejected","reason":"outlier","staging_id":staging.id,"row":row}

    # --- Transform to main
    cat = classify_category(f"{commodity} {brand or ''}")
    # Reject anything that is not one of our allowed commodities (classifies as "other")
    if cat == "other":
        # mark the staging row as rejected for audit visibility, then stop
        staging.status = "rejected"
        staging.note = ((staging.note or "") + " | unsupported commodity").strip(" |")
        sess.add(staging)
        sess.commit()
        sess.refresh(staging)
        return {
            "status": "rejected",
            "reason": "unsupported commodity",
            "staging_id": staging.id,
            "row": row,
        }

    price_ngn = to_ngn(price_value, currency)
    mp = MainPrice(
        commodity=commodity, brand=brand, category=cat,
        price_ngn=price_ngn, currency_original=currency,
        unit_value=uval, unit_name=uname, state=state, city=city, market=market,
        effective_date=date_uploaded,
        commodity_lc=commodity.lower(), brand_lc=(brand.lower() if brand else None),
    )
    sess.add(mp); sess.commit(); sess.refresh(mp)

    # --- Aggregate daily median for that location/commodity/day
    day = mp.effective_date.isoformat()
    key = f"{mp.commodity}|{mp.state}|{mp.city}|{mp.market or ''}|{day}"
    vals = fetch_scalar_list(sess, select(MainPrice.price_ngn).where(
        MainPrice.commodity_lc == mp.commodity_lc,
        MainPrice.state == mp.state, MainPrice.city == mp.city,
        (MainPrice.market == mp.market) if mp.market is not None else MainPrice.market.is_(None),
        MainPrice.effective_date == mp.effective_date
    ))
    median = float(np.median(vals)) if vals else mp.price_ngn

    old = sess.get(AggregateDaily, key)
    if old:
        sess.delete(old); sess.commit()
    sess.add(AggregateDaily(
        id=key, commodity=mp.commodity, state=mp.state, city=mp.city, market=mp.market,
        day=day, median_price_ngn=round(median,2), n=len(vals)
    )); sess.commit()

    return {
        "status":"accepted",
        "staging_id": staging.id,
        "main_id": mp.id,
        "standardized": {
            "commodity": mp.commodity, "brand": mp.brand, "category": mp.category,
            "price_ngn": mp.price_ngn, "currency_original": mp.currency_original,
            "unit_value": mp.unit_value, "unit_name": mp.unit_name,
            "state": mp.state, "city": mp.city, "market": mp.market,
            "effective_date": day
        }
    }
