# validators.py
from sqlmodel import Session, select
from backend.models import Locations
from backend.utils import normalize_market

def validate_or_upsert_location(sess: Session, state: str, city: str, market: str | None) -> None:
    state = (state or "").strip()
    city  = (city  or "").strip()
    market = normalize_market(market)
    exists = sess.exec(
        select(Locations).where(
            Locations.state == state,
            Locations.city == city,
            (Locations.market == market) if market is not None else Locations.market.is_(None)
        )
    ).first()
    if not exists:
        sess.add(Locations(state=state, city=city, market=market)); sess.commit()
