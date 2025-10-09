# models.py
from datetime import datetime, date, timezone
from typing import Optional
from sqlmodel import SQLModel, Field as SQLField, Index

class Locations(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    state: str = SQLField(index=True)
    city: str = SQLField(index=True)
    market: Optional[str] = SQLField(default=None, index=True)
    __table_args__ = (Index("idx_locations_unique", "state", "city", "market", unique=True),)

class StagingPrice(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    commodity: str = SQLField(index=True)
    brand: Optional[str] = SQLField(default=None, index=True)
    raw_price: str
    price_value: float
    currency_original: str = "NGN"
    unit_text: Optional[str] = None
    unit_value: float = 1.0
    unit_name: str = "unit"
    state: str = SQLField(index=True)
    city: str = SQLField(index=True)
    market: Optional[str] = SQLField(default=None, index=True)
    date_uploaded: date
    source_type: str = "manual"
    note: Optional[str] = None
    valid: bool = True
    is_outlier: bool = False
    created_at: datetime = SQLField(default_factory=lambda: datetime.now(timezone.utc))
    # normalized
    commodity_lc: str = SQLField(default="", index=True)
    brand_lc: Optional[str] = SQLField(default=None, index=True)

class MainPrice(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    commodity: str = SQLField(index=True)
    brand: Optional[str] = SQLField(default=None, index=True)
    category: str = SQLField(index=True)
    price_ngn: float
    currency_original: str
    unit_value: float
    unit_name: str
    state: str = SQLField(index=True)
    city: str = SQLField(index=True)
    market: Optional[str] = SQLField(default=None, index=True)
    effective_date: date = SQLField(index=True)
    processed_at: datetime = SQLField(default_factory=lambda: datetime.now(timezone.utc))
    # normalized
    commodity_lc: str = SQLField(default="", index=True)
    brand_lc: Optional[str] = SQLField(default=None, index=True)

class AggregateDaily(SQLModel, table=True):
    id: str = SQLField(primary_key=True)  # commodity|state|city|market|YYYY-MM-DD
    commodity: str = SQLField(index=True)
    state: str = SQLField(index=True)
    city: str = SQLField(index=True)
    market: Optional[str] = SQLField(default=None, index=True)
    day: str = SQLField(index=True)
    median_price_ngn: float
    n: int
