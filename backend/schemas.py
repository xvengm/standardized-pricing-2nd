# schemas.py
from datetime import datetime, date
from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field

class ManualEntry(BaseModel):
    commodity: str
    brand: Optional[str] = None
    unit_value: float = Field(1.0, ge=0.001)
    unit_name: str = Field("unit")
    price_text: str
    currency: Optional[str] = None
    state: str
    city: str
    market: Optional[str] = None
    date_uploaded: Optional[date] = None

class UploadSummary(BaseModel):
    accepted: int
    rejected: int
    errors: int
    items: List[Dict[str, Any]]

class PriceResponse(BaseModel):
    commodity: str
    brand: Optional[str]
    location: Dict[str,str]
    price_ngn: Optional[float]
    unit: Dict[str,Any]
    as_of: Optional[datetime] = None
    effective_date: Optional[date] = None
    category: Optional[str] = None

class LocationUpsert(BaseModel):
    state: str
    city: str
    market: Optional[str] = None
