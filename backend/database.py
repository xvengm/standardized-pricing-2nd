# backend/database.py
import os
from typing import List, Tuple
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlmodel import SQLModel, create_engine, Session, select
from sqlalchemy import func, text

# ✅ package-aware imports
from backend.models import Locations, StagingPrice, MainPrice, AggregateDaily

# -------- Env & engine --------
load_dotenv()

DB_URL = os.getenv("PRICING_DB_URL", "sqlite:///data/pricing.db")

# sqlite needs check_same_thread=False for FastAPI multithreading
connect_args = {"check_same_thread": False} if DB_URL.startswith("sqlite") else {}
engine = create_engine(DB_URL, echo=False, connect_args=connect_args)

# Optional AI flag (used by /health)
try:
    import google.generativeai as _genai  # type: ignore
    if os.getenv("GOOGLE_API_KEY"):
        _genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        GEMINI_AVAILABLE = True
    else:
        GEMINI_AVAILABLE = False
except Exception:
    GEMINI_AVAILABLE = False


# -------- Session helpers --------
def get_session() -> Session:
    """Simple helper to open a Session; prefer `with Session(engine) as sess` in code."""
    return Session(engine)


@contextmanager
def session_scope():
    """Optional context manager if you want auto-commit/rollback behavior."""
    sess = Session(engine)
    try:
        yield sess
        sess.commit()
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


# -------- Migrations (safe/idempotent) --------
def _sqlite_table_exists(conn, table: str) -> bool:
    row = conn.exec_driver_sql(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=:t",
        {"t": table},
    ).fetchone()
    return row is not None


def _sqlite_columns(conn, table: str) -> List[str]:
    if not _sqlite_table_exists(conn, table):
        return []
    rows = conn.exec_driver_sql(f"PRAGMA table_info({table});").fetchall()
    # row tuple: (cid, name, type, notnull, dflt_value, pk)
    return [r[1] for r in rows]


def _add_column(conn, table: str, coldef: str):
    conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {coldef};")


def migrate_sqlite_schema():
    """
    Create tables if missing and add any new columns introduced by newer app versions.
    Safe to run on every startup.
    """
    # Ensure base tables exist
    SQLModel.metadata.create_all(engine)

    if not DB_URL.startswith("sqlite"):
        # in Postgres/MySQL you'd use Alembic; keep sqlite pragmatic here
        return

    # Add missing columns on existing SQLite DBs
    with engine.connect() as conn:
        # ---- MAINPRICE ----
        mp_cols = set(_sqlite_columns(conn, "mainprice"))
        needed_mp = {
            "commodity": "commodity TEXT",
            "commodity_lc": "commodity_lc TEXT",
            "brand": "brand TEXT",
            "brand_lc": "brand_lc TEXT",
            "category": "category TEXT",
            "price_ngn": "price_ngn REAL",
            "currency_original": "currency_original TEXT",
            "unit_value": "unit_value REAL",
            "unit_name": "unit_name TEXT",
            "state": "state TEXT",
            "state_lc": "state_lc TEXT",
            "city": "city TEXT",
            "city_lc": "city_lc TEXT",
            "market": "market TEXT",
            "effective_date": "effective_date DATE",
            "processed_at": "processed_at TIMESTAMP",
        }
        for name, coldef in needed_mp.items():
            if name not in mp_cols:
                _add_column(conn, "mainprice", coldef)

        # ---- STAGINGPRICE ----
        sp_cols = set(_sqlite_columns(conn, "stagingprice"))
        needed_sp = {
            "commodity": "commodity TEXT",
            "brand": "brand TEXT",
            "raw_price": "raw_price TEXT",
            "price_value": "price_value REAL",
            "currency_original": "currency_original TEXT",
            "unit_text": "unit_text TEXT",
            "unit_value": "unit_value REAL",
            "unit_name": "unit_name TEXT",
            "state": "state TEXT",
            "city": "city TEXT",
            "market": "market TEXT",
            "date_uploaded": "date_uploaded DATE",
            "source_type": "source_type TEXT",
            "note": "note TEXT",
            "valid": "valid BOOLEAN",
            "is_outlier": "is_outlier BOOLEAN",
            "created_at": "created_at TIMESTAMP",
        }
        for name, coldef in needed_sp.items():
            if name not in sp_cols:
                _add_column(conn, "stagingprice", coldef)

        # ---- AGGREGATEDAILY ----
        ad_cols = set(_sqlite_columns(conn, "aggregatedaily"))
        needed_ad = {
            "commodity": "commodity TEXT",
            "state": "state TEXT",
            "city": "city TEXT",
            "market": "market TEXT",
            "day": "day TEXT",
            "median_price_ngn": "median_price_ngn REAL",
            "n": "n INTEGER",
        }
        for name, coldef in needed_ad.items():
            if name not in ad_cols:
                _add_column(conn, "aggregatedaily", coldef)


# -------- Seeding --------
def seed_locations_if_empty():
    """
    Seed a small, useful set of default locations if the Locations table is empty.
    """
    with Session(engine) as sess:
        total = sess.exec(select(func.count(Locations.id))).one()
        if total and int(total) > 0:
            return

        seeds: List[Tuple[str, str, str]] = [
            ("Lagos", "Ikeja", "Computer Village"),
            ("Lagos", "Lekki", "Ajah Market"),
            ("Lagos", "Agege", "Agege Market"),
            ("Abuja", "Wuse", "Wuse Market"),
            ("Abuja", "Garki", "Area 1 Market"),
            ("Rivers", "Port Harcourt", "Oil Mill Market"),
            ("Rivers", "Port Harcourt", "Next Superstore"),
        ]
        for state, city, market in seeds:
            sess.add(Locations(state=state, city=city, market=market))
        sess.commit()


# -------- Public initializer --------
def init_db():
    """
    Call this at application startup.
    """
    # Create tables (no-op if already exist)
    SQLModel.metadata.create_all(engine)
    # Bring older SQLite DBs up-to-date
    migrate_sqlite_schema()
    # Seed default locations if empty
    seed_locations_if_empty()
