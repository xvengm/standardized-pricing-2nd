# backend/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database import init_db
from backend.routers.admin import router as admin_router
from backend.routers.user  import router as user_router

app = FastAPI(title="Standardized Pricing API (AI)", version="4.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.on_event("startup")
def _startup():
    init_db()

app.include_router(admin_router, prefix="/admin", tags=["admin"])
app.include_router(user_router,  prefix="",       tags=["user"])

@app.get("/")
def health():
    from backend.database import DB_URL
    from backend.ai import GEMINI_AVAILABLE
    return {
        "service": "Standardized Pricing API (AI)",
        "db": DB_URL,
        "gemini_available": GEMINI_AVAILABLE
    }
