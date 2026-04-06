"""
Email Sender Microservice — FastAPI wrapper around sender.py
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from sender import send_email, get_daily_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Email Sender service starting up")
    yield
    logger.info("Email Sender service shutting down")


app = FastAPI(
    title="Email Sender Service",
    description="Sends cold emails with resume attachment via SMTP (Gmail / AWS SES)",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Pydantic Models ──────────────────────────────────────────
class SendRequest(BaseModel):
    to_email: str = Field(..., examples=["hr@techstartup.com"])
    subject: str = Field(..., examples=["Backend & AI Developer — Open to Opportunities"])
    body: str = Field(..., examples=["Hi Priya, ..."])
    resume_path: str = Field("/app/assets/resume.pdf")
    dry_run: Optional[bool] = None  # None = use env default


class SendResponse(BaseModel):
    success: bool
    message: str
    timestamp: str


class StatsResponse(BaseModel):
    date: str
    sent_today: int
    remaining: int
    limit: int


class StatusResponse(BaseModel):
    status: str


# ── Routes ───────────────────────────────────────────────────
@app.get("/health", response_model=StatusResponse)
async def health():
    return {"status": "ok"}


@app.post("/send-email", response_model=SendResponse)
async def send(req: SendRequest):
    """Send an email with an optional resume attachment."""
    result = send_email(
        to_email=req.to_email,
        subject=req.subject,
        body=req.body,
        resume_path=req.resume_path,
        dry_run=req.dry_run,
    )
    return SendResponse(**result)


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Get today's email send statistics."""
    return get_daily_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
