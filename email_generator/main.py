"""
Email Generator Microservice — FastAPI wrapper around generator.py
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from generator import generate_email

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Email Generator service starting up")
    yield
    logger.info("Email Generator service shutting down")


app = FastAPI(
    title="Email Generator Service",
    description="Generates personalized cold emails using a local LLM (Ollama) or cloud providers",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Pydantic Models ──────────────────────────────────────────
class ContactInput(BaseModel):
    name: str = Field(..., examples=["Priya Sharma"])
    email: str = Field(..., examples=["priya@startup.com"])
    company: str = Field(..., examples=["TechStartup AI"])
    title: str = Field("Recruiter", examples=["HR Manager"])
    linkedin_url: Optional[str] = None
    company_description: Optional[str] = Field(
        None, examples=["AI-first SaaS company building developer tools"]
    )


class EmailResponse(BaseModel):
    subject: str
    body: str
    contact_email: str
    contact_name: str


class StatusResponse(BaseModel):
    status: str


# ── Routes ───────────────────────────────────────────────────
@app.get("/health", response_model=StatusResponse)
async def health():
    return {"status": "ok"}


@app.post("/generate-email", response_model=EmailResponse)
async def generate(contact: ContactInput):
    """Generate a personalized cold email for the given contact."""
    result = await generate_email(contact.model_dump())
    return EmailResponse(
        subject=result["subject"],
        body=result["body"],
        contact_email=contact.email,
        contact_name=contact.name,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
