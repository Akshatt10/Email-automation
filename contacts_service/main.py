"""
Contacts Fetching Microservice
Fetches HR/recruiter contacts from Apollo.io and Hunter.io,
stores them in a local SQLite database for tracking.
"""

import os
import logging
import sqlite3
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional

import csv
from io import StringIO
import httpx
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from pydantic import BaseModel, Field

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────
DB_PATH = "/app/data/contacts.db"


# ── Database helpers ─────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS contacts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            email       TEXT UNIQUE,
            company     TEXT,
            title       TEXT,
            linkedin_url TEXT,
            company_description TEXT,
            status      TEXT DEFAULT 'pending',
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
            sent_at     TEXT
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialised at %s", DB_PATH)


# ── Lifespan ─────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Contacts Service",
    description="Fetches and stores recruiter/HR contacts from Apollo.io & Hunter.io",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Pydantic Models ──────────────────────────────────────────
class FetchRequest(BaseModel):
    """
    Request model for n8n to pull the next batch of pending contacts.
    """
    limit: int = Field(50, ge=1, le=200)


class ContactOut(BaseModel):
    id: int
    name: str
    email: Optional[str] = None
    company: Optional[str] = None
    title: Optional[str] = None
    linkedin_url: Optional[str] = None
    company_description: Optional[str] = None
    status: str = "pending"
    created_at: Optional[str] = None
    sent_at: Optional[str] = None


class StatusResponse(BaseModel):
    status: str





# ── Upsert contact into DB ──────────────────────────────────
def upsert_contact(contact: dict) -> Optional[int]:
    """Insert contact if email doesn't already exist. Returns row id or None."""
    conn = get_db()
    try:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO contacts
                (name, email, company, title, linkedin_url, company_description, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                contact.get("name"),
                contact.get("email"),
                contact.get("company"),
                contact.get("title"),
                contact.get("linkedin_url"),
                contact.get("company_description"),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        if cursor.rowcount > 0:
            return cursor.lastrowid
        return None  # duplicate
    finally:
        conn.close()


# ── Routes ───────────────────────────────────────────────────
@app.get("/health", response_model=StatusResponse)
async def health():
    return {"status": "ok"}


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """
    Parses an Apollo.io exported CSV file and adds contacts to the local database.
    Skips duplicates based on email.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "File must be a CSV")

    contents = await file.read()
    text = contents.decode("utf-8-sig")  # handle BOM if present
    reader = csv.DictReader(StringIO(text))

    saved_count = 0
    duplicate_count = 0
    skipped_count = 0

    for row in reader:
        email = row.get("Email", "").strip()
        
        # We only want contacts that actually have an email
        if not email or email.lower() == "not available":
            skipped_count += 1
            continue

        first_name = row.get("First Name", "").strip()
        last_name = row.get("Last Name", "").strip()
        
        c = {
            "name": f"{first_name} {last_name}".strip(),
            "email": email,
            "company": row.get("Company Name"),
            "title": row.get("Title"),
            "linkedin_url": row.get("Person Linkedin Url"),
            "company_description": row.get("Keywords", ""), # Use keywords as brief context
        }

        row_id = upsert_contact(c)
        if row_id:
            saved_count += 1
            logger.info("Imported via CSV: %s <%s>", c["name"], c["email"])
        else:
            duplicate_count += 1

    return {
        "message": "CSV processing complete",
        "saved": saved_count,
        "duplicates_skipped": duplicate_count,
        "no_email_skipped": skipped_count
    }


@app.post("/fetch-contacts", response_model=list[ContactOut])
async def fetch_contacts(req: FetchRequest):
    """
    Instead of calling Apollo, this now acts as a queue puller.
    It fetches 'limit' number of pending contacts directly from the local SQLite database 
    and hands them to n8n for processing.
    """
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM contacts WHERE status = 'pending' ORDER BY created_at ASC LIMIT ?",
            (req.limit,)
        ).fetchall()
        
        contacts = [dict(r) for r in rows]
        logger.info("Handed %d pending contacts to n8n", len(contacts))
        return contacts
    finally:
        conn.close()


@app.get("/contacts", response_model=list[ContactOut])
async def list_contacts(status: Optional[str] = Query(None, examples=["pending", "sent", "failed"])):
    """List all contacts, optionally filtered by status."""
    conn = get_db()
    try:
        if status:
            rows = conn.execute(
                "SELECT * FROM contacts WHERE status = ? ORDER BY created_at DESC", (status,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM contacts ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.patch("/contacts/{contact_id}/status")
async def update_status(contact_id: int, status: str = Query(...)):
    """Update a contact's status (used by the email sender)."""
    conn = get_db()
    try:
        sent_at = datetime.now(timezone.utc).isoformat() if status == "sent" else None
        conn.execute(
            "UPDATE contacts SET status = ?, sent_at = ? WHERE id = ?",
            (status, sent_at, contact_id),
        )
        conn.commit()
        return {"updated": True, "id": contact_id, "status": status}
    finally:
        conn.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
