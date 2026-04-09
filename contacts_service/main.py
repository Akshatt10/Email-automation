"""
Contacts Service for storing and managing job outreach contacts.
"""

import os
import logging
import sqlite3
import csv
import shutil
import asyncio
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional
from io import StringIO
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DB_PATH = "/app/data/contacts.db"
INGEST_PATH = "/app/ingest"

def process_csv_content(text: str) -> dict:
    reader = csv.DictReader(StringIO(text))
    saved_count = 0
    duplicate_count = 0
    skipped_count = 0

    for row in reader:
        email = row.get("Email", "").strip()
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
            "company_description": row.get("Keywords", ""),
        }

        row_id = upsert_contact(c)
        if row_id:
            saved_count += 1
        else:
            duplicate_count += 1
    
    return {
        "saved": saved_count,
        "duplicates": duplicate_count,
        "skipped": skipped_count
    }

async def auto_ingest_worker():
    """Background task scanning for CSV files in the ingest folder."""
    os.makedirs(INGEST_PATH, exist_ok=True)
    logger.info("Auto-ingest worker started. Watching %s", INGEST_PATH)

    while True:
        try:
            files = [f for f in os.listdir(INGEST_PATH) if f.endswith(".csv")]
            for filename in files:
                file_path = os.path.join(INGEST_PATH, filename)
                logger.info("Auto-importing: %s", filename)

                try:
                    with open(file_path, "r", encoding="utf-8-sig") as f:
                        stats = process_csv_content(f.read())
                    
                    shutil.move(file_path, f"{file_path}.processed")
                    logger.info("Successfully ingested %s: Saved %d, Duplicates %d", 
                                filename, stats["saved"], stats["duplicates"])
                except Exception as e:
                    logger.error("Error processing %s: %s", filename, e)
        except Exception as e:
            logger.error("Auto-ingest worker loop error: %s", e)
        
        await asyncio.sleep(60)

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    asyncio.create_task(auto_ingest_worker())
    yield

app = FastAPI(
    title="Contacts Service",
    description="Manages recruiter contacts and queue for outreach",
    version="1.0.0",
    lifespan=lifespan,
)

class FetchRequest(BaseModel):
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

def upsert_contact(contact: dict) -> Optional[int]:
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
        return None
    finally:
        conn.close()

@app.get("/health", response_model=StatusResponse)
async def health():
    return {"status": "ok"}

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "File must be a CSV")

    contents = await file.read()
    text = contents.decode("utf-8-sig")
    stats = process_csv_content(text)

    return {
        "message": "CSV processing complete",
        "saved": stats["saved"],
        "duplicates_skipped": stats["duplicates"],
        "no_email_skipped": stats["skipped"]
    }

@app.post("/fetch-contacts", response_model=list[ContactOut])
async def fetch_contacts(req: FetchRequest):
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

@app.get("/contacts/count")
async def count_contacts():
    conn = get_db()
    try:
        total = conn.execute("SELECT COUNT(*) FROM contacts").fetchone()[0]
        status_counts = conn.execute(
            "SELECT status, COUNT(*) FROM contacts GROUP BY status"
        ).fetchall()
        breakdown = {row[0]: row[1] for row in status_counts}
        return {"total": total, "breakdown": breakdown}
    finally:
        conn.close()

@app.patch("/contacts/{contact_id}/status")
async def update_status(contact_id: int, status: str = Query(...)):
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
