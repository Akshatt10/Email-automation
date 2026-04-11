"""
Email sending logic using smtplib.
"""

import os
import json
import logging
import smtplib
from datetime import datetime, timezone, date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.header import Header
from pathlib import Path

logger = logging.getLogger(__name__)

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "50"))
DRY_RUN = os.getenv("DRY_RUN", "true").lower() in ("true", "1", "yes")
RATE_LIMITER_PATH = "/app/data/rate_limiter.json"

def _load_rate_data() -> dict:
    os.makedirs(os.path.dirname(RATE_LIMITER_PATH), exist_ok=True)
    if not Path(RATE_LIMITER_PATH).exists():
        return {"date": "", "count": 0}
    try:
        with open(RATE_LIMITER_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {"date": "", "count": 0}

def _save_rate_data(data: dict):
    with open(RATE_LIMITER_PATH, "w") as f:
        json.dump(data, f, indent=2)

def check_rate_limit() -> tuple[bool, int]:
    """Returns (allowed, remaining). Resets counter on new date."""
    data = _load_rate_data()
    today = date.today().isoformat()
    if data.get("date") != today:
        data = {"date": today, "count": 0}
        _save_rate_data(data)
    remaining = DAILY_LIMIT - data["count"]
    return remaining > 0, remaining

def increment_rate_counter():
    data = _load_rate_data()
    today = date.today().isoformat()
    if data.get("date") != today:
        data = {"date": today, "count": 0}
    data["count"] += 1
    _save_rate_data(data)

def attach_resume(msg: MIMEMultipart, resume_path: str) -> bool:
    """Attaches a PDF resume. Returns True if successful."""
    path = Path(resume_path)
    if not path.exists():
        logger.warning("Resume not found at %s", resume_path)
        return False
    try:
        with open(path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{path.name}"')
        msg.attach(part)
        return True
    except Exception as e:
        logger.error("Failed to attach resume: %s", e)
        return False

def send_email(to_email: str, subject: str, body: str, resume_path: str = "/app/assets/resume.pdf", dry_run: bool | None = None) -> dict:
    """Send email with resume. Returns status dict."""
    if dry_run is None: dry_run = DRY_RUN
    now = datetime.now(timezone.utc).isoformat()

    allowed, remaining = check_rate_limit()
    if not allowed:
        msg = f"Daily limit reached ({DAILY_LIMIT})."
        logger.warning(msg)
        return {"success": False, "message": msg, "timestamp": now, "is_dry_run": dry_run}

    email_msg = MIMEMultipart()
    email_msg["From"] = SMTP_USER
    email_msg["To"] = to_email
    email_msg["Subject"] = Header(subject, "utf-8")
    email_msg.attach(MIMEText(body, "plain", "utf-8"))
    attach_resume(email_msg, resume_path)

    if dry_run:
        logger.info("[DRY RUN] To: %s | Subject: %s", to_email, subject)
        increment_rate_counter()
        return {"success": True, "message": f"[DRY RUN] Logged for {to_email}", "timestamp": now, "is_dry_run": True}

    if not SMTP_USER or not SMTP_PASS:
        msg = "SMTP credentials missing."
        logger.error(msg)
        return {"success": False, "message": msg, "timestamp": now, "is_dry_run": False}

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(email_msg)
        increment_rate_counter()
        logger.info("[SENT] %s", to_email)
        return {"success": True, "message": f"Sent to {to_email}", "timestamp": now, "is_dry_run": False}
    except Exception as e:
        logger.error("Email failed: %s", e)
        return {"success": False, "message": str(e), "timestamp": now, "is_dry_run": False}

def get_daily_stats() -> dict:
    data = _load_rate_data()
    today = date.today().isoformat()
    if data.get("date") != today:
        return {"date": today, "sent_today": 0, "remaining": DAILY_LIMIT, "limit": DAILY_LIMIT}
    return {
        "date": today,
        "sent_today": data["count"],
        "remaining": DAILY_LIMIT - data["count"],
        "limit": DAILY_LIMIT,
    }
