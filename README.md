# 🚀 Offline-First Job Hunt Automation

An end-to-end automated cold email pipeline for job hunting that runs **100% locally** on your machine. You simply export a CSV of leads from Apollo.io, and the system seamlessly generates hyper-personalized emails using a **local LLM** (Ollama/Llama 3.1) and sends them with your physical resume attached — all orchestrated automatically by **n8n**.

```text
┌─────────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Upload CSV     │────▶│ n8n          │────▶│ Email        │────▶│ Email Sender │
│ (Apollo Leads)  │     │ Scheduler    │     │ Generator    │     │ (SMTP/Gmail) │
└─────────────────┘     └──────────────┘     │ (Ollama LLM) │     └──────────────┘
                                             └──────────────┘
```

---

## 📋 Prerequisites

- **Docker** & **Docker Compose** (v2+)
- **macOS** (tested on M-series Apple Silicon) or Linux
- **Ollama Desktop** (optional but recommended for stability)
- **SMTP credentials** (e.g., Gmail App Password)

---

## ⚡ Quick Start

### 1. Clone & configure environment

```bash
git clone <your-repo-url> job-automation
cd "job automation"
cp .env.example .env
cp config.example.yaml config.yaml
```

Open `.env` and fill in your SMTP credentials:

```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
# IMPORTANT: Remove all spaces from the Google App password if you copy it!
SMTP_PASS=your_gmail_app_password     
DRY_RUN=true                          # set to false when ready to send real emails
```

> **Gmail Users**: You must generate an [App Password](https://myaccount.google.com/apppasswords) (requires 2FA enabled). Your normal Gmail password will NOT work.

### 2. Add your resume & profile context

1. **Resume PDF**: Copy your resume to the assets folder:
   ```bash
   cp /path/to/your/resume.pdf assets/resume.pdf
   ```
2. **Profile Configuration**: Open `config.yaml` and update your personal details (name, skills, GitHub, bio, projects). The Artificial Intelligence uses this file as its "brain" to personalize every single email.

### 3. Start the stack

```bash
docker compose up -d --build
```

### 4. Setup Llama 3.1 (Local AI)

```bash
docker exec -it ollama_llm ollama pull llama3.1:8b
```
*(This downloads ~4.7 GB. You only need to do this once — the model persists natively on your machine).*

### 5. Ingest your Apollo Leads

1. Go to [Apollo.io](https://www.apollo.io), build a list of target companies/recruiters, and click **Export CSV**.
2. Drop that `.csv` file into the root of this project folder.
3. Run this command to digest the CSV into the local database (it automatically skips duplicates!):
   ```bash
   curl -X POST -F "file=@your-apollo-export.csv" http://localhost:8000/upload-csv
   ```

### 6. Import the n8n logic

1. Open [http://localhost:5678](http://localhost:5678)
2. Log in with the credentials from your `.env` (default: `admin` / `admin123`)
3. Click **"Add workflow"** → **⋯ menu** → **"Import from file"**
4. Select `n8n_workflows/workflow.json`
5. Click the toggle switch at the top right to **"Active"** to enable your nightly schedule!

---

## 🔧 Architecture & Privacy

This system was deliberately refactored from a cloud-based API scraper into a **Strictly Offline Pipeline** for two reasons:
1. **Privacy**: Your resume data and contact lists never leave your local machine (no OpenAI ingestion).
2. **Cost**: You incur $0 in API costs.

### The Microservices

| Service | Tech | Description |
|---|---|---|
| **Contacts Queue** | FastAPI + SQLite | Exposes an `/upload-csv` ingestion endpoint and stores contacts securely in an offline database. Feeds `pending` contacts to n8n batch-by-batch. |
| **Email Generator** | FastAPI + LangChain + Ollama | Merges the contact's Apollo keywords with your `config.yaml` profile to generate hyper-personalized cold emails using Llama 3.1. |
| **Email Sender** | FastAPI + smtplib | Wraps the generated email in proper UTF-8 headers, physically attaches `assets/resume.pdf`, checks daily rate limits to protect your Gmail reputation, and sends the email. |
| **n8n** | Node.js | The visual orchestrator. Coordinates the 1-by-1 processing loop, handles success/failure routing, and triggers the schedule. |

---

## 🧪 Testing

### Dry-run mode (default)
Always ensure `DRY_RUN=true` is set in your `.env` when you install the project. In dry-run mode, the AI will generate the email, but the Sender will only print the result to the console instead of firing it over SMTP.

If you want to read what the AI drafted without checking logs, you can ping the generator directly:
```bash
curl -X POST http://localhost:8001/generate-email \
  -H "Content-Type: application/json" \
  -d '{
        "name": "Jane Doe",
        "email": "jane@apple.com",
        "company": "Apple",
        "title": "Engineering Manager",
        "company_description": "Consumer electronics, computer software, machine learning"
      }'
```

---

## 🛑 Operations

```bash
# Monitor the rate-limit quota and daily sends
curl http://localhost:8002/stats

# Check how many leads are pending vs sent
curl "http://localhost:8000/contacts?status=pending"

# Shut down the automation suite
docker compose down
```

---

## ⚠️ Important Notes

- **Google UI Bug:** Never copy a Google App Password directly into `.env` without deleting the spaces first. Google's web UI uses invisible Unicode spaces that will crash the Python `smtplib` encoder.
- **Rate Limiting**: The default is 50 emails/day (hard stop). You can change this via `DAILY_LIMIT` in `.env`. Google naturally caps personal Gmails at ~500 sends/day, so stay well below that to prevent spam algorithms from flagging your account.
- **Fail-Safes**: If you restart the script, contacts with `status=sent` are inherently skipped. The same email address can never be uploaded to the database twice. 

Happy hunting!
