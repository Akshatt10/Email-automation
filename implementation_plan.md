# Automated Job-Hunt Cold Email System

Build an end-to-end automated pipeline to find contacts, generate personalized emails, and send them with resume attachments, all orchestrated via n8n.

## User Review Required

> [!IMPORTANT]
> **API Keys Required**: You'll need to provide your own API keys for:
> - Apollo.io (for contact searching)
> - Hunter.io (optional, for email verification/domain search)
> - OpenAI/Gemini (if not using local Llama)
> - SMTP Credentials (for Gmail/AWS SES)

> [!TIP]
> **Local LLM**: I will include an Ollama container in the `docker-compose.yml` to run Llama 3 locally on your M4 Mac. This will require an initial pull of the model (7B/8B version recommended).

## Proposed Changes

### Phase 1: Infrastructure & Orchestration

#### [NEW] [docker-compose.yml](file:///Users/akshattyagi/Developer/job%20automation/docker-compose.yml)
Define services:
- **n8n**: Main orchestrator.
- **Postgres**: Backend for n8n.
- **Ollama**: Local LLM runner.
- **Microservices**: `contacts-service`, `email-generator`, `email-sender`.

#### [NEW] [.env.example](file:///Users/akshattyagi/Developer/job%20automation/.env.example)
Template for all required environment variables.

### Phase 2: Contact Fetching Microservice

#### [NEW] `contacts_service/`
- **FastAPI**: `POST /fetch-contacts` to call Apollo/Hunter.
- **SQLite**: Local storage for deduplication and status tracking.
- **Deduplication**: Logic to skip already processed emails.

### Phase 3: AI Email Personalization

#### [NEW] `email_generator/`
- **FastAPI**: `POST /generate-email`.
- **LangChain**: Integrate with Ollama (Local) or Gemini/OpenAI (Cloud).
- **Prompt Engineering**: Personalized content based on company context and your `config.yaml`.

### Phase 4: Email Sender

#### [NEW] `email_sender/`
- **Python smtplib**: Handle Gmail/SES SMTP.
- **Attachment**: Logic to attach `assets/resume.pdf`.
- **Rate Limiting**: Daily limit check using `rate_limiter.json`.

### Phase 5: n8n Workflow

#### [NEW] [workflow.json](file:///Users/akshattyagi/Developer/job%20automation/n8n_workflows/workflow.json)
- Daily cron trigger (9:00 AM IST).
- Batch processing of contacts.
- Integration between the microservices.

### Phase 6: Project Configuration

#### [NEW] [config.yaml](file:///Users/akshattyagi/Developer/job%20automation/config.yaml)
Stores your personal details (Bio, GitHub, Projects) for the LLM prompt.

#### [NEW] [README.md](file:///Users/akshattyagi/Developer/job%20automation/README.md)
Comprehensive setup and running guide.

## Open Questions

1. **Local LLM Performance**: Since we're running Ollama in Docker on Mac, it might be slower than running it natively. Would you prefer I include instructions to run Ollama natively on your Mac and just have the containers point to `host.docker.internal:11434`? (Recommended for M4 performance).
2. **Resume Location**: Can you confirm you'll provide a `resume.pdf` in the `assets/` folder?
3. **Daily Limit**: You mentioned 50/day. Should we make this a hard stop or just a warning in the logs?

## Verification Plan

### Automated Tests
- `pytest` for internal logic in each microservice.
- Docker-compose health checks to ensure service connectivity.

### Manual Verification
- Testing single contact fetch and email generation via FastAPI Swagger UI (`/docs`).
- Dry-run mode for the `email_sender` to verify content before sending real emails.
- Manual trigger in n8n UI.
