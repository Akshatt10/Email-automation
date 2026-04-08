"""
Core email generation logic using LangChain + Ollama (local) or cloud LLMs.
Generates a personalized cold email subject + body for a given contact.

V2 — Rewritten with advanced prompt engineering for higher reply rates.
"""

import os
import logging
from typing import Optional

import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.llms import Ollama
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Output schema ────────────────────────────────────────────
class EmailOutput(BaseModel):
    subject: str = Field(description="Email subject line — curiosity-driven or value-prop")
    body: str = Field(description="Email body — 100-130 words, company-first, no sign-off")


# ── Load personal config ─────────────────────────────────────
def load_profile(config_path: str = "/app/config.yaml") -> dict:
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f).get("profile", {})
    except FileNotFoundError:
        logger.warning("config.yaml not found at %s — using defaults", config_path)
        return {
            "name": "A Software Developer",
            "skills": ["Python", "FastAPI"],
            "years_of_experience": 1,
            "bio": "Backend developer focused on AI systems.",
        }


# ── Fallback template (also improved) ───────────────────────
FALLBACK_TEMPLATE = {
    "subject": "Backend + AI systems experience — relevant to {company}",
    "body": (
        "Hi {name},\n\n"
        "I've been building production backend systems and AI-powered pipelines "
        "— the kind of work that maps directly to what {company} is doing.\n\n"
        "I've attached my resume with specifics on what I've shipped. "
        "Would a 15-minute call make sense to see if there's a fit?"
    ),
}


# ── Build the LLMs with fallback chain ───────────────────────
def get_all_llms():
    """
    Returns a list of (name, llm) tuples in priority order.
    Priority: Gemini (free) → OpenAI (cheap) → Ollama (local/slow).
    If one fails (quota, rate limit), the caller tries the next.
    """
    llms = []
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")

    # 1st priority: Gemini (free tier — 1500 req/day, 15 req/min)
    if gemini_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llms.append(("Gemini 2.5 Flash", ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.6,
                google_api_key=gemini_key,
            )))
        except ImportError:
            logger.warning("langchain-google-genai not installed, skipping Gemini")

    # 2nd priority: OpenAI (gpt-4o-mini — ~$0.0003 per email)
    if openai_key:
        try:
            from langchain_openai import ChatOpenAI
            llms.append(("OpenAI GPT-4o-mini", ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.75,
                api_key=openai_key,
            )))
        except ImportError:
            logger.warning("langchain-openai not installed, skipping OpenAI")

    # 3rd priority: Local Ollama (free but slow on CPU)
    llms.append(("Ollama llama3.1:8b", Ollama(
        model="llama3.1:8b",
        base_url=ollama_host,
        temperature=0.6,
    )))

    return llms


# ── V2 Prompts ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a sharp, direct email copywriter helping a developer land interviews.
Your emails get replies because they feel human, specific, and brief — not like a mass blast.

VOICE: Confident without being arrogant. Curious without being sycophantic.
Tone benchmark: how a senior engineer writes a Slack message to someone they respect but don't know yet.
This candidate is a BUILDER — someone who figures things out and ships. Every email should subtly
convey: "I get things done. I don't wait. I build." Never say it directly — SHOW it through
what they've built and how they describe it.

STRUCTURE (follow this exactly):
1. Opening (1 sentence) — a specific, genuine observation about the company or their work.
   NOT "I came across your company" — that's filler.
   YES "Your approach to [something from company_description] caught my attention."
2. Value bridge (2-3 sentences) — what the candidate builds/has done that's relevant to THIS company.
   CRITICAL: Focus ONLY on my backend experience handling systems and my RAG project.
   Keep it simple: I build backend systems and I built an AI Knowledge Vault (RAG System).
   Do NOT mention 50 emails a day, n8n, or other projects. Just backend systems + RAG project.
   Example: "I built X from scratch that does Y" not "I have N years of X experience"
3. The ask (1 sentence) — a soft, specific call to action. A 15-minute call. Not "I'd love to connect."
4. Resume mention (1 sentence) — weave it in naturally, not as a separate awkward line.

HARD RULES:
- Body must be 100-130 words. Count carefully. Under 100 = too thin. Over 140 = too long.
- Zero clichés: ban "passionate", "go-getter", "perfect fit", "I believe", "I feel", "excited to"
- No filler openers: ban "I hope this email finds you well", "I came across your company", "My name is"
- Never beg: ban "give me a chance", "I'm a quick learner", "eager to", "would love the opportunity"
- The subject line must create curiosity or state a specific value prop — never just a job title
- Do NOT write a sign-off or closing name — it will be appended separately
- The email should read like someone who SHIPS, not someone who's looking for permission
- Write as if this is the only email this recruiter will read today that doesn't bore them

OUTPUT: valid JSON only. No markdown, no preamble, no explanation.
{{
  "subject": "...",
  "body": "..."
}}"""

USER_PROMPT_TEMPLATE = """Write a cold outreach email for a job application.

COMPANY CONTEXT (read this first — the email opens with THEM, not the candidate):
- Company: {company}
- What they do: {company_description}
- You're writing to: {contact_name}, {contact_title}

CANDIDATE PROFILE (use this to show relevance to the company above):
- Name: {my_name}
- Target role: {my_title}
- Experience: {my_experience} year(s) building real things
- Core skills: {my_skills}
- Background: {my_bio}
- Notable work: {my_projects}

CANDIDATE IDENTITY (this shapes the TONE — critical):
- Communication tone: {comm_tone}
- Things to emphasize: {comm_emphasize}
- Things to avoid: {comm_avoid}
- The candidate has 1 year of experience but ships real things. Let the work speak for itself.
- Tone down the intensity. Don't sound like you're aggressively selling yourself. Just state facts casually.
- NEVER list all projects. Focus ONLY on backend systems engineering and the AI Knowledge Vault (RAG System).

SUBJECT LINE OPTIONS — pick the best fit or invent better:
  Formula A: [Specific skill] + [relevant to their work] → e.g. "FastAPI + LLM pipelines — relevant to {company}'s stack"
  Formula B: Outcome statement → e.g. "Built 3 production AI tools — want to bring that to {company}"
  Formula C: Direct value prop → e.g. "{my_title} with {company_description} experience — open to a chat"

OPENING SENTENCE — do NOT use these (they're trash):
  BAD: "I came across {company} and was impressed..."
  BAD: "My name is {my_name} and I am a developer..."
  BAD: "I hope this message finds you well..."
  GOOD: Start with a specific, true observation about {company}'s work based on their description.
  GOOD: Or start with what you've built that directly maps to their problem space.

CLOSING — end with ONE soft ask. Suggest a specific short call (15 min), not a vague "connect".
Do NOT write "Best regards" or the candidate's name. Stop after the call to action.
Do NOT specify a timeframe like "next week" or "tomorrow". Just ask if they are open to a call.

Now write the email. JSON only."""


# ── Banned phrase detection ──────────────────────────────────
BANNED_PHRASES = [
    "i hope this email finds you",
    "i came across your company",
    "i am a passionate",
    "perfect fit",
    "my name is",
    "i am writing to",
    "i feel",
    "i believe i",
    "excited to",
    "go-getter",
    "give me a chance",
    "i'm a quick learner",
    "eager to",
    "would love the opportunity",
    "i am eager",
    "passionate about",
    "looking for an opportunity",
    "next week",
    "tomorrow",
    "this week"
]


def build_chain(llm):
    """Build the LangChain prompt → LLM → parser chain for a given LLM."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT_TEMPLATE),
    ])
    parser = JsonOutputParser(pydantic_object=EmailOutput)
    return prompt | llm | parser


# ── Main generation function ────────────────────────────────
async def generate_email(contact: dict, config_path: str = "/app/config.yaml") -> dict:
    """
    Generate a personalized cold email for the given contact.
    Returns {"subject": "...", "body": "..."}.
    Tries each LLM in the fallback chain (Gemini → OpenAI → Ollama).
    Falls back to a template only if ALL LLMs fail.
    """
    profile = load_profile(config_path)

    # Format projects as a readable list for the LLM
    projects_list = profile.get("notable_projects", [])
    formatted_projects = "\n".join([
        f"- {p.get('name', 'Project')}: {p.get('description', '').strip()}"
        for p in projects_list
    ]) if projects_list else "No specific projects listed."

    # Format communication style for the LLM
    comm_style = profile.get("communication_style", {})

    prompt_vars = {
        "my_name": profile.get("name", "Developer"),
        "my_title": profile.get("title", "Software Developer"),
        "my_experience": profile.get("years_of_experience", 1),
        "my_skills": ", ".join(profile.get("skills", ["Python"])),
        "my_bio": profile.get("bio", ""),
        "my_projects": formatted_projects,
        "contact_name": contact.get("name", "Hiring Manager"),
        "contact_title": contact.get("title", "Recruiter"),
        "company": contact.get("company", "your company"),
        "company_description": contact.get("company_description", "a technology company"),
        "comm_tone": comm_style.get("tone", "direct and confident"),
        "comm_emphasize": ", ".join(comm_style.get("emphasize", ["impact", "things shipped"])),
        "comm_avoid": ", ".join(comm_style.get("avoid", ["generic phrases", "buzzwords"])),
    }

    # ── Try each LLM in the fallback chain ──
    all_llms = get_all_llms()
    last_error = None

    for llm_name, llm in all_llms:
        try:
            logger.info("Trying %s for %s at %s", llm_name, contact.get("name"), contact.get("company"))
            chain = build_chain(llm)
            result = await chain.ainvoke(prompt_vars)

            # Validate we got both keys
            if not isinstance(result, dict) or "subject" not in result or "body" not in result:
                raise ValueError(f"Unexpected LLM output format: {result}")

            # ── Word count enforcement ──
            body_word_count = len(result["body"].split())

            # Too short — LLM got lazy, retry once with same provider
            if body_word_count < 60:
                logger.warning("[%s] Email too short (%d words) — retrying once", llm_name, body_word_count)
                result = await chain.ainvoke(prompt_vars)
                body_word_count = len(result["body"].split())

            # Too long — LLM ignored word limit, retry with compression instruction
            if body_word_count > 160:
                logger.warning("[%s] Email too long (%d words) — requesting compression", llm_name, body_word_count)
                compressed_vars = {**prompt_vars}
                compressed_vars["my_bio"] = (
                    f"CRITICAL: Your last attempt was {body_word_count} words. "
                    f"The hard limit is 130. Cut ruthlessly. " + compressed_vars["my_bio"]
                )
                result = await chain.ainvoke(compressed_vars)

            # ── Banned phrase detection (log-only, don't block) ──
            body_lower = result["body"].lower()
            for phrase in BANNED_PHRASES:
                if phrase in body_lower:
                    logger.warning("[%s] Banned phrase detected: '%s' — flagging for review", llm_name, phrase)

            # ── Append signature ──
            sign_off = f"\n\nBest regards,\n{profile.get('name', 'Developer')}"
            if profile.get("github"):
                sign_off += f"\nGitHub: {profile['github']}"
            if profile.get("linkedin"):
                sign_off += f"\nLinkedIn: {profile['linkedin']}"

            result["body"] = result["body"].rstrip() + sign_off

            logger.info(
                "✅ [%s] Generated email for %s at %s (%d words)",
                llm_name, contact.get("name"), contact.get("company"), body_word_count
            )
            return result

        except Exception as e:
            last_error = e
            logger.warning("❌ [%s] Failed: %s — trying next provider", llm_name, str(e)[:200])
            continue

    # ── All LLMs failed — use fallback template ──
    logger.error("All LLM providers failed. Last error: %s — using fallback template", last_error)
    return {
        "subject": FALLBACK_TEMPLATE["subject"].format(
            company=contact.get("company", "Your Company")
        ),
        "body": FALLBACK_TEMPLATE["body"].format(
            name=contact.get("name", "Hiring Manager"),
            company=contact.get("company", "your company"),
        ) + f"\n\nBest regards,\n{profile.get('name', 'Developer')}",
    }

