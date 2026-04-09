"""
Email generation logic for personalized job outreach emails.
"""

import os
import logging
import yaml
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.llms import Ollama
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class EmailOutput(BaseModel):
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body (100-130 words)")

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

def get_all_llms():
    """Returns prioritized list of LLM providers: Gemini -> OpenAI -> Ollama."""
    llms = []
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")

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

    llms.append(("Ollama llama3.1:8b", Ollama(
        model="llama3.1:8b",
        base_url=ollama_host,
        temperature=0.6,
    )))

    return llms

SYSTEM_PROMPT = """You are a sharp, direct email copywriter helping a developer land interviews.
Your emails get replies because they feel human, specific, and brief.

VOICE: Confident without being arrogant. Senior engineer tone.
Candidate identity: BUILDER who ships.

STRUCTURE (follow exactly):
1. Opening (1 sentence) — Genuine observation about the company. No filler.
2. Value bridge (2-3 sentences) — Relevant projects and focus areas.
   CRITICAL: Focus ONLY on {comm_focus_areas} and {comm_highlight_project}.
   Do NOT mention {comm_avoid_projects}.
3. The ask (1 sentence) — Soft CTA for 15-minute call.
4. Resume mention (1 sentence) — Woven in naturally.

HARD RULES:
- Body: 100-130 words.
- Zero clichés: passionate, go-getter, perfect fit, excited to.
- No filler openers: "I hope this finds you well", "My name is".
- Subject line: Curiosity or specific value prop.
- No sign-off/closing name.

OUTPUT: valid JSON only.
{{
  "subject": "...",
  "body": "..."
}}"""

USER_PROMPT_TEMPLATE = """Write a cold outreach email for a job application.

COMPANY:
- Company: {company}
- What they do: {company_description}
- Contact: {contact_name}, {contact_title}

CANDIDATE:
- Name: {my_name}
- Role: {my_title}
- Experience: {my_experience} year(s)
- Skills: {my_skills}
- Bio: {my_bio}
- Projects: {my_projects}

IDENTITY:
- Tone: {comm_tone}
- Emphasize: {comm_emphasize}
- Avoid: {comm_avoid}
- Focus focal points: {comm_focus_areas}, {comm_highlight_project}.

Now write the email. JSON only."""

BANNED_PHRASES = [
    "i hope this email finds you", "i came across your company", "i am a passionate",
    "perfect fit", "my name is", "i am writing to", "i feel", "i believe i",
    "excited to", "go-getter", "give me a chance", "i'm a quick learner",
    "eager to", "would love the opportunity", "i am eager", "passionate about",
    "looking for an opportunity", "next week", "tomorrow", "this week"
]

def build_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT_TEMPLATE),
    ])
    parser = JsonOutputParser(pydantic_object=EmailOutput)
    return prompt | llm | parser

async def generate_email(contact: dict, config_path: str = "/app/config.yaml") -> dict:
    """Generate personalized email using fallback LLM chain (Gemini -> OpenAI -> Ollama)."""
    profile = load_profile(config_path)
    projects_list = profile.get("notable_projects", [])
    formatted_projects = "\n".join([
        f"- {p.get('name', 'Project')}: {p.get('description', '').strip()}"
        for p in projects_list
    ]) if projects_list else "No projects listed."

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
        "comm_focus_areas": comm_style.get("focus_areas", "relevant technical experience"),
        "comm_highlight_project": comm_style.get("highlight_project", "a core technical project"),
        "comm_avoid_projects": comm_style.get("avoid_projects", "irrelevant or basic projects"),
    }

    all_llms = get_all_llms()
    last_error = None

    for llm_name, llm in all_llms:
        try:
            logger.info("Trying %s for %s at %s", llm_name, contact.get("name"), contact.get("company"))
            chain = build_chain(llm)
            result = await chain.ainvoke(prompt_vars)

            if not isinstance(result, dict) or "subject" not in result or "body" not in result:
                raise ValueError("Unexpected LLM format")

            body_word_count = len(result["body"].split())

            # Retry logic for length constraints
            if body_word_count < 60:
                result = await chain.ainvoke(prompt_vars)
                body_word_count = len(result["body"].split())

            if body_word_count > 160:
                compressed_vars = {**prompt_vars}
                compressed_vars["my_bio"] = f"CRITICAL: Too long ({body_word_count} words). Hard limit 130. " + compressed_vars["my_bio"]
                result = await chain.ainvoke(compressed_vars)

            body_lower = result["body"].lower()
            for phrase in BANNED_PHRASES:
                if phrase in body_lower:
                    logger.warning("[%s] Banned phrase: '%s'", llm_name, phrase)

            sign_off = f"\n\nBest regards,\n{profile.get('name', 'Developer')}"
            if profile.get("github"): sign_off += f"\nGitHub: {profile['github']}"
            if profile.get("linkedin"): sign_off += f"\nLinkedIn: {profile['linkedin']}"

            result["body"] = result["body"].rstrip() + sign_off
            logger.info("✅ Generated for %s at %s", contact.get("name"), contact.get("company"))
            return result
        except Exception as e:
            last_error = e
            logger.warning("❌ [%s] Failed: %s", llm_name, str(e)[:100])
            continue

    logger.error("All LLMs failed. Using fallback template.")
    return {
        "subject": FALLBACK_TEMPLATE["subject"].format(company=contact.get("company", "Company")),
        "body": FALLBACK_TEMPLATE["body"].format(
            name=contact.get("name", "Hiring Manager"),
            company=contact.get("company", "company"),
        ) + f"\n\nBest regards,\n{profile.get('name', 'Developer')}",
    }

