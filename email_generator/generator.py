"""
Core email generation logic using LangChain + Ollama (local) or cloud LLMs.
Generates a personalized cold email subject + body for a given contact.
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
    subject: str = Field(description="Email subject line — concise and compelling")
    body: str = Field(description="Email body — max 150 words, professional, value-focused")


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
            "bio": "Backend developer passionate about AI.",
        }


# ── Fallback template ───────────────────────────────────────
FALLBACK_TEMPLATE = {
    "subject": "Experienced Backend & AI Developer — Open to Opportunities at {company}",
    "body": (
        "Hi {name},\n\n"
        "I came across {company} and was impressed by your work in the AI space. "
        "I'm a backend and AI developer with experience building production APIs, "
        "LLM-powered applications, and data pipelines.\n\n"
        "I'd love to explore how I can contribute to your team. "
        "I've attached my resume for your reference.\n\n"
        "Looking forward to hearing from you.\n\n"
        "Best regards"
    ),
}


# ── Build the LLM chain ─────────────────────────────────────
def get_llm():
    """
    Priority:
    1. Local Ollama (llama3.1:8b) — default, zero cost
    2. OpenAI (if OPENAI_API_KEY set)
    3. Google Gemini (if GEMINI_API_KEY set)
    """
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Try cloud providers first if keys are present
    openai_key = os.getenv("OPENAI_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")

    if openai_key:
        try:
            from langchain_openai import ChatOpenAI
            logger.info("Using OpenAI GPT-4o-mini as LLM provider")
            return ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.7,
                api_key=openai_key,
            )
        except ImportError:
            logger.warning("langchain-openai not installed, falling back to Ollama")

    if gemini_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            logger.info("Using Google Gemini as LLM provider")
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.7,
                google_api_key=gemini_key,
            )
        except ImportError:
            logger.warning("langchain-google-genai not installed, falling back to Ollama")

    # Default: local Ollama
    logger.info("Using local Ollama at %s", ollama_host)
    return Ollama(
        model="llama3.1:8b",
        base_url=ollama_host,
        temperature=0.7,
    )


SYSTEM_PROMPT = """You are a professional job-seeker assistant helping write concise, personalized cold emails.

Rules:
- Never sound desperate or needy
- Focus on VALUE you bring to the company
- Keep the email body under 150 words
- Be specific about why this company interests you
- Sound natural, not robotic
- Don't use clichés like "I'm a perfect fit" or "passionate go-getter"
- Always explicitly mention that your resume is attached for their reference
- End with a soft call to action (suggest a brief call or ask to chat)
- Do NOT include any sign-off (e.g. do not write "Best regards" or leave your name at the end), just output the body paragraph

You MUST respond with valid JSON containing exactly two keys:
- "subject": A compelling email subject line
- "body": The email body text (plain text, use \\n for newlines)"""

USER_PROMPT_TEMPLATE = """Write a cold email for a job application.

ABOUT ME:
- Name: {my_name}
- Role: {my_title}
- Experience: {my_experience} year(s)
- Key Skills: {my_skills}
- Bio: {my_bio}

TARGET CONTACT:
- Name: {contact_name}
- Title: {contact_title}
- Company: {company}
- Company Description: {company_description}

Write a personalized cold email that highlights why I'd be a great addition to {company}. Reference something specific about the company if possible."""


def build_chain():
    """Build the LangChain prompt → LLM → parser chain."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT_TEMPLATE),
    ])
    llm = get_llm()
    parser = JsonOutputParser(pydantic_object=EmailOutput)
    return prompt | llm | parser


# ── Main generation function ────────────────────────────────
async def generate_email(contact: dict, config_path: str = "/app/config.yaml") -> dict:
    """
    Generate a personalized cold email for the given contact.
    Returns {"subject": "...", "body": "..."}.
    Falls back to a template if the LLM call fails.
    """
    profile = load_profile(config_path)

    prompt_vars = {
        "my_name": profile.get("name", "Developer"),
        "my_title": profile.get("title", "Software Developer"),
        "my_experience": profile.get("years_of_experience", 1),
        "my_skills": ", ".join(profile.get("skills", ["Python"])),
        "my_bio": profile.get("bio", ""),
        "contact_name": contact.get("name", "Hiring Manager"),
        "contact_title": contact.get("title", "Recruiter"),
        "company": contact.get("company", "your company"),
        "company_description": contact.get("company_description", "a technology company"),
    }

    try:
        chain = build_chain()
        result = await chain.ainvoke(prompt_vars)

        # Validate we got both keys
        if not isinstance(result, dict) or "subject" not in result or "body" not in result:
            raise ValueError(f"Unexpected LLM output format: {result}")

        # Append signature
        sign_off = f"\n\nBest regards,\n{profile.get('name', 'Developer')}"
        if profile.get("github"):
            sign_off += f"\nGitHub: {profile['github']}"
        if profile.get("linkedin"):
            sign_off += f"\nLinkedIn: {profile['linkedin']}"

        result["body"] = result["body"].rstrip() + sign_off

        logger.info("Generated email for %s at %s", contact.get("name"), contact.get("company"))
        return result

    except Exception as e:
        logger.error("LLM generation failed: %s — using fallback template", e)
        return {
            "subject": FALLBACK_TEMPLATE["subject"].format(
                company=contact.get("company", "Your Company")
            ),
            "body": FALLBACK_TEMPLATE["body"].format(
                name=contact.get("name", "Hiring Manager"),
                company=contact.get("company", "your company"),
            ) + f"\n\nBest regards,\n{profile.get('name', 'Developer')}",
        }
