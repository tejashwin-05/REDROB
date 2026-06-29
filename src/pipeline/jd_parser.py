"""
JD Parser Node — uses Groq LLM to semantically understand the job description
and extract structured requirements.
"""

from __future__ import annotations

import json

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.models.job_description import JobDescription, ParsedJD
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are an expert technical recruiter. Your job is to deeply understand a job description
and extract structured requirements that can be used to rank candidates.

Extract the following as JSON (return ONLY valid JSON, no markdown code fences):

{{
  "role_title": "exact role title",
  "role_summary": "1-2 sentence summary of what this role is about",
  "required_skills": ["skill1", "skill2"],
  "preferred_skills": ["skill1"],
  "required_experience_years": 5,
  "max_experience_years": 9,
  "industries_preferred": ["industry1"],
  "education_requirements": "description of education needs",
  "key_responsibilities": ["responsibility1", "responsibility2"],
  "disqualifiers": ["disqualifier1"],
  "work_mode": "remote or hybrid or onsite or flexible or null",
  "location_preference": "location string or any"
}}

Rules:
- required_skills: focus on what the role NEEDS, not just mentions
- disqualifiers: anything that would make a candidate clearly wrong for the role
- Be precise: don't over-inflate required_skills; only true requirements
- Return ONLY valid JSON, no markdown code fences, no commentary
"""

_HUMAN_PROMPT = "Job Description:\n\n{jd_text}"


def parse_job_description(jd: JobDescription, groq_api_key: str) -> ParsedJD:
    """
    Use Groq LLM to parse and structure the job description.
    Returns a ParsedJD Pydantic model.
    """
    logger.info("Parsing job description with Groq LLM...")

    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        max_tokens=2048,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", _HUMAN_PROMPT),
    ])

    chain = prompt | llm | JsonOutputParser()
    raw: dict = chain.invoke({"jd_text": jd.raw_text})

    # Build ParsedJD, with safe defaults for missing fields
    parsed = ParsedJD(
        role_title=raw.get("role_title", "Unknown Role"),
        role_summary=raw.get("role_summary", ""),
        required_skills=raw.get("required_skills", []),
        preferred_skills=raw.get("preferred_skills", []),
        required_experience_years=float(raw.get("required_experience_years", 0)),
        max_experience_years=raw.get("max_experience_years"),
        industries_preferred=raw.get("industries_preferred", []),
        education_requirements=raw.get("education_requirements", ""),
        key_responsibilities=raw.get("key_responsibilities", []),
        disqualifiers=raw.get("disqualifiers", []),
        work_mode=raw.get("work_mode"),
        location_preference=raw.get("location_preference"),
        embedding_text="",  # will be set below
    )

    # Build the embedding-optimised text
    parsed.embedding_text = parsed.build_embedding_text()

    logger.info(f"JD parsed → role: '{parsed.role_title}', "
                f"required_skills: {len(parsed.required_skills)}, "
                f"min_exp: {parsed.required_experience_years}y")
    return parsed
