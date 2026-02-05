from enum import Enum


class AnalysisMode(Enum):
    STANDARD = "standard"
    NO_COMPANY = "no_company"
    FAILURE = "failure"


LANGUAGE_INSTRUCTIONS = {
    "en": "Please respond in English.",
    "ja": "日本語で回答してください。",
    "zh": "请用中文回答。",
}

STANDARD_TEMPLATE = """You are a senior interview consultant. Your task is to analyze an interview conversation and help the candidate understand their performance and improve.

## Background Information
Company: {company_name}
Company Overview: {search_summary}
Position: {job_title}
Job Requirements:
{job_description}

## Interview Transcript
{transcript}

## Analysis Requirements
1. Break the interview into individual Q&A exchanges.
2. For each exchange, give a qualitative rating (Good / Neutral / Bad) with a one or two sentence explanation.
   - Criteria: relevance to the question, demonstration of skills required by the job, clarity and structure of expression.
3. Infer the interviewer's focus areas and questioning logic.
4. Provide 3 specific, actionable improvement suggestions for the next round of interviews.

{language_instruction}
Be professional but not stiff.
"""

NO_COMPANY_TEMPLATE = """You are a senior interview consultant. Your task is to analyze an interview conversation and help the candidate understand their performance and improve.

## Position Information
Position: {job_title}
Job Requirements:
{job_description}

## Interview Transcript
{transcript}

## Analysis Requirements
1. Break the interview into individual Q&A exchanges.
2. For each exchange, give a qualitative rating (Good / Neutral / Bad) with a one or two sentence explanation.
   - Criteria: relevance to the question, demonstration of skills required by the job, clarity and structure of expression.
3. Infer the interviewer's focus areas and questioning logic.
4. Provide 3 specific, actionable improvement suggestions for the next round of interviews.

{language_instruction}
Be professional but not stiff.
"""

FAILURE_TEMPLATE = """You are a supportive and experienced interview consultant. The candidate did not pass this interview. Your goal is to help them learn from this experience and prepare better for future opportunities.

## Position Information
Position: {job_title}
Job Requirements:
{job_description}

{company_section}

## Interview Transcript
{transcript}

## Analysis Requirements
1. **Interview Summary**: Briefly summarize the overall interview flow and key discussion points.
2. **Detailed Q&A Analysis**: Break the interview into individual Q&A exchanges. For each exchange, give a qualitative rating (Good / Neutral / Bad) with a one or two sentence explanation.
3. **Key Improvement Areas**: Identify the 3 most important areas where the candidate can improve, with specific examples from the transcript and concrete suggestions.
4. **What Went Well**: Highlight the positive aspects of the candidate's performance — these are strengths to carry forward.
5. **Encouragement**: End with a brief, genuine message of encouragement. Remind the candidate that every interview is a learning opportunity and they are making progress.

{language_instruction}
Be warm, supportive, and constructive. Focus on growth rather than shortcomings.
"""


def determine_mode(is_rejected: bool, company_name: str, search_summary: str) -> AnalysisMode:
    """Determine which prompt mode to use.

    Priority: FAILURE > STANDARD > NO_COMPANY
    """
    if is_rejected:
        return AnalysisMode.FAILURE
    if company_name and search_summary:
        return AnalysisMode.STANDARD
    return AnalysisMode.NO_COMPANY


def build_prompt(
    transcript: str,
    job_title: str,
    job_description: str,
    mode: AnalysisMode,
    detected_language: str = "en",
    company_name: str = "",
    search_summary: str = "",
) -> str:
    """Build the final prompt based on the analysis mode."""
    lang_key = detected_language if detected_language in LANGUAGE_INSTRUCTIONS else "en"
    language_instruction = LANGUAGE_INSTRUCTIONS[lang_key]

    if mode == AnalysisMode.STANDARD:
        return STANDARD_TEMPLATE.format(
            company_name=company_name,
            search_summary=search_summary,
            job_title=job_title,
            job_description=job_description or "Not provided",
            transcript=transcript,
            language_instruction=language_instruction,
        )

    if mode == AnalysisMode.NO_COMPANY:
        return NO_COMPANY_TEMPLATE.format(
            job_title=job_title,
            job_description=job_description or "Not provided",
            transcript=transcript,
            language_instruction=language_instruction,
        )

    # FAILURE mode
    company_section = ""
    if company_name and search_summary:
        company_section = f"## Company Background\nCompany: {company_name}\n{search_summary}"
    elif company_name:
        company_section = f"## Company\n{company_name}"

    return FAILURE_TEMPLATE.format(
        job_title=job_title,
        job_description=job_description or "Not provided",
        company_section=company_section,
        transcript=transcript,
        language_instruction=language_instruction,
    )
