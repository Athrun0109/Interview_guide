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

## Output Format
Structure your response with these exact section headers:

### Overall Impression
(2-3 sentences summarizing the candidate's overall performance)

### Q&A Breakdown and Analysis
(Break the interview into individual Q&A exchanges. For each exchange, give a qualitative rating Good/Neutral/Bad with explanation)

### Interviewer's Focus Areas
(Infer the interviewer's focus areas and questioning logic)

### Improvement Suggestions
(Provide 3 specific, actionable improvement suggestions)

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

## Output Format
Structure your response with these exact section headers:

### Overall Impression
(2-3 sentences summarizing the candidate's overall performance)

### Q&A Breakdown and Analysis
(Break the interview into individual Q&A exchanges. For each exchange, give a qualitative rating Good/Neutral/Bad with explanation)

### Interviewer's Focus Areas
(Infer the interviewer's focus areas and questioning logic)

### Improvement Suggestions
(Provide 3 specific, actionable improvement suggestions)

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

## Output Format
Structure your response with these exact section headers:

### Overall Impression
(2-3 sentences summarizing the interview and providing encouragement)

### Q&A Breakdown and Analysis
(Break the interview into individual Q&A exchanges. For each exchange, give a qualitative rating Good/Neutral/Bad with explanation)

### What Went Well
(Highlight the positive aspects — strengths to carry forward)

### Improvement Suggestions
(Identify 3 key improvement areas with specific examples and concrete suggestions)

{language_instruction}
Be warm, supportive, and constructive. Focus on growth rather than shortcomings.
"""


EXPORTABLE_PROMPT_TEMPLATE = """I just finished a job interview and would like your help analyzing my performance.

## Position Information
- Position: {job_title}
- Company: {company_name}
{job_description_section}

## Interview Status
{status_note}

## Full Interview Transcript
{transcript}

## What I Need
Please analyze this interview and provide:
1. Overall impression of my performance (2-3 sentences)
2. Breakdown of each Q&A exchange with ratings (Good/Neutral/Bad) and brief explanations
3. The interviewer's apparent focus areas and questioning strategy
4. 3 specific, actionable suggestions for improvement

{language_instruction}
"""


def build_exportable_prompt(
    transcript: str,
    job_title: str,
    job_description: str,
    company_name: str,
    is_rejected: bool,
    detected_language: str = "en",
) -> str:
    """Build a self-contained prompt that users can copy to any LLM."""
    lang_key = detected_language if detected_language in LANGUAGE_INSTRUCTIONS else "en"
    language_instruction = LANGUAGE_INSTRUCTIONS[lang_key]

    job_description_section = ""
    if job_description:
        job_description_section = f"- Job Requirements:\n{job_description}"

    status_note = "I was rejected for this position. Please be supportive while giving honest feedback." if is_rejected else "Waiting for results or proceeding to next round."

    return EXPORTABLE_PROMPT_TEMPLATE.format(
        job_title=job_title or "Not specified",
        company_name=company_name or "Not specified",
        job_description_section=job_description_section,
        status_note=status_note,
        transcript=transcript,
        language_instruction=language_instruction,
    )


def parse_exported_prompt(prompt_text: str) -> dict:
    """Parse an exported prompt to extract job info and transcript.

    Returns a dict with keys: job_title, company_name, job_description, transcript, is_rejected
    """
    import re

    result = {
        "job_title": "",
        "company_name": "",
        "job_description": "",
        "transcript": "",
        "is_rejected": False,
    }

    # Extract Position
    position_match = re.search(r"-\s*Position:\s*(.+)", prompt_text)
    if position_match:
        value = position_match.group(1).strip()
        if value.lower() != "not specified":
            result["job_title"] = value

    # Extract Company
    company_match = re.search(r"-\s*Company:\s*(.+)", prompt_text)
    if company_match:
        value = company_match.group(1).strip()
        if value.lower() != "not specified":
            result["company_name"] = value

    # Extract Job Requirements
    jd_match = re.search(r"-\s*Job Requirements:\s*\n(.*?)(?=\n##|\Z)", prompt_text, re.DOTALL)
    if jd_match:
        result["job_description"] = jd_match.group(1).strip()

    # Extract Interview Status (check if rejected)
    if "rejected" in prompt_text.lower() and "was rejected" in prompt_text.lower():
        result["is_rejected"] = True

    # Extract Transcript
    transcript_match = re.search(
        r"##\s*Full Interview Transcript\s*\n(.*?)(?=\n##\s*What I Need|\Z)",
        prompt_text,
        re.DOTALL
    )
    if transcript_match:
        result["transcript"] = transcript_match.group(1).strip()

    return result


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
