from google import genai

from config import GEMINI_TEMPERATURE, GEMINI_MAX_OUTPUT_TOKENS
from modules.prompts import AnalysisMode, build_prompt

# Available Gemini models for analysis
GEMINI_MODELS = {
    "gemini-2.0-flash": "Gemini 2.0 Flash (fast, recommended)",
    "gemini-2.0-flash-lite": "Gemini 2.0 Flash Lite (fastest, lower quality)",
    "gemini-1.5-flash": "Gemini 1.5 Flash (stable)",
    "gemini-1.5-pro": "Gemini 1.5 Pro (best quality, slower)",
}


def analyze_interview(
    transcript: str,
    job_title: str,
    job_description: str,
    mode: AnalysisMode,
    api_key: str,
    model: str = "gemini-2.0-flash",
    detected_language: str = "en",
    company_name: str = "",
    search_summary: str = "",
) -> str:
    """Send interview transcript to Gemini for analysis and return the report text."""
    prompt = build_prompt(
        transcript=transcript,
        job_title=job_title,
        job_description=job_description,
        mode=mode,
        detected_language=detected_language,
        company_name=company_name,
        search_summary=search_summary,
    )

    client = genai.Client(
        api_key=api_key,
        vertexai=False,
        http_options={"api_version": "v1beta"},
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "temperature": GEMINI_TEMPERATURE,
            "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS,
        },
    )

    return response.text
