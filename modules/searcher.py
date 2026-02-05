import requests

from config import SERPER_ENDPOINT


def search_company(company_name: str, job_title: str, api_key: str) -> str:
    """Search for company background information using Serper.dev API.

    Returns a summary string, or empty string on any failure (triggering graceful degradation).
    """
    if not api_key or not company_name:
        return ""

    query = f"{company_name} {job_title} company overview"

    try:
        resp = requests.post(
            SERPER_ENDPOINT,
            json={"q": query, "num": 5},
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return ""

    parts = []

    # Extract knowledge graph if available
    kg = data.get("knowledgeGraph", {})
    if kg:
        if kg.get("title"):
            parts.append(kg["title"])
        if kg.get("description"):
            parts.append(kg["description"])

    # Extract organic results snippets
    for item in data.get("organic", [])[:3]:
        snippet = item.get("snippet", "")
        if snippet:
            parts.append(snippet)

    return "\n".join(parts)
