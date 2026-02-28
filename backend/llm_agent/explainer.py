from backend.llm_agent.ollama_client import query_ollama

def _build_prompt(comparison_dict, sepsis_rate):
    feature_lines = "\n".join(
        [f"{k}: {v}" for k, v in comparison_dict.items()]
    )

    return f"""
You are a clinical decision support assistant.

A patient's physiologic profile is compared to similar ICU patients.

Neighborhood sepsis prevalence: {sepsis_rate:.2f}

Feature comparison:
{feature_lines}

Write a concise clinical interpretation of similarity and risk.
"""


def generate_explanation(comparison_dict, sepsis_rate):
    """
    Generate explanation using LLM.
    If API fails → return rule-based fallback.
    """

    prompt = _build_prompt(comparison_dict, sepsis_rate)

    # ---------- BLOCKER HANDLING ----------
    if not query_ollama:
        return _fallback_explanation(sepsis_rate)

    try:
        response = query_ollama(prompt)
        return response

    except Exception as e:
        print("LLM API ERROR:", str(e))
        return _fallback_explanation(sepsis_rate)


def _fallback_explanation(sepsis_rate):
    """
    Used if API fails or not configured.
    """

    if sepsis_rate > 0.6:
        risk = "high"
    elif sepsis_rate > 0.3:
        risk = "moderate"
    else:
        risk = "low"

    return f"""
LLM explanation unavailable.

Based on neighborhood similarity,
estimated sepsis risk appears {risk}
(sepsis prevalence = {sepsis_rate:.2f}).
"""
