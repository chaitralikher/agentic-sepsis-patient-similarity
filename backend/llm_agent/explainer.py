from backend.llm_agent.ollama_client import query_ollama

def _build_prompt(comparison_dict, sepsis_rate):
    feature_lines = "\n".join(
        [f"{k}: {v}" for k, v in comparison_dict.items()]
    )

    return f"""
You are a clinical data interpretation assistant.

An ICU patient's physiologic profile has been compared to similar patients using a similarity algorithm.

Neighborhood sepsis prevalence: {sepsis_rate:.2f}

Feature comparison between the index patient and similar patients:
{feature_lines}

Instructions:
- Use ONLY the information provided in the feature comparison.
- Describe whether the patient's values are higher, lower, or similar relative to neighbors.
- Explain possible physiologic implications cautiously using general language.
- Do NOT diagnose diseases.
- Do NOT predict outcomes.
- Do NOT infer unobserved variables.
- Do NOT recommend treatments.
- Do NOT introduce clinical concepts that are not directly related to the listed features.

Start the explanation with:
"The index patient shows physiologic similarity to neighboring ICU patients with respect to..."

First list 3–4 key physiologic similarities as bullet points.
Then provide a short summary paragraph in concise 3–5 sentence clinical-style interpretation summarizing the similarity patterns.

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
