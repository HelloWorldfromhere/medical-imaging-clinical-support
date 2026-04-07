"""
LLM Provider — Swappable backend for clinical summary generation.

Supports:
    - google_gemini (default, free tier)
    - groq (free tier, fast, Llama/Mixtral)
    - stub (no API needed, for testing)

To switch providers, change LLM_PROVIDER in .env:
    LLM_PROVIDER=google_gemini   (default)
    LLM_PROVIDER=groq
    LLM_PROVIDER=stub

Usage:
    from api.llm_provider import generate_clinical_summary
    summary = generate_clinical_summary(query, cnn_prediction, confidence, chunks)
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google_gemini")

CLINICAL_SYSTEM_PROMPT = """You are a medical AI assistant providing clinical decision support.
You synthesize retrieved medical literature into actionable clinical summaries.

Your response must include:
1. **Clinical Interpretation** — What the findings suggest based on the retrieved literature
2. **Differential Diagnoses** — Other conditions to consider, ranked by likelihood
3. **Recommended Next Steps** — Diagnostic tests, imaging, or labs to order
4. **Treatment Considerations** — Evidence-based treatment options from the literature
5. **Safety Warnings** — Infection control precautions, drug interactions, or urgent actions needed

Rules:
- Cite sources using [Source N] format matching the chunk numbers provided
- If the retrieved literature doesn't adequately cover a topic, say so explicitly
- Always end with a disclaimer that this is AI-generated and requires physician review
- Be concise but thorough — aim for 200-400 words
- Prioritize patient safety in all recommendations"""


def _build_prompt(query, cnn_prediction, confidence, chunks):
    """Build the LLM prompt with retrieved context."""
    context_block = "\n\n".join(
        f"[Source {i+1}] (PMID: {chunk['doc_id']}, relevance: {chunk['similarity']:.3f})\n{chunk['chunk_text']}"
        for i, chunk in enumerate(chunks)
    )

    prompt = f"""A clinical query has been submitted to the Medical RAG system.

**AI Classification Result:**
- Prediction: {cnn_prediction}
- Confidence: {confidence:.1%}

**Clinical Query:**
{query}

**Retrieved Medical Literature ({len(chunks)} sources):**
{context_block}

Based on the retrieved literature above, provide a clinical summary following the required format."""

    return prompt


def _generate_gemini(prompt):
    """Generate using Google Gemini API."""
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in .env")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    response = model.generate_content(
        [
            {"role": "user", "parts": [CLINICAL_SYSTEM_PROMPT + "\n\n" + prompt]}
        ],
        generation_config=genai.types.GenerationConfig(
            temperature=0.3,
            max_output_tokens=800,
        ),
    )

    return response.text


def _generate_groq(prompt):
    """Generate using Groq API (free tier, Llama/Mixtral)."""
    import requests

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in .env")

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": CLINICAL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 800,
        },
    )

    if response.status_code != 200:
        raise ValueError(f"Groq API error: {response.status_code} {response.text}")

    return response.json()["choices"][0]["message"]["content"]


def _generate_stub(prompt):
    """Stub response for testing without API keys."""
    return (
        "**Clinical Interpretation**\n"
        "This is a placeholder response. To enable real clinical summaries, "
        "configure an LLM provider in your .env file:\n"
        "- Set LLM_PROVIDER=google_gemini and add GEMINI_API_KEY\n"
        "- Or set LLM_PROVIDER=groq and add GROQ_API_KEY\n\n"
        "*Disclaimer: This is an AI-generated educational demo, not clinical advice.*"
    )


PROVIDERS = {
    "google_gemini": _generate_gemini,
    "groq": _generate_groq,
    "stub": _generate_stub,
}


def generate_clinical_summary(query, cnn_prediction, confidence, chunks):
    """
    Generate a clinical summary from retrieved chunks.

    Args:
        query: Clinical query text
        cnn_prediction: CNN model prediction (e.g., "pneumonia")
        confidence: CNN confidence score (0-1)
        chunks: List of dicts with keys: doc_id, chunk_text, similarity

    Returns:
        str: Generated clinical summary
    """
    provider = LLM_PROVIDER
    if provider not in PROVIDERS:
        logger.warning(f"Unknown LLM provider '{provider}', falling back to stub")
        provider = "stub"

    prompt = _build_prompt(query, cnn_prediction, confidence, chunks)

    try:
        result = PROVIDERS[provider](prompt)
        logger.info(f"LLM generation complete ({provider}): {len(result)} chars")
        return result
    except Exception as e:
        logger.error(f"LLM generation failed ({provider}): {e}")
        return f"*LLM generation failed: {str(e)}*\n\nRetrieved {len(chunks)} relevant sources — see chunks above for raw evidence."
