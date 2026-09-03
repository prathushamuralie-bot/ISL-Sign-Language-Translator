"""
FULLY OFFLINE Multilingual Grammar Correction Module
-------------------------------------------------------
Uses a small open-source LLM running locally on your machine via Ollama.
No internet needed after the one-time model download. No API key,
no billing, no cloud calls. Understands meaning (unlike rule-based
tools), so it can fix things like "my name Prathusha" -> "My name is
Prathusha" correctly, in any language, without changing meaning or
names.

SETUP (one-time):
    1. Install Ollama: https://ollama.com/download  (Windows/Mac/Linux)
    2. Open a terminal and run:
           ollama pull llama3.2
       (downloads a ~2GB small model, one-time, needs internet ONCE
        for this download only)
    3. Install the python client:
           pip install ollama
    4. After this, everything runs 100% offline — no internet needed
       to use correct_grammar_offline() ever again.
"""

import ollama


MODEL = "llama3.2:1b"  # SMALLER model (~1.3GB) - much faster than the 3B version, still good enough for short-sentence grammar fixes


def correct_grammar_offline(text: str, model: str = MODEL) -> dict:
    """
    Corrects grammar of the input sentence using a local LLM (via Ollama).
    Works fully offline. Detects the language automatically and keeps
    the correction in the SAME language, without changing meaning or names.

    Args:
        text: raw sentence (e.g. ISL-translated output)
        model: Ollama model name (default "llama3.2")

    Returns:
        dict with keys: original, corrected
    """
    if not text or not text.strip():
        return {"original": text, "corrected": text}

    system_prompt = (
        "You are a strict grammar-only correction engine for a sign-language "
        "translator app. You will receive a short sentence that may have grammar "
        "mistakes (missing verbs like 'is/am/are', wrong word order, missing "
        "tense markers, missing articles, etc).\n\n"
        "ABSOLUTE RULES — breaking any of these is a critical failure:\n"
        "1. Detect the language of the input automatically.\n"
        "2. Correct ONLY the grammar, using that SAME language's own grammar rules.\n"
        "3. NEVER translate to another language. Output must stay in the input language.\n"
        "4. NEVER add ANY new information, words, places, activities, or continuation "
        "that was not in the original sentence. Do NOT continue or extend the sentence.\n"
        "5. NEVER change names, proper nouns, or numbers.\n"
        "6. If the sentence is already grammatically correct, return it EXACTLY unchanged.\n"
        "7. The corrected sentence must have almost the same number of words as the "
        "input — you are only fixing grammar, not writing a new sentence.\n"
        "8. Reply with ONLY the corrected sentence. No explanation, no quotes, "
        "no extra text — just the corrected sentence itself.\n\n"
        "EXAMPLES:\n"
        "Input: my name Prathusha\n"
        "Output: My name is Prathusha\n\n"
        "Input: I Sri\n"
        "Output: I am Sri\n\n"
        "Input: she eating food now\n"
        "Output: She is eating food now\n\n"
        "Input: Hello Thank You\n"
        "Output: Hello Thank You"
    )

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        keep_alive="30m",  # keep model loaded in memory between calls -> avoids reload delay
        options={
            "num_predict": 60,   # cap output length - short sentences, no need for long generations
            "temperature": 0.1,  # more deterministic + slightly faster
        },
    )

    corrected = response["message"]["content"].strip()
    # Strip accidental quote marks the model sometimes adds
    corrected = corrected.strip('"').strip("'").strip()

    # SAFETY NET: small models sometimes hallucinate extra content (e.g. "I Sri"
    # -> "I Sri am going to market"). If the corrected sentence has grown by
    # more than 3 words compared to the original, something was likely
    # invented — fall back to the original text instead of risking wrong info.
    original_word_count = len(text.split())
    corrected_word_count = len(corrected.split())
    if corrected_word_count > original_word_count + 3:
        print(f"[grammar_correct] Rejected hallucinated output: '{corrected}' (kept original)")
        corrected = text

    return {"original": text, "corrected": corrected}


if __name__ == "__main__":
    samples = [
        "my name Prathusha",
        "I go college everyday",
        "she eating food now",
        "yesterday I meeting friend",
        "என் பெயர் பிரதுஷா ரெண்டு",
        "நான் நேற்று பள்ளி போனேன் இல்லை",
    ]
    for s in samples:
        result = correct_grammar_offline(s)
        print(f"{result['original']}  -->  {result['corrected']}")
