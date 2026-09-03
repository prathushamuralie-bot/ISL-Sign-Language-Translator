"""
FREE Multilingual Grammar Correction Module (No API cost)
-----------------------------------------------------------
Runs LanguageTool locally on your machine — fully offline after first
download, no API key, no billing, no rate limit.

Install first:
    pip install language-tool-python
"""

import language_tool_python


def correct_grammar_local(text: str, lang: str = "en-US") -> dict:
    """
    Corrects grammar mistakes only (missing verbs, wrong tense,
    subject-verb agreement, word order). Spell-checking is OFF so
    proper nouns / names (e.g. "Prathusha") never get changed —
    meaning is preserved.
    """
    if not text or not text.strip():
        return {"original": text, "corrected": text}

    tool = language_tool_python.LanguageTool(lang)

    tool.disabled_rules = {
        "MORFOLOGIK_RULE_EN_US",
        "HUNSPELL_RULE",
    }
    tool.disabled_categories = {"TYPOS", "CASING"}

    matches = tool.check(text)

    safe_matches = []
    for m in matches:
        original_word = text[m.offset: m.offset + m.errorLength]
        if original_word[:1].isupper() and original_word.isalpha():
            if m.category not in ("TYPOS", "CASING"):
                safe_matches.append(m)
        else:
            safe_matches.append(m)

    corrected = language_tool_python.utils.correct(text, safe_matches)
    return {"original": text, "corrected": corrected}


if __name__ == "__main__":
    samples = [
        "my name Prathusha",
        "I go college everyday",
        "she eating food now",
        "yesterday I meeting friend",
    ]
    for s in samples:
        result = correct_grammar_local(s, lang="en-US")
        print(f"{result['original']}  -->  {result['corrected']}")