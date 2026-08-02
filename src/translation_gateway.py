import os

TRANSLATION_PROVIDER = os.getenv("TRANSLATION_PROVIDER", "none")

def detect_user_language(text: str) -> str:
    # Placeholder for future language detection API
    return "UNKNOWN"

def translate_to_english(text: str, source_lang: str) -> str:
    if TRANSLATION_PROVIDER == "none" or TRANSLATION_PROVIDER == "mock":
        return text
    elif TRANSLATION_PROVIDER == "google_cloud_later":
        raise NotImplementedError("Google Cloud Translation is not yet implemented.")
    return text

def translate_from_english(text: str, target_lang: str) -> str:
    if TRANSLATION_PROVIDER == "none" or TRANSLATION_PROVIDER == "mock":
        return text
    elif TRANSLATION_PROVIDER == "google_cloud_later":
        raise NotImplementedError("Google Cloud Translation is not yet implemented.")
    return text

def protect_terms_before_translation(text: str, protected_terms: list) -> tuple:
    protected_map = {}
    for i, term in enumerate(protected_terms):
        placeholder = f"__PROTECTED_{i}__"
        protected_map[placeholder] = term
        text = text.replace(term, placeholder)
    return text, protected_map

def restore_protected_terms(text: str, protected_map: dict) -> str:
    for placeholder, term in protected_map.items():
        text = text.replace(placeholder, term)
    return text
