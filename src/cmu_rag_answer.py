import json
import typing
from openai import OpenAI, APIConnectionError, APIError
import os
import sys
import re
import gc
import torch

# Make the repo-root config.py importable regardless of how the entry point
# (Streamlit, pytest, run_backend_tests.py) set up sys.path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import config

def get_truncated_history(history):
    return [{'role': msg['role'], 'content': msg['content'][:300] + '...' if len(msg['content']) > 300 else msg['content']} for msg in history]

HOTEL_METADATA_CACHE = None

def load_hotel_metadata():
    global HOTEL_METADATA_CACHE
    if HOTEL_METADATA_CACHE is None:
        try:
            metadata_path = os.path.join(_REPO_ROOT, "data", "cmu_hotel_metadata.json")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                HOTEL_METADATA_CACHE = json.load(f)
        except (OSError, json.JSONDecodeError):
            HOTEL_METADATA_CACHE = {}
    return HOTEL_METADATA_CACHE

from cmu_retrieve import search
from travelmind_scoring import (
    calculate_travelmind_score,
    build_strengths,
    build_cautions,
)
from hotel_feature_verbalizer import (
    build_grounded_hotel_answer,
    coalesce_hotel_rewrite_sentences,
    get_recorded_room_types,
    get_hotel_feature_facts,
    get_verified_amenities,
    join_english,
    validate_hotel_feature_rewrite,
)

# Reconfigure stdout for Windows emoji support
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

TOP_K_RETRIEVAL = 10
TOP_K_FOR_LLM = 3
DEBUG = False


def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clear_python_gpu_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("Python GPU cache cleared.")


_CACHED_FOUNDRY_BASE_URL = None

def get_foundry_base_url(force_refresh=False):
    """Start/reuse Foundry and return its OpenAI-compatible ``/v1`` URL.

    Existing 0.x CLI installations are preferred so their already-downloaded
    CUDA models can be reused. SDK 1.x remains a portable fallback.
    """
    global _CACHED_FOUNDRY_BASE_URL
    if _CACHED_FOUNDRY_BASE_URL is not None and not force_refresh:
        return _CACHED_FOUNDRY_BASE_URL

    from foundry_runtime import ensure_legacy_runtime, ensure_runtime

    legacy = ensure_legacy_runtime(force_restart=force_refresh)
    if legacy is not None:
        endpoint, _ = legacy
    else:
        _, _, endpoint = ensure_runtime(force_restart=force_refresh)

    if DEBUG:
        print("Foundry endpoint:", endpoint)
    _CACHED_FOUNDRY_BASE_URL = endpoint
    return endpoint


def get_foundry_client_and_model():
    """Builds an OpenAI client against the cached Foundry endpoint. If the
    cached endpoint has gone stale (e.g. Foundry was restarted on a new
    port) and the connection fails, re-discovers the endpoint once and
    retries before giving up."""
    try:
        base_url = get_foundry_base_url()
        client = OpenAI(base_url=base_url, api_key='not-needed', timeout=300.0)
        model_id = get_available_model_id(client)
        return client, model_id
    except APIConnectionError:
        base_url = get_foundry_base_url(force_refresh=True)
        client = OpenAI(base_url=base_url, api_key='not-needed', timeout=300.0)
        model_id = get_available_model_id(client)
        return client, model_id


def create_chat_completion_with_retry(make_request):
    """make_request(client, model_id) -> streamed response. Retries once
    with a freshly discovered Foundry endpoint if the cached one raises a
    connection error (e.g. Foundry restarted on a different port)."""
    client, model_id = get_foundry_client_and_model()
    try:
        return make_request(client, model_id)
    except APIConnectionError:
        base_url = get_foundry_base_url(force_refresh=True)
        client = OpenAI(base_url=base_url, api_key='not-needed', timeout=300.0)
        model_id = get_available_model_id(client)
        return make_request(client, model_id)


def get_available_model_id(client):
    """Return the concrete loaded variant for the configured model alias."""
    try:
        models = client.models.list()
        model_ids = [model.id for model in models.data]
    except (APIConnectionError, APIError) as e:
        raise RuntimeError(
            f"Could not reach Foundry to verify model '{config.MODEL_ALIAS}' ({e}). "
            "Run: .venv\\Scripts\\python.exe scripts\\setup_foundry_runtime.py"
        ) from e

    if config.MODEL_ID in model_ids:
        selected_model = config.MODEL_ID
    else:
        prefix = config.MODEL_ALIAS.lower() + "-"
        matching_ids = [
            model_id for model_id in model_ids
            if model_id.lower() == config.MODEL_ALIAS.lower()
            or model_id.lower().startswith(prefix)
        ]
        selected_model = matching_ids[0] if len(matching_ids) == 1 else None

    if selected_model is None:
        raise RuntimeError(
            f"Configured model alias '{config.MODEL_ALIAS}' is not loaded on the Foundry endpoint. "
            f"Models currently available: {model_ids or '(none)'}. "
            "Run: .venv\\Scripts\\python.exe scripts\\setup_foundry_runtime.py"
        )

    if DEBUG:
        print("Using pinned model variant:", selected_model)
    return selected_model

def verbalize_amenity(feature_name, status, language="en"):
    fname_en = feature_name.replace("_", " ").lower()

    if status == "YES":
        return f"This hotel offers {fname_en}."
    elif status == "NO":
        return f"Unfortunately, this hotel does not have {fname_en}."
    else:
        return ""

def verbalize_room_info(room_type, status, language="en"):
    rname_en = room_type.replace("_", " ").lower()

    if status == "YES":
        return f"Also, {rname_en} options are available."
    elif status == "NO":
        return f"Based on current information, {rname_en} options are not visible."
    else:
        return ""



def build_hotel_context(query, card, index, lang_code="en"):
    """Compact, allowlisted card context for legacy model-backed callers."""
    facts = get_hotel_feature_facts(card)
    hotel_class = str(card.get("hotel_class") or "").strip()
    if hotel_class.casefold() in {"", "unknown", "none", "null", "nan"}:
        hotel_class = "Not stated"
    phone = _clean_public_card_value(card.get("phone")) or "Not stated"
    map_link = _safe_map_link(card.get("map_link")) or "Not stated"
    amenities_str = join_english(facts["amenities"]) or "None supplied"
    room_types_str = join_english(facts["room_types"]) or "None supplied"

    context = f"""
Hotel Card {index}:
- Hotel name: {facts['hotel_name']}
- City: {facts['location']}
- Hotel Class (Star Rating): {hotel_class}
- Verified phone: {phone}
- Verified map link: {map_link}
- TravelMind suitability score: {facts['score'] or 'Not stated'}
- Verified amenities: {amenities_str}
- Recorded room types (static categories, not live availability): {room_types_str}
"""

    return context.strip()


_UNKNOWN_CARD_VALUES = {"", "unknown", "none", "null", "nan", "n/a", "not available"}


def normalize_hotel_reference(value) -> str:
    """Normalize a hotel reference without relying on locale-sensitive text."""
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _clean_public_card_value(value) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if text.casefold() in _UNKNOWN_CARD_VALUES:
        return ""
    return text


def safe_map_link(value) -> str:
    """Allow only the Google Maps URLs created by the card builder."""
    link = _clean_public_card_value(value)
    if re.match(r"^https://(?:www\.)?google\.com/maps/", link, re.IGNORECASE):
        return link
    if re.match(r"^https://maps\.google\.com/", link, re.IGNORECASE):
        return link
    return ""


# Compatibility for older internal imports/tests.
_safe_map_link = safe_map_link


def clamp_selected_hotel_index(last_hotel_cards, selected_index=0) -> int:
    """Return a safe selection index; stale/invalid state always resets to zero."""
    cards = list(last_hotel_cards or [])
    if not cards:
        return 0
    try:
        index = int(selected_index)
    except (TypeError, ValueError):
        return 0
    return index if 0 <= index < len(cards) else 0


def find_referenced_hotel_index(
    last_hotel_cards,
    question="",
    requested_hotel_name=None,
):
    """Find an explicit prior-card reference, preferring the router's name."""
    cards = list(last_hotel_cards or [])
    requested = normalize_hotel_reference(requested_hotel_name)
    question_norm = normalize_hotel_reference(question)

    def matches(reference, hotel_name):
        if not reference or not hotel_name:
            return False
        without_the = hotel_name.removeprefix("the ")
        return (
            reference == hotel_name
            or reference in hotel_name
            or hotel_name in reference
            or (len(without_the) > 3 and without_the in reference)
        )

    if requested:
        for index, card in enumerate(cards):
            if matches(requested, normalize_hotel_reference(card.get("hotel_name"))):
                return index
    for index, card in enumerate(cards):
        if matches(question_norm, normalize_hotel_reference(card.get("hotel_name"))):
            return index
    return None


def resolve_hotel_selection(
    last_hotel_cards,
    question="",
    requested_hotel_name=None,
    selected_index=0,
):
    """Resolve explicit name -> current selection -> first card, in that order."""
    cards = list(last_hotel_cards or [])
    if not cards:
        return None, 0
    referenced_index = find_referenced_hotel_index(
        cards,
        question=question,
        requested_hotel_name=requested_hotel_name,
    )
    index = (
        referenced_index
        if referenced_index is not None
        else clamp_selected_hotel_index(cards, selected_index)
    )
    return cards[index], index


def next_hotel_index(last_hotel_cards, selected_index=0):
    """Return the next result index, or ``None`` when the list is exhausted."""
    cards = list(last_hotel_cards or [])
    if not cards:
        return None
    next_index = clamp_selected_hotel_index(cards, selected_index) + 1
    return next_index if next_index < len(cards) else None


def _amenity_status(card, amenity_key) -> str:
    amenities = (card or {}).get("amenities", {})
    if not isinstance(amenities, dict):
        return "UNKNOWN"
    value = str(amenities.get(amenity_key, "UNKNOWN")).upper()
    return value if value in {"YES", "NO"} else "UNKNOWN"


def build_amenity_followup_answer(
    last_hotel_cards,
    amenity_key,
    question="",
    requested_hotel_name=None,
    selected_index=0,
):
    """Answer pool/breakfast follow-ups from card facts and return new selection."""
    cards = list(last_hotel_cards or [])
    label = "a pool" if amenity_key == "pool" else "breakfast"
    if not cards:
        return (
            f"I do not have previous hotel results to check for {label}. "
            "Please search for a supported city first.",
            0,
        )

    referenced_index = find_referenced_hotel_index(
        cards,
        question=question,
        requested_hotel_name=requested_hotel_name,
    )
    q_norm = normalize_hotel_reference(question)
    asks_across_results = referenced_index is None and bool(
        re.search(
            r"\b(?:which|what|any|all|hotels|ones|options|list|who)\b",
            q_norm,
        )
    )
    current_index = clamp_selected_hotel_index(cards, selected_index)

    if asks_across_results:
        confirmed = [
            _clean_public_card_value(card.get("hotel_name")) or "Unnamed hotel"
            for card in cards
            if _amenity_status(card, amenity_key) == "YES"
        ]
        if confirmed:
            return (
                f"Based on the verified hotel profiles, {join_english(confirmed)} "
                f"list{'s' if len(confirmed) == 1 else ''} {label}.",
                current_index,
            )
        if all(_amenity_status(card, amenity_key) == "NO" for card in cards):
            return (
                f"None of the current hotel profiles lists {label}.",
                current_index,
            )
        return (
            f"The current hotel profiles do not confirm {label} for any of the displayed options.",
            current_index,
        )

    card, resolved_index = resolve_hotel_selection(
        cards,
        question=question,
        requested_hotel_name=requested_hotel_name,
        selected_index=current_index,
    )
    name = _clean_public_card_value(card.get("hotel_name")) or "The selected hotel"
    status = _amenity_status(card, amenity_key)
    if status == "YES":
        answer = f"**{name}** lists {label} in its verified hotel profile."
    elif status == "NO":
        answer = f"**{name}** does not list {label} in its hotel profile."
    else:
        answer = (
            f"TravelMind's current record for **{name}** does not confirm "
            f"whether it offers {label}."
        )
    return answer, resolved_index


def build_grounded_followup_answer(
    question,
    last_hotel_cards,
    selected_index=0,
    requested_hotel_name=None,
):
    """Build a deterministic answer for common hotel-card follow-ups.

    The returned index is the single selection state the UI should persist.
    No live price/availability claim is ever inferred from static room types.
    """
    cards = list(last_hotel_cards or [])
    if not cards:
        return (
            "I do not have previous hotel results to inspect. Please search for a supported city first.",
            0,
        )
    card, index = resolve_hotel_selection(
        cards,
        question=question,
        requested_hotel_name=requested_hotel_name,
        selected_index=selected_index,
    )
    q = normalize_hotel_reference(question)
    name = _clean_public_card_value(card.get("hotel_name")) or "The selected hotel"

    if re.search(r"\b(?:phone|telephone|contact|call|number)\b", q):
        phone = _clean_public_card_value(card.get("phone"))
        if phone:
            return f"The verified phone number listed for **{name}** is {phone}.", index
        return (
            f"TravelMind's current record for **{name}** does not include a verified phone number.",
            index,
        )

    if re.search(r"\b(?:where|location|located|address|map|directions)\b", q):
        location = _clean_public_card_value(card.get("location"))
        map_link = _safe_map_link(card.get("map_link"))
        if location and map_link:
            return f"**{name}** is listed in {location}. [View it on the map]({map_link}).", index
        if location:
            return f"**{name}** is listed in {location}; a verified map link is not available in the current record.", index
        if map_link:
            return f"TravelMind's current record does not state a location for **{name}**, but it includes this [verified map link]({map_link}).", index
        return f"TravelMind's current record for **{name}** does not include a verified location or map link.", index

    if re.search(r"\b(?:score|rating|rated)\b", q) and not re.search(
        r"\b(?:review|guest|stars?|class)\b", q
    ):
        score = get_hotel_feature_facts(card)["score"]
        if score:
            return f"**{name}** has a TravelMind suitability score of {score}.", index
        return f"TravelMind's current record for **{name}** does not include a suitability score.", index

    if re.search(r"\b(?:class|stars?|star rating)\b", q):
        hotel_class = _clean_public_card_value(card.get("hotel_class"))
        if hotel_class:
            return f"The hotel class listed for **{name}** is {hotel_class} stars.", index
        return f"TravelMind's current record for **{name}** does not include a verified hotel class.", index

    if re.search(r"\b(?:rooms?|room types?|accommodation types?)\b", q):
        room_types = get_recorded_room_types(card, limit=6)
        if room_types:
            return (
                f"The static profile for **{name}** records {join_english(room_types)}. "
                "These are room categories, not live availability.",
                index,
            )
        return f"TravelMind's current record for **{name}** does not include verified room types or live availability.", index

    if re.search(r"\b(?:amenities|amenity|facilities|features|offers|have|has)\b", q):
        amenities = get_verified_amenities(card, limit=8)
        if amenities:
            return f"Verified amenities for **{name}** include {join_english(amenities)}.", index
        return f"TravelMind's current record for **{name}** does not include verified amenity details.", index

    facts = get_hotel_feature_facts(card, amenity_limit=8, room_limit=6)
    details = []
    location = _clean_public_card_value(card.get("location"))
    hotel_class = _clean_public_card_value(card.get("hotel_class"))
    if location:
        details.append(f"it is listed in {location}")
    if hotel_class:
        details.append(f"its listed class is {hotel_class} stars")
    if facts["score"]:
        details.append(f"its TravelMind score is {facts['score']}")
    if facts["amenities"]:
        details.append(f"verified amenities include {join_english(facts['amenities'])}")
    if facts["room_types"]:
        details.append(f"recorded room types include {join_english(facts['room_types'])}")
    if details:
        return f"For **{name}**, " + "; ".join(details) + ".", index
    return f"TravelMind's current record for **{name}** does not contain further verified details.", index


def stream_and_strip_think(response, lang_code):
    """Stream chunks, filtering <think>...</think> blocks safely even when tags split across chunks.
    Yields dicts: {"type": "think"|"answer", "content": str}
    """
    buffer = ""
    in_think = False

    OPEN_TAG = "<think>"
    CLOSE_TAG = "</think>"

    for chunk in response:
        delta = chunk.choices[0].delta.content or ""
        if not delta:
            continue
        buffer += delta

        # Process buffer until no more complete tags can be handled
        while True:
            if in_think:
                close_pos = buffer.find(CLOSE_TAG)
                if close_pos != -1:
                    # Yield think content up to close tag
                    think_content = buffer[:close_pos]
                    if think_content:
                        yield {"type": "think", "content": think_content}
                    buffer = buffer[close_pos + len(CLOSE_TAG):]
                    in_think = False
                else:
                    # Check if we might be in a partial close tag at the end
                    partial = False
                    for end_len in range(1, len(CLOSE_TAG)):
                        if buffer.endswith(CLOSE_TAG[:end_len]):
                            partial = True
                            break
                    if partial:
                        # Hold back potential partial tag, yield everything before it
                        safe_end = len(buffer) - len(CLOSE_TAG) + 1
                        if safe_end > 0:
                            yield {"type": "think", "content": buffer[:safe_end]}
                            buffer = buffer[safe_end:]
                    else:
                        yield {"type": "think", "content": buffer}
                        buffer = ""
                    break
            else:
                open_pos = buffer.find(OPEN_TAG)
                if open_pos != -1:
                    # Yield answer content before open tag
                    if open_pos > 0:
                        yield {"type": "answer", "content": buffer[:open_pos]}
                    buffer = buffer[open_pos + len(OPEN_TAG):]
                    in_think = True
                else:
                    # Check for partial open tag at end
                    partial = False
                    for end_len in range(1, len(OPEN_TAG)):
                        if buffer.endswith(OPEN_TAG[:end_len]):
                            partial = True
                            break
                    if partial:
                        safe_end = len(buffer) - len(OPEN_TAG) + 1
                        if safe_end > 0:
                            yield {"type": "answer", "content": buffer[:safe_end]}
                            buffer = buffer[safe_end:]
                    else:
                        yield {"type": "answer", "content": buffer}
                        buffer = ""
                    break

    # Flush any remaining buffer
    if buffer:
        yield {"type": "think" if in_think else "answer", "content": buffer}


def extract_answer(text: str) -> str:
    """Return only the user-facing portion of a completed model stream.

    Foundry applies stop sequences before returning them, so a successful
    completion normally contains ``<answer>`` but not ``</answer>``.  Both
    the closed and opening-only forms are supported.  If no opening tag is
    present, only a clearly recognisable first meta-commentary paragraph is
    removed; otherwise the model text passes through unchanged.
    """
    if text is None:
        return text

    raw_text = str(text).strip()
    if not raw_text:
        return raw_text

    opening_match = re.search(r"<answer\s*>", raw_text, re.IGNORECASE)
    if opening_match:
        answer_text = raw_text[opening_match.end():]
        closing_match = re.search(r"</answer\s*>", answer_text, re.IGNORECASE)
        if closing_match:
            answer_text = answer_text[:closing_match.start()]
        return answer_text.strip()

    # Be tolerant of a provider that returns the closing delimiter but not
    # the opening one. This also prevents a raw HTML-like tag reaching the UI.
    raw_text = re.sub(r"\s*</answer\s*>\s*$", "", raw_text, flags=re.IGNORECASE)

    meta_preamble = re.compile(
        r"^\s*(?:[-*>#`]+\s*)?(?:analysis\s*:\s*)?"
        r"(?:okay\b|let me\b|i must\b|the user\b|i know (?:the )?rules\b|"
        r"i need to\b|i should\b)",
        re.IGNORECASE,
    )

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\r?\n\s*\r?\n", raw_text)
        if paragraph.strip()
    ]
    first_public_paragraph = 0
    while (
        first_public_paragraph < len(paragraphs)
        and meta_preamble.match(paragraphs[first_public_paragraph])
    ):
        first_public_paragraph += 1
    if 0 < first_public_paragraph < len(paragraphs):
        return "\n\n".join(paragraphs[first_public_paragraph:]).strip()

    # Some local models separate the preamble with a single newline rather
    # than a blank line. Treat the first non-empty line as the first paragraph
    # only when it independently matches the same narrow meta pattern.
    lines = raw_text.splitlines()
    first_nonempty = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first_nonempty is not None and meta_preamble.match(lines[first_nonempty]):
        remainder = "\n".join(lines[first_nonempty + 1:]).strip()
        if remainder:
            return remainder

    return raw_text


def stream_extract_answer(response, lang_code):
    """Accumulate a completion and emit only its extracted final answer.

    Extraction intentionally happens after the model stream completes. This
    makes the no-tag preamble fallback deterministic and guarantees callers
    accumulate clean text before passing it to ``validate_answer()``. Think
    chunks remain separate for compatibility with existing callers.
    """
    answer_parts = []
    for chunk in stream_and_strip_think(response, lang_code):
        if chunk["type"] == "think":
            yield chunk
        else:
            answer_parts.append(chunk["content"])

    extracted = extract_answer("".join(answer_parts))
    if extracted:
        yield {"type": "answer", "content": extracted}


def generate_llm_answer(
    query,
    hotel_context_str,
    chat_history,
    location,
    lang_code="en",
    hotel_cards=None,
    query_requirements=None,
):
    structured_cards = list(hotel_cards or [])[:3]
    structured_fallback = safe_card_based_fallback_answer(
        user_query=query,
        hotel_cards=structured_cards,
        query_requirements=query_requirements,
        city=location,
        language=lang_code,
    )

    try:
        # Normal UI path: code first creates a fact-complete canonical draft.
        # Qwen sees only that compact allowlist and its rewrite is accepted
        # only after a strict, hotel-by-hotel local fact gate.
        if structured_cards:
            import prompt_builders

            prompt = prompt_builders.build_hotel_feature_rewrite_prompt()
            structured_max_tokens = 160 + 80 * (len(structured_cards) - 1)

            def make_structured_request(client, model_id):
                return client.chat.completions.create(
                    model=model_id,
                    messages=typing.cast(
                        typing.Any,
                        [
                            {"role": "system", "content": prompt},
                            {
                                "role": "user",
                                "content": structured_fallback,
                            },
                        ],
                    ),
                    temperature=0.3,
                    top_p=0.9,
                    max_tokens=structured_max_tokens,
                    stream=True,
                    stop=["</answer>"],
                    timeout=60.0,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False}
                    },
                )

            response = create_chat_completion_with_retry(make_structured_request)
            answer_parts = []
            for chunk in stream_extract_answer(response, lang_code):
                if chunk.get("type") == "answer":
                    answer_parts.append(chunk.get("content", ""))
            candidate = coalesce_hotel_rewrite_sentences(
                "".join(answer_parts).strip(), structured_cards
            )
            passed, rejection_reasons = validate_hotel_feature_rewrite(
                candidate,
                structured_cards,
                query_requirements=query_requirements,
            )
            if passed:
                print(
                    "[HOTEL_FEATURE_GATE] Qwen rewrite accepted.",
                    file=sys.stderr,
                )
                yield {"type": "answer", "content": candidate}
            else:
                print(
                    "[HOTEL_FEATURE_GATE] Qwen rewrite rejected; using "
                    f"grounded draft: {', '.join(rejection_reasons)}",
                    file=sys.stderr,
                )
                yield {"type": "answer", "content": structured_fallback}
            return

        meta = load_hotel_metadata()
        total_hotels_in_city = sum(1 for h in meta.values() if h.get('city', '').lower() == location.lower()) if location else 0

        target_lang = "English"

        import prompt_builders
        style_instruction = prompt_builders.get_style_instruction(target_lang)
        prompt = prompt_builders.build_final_answer_prompt(
            target_language=target_lang,
            intent="hotel_search",
            requested_location=location,
            hotel_context_str=hotel_context_str,
            total_hotels_in_city=total_hotels_in_city,
            style_instruction=style_instruction
        )

        def make_request(client, model_id):
            return client.chat.completions.create(
                model=model_id,
                messages=typing.cast(typing.Any, [{'role': 'system', 'content': prompt}] + get_truncated_history(chat_history) + [{'role': 'user', 'content': query}]),
                temperature=0.3,
                frequency_penalty=0.5,
                max_tokens=240,
                stream=True,
                stop=["</answer>"],
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False}
                },
            )

        response = create_chat_completion_with_retry(make_request)
        yield from stream_extract_answer(response, lang_code)
    except Exception as e:
        import traceback
        traceback.print_exc()
        if structured_cards:
            print(
                "[HOTEL_FEATURE_GATE] Qwen unavailable; using grounded draft.",
                file=sys.stderr,
            )
            yield {"type": "answer", "content": structured_fallback}
        else:
            warning = "⚠️ **System Warning:** Foundry Local is not ready. Please run `.venv\\Scripts\\python.exe scripts\\setup_foundry_runtime.py`.\n\n"
            yield {"type": "answer", "content": warning + structured_fallback}


def get_deterministic_conversational_reply(query):
    q = normalize_hotel_reference(query)
    combined_greeting = re.fullmatch(
        r"(?:hi|hello|hey)(?: there| travelmind)? "
        r"(?:how are you|how is it going|how s it going)",
        q,
    )
    if combined_greeting:
        return "Hi! I'm ready to help. Which supported city would you like hotel recommendations for?"
    if q in {"hello", "hi", "hey", "good morning", "good afternoon", "good evening", "merhaba", "selam"}:
        return "Hello! I am TravelMind. Which supported city would you like hotel recommendations for?"
    if q in {"thanks", "thank you", "thanks a lot", "thank you very much", "tesekkurler", "tesekkur ederim"}:
        return "You're welcome! I can help whenever you need another hotel recommendation."
    if q in {"how are you", "how is it going"}:
        return "I'm ready to help. Tell me a supported city or ask about one of the displayed hotels."
    if q in {"who are you", "what are you", "what is your name", "whats your name"}:
        return "I am TravelMind, a hotel recommendation assistant for supported cities."
    if q in {"what can you do", "how can you help"}:
        return "I can rank hotels in supported cities and answer grounded questions about displayed hotel profiles and guest reviews."
    return None


def generate_conversational_answer(query, lang_code, chat_history):
    deterministic_reply = get_deterministic_conversational_reply(query)
    if deterministic_reply:
        yield {"type": "answer", "content": deterministic_reply}
        return
    try:
        target_lang = "English"
        import prompt_builders
        prompt = prompt_builders.build_conversational_answer_prompt(
            target_language=target_lang
        )

        def make_request(client, model_id):
            return client.chat.completions.create(
                model=model_id,
                messages=typing.cast(typing.Any, [{'role': 'system', 'content': prompt}] + get_truncated_history(chat_history) + [{'role': 'user', 'content': f"{query}\n/no_think"}]),
                temperature=0.2,
                max_tokens=120,
                stream=True,
                stop=["</answer>"],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

        response = create_chat_completion_with_retry(make_request)
        emitted_answer = False
        for chunk in stream_extract_answer(response, lang_code):
            if chunk.get("type") == "answer" and chunk.get("content", "").strip():
                emitted_answer = True
            yield chunk
        if not emitted_answer:
            yield {"type": "answer", "content": "I am TravelMind, a hotel recommendation assistant. Tell me a supported city and I can help you find a hotel."}
    except Exception:
        import traceback
        traceback.print_exc()
        yield {"type": "answer", "content": "I am TravelMind, a hotel recommendation assistant. Tell me a supported city and I can help you find a hotel."}

def generate_followup_answer(query, context_str, lang_code, chat_history):
    try:
        target_lang = "English"

        import prompt_builders
        style_instruction = prompt_builders.get_style_instruction(target_lang, is_followup=True)
        prompt = prompt_builders.build_followup_prompt(
            target_language=target_lang,
            hotel_context_str=context_str if isinstance(context_str, str) else "",
            style_instruction=style_instruction
        )

        def make_request(client, model_id):
            return client.chat.completions.create(
                model=model_id,
                messages=typing.cast(typing.Any, [{'role': 'system', 'content': prompt}] + get_truncated_history(chat_history) + [{'role': 'user', 'content': f"{query}\n/no_think"}]),
                temperature=0.2,
                frequency_penalty=0.2,
                max_tokens=180,
                stream=True,
                stop=["</answer>"],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

        response = create_chat_completion_with_retry(make_request)
        emitted_answer = False
        for chunk in stream_extract_answer(response, lang_code):
            if chunk.get("type") == "answer" and chunk.get("content", "").strip():
                emitted_answer = True
            yield chunk
        if not emitted_answer:
            yield {"type": "answer", "content": "I could not produce a reliable answer from the current hotel records. Please ask about a specific displayed hotel's verified details."}
    except Exception:
        import traceback
        traceback.print_exc()
        yield {"type": "answer", "content": "I could not produce a reliable answer from the current hotel records. Please ask about a specific displayed hotel's verified details."}


def generate_review_answer(
    hotel_card,
    review_chunks,
    question,
    lang_code="en",
    chat_history=None,
):
    """Generate a review-grounded answer for one selected hotel."""
    from review_summarizer import summarize_common_review_question

    deterministic_answer = summarize_common_review_question(
        hotel_card=hotel_card,
        review_chunks=review_chunks,
        question=question,
    )
    if deterministic_answer:
        yield {"type": "answer", "content": deterministic_answer}
        return

    try:
        import prompt_builders

        prompt = prompt_builders.build_review_answer_prompt(
            hotel_card=hotel_card,
            review_chunks=review_chunks,
            question=question,
        )

        def make_request(client, model_id):
            return client.chat.completions.create(
                model=model_id,
                messages=typing.cast(
                    typing.Any,
                    [{"role": "system", "content": prompt}]
                    + get_truncated_history(chat_history or [])
                    + [{"role": "user", "content": f"{question}\n/no_think"}],
                ),
                temperature=0.15,
                frequency_penalty=0.2,
                max_tokens=180,
                stream=True,
                stop=["</answer>"],
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False}
                },
            )

        response = create_chat_completion_with_retry(make_request)
        emitted_answer = False
        for chunk in stream_extract_answer(response, lang_code):
            if chunk.get("type") == "answer" and chunk.get("content", "").strip():
                emitted_answer = True
            yield chunk
        if not emitted_answer:
            yield {
                "type": "answer",
                "content": (
                    "The retrieved guest-review excerpts did not produce a "
                    "reliable answer to this question. I cannot identify a "
                    "clear majority view from those excerpts. No price, "
                    "availability, or booking claim can be inferred from them."
                ),
            }
    except Exception:
        import traceback

        traceback.print_exc()
        yield {
            "type": "answer",
            "content": (
                "I could not reach the local AI service to summarize the guest "
                "reviews. Please prepare Foundry Local and try again."
            ),
        }


def consume_generator(generator, console):
    import sys
    full_answer = ""
    console.print("\n[bold green]TravelMind:[/bold green] ", end="")
    for chunk in generator:
        if isinstance(chunk, dict):
            if chunk["type"] == "think":
                sys.stdout.write(f"\033[90m{chunk['content']}\033[0m")
                sys.stdout.flush()
            else:
                full_answer += chunk["content"]
                sys.stdout.write(chunk["content"])
                sys.stdout.flush()
        else:
            full_answer += chunk
            sys.stdout.write(chunk)
            sys.stdout.flush()
    print()
    return full_answer

def _legacy_fast_route_query(user_query, session_state=None) -> dict:
    import time
    start_time = time.time()

    q_lower = user_query.lower().strip()
    result = None

    # 0. Resolve an explicitly named hotel and route obvious review questions
    # before the broader specific-hotel follow-up rule can consume them.
    last_cards = session_state.get("last_hotel_cards", []) if session_state else []
    matched_hotel_name = None
    for card in last_cards:
        original_name = str(card.get("hotel_name", "")).strip()
        normalized_name = original_name.lower()
        name_without_article = (
            normalized_name[4:].strip()
            if normalized_name.startswith("the ")
            else normalized_name
        )
        if (
            len(normalized_name) > 3
            and (
                normalized_name in q_lower
                or (len(name_without_article) > 3 and name_without_article in q_lower)
            )
        ):
            matched_hotel_name = original_name
            break

    review_phrases = (
        "what do guests say",
        "what did guests say",
        "what do reviews say",
        "what did reviews say",
        "guest reviews",
        "guest feedback",
        "review feedback",
        "any complaints",
        "complaints about",
        "noise complaints",
        "value for money",
    )
    review_aspect_question = re.search(
        r"\bhow\s+(?:is|are|was|were)\s+(?:the\s+)?"
        r"(?:rooms?|service|cleanliness|value|staff)\b",
        q_lower,
    )
    review_aspect_feedback = re.search(
        r"(?:\b(?:reviews?|feedback|complaints?)\b.*"
        r"\b(?:rooms?|service|cleanliness|value|staff)\b|"
        r"\b(?:rooms?|service|cleanliness|value|staff)\b.*"
        r"\b(?:reviews?|feedback|complaints?)\b)",
        q_lower,
    )
    looks_like_review_question = (
        any(phrase in q_lower for phrase in review_phrases)
        or review_aspect_question is not None
        or review_aspect_feedback is not None
        or re.search(r"\b(?:noisy|noise)\b", q_lower) is not None
    )
    if last_cards and looks_like_review_question:
        routed = {
            "intent": "review_question",
            "requested_hotel_name": matched_hotel_name,
        }
        print(f"[ROUTER] Caught review question in {time.time() - start_time:.3f}s")
        return routed

    if matched_hotel_name:
        print(f"[ROUTER] Caught context hotel name: {matched_hotel_name}")
        return {
            "intent": "specific_hotel_info",
            "requested_hotel_name": matched_hotel_name,
        }

    # Check if a supported city is mentioned
    supported_cities = ["dallas", "chicago", "new york", "san francisco", "boston", "washington", "san diego", "houston", "denver", "los angeles", "seattle", "san antonio", "phoenix", "philadelphia", "memphis", "baltimore", "san jose", "detroit", "austin", "indianapolis", "jacksonville", "charlotte", "columbus", "fort worth", "el paso"]
    has_city = any(c in q_lower for c in supported_cities)

    # 1. Price check
    if any(p in q_lower for p in ["price", "how much", "per night"]):
        result = {"intent": "price_question"}

    # 2. Pool follow-up
    elif not has_city and "pool" in q_lower:
        result = {"intent": "followup_pool"}

    # 3. Breakfast follow-up
    elif not has_city and "breakfast" in q_lower:
        result = {"intent": "followup_breakfast"}

    # 4. Other hotel follow-up
    elif not has_city and any(o in q_lower for o in ["other hotel", "another hotel"]):
        result = {"intent": "followup_other_hotel"}

    # 5. Score / Class explanation
    elif any(s in q_lower for s in ["score", "calculate", "class"]):
        if any(s in q_lower for s in ["how", "what"]):
            if "class" in q_lower:
                result = {"intent": "class_explanation"}
            else:
                result = {"intent": "score_explanation"}

    # 6. Unsupported Locations
    unsupported_cities = ["paris", "istanbul", "vienna", "miami", "las vegas", "london", "tokyo", "rome", "berlin", "madrid"]
    if result is None:
        for uc in unsupported_cities:
            if uc in q_lower:
                result = {"intent": "unsupported_location", "location": uc.capitalize()}
                break

    # 7. Supported city + hotel/amenity signal -> hotel_search
    if result is None and has_city:
        hotel_signals = ["hotel", "stay", "room", "breakfast", "wifi", "wi-fi", "pool", "central", "clean", "suggest", "recommend", "suite", "single", "double"]
        specific_signals = ["tell me about", "about"]

        found_city = next((c for c in supported_cities if c in q_lower), None)

        reqs = {}
        if "breakfast" in q_lower:
            reqs["breakfast"] = "REQUIRED"
        if "wifi" in q_lower or "wi-fi" in q_lower:
            reqs["wifi"] = "REQUIRED"
        if "pool" in q_lower:
            reqs["pool"] = "REQUIRED"
        if "suite" in q_lower or "suit" in q_lower:
            reqs["suite"] = "REQUIRED"
        if "single" in q_lower:
            reqs["single_room"] = "REQUIRED"
        if "double" in q_lower:
            reqs["double_room"] = "REQUIRED"

        if any(s in q_lower for s in specific_signals):
            result = {"intent": "specific_hotel_info", "city": found_city.capitalize(), "requirements": reqs}
        elif any(s in q_lower for s in hotel_signals):
            result = {"intent": "hotel_search", "city": found_city.capitalize(), "requirements": reqs}

    if result:
        print(f"[TIMING] fast_route_query matched '{result['intent']}' in {time.time() - start_time:.3f}s")
        return result

    print(f"[TIMING] fast_route_query fell through in {time.time() - start_time:.3f}s")
    return None


SUPPORTED_CITIES = (
    "dallas", "chicago", "new york", "san francisco", "boston", "washington",
    "san diego", "houston", "denver", "los angeles", "seattle", "san antonio",
    "phoenix", "philadelphia", "memphis", "baltimore", "san jose", "detroit",
    "austin", "indianapolis", "jacksonville", "charlotte", "columbus",
    "fort worth", "el paso",
)


def _initial_requested_hotel_name(query, city=None):
    """Extract conservative ``... Hotel`` names from initial city searches."""
    text = re.sub(r"\s+", " ", str(query or "")).strip()
    match = re.search(
        r"\b(?:show|find|search(?: for)?|tell me about)\s+(?:me\s+)?"
        r"(?P<name>(?:the\s+)?[a-z0-9&' .-]{2,80}?\bhotel)"
        r"(?=\s+(?:in|at|near)\b|[?.!,]|$)",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    name = re.sub(r"\s+", " ", match.group("name")).strip(" ,.-")
    if normalize_hotel_reference(name) in {"hotel", "a hotel", "an hotel", "some hotel"}:
        return None
    return name


def fast_route_query(user_query, session_state=None) -> dict:
    """Deterministic routes for common/critical intents before local Qwen."""
    query = str(user_query or "").strip()
    q = normalize_hotel_reference(query)
    cards = list((session_state or {}).get("last_hotel_cards", []) or [])
    selected_index = (session_state or {}).get("selected_hotel_index", 0)
    matched_index = find_referenced_hotel_index(cards, question=query)
    matched_name = (
        _clean_public_card_value(cards[matched_index].get("hotel_name"))
        if matched_index is not None
        else None
    )
    found_city = next((city for city in SUPPORTED_CITIES if city in q), None)
    unsupported = next(
        (city for city in ("paris", "istanbul", "vienna", "miami", "las vegas", "london", "tokyo", "rome", "berlin", "madrid") if city in q),
        None,
    )

    if q in {"bye", "goodbye", "exit", "quit", "see you"}:
        return {"intent": "exit"}
    if get_deterministic_conversational_reply(query):
        return {"intent": "general_chat", "location": None, "query_requirements": {}}

    out_of_scope_terms = (
        "flight", "airfare", "plane ticket", "visa", "passport", "weather",
        "itinerary", "car rental", "rent a car", "train ticket", "bus ticket",
        "exchange rate", "currency rate", "phone brand", "write code", "programming",
    )
    if any(term in q for term in out_of_scope_terms):
        return {"intent": "out_of_scope"}

    if re.search(r"\b(?:price|prices|cost|costs|rate|rates|how much|per night|book|booking|availability)\b", q):
        return {"intent": "price_question", "requested_hotel_name": matched_name}

    review_question = bool(
        any(phrase in q for phrase in (
            "what do guests say", "what did guests say", "what do reviews say",
            "guest reviews", "guest feedback", "any complaints", "complaints about",
            "noise complaints", "value for money",
        ))
        or re.search(r"\b(?:reviews?|guests?|feedback|complaints?|noise|noisy)\b", q)
        or re.search(r"\bhow (?:is|are|was|were) (?:the )?(?:rooms?|service|cleanliness|value|staff)\b", q)
    )
    if review_question:
        return {"intent": "review_question", "requested_hotel_name": matched_name}

    reqs = {}
    for token, key in (("breakfast", "breakfast"), ("pool", "pool"), ("wifi", "wifi"), ("wi fi", "wifi"), ("parking", "parking"), ("suite", "suite"), ("single", "single_room"), ("double", "double_room")):
        if token in q:
            reqs[key] = "REQUIRED"

    initial_requested_name = _initial_requested_hotel_name(query, found_city) if found_city else None
    if found_city and initial_requested_name:
        return {
            "intent": "hotel_search",
            "city": found_city.title(),
            "location": found_city.title(),
            "requested_hotel_name": initial_requested_name,
            "requirements": reqs,
            "query_requirements": reqs,
        }

    if not found_city and re.search(r"\bpool\b", q):
        return {"intent": "followup_pool", "requested_hotel_name": matched_name}
    if not found_city and re.search(r"\bbreakfast\b", q):
        return {"intent": "followup_breakfast", "requested_hotel_name": matched_name}

    if not found_city and re.search(r"\b(?:next (?:one|option|hotel)|another (?:one|option|hotel)|other (?:one|option|hotel)|different hotel|any other options?)\b", q):
        return {"intent": "follow_up"}

    global_score = bool(re.search(r"\b(?:how|what).*(?:score).*(?:calculat|determin|mean|work)|\bwhat is (?:the )?travelmind score\b", q))
    global_class = bool(re.search(r"\b(?:how|what).*(?:hotel class|star rating).*(?:calculat|determin|mean|work)|\bwhat is (?:the )?hotel class\b", q))
    if global_score and matched_name is None:
        return {"intent": "score_explanation"}
    if global_class and matched_name is None:
        return {"intent": "class_explanation"}

    detail_signal = bool(re.search(
        r"\b(?:tell me more|details?|about (?:it|there|that hotel)|phone|telephone|contact|call|where|location|located|address|map|directions|score|rating|rated|class|stars?|amenities|amenity|facilities|features|room types?|rooms?|does it have|what does it have)\b",
        q,
    ))
    if cards and (matched_name or detail_signal):
        selected_card, resolved_index = resolve_hotel_selection(
            cards, query, matched_name, selected_index
        )
        return {
            "intent": "specific_hotel_info",
            "requested_hotel_name": _clean_public_card_value(selected_card.get("hotel_name")),
            "selected_hotel_index": resolved_index,
        }

    if unsupported:
        return {"intent": "unsupported_location", "location": unsupported.title()}

    if found_city:
        hotel_signal = bool(re.search(r"\b(?:hotel|stay|accommodation|room|suggest|recommend|find|show|search|pool|breakfast|wifi|parking|suite)\b", q))
        if hotel_signal:
            result = {
                "intent": "hotel_search",
                "city": found_city.title(),
                "location": found_city.title(),
                "requirements": reqs,
                "query_requirements": reqs,
            }
            return result

    if re.search(r"\b(?:hotel|stay|accommodation)\b", q) and re.search(r"\b(?:find|show|recommend|suggest|looking|need|want)\b", q):
        return {"intent": "missing_location", "location": None, "query_requirements": reqs}
    return None

def get_llm_intent_and_location(query: str, chat_history: list) -> dict:
    q_lower = query.lower().strip()

    # Fast-path for greetings
    greetings = ["hello", "hi", "hey", "good morning", "good evening", "how are you"]
    if q_lower in greetings or any(q_lower == g for g in greetings) or (len(q_lower.split()) <= 2 and any(g in q_lower for g in greetings)):
        return {"intent": "general_chat", "location": None, "filters": {}}

    try:
        import typing, json

        import prompt_builders
        system_prompt = prompt_builders.build_router_system_prompt()

        def make_request(client, model_id):
            return client.chat.completions.create(
                model=model_id,
                messages=typing.cast(typing.Any, [{"role": "system", "content": system_prompt}] + get_truncated_history(chat_history) + [{"role": "user", "content": query}]),
                temperature=0.1,
                frequency_penalty=0.0,
                max_tokens=220,
                stream=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

        response = create_chat_completion_with_retry(make_request)
        content = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
        import re
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        match = re.search(r'\{.*\}', content, re.DOTALL)
        parsed = None
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        if not parsed:
            parsed = {"intent": "out_of_scope", "location": None, "query_requirements": {}}

        # Hard fallback heuristic
        q_lower = query.lower()
        import difflib
        is_hotel_query = False
        words = q_lower.replace("'", " ").replace('"', " ").split()
        if "hotel" in q_lower or "stay" in q_lower or "accommodation" in q_lower:
            is_hotel_query = True
        else:
            if difflib.get_close_matches("hotel", words, n=1, cutoff=0.7):
                is_hotel_query = True

        if parsed.get("intent") in ["general_chat", "missing_location", "out_of_scope"] and is_hotel_query:
            cities_map = {
                "dallas": "Dallas, TX", "chicago": "Chicago, IL", "new york": "New York City, NY",
                "san francisco": "San Francisco, CA", "boston": "Boston, MA", "washington": "Washington DC, DC",
                "san diego": "San Diego, CA", "houston": "Houston, TX", "denver": "Denver, CO",
                "los angeles": "Los Angeles, CA", "seattle": "Seattle, WA", "san antonio": "San Antonio, TX",
                "phoenix": "Phoenix, AZ", "philadelphia": "Philadelphia, PA", "memphis": "Memphis, TN",
                "baltimore": "Baltimore, MD", "san jose": "San Jose, CA", "detroit": "Detroit, MI",
                "austin": "Austin, TX", "indianapolis": "Indianapolis, IN", "jacksonville": "Jacksonville, FL",
                "charlotte": "Charlotte, NC", "columbus": "Columbus, OH", "fort worth": "Fort Worth, TX",
                "el paso": "El Paso, TX"
            }
            import difflib
            words = q_lower.split()
            bigrams = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
            all_tokens = words + bigrams

            best_match = None
            best_val = None

            for c_key, c_val in cities_map.items():
                if c_key in q_lower:
                    best_match = c_key
                    best_val = c_val
                    break

                matches = difflib.get_close_matches(c_key, all_tokens, n=1, cutoff=0.75)
                if matches:
                    best_match = matches[0]
                    best_val = c_val
                    break

            if best_match:
                parsed["intent"] = "hotel_search"
                parsed["location"] = best_val
            else:
                parsed["intent"] = "missing_location"
                parsed["location"] = None
        if parsed.get("intent") == "hotel_search":
            if not parsed.get("location") or str(parsed.get("location")).strip() == "":
                parsed["intent"] = "missing_location"
            else:
                loc_lower = str(parsed["location"]).lower()
                cities_map_keys = [
                    "dallas", "chicago", "new york", "san francisco", "boston", "washington",
                    "san diego", "houston", "denver", "los angeles", "seattle", "san antonio",
                    "phoenix", "philadelphia", "memphis", "baltimore", "san jose", "detroit",
                    "austin", "indianapolis", "jacksonville", "charlotte", "columbus",
                    "fort worth", "el paso"
                ]
                is_supported = any(c in loc_lower for c in cities_map_keys)
                if not is_supported:
                    parsed["intent"] = "unsupported_location"

        allowed_intents = {
            "hotel_search", "preference_refinement", "follow_up", "score_explanation",
            "class_explanation", "price_question", "specific_hotel_info",
            "review_question", "general_chat", "missing_location",
            "unsupported_location", "out_of_scope", "exit",
        }
        if parsed.get("intent") not in allowed_intents:
            parsed["intent"] = "out_of_scope"
        if parsed.get("intent") == "general_chat" and not get_deterministic_conversational_reply(query):
            parsed["intent"] = "out_of_scope"
        return parsed
    except Exception as e:
        print(f"Error in intent routing: {e}")
        q = normalize_hotel_reference(query)
        if re.search(r"\b(?:hotel|stay|accommodation)\b", q):
            return {"intent": "missing_location", "location": None, "query_requirements": {}}
        return {"intent": "out_of_scope", "location": None, "query_requirements": {}}

def generate_out_of_scope_answer(user_query=None, language="en", chat_history=None):
    return (
        "TravelMind currently supports hotel and accommodation recommendations only for supported cities. "
        "It does not provide live prices, availability, booking, flight, visa, or general travel planning information."
    )

def safe_card_based_fallback_answer(
    user_query=None,
    hotel_cards=None,
    query_requirements=None,
    city=None,
    language="en"
):
    """Return a complete local answer when Qwen is unavailable or rejected."""
    return build_grounded_hotel_answer(
        hotel_cards=list(hotel_cards or [])[:3],
        city=city,
        query_requirements=query_requirements,
    )
