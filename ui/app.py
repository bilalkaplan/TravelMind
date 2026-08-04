import streamlit as st

# 1. MUST BE ADDED AT THE TOP SO THE "STREAMLIT" LABEL DISAPPEARS!
st.set_page_config(
    page_title="TravelMind - AI Travel Agent",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import sys
import os
from pathlib import Path

# Disable the browser's automatic translation (fixes placeholder bugs)
st.html("""
<script>
    window.parent.document.documentElement.setAttribute("translate", "no");
    window.parent.document.documentElement.classList.add("notranslate");
</script>
""", unsafe_allow_javascript=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cmu_rag_answer import (
    build_amenity_followup_answer,
    build_grounded_followup_answer,
    clamp_selected_hotel_index,
    generate_conversational_answer,
    generate_llm_answer,
    generate_out_of_scope_answer_stream,
    generate_review_answer,
    get_llm_intent_and_location,
    resolve_hotel_selection,
    safe_map_link,
)

def sanitize_before_render(answer):
    if not answer or not str(answer).strip():
        return None

    sanitized = str(answer)
    forbidden = [
        "Okay, the user",
        "The user is asking",
        "<think>",
        "</think>",
        "Chunk Type:",
        "[Insert evidence summary here]",
    ]
    for x in forbidden:
        # Case insensitive replacement
        import re
        sanitized = re.sub(re.escape(x), "", sanitized, flags=re.IGNORECASE)

    if not sanitized.strip():
        return None

    return sanitized
from cmu_retrieve import search, search_reviews_for_hotel
from travelmind_scoring import WEIGHTS
from answer_validator import validate_answer, extract_final_answer
from hotel_card_builder import build_hotel_cards


def collect_llm_answer(generator):
    """Collect only public answer tokens, then remove the answer delimiter.

    Raw tokens are deliberately not rendered because a reasoning preamble or
    delimiter must never flash in the UI before extraction and validation.
    """
    answer_parts = []
    for chunk in generator:
        if isinstance(chunk, dict):
            if chunk.get("type") == "answer":
                answer_parts.append(str(chunk.get("content", "")))
        elif chunk is not None:
            answer_parts.append(str(chunk))
    return extract_final_answer("".join(answer_parts))


def select_review_hotel(last_hotel_cards, question, requested_hotel_name=None, selected_index=0):
    """Resolve a review target using the same selection state as follow-ups."""
    return resolve_hotel_selection(
        last_hotel_cards,
        question=question,
        requested_hotel_name=requested_hotel_name,
        selected_index=selected_index,
    )


def build_score_explanation_text():
    labels = {
        "location_match": "Location match",
        "hotel_class": "Hotel class/stars",
        "amenities_match": "Requested amenities",
        "room_type_match": "Requested room type",
        "review_overall": "Overall guest rating",
        "review_service": "Service rating",
        "review_rooms": "Room rating",
        "review_cleanliness": "Cleanliness rating",
        "review_volume": "Review volume",
    }
    components = ", ".join(
        f"{labels[key]} ({weight}%)" for key, weight in WEIGHTS.items()
    )
    return (
        f"The TravelMind suitability score uses {components}. "
        "Guest ratings are normalized from 1–5 to 0–100, and review volume uses logarithmic normalization. "
        "When a hotel has no data for a component, that component is skipped and the remaining weights are renormalized, so missing data is not treated as a zero."
    )


@st.cache_resource
def load_retrieval_resources():
    """Loads the embedding model, the embedding matrix, and the row index
    once per server process instead of on every rerun/query."""
    from cmu_retrieve import get_or_load_embedding_model, get_or_load_matrix, get_or_load_row_index
    get_or_load_embedding_model()
    get_or_load_matrix()
    get_or_load_row_index()
    return True

try:
    load_retrieval_resources()
except Exception as exc:
    print(f"[STARTUP] Retrieval artifacts could not be loaded: {exc}", file=sys.stderr)
    st.error(
        "TravelMind could not load its local retrieval artifacts. "
        "Run `.venv\\Scripts\\python.exe scripts\\verify_runtime_artifacts.py` "
        "from the project folder, then restart the app."
    )
    st.stop()


@st.cache_resource
def log_and_validate_llm_model():
    """Logs the pinned LLM model id once per server process and verifies
    it's actually loaded on the Foundry endpoint. Does not crash the app if
    Foundry isn't up yet -- per-query calls already retry/discover the
    endpoint live and show a graceful in-chat warning on failure."""
    import config
    from cmu_rag_answer import get_foundry_client_and_model
    try:
        _, model_id = get_foundry_client_and_model()
        print(f"[STARTUP] Using Foundry model: {model_id}", file=sys.stderr)
        return model_id
    except Exception as e:
        print(f"[STARTUP] Configured model '{config.MODEL_ID}' could not be verified: {e}", file=sys.stderr)
        return None

log_and_validate_llm_model()


def append_message(msg):
    st.session_state.messages.append(msg)
    if os.getenv("TRAVELMIND_CHAT_LOG", "").strip().lower() not in {"1", "true", "yes"}:
        return

    import datetime
    try:
        log_dir = PROJECT_ROOT / "ui_test_logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = log_dir / "chat_history.log"
        with open(log_file, 'a', encoding='utf-8') as f:
            ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            role = msg.get("role", "UNKNOWN").upper()
            content = msg.get("content", "")
            f.write(f"[{ts}] {role}:\n{content}\n" + "-" * 40 + "\n")
    except Exception:
        pass

# TravelMind is English-only; this is a fixed dictionary, not a language switcher.
t = {
    "code": "en",
    "hero_title": "Find your hotel with TravelMind",
    "hero_subtitle": "Search among thousands of hotels in 25 popular U.S. cities to find the one that's right for you.",
    "chat_placeholder": "What kind of hotel are you looking for? (e.g., A luxury hotel with a pool in New York)",
    "analyzing": "TravelMind is analyzing...",
    "exit_msg": "Goodbye! Have a great trip.",
    "unsupported_loc": "Unfortunately, I only serve supported US cities right now.",
    "missing_loc": "Please specify the city you want to search in. (e.g., Boston, Chicago, Miami)",
    "not_found": "Sorry, I couldn't find a hotel matching your criteria in {}.",
    "unknown_hotel": "Unknown Hotel",
    "class_label": "Class",
    "amenities_label": "Amenities",
    "score_label": "TravelMind Score",
    "map": "View on Map",
    "footer": "Bilal Kocakaplan, 2026. All Rights Reserved.",
    "theme_label": "Theme",
    "dark_theme": "🌙 Dark",
    "light_theme": "☀️ Light",
}

if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_hotel_index" not in st.session_state:
    st.session_state.selected_hotel_index = 0
if "last_hotel_cards" not in st.session_state:
    st.session_state.last_hotel_cards = []
if "theme_internal" not in st.session_state:
    st.session_state.theme_internal = "dark"
if "current_location" not in st.session_state:
    st.session_state.current_location = None

if st.session_state.theme_internal == "dark":
    bg_color = "#121212"
    text_color = "#e0e0e0"
    chat_bg = "#1e1e1e"
    chat_text = "#ffffff"
    input_bg = "#2b2b2b"
    input_text = "#ffffff"
    gradient_end = "rgba(18, 18, 18, 1)"
    nav_bg = "rgba(15, 23, 42, 0.98)"
    nav_border = "rgba(255, 255, 255, 0.1)"
    nav_text = "#ffffff"
else:
    bg_color = "#ffffff"
    text_color = "#111111"
    chat_bg = "#0f172a"
    chat_text = "#ffffff"
    input_bg = "#ffffff"
    input_text = "#111111"
    gradient_end = "rgba(255, 255, 255, 1)"
    nav_bg = "rgba(245, 248, 250, 0.98)"
    nav_border = "rgba(0, 0, 0, 0.1)"
    nav_text = "#0f172a"

HERO_BG_URL = "https://images.unsplash.com/photo-1499092346589-b9b6be3e94b2?q=80&w=3840&auto=format&fit=crop"

import base64
logo_path = Path(__file__).resolve().parent / "assets" / "logo_cropped.png"
logo_base64 = ""
if logo_path.exists():
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode("utf-8")

app_bg = f"""
    .stApp {{
        background: linear-gradient(to bottom, rgba(0,0,0,0.5) 0%, rgba(0,0,0,0.7) 50%, {gradient_end} 100%), url('{HERO_BG_URL}') no-repeat center center fixed !important;
        background-size: cover !important;
        background-attachment: fixed !important;
        color: {text_color} !important;
        font-family: 'Inter', sans-serif !important;
    }}
    """

custom_css = f"""
<style>
    {app_bg}
    .custom-footer {{
        text-align: center;
        padding: 20px;
        font-size: 13px;
        color: {text_color} !important;
        opacity: 0.7;
    }}
    /* Chat input fixes */
    div[data-testid="stBottom"],
    div[data-testid="stBottom"] > div,
    div[data-testid="stBottomBlockContainer"] {{
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
    }}
    div[data-testid="stChatInput"] {{
        background-color: transparent !important;
    }}
    div[data-testid="stChatInput"] > div {{
        background-color: {chat_bg} !important;
        border: 1px solid {nav_border} !important;
    }}
    div[data-testid="stChatInput"] textarea {{
        color: {chat_text} !important;
        background-color: transparent !important;
    }}
    div[data-testid="stChatInput"] button svg {{
        fill: {chat_text} !important;
        color: {chat_text} !important;
    }}

    div[data-testid="stChatMessage"] {{
        background-color: {chat_bg} !important;
        border: 1px solid {nav_border} !important;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }}
    div[data-testid="stChatMessage"] * {{
        color: {chat_text} !important;
    }}

    /* Selectbox and input fixes (specific to the React Aria ComboBox API) */
    div[data-testid="stSelectbox"] .react-aria-ComboBox div[role="group"] {{
        background-color: {input_bg} !important;
        border: 1px solid {nav_border} !important;
    }}
    div[data-testid="stSelectbox"] .react-aria-ComboBox div[role="group"] input {{
        color: {input_text} !important;
        background-color: transparent !important;
        pointer-events: none !important;
        cursor: pointer !important;
    }}
    div[data-testid="stSelectbox"] .react-aria-ComboBox div[role="group"] button svg {{
        fill: {input_text} !important;
        color: {input_text} !important;
    }}

    /* Radio button text color fix */
    div[data-testid="stRadio"] div[role="radiogroup"] p {{
        color: {nav_text} !important;
    }}

    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden !important;}}
    footer {{visibility: hidden;}}

    .block-container {{
        padding-top: 100px !important;
        padding-bottom: 5rem !important;
        max-width: 900px !important;
    }}

    /* Full-width navbar look (dark navy theme) */
    div[data-testid="stHorizontalBlock"]:first-of-type:not(div[data-testid="stChatMessage"] *) {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        max-width: none;
        background-color: {nav_bg};
        backdrop-filter: blur(15px);
        padding: 15px 5vw;
        border-radius: 0;
        border-bottom: 1px solid {nav_border};
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.8);
        z-index: 999999;
        align-items: center;
    }}

    /* Align elements inside the navbar */
    div[data-testid="stHorizontalBlock"]:first-of-type:not(div[data-testid="stChatMessage"] *) * {{
        color: {nav_text} !important;
    }}

    /* Navbar label style for the pill-shaped headings */
    div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="stWidgetLabel"] p {{
        background: rgba(128, 128, 128, 0.2) !important;
        padding: 4px 10px !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        font-size: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        display: inline-block !important;
        margin-bottom: 5px !important;
        color: {nav_text} !important;
    }}

    .hero-container {{
        text-align: center;
        margin-top: 10vh;
        margin-bottom: 50px;
    }}
    .hero-title {{
        font-size: 4rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 15px;
        text-shadow: 0px 4px 15px rgba(0,0,0,0.6);
        letter-spacing: -1px;
    }}
    .hero-subtitle {{
        font-size: 1.4rem;
        font-weight: 400;
        color: #f0f0f0;
        text-shadow: 0px 2px 10px rgba(0,0,0,0.8);
    }}

    .custom-footer {{
        text-align: center;
        padding: 15px;
        font-size: 12px;
        margin-top: 30px;
        color: {text_color} !important;
        opacity: 0.7;
    }}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# TOP NAVBAR - Full width with logo
nav_col1, nav_space, nav_col3 = st.columns([3, 4, 1.5])

with nav_col1:
    if logo_base64:
        st.markdown(f'<img src="data:image/png;base64,{logo_base64}" style="height: 80px; margin-top: 5px; object-fit: contain;">', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="font-size: 26px; font-weight: 900; color: #ffffff; letter-spacing: 1px; margin-top: 4px;">TravelMind</div>', unsafe_allow_html=True)

def on_theme_change():
    if st.session_state.theme_selector == t["dark_theme"]:
        st.session_state.theme_internal = "dark"
    else:
        st.session_state.theme_internal = "light"

with nav_col3:
    st.radio(
        t["theme_label"],
        [t["dark_theme"], t["light_theme"]],
        index=0 if st.session_state.theme_internal == "dark" else 1,
        key="theme_selector",
        horizontal=True,
        on_change=on_theme_change,
        label_visibility="collapsed"
    )


def render_hotel_card(card):
    hotel_name = card.get('hotel_name', t['unknown_hotel'])
    location = card.get('location', '')
    h_class = card.get('hotel_class', '')
    if not h_class or str(h_class).lower() in ["nan", "none", "unknown", "null", ""]:
        h_class = "Not Specified"

    score = card.get('travelmind_score', 0)
    map_url = safe_map_link(card.get('map_link'))
    phone_number = card.get('phone', 'UNKNOWN')

    from hotel_feature_verbalizer import (
        get_recorded_room_types,
        get_verified_amenities,
        join_english,
    )

    nice_am = get_verified_amenities(card, limit=6)
    rooms_list = get_recorded_room_types(card, limit=4)

    with st.container():
        st.markdown("---")
        st.markdown(f"#### 🏨 {hotel_name}")
        c1, c2 = st.columns([2.5, 1.5])
        with c1:
            st.write(f"📍 **{location}** | {t['class_label']}: {h_class}")
            if phone_number and phone_number != 'UNKNOWN' and str(phone_number).lower() not in ['none', 'null']:
                st.write(f"📞 **Phone:** {phone_number}")
            detail_sentences = []
            if nice_am:
                detail_sentences.append(
                    "Verified amenities include " + join_english(nice_am) + "."
                )
            if rooms_list:
                detail_sentences.append(
                    "Recorded room types include " + join_english(rooms_list) + "."
                )
            if detail_sentences:
                st.write(f"**Hotel details:** {' '.join(detail_sentences)}")
            else:
                st.write(
                    "**Hotel details:** Detailed amenity and room-type "
                    "information could not be verified in our current records."
                )

            req_warnings = card.get("requirement_satisfaction", {})
            if req_warnings:
                for req_key, req_status in req_warnings.items():
                    if req_status == "UNKNOWN" or req_status == "MISSING":
                        translated_key = req_key.replace('_', ' ').capitalize()
                        warn_msg = f"{translated_key}: Cannot be confirmed from our current records."
                        st.write(f"⚠️ **{warn_msg}**")

        with c2:
            st.metric(t["score_label"], f"{score:.1f}/100")
            if map_url:
                st.markdown(f"👉 **[{t['map']}]({map_url})**")


for msg in st.session_state.messages:
    avatar = "🌍" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if "hotels" in msg and msg["hotels"]:
            for card in msg["hotels"]:
                render_hotel_card(card)

if len(st.session_state.messages) == 0:
    st.markdown(f"""
    <div class="hero-container" translate="no">
        <div class="hero-title">{t["hero_title"]}</div>
        <div class="hero-subtitle">{t["hero_subtitle"]}</div>
    </div>
    """, unsafe_allow_html=True)

if prompt := st.chat_input(t["chat_placeholder"]):
    append_message({"role": "user", "content": prompt})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    prompt = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant", avatar="🌍"):
        with st.spinner(t["analyzing"]):
            from cmu_rag_answer import fast_route_query
            fast_res = fast_route_query(prompt, st.session_state)
            router_res = None

            if fast_res:
                intent = fast_res.get("intent")
                parsed_location = fast_res.get("location", fast_res.get("city", None))
                query_requirements = fast_res.get("query_requirements", fast_res.get("requirements", {}))
                route_res = fast_res
            else:
                router_res = get_llm_intent_and_location(prompt, st.session_state.messages[:-1])
                intent = router_res.get("intent", "general_chat")
                parsed_location = router_res.get("location", None)
                query_requirements = router_res.get("query_requirements", {})
                route_res = router_res

            if parsed_location:
                st.session_state.current_location = parsed_location

            effective_location = st.session_state.current_location

            if intent == "exit":
                response_text = t["exit_msg"]
                st.markdown(response_text)
                append_message({"role": "assistant", "content": response_text})
                st.rerun()

            elif intent == "unsupported_location":
                answer_container = st.empty()
                city_name = parsed_location if parsed_location else "this city"
                full_answer = f"TravelMind currently does not support hotel recommendations for {city_name}. Our system only works for supported cities. It does not provide live prices, availability, or booking information."

                answer_container.markdown(full_answer)
                append_message({"role": "assistant", "content": full_answer})
                st.rerun()

            elif intent == "missing_location" and not effective_location:
                response_text = t["missing_loc"]
                st.markdown(response_text)
                append_message({"role": "assistant", "content": response_text})
                st.rerun()

            elif intent == "out_of_scope":
                answer_container = st.empty()
                generator = generate_out_of_scope_answer_stream(prompt, t["code"], st.session_state.messages[:-1])
                full_answer = sanitize_before_render(collect_llm_answer(generator))
                if full_answer is None:
                    full_answer = "I am a hotel recommendation assistant. I cannot answer questions outside of hotel search and travel accommodations."
                answer_container.markdown(full_answer)
                append_message({"role": "assistant", "content": full_answer})
                st.rerun()

            elif intent == "class_explanation":
                answer_container = st.empty()
                full_answer = (
                    "The Hotel Class is an official metadata value directly pulled from the Tripadvisor dataset "
                    "and is not calculated by our system. It simply represents the hotel's registered official "
                    f"star rating (e.g., 4.0, 5.0). It accounts for {WEIGHTS['hotel_class']}% of the final TravelMind suitability score."
                )
                answer_container.markdown(full_answer)
                append_message({"role": "assistant", "content": full_answer})
                st.rerun()

            elif intent == "score_explanation":
                answer_container = st.empty()
                full_answer = build_score_explanation_text()
                answer_container.markdown(full_answer)
                append_message({"role": "assistant", "content": full_answer})
                st.rerun()

            elif intent == "review_question":
                answer_container = st.empty()
                last_cards = st.session_state.get("last_hotel_cards", [])
                if not last_cards:
                    final_display = (
                        "I do not have a previous hotel result to review yet. "
                        "Please search for a supported city first."
                    )
                else:
                    review_card, review_index = select_review_hotel(
                        last_cards,
                        prompt,
                        route_res.get("requested_hotel_name"),
                        st.session_state.get("selected_hotel_index", 0),
                    )
                    st.session_state.selected_hotel_index = review_index
                    review_chunks = search_reviews_for_hotel(
                        review_card,
                        prompt,
                        k=8,
                    )
                    if not review_chunks:
                        final_display = (
                            f"I could not find review excerpts for **{review_card.get('hotel_name', 'this hotel')}** "
                            "that answer this question."
                        )
                    else:
                        generator = generate_review_answer(
                            hotel_card=review_card,
                            review_chunks=review_chunks,
                            question=prompt,
                            lang_code=t["code"],
                            chat_history=st.session_state.messages[:-1],
                        )
                        review_answer = collect_llm_answer(generator)
                        validation = validate_answer(
                            review_answer,
                            [review_card],
                            "review_question",
                            None,
                            t["code"],
                            evidence_text=review_chunks,
                            allowed_hotel_names=[review_card.get("hotel_name")],
                        )
                        print(
                            "[REVIEW VALIDATOR] "
                            f"hotel={review_card.get('hotel_name')} "
                            f"chunks={len(review_chunks)} "
                            f"fallback={validation['needs_fallback']} "
                            f"blocking={validation['blocking_issues']} "
                            f"warnings={validation['warnings']}",
                            file=sys.stderr,
                        )
                        final_display = validation["sanitized_answer"]

                    final_display = sanitize_before_render(final_display)
                    if final_display is None:
                        final_display = (
                            "The retrieved reviews do not contain enough relevant information "
                            "to answer that question reliably."
                        )

                answer_container.markdown(final_display)
                append_message({"role": "assistant", "content": final_display})
                st.rerun()

            elif intent == "general_chat":
                answer_container = st.empty()
                generator = generate_conversational_answer(prompt, t["code"], st.session_state.messages[:-1])
                full_answer = sanitize_before_render(collect_llm_answer(generator))
                if full_answer is None:
                    full_answer = "I am a hotel recommendation assistant. How can I help you with your travel plans today?"
                answer_container.markdown(full_answer)
                append_message({"role": "assistant", "content": full_answer})
                st.rerun()

            elif intent in ["price_question", "followup_pool", "followup_breakfast", "followup_other_hotel", "follow_up", "specific_hotel_info"]:
                last_cards = st.session_state.get("last_hotel_cards", [])
                selected_idx = clamp_selected_hotel_index(
                    last_cards,
                    st.session_state.get("selected_hotel_index", 0),
                )
                st.session_state.selected_hotel_index = selected_idx
                answer_container = st.empty()

                if intent == "price_question":
                    response_text = "TravelMind does not provide live pricing or availability because our system does not contain real-time booking data."
                    answer_container.markdown(response_text)
                    append_message({"role": "assistant", "content": response_text})
                    st.rerun()

                elif intent == "followup_pool":
                    response_text, selected_idx = build_amenity_followup_answer(
                        last_cards,
                        "pool",
                        question=prompt,
                        requested_hotel_name=route_res.get("requested_hotel_name"),
                        selected_index=selected_idx,
                    )
                    st.session_state.selected_hotel_index = selected_idx

                    answer_container.markdown(response_text)
                    append_message({"role": "assistant", "content": response_text})
                    st.rerun()

                elif intent == "followup_breakfast":
                    response_text, selected_idx = build_amenity_followup_answer(
                        last_cards,
                        "breakfast",
                        question=prompt,
                        requested_hotel_name=route_res.get("requested_hotel_name"),
                        selected_index=selected_idx,
                    )
                    st.session_state.selected_hotel_index = selected_idx

                    answer_container.markdown(response_text)
                    append_message({"role": "assistant", "content": response_text})
                    st.rerun()

                elif intent in ["followup_other_hotel", "follow_up"]:
                    all_cards = st.session_state.get("last_search_all_cards", [])
                    shown_count = st.session_state.get("shown_hotel_count", len(last_cards))
                    remaining_cards = all_cards[shown_count:shown_count + 3]

                    if not all_cards:
                        response_text = "I do not have previous hotel results to choose another hotel from. Please search for a city first."
                        answer_container.markdown(response_text)
                        append_message({"role": "assistant", "content": response_text})
                        st.rerun()
                    elif not remaining_cards:
                        response_text = "I have already shown all the hotels available for this search. Please try a different city or adjust your requirements."
                        answer_container.markdown(response_text)
                        append_message({"role": "assistant", "content": response_text})
                        st.rerun()
                    else:
                        from cmu_rag_answer import build_hotel_context
                        stored_requirements = st.session_state.get("last_search_query_requirements", {})
                        hotel_context_str = ""
                        for i, result in enumerate(remaining_cards, start=1):
                            hotel_context_str += build_hotel_context(prompt, result, i, lang_code="en") + "\n\n"

                        generator = generate_llm_answer(
                            query=prompt,
                            hotel_context_str=hotel_context_str,
                            chat_history=st.session_state.messages[:-1],
                            location=effective_location,
                            lang_code=t["code"],
                            hotel_cards=remaining_cards,
                            query_requirements=stored_requirements,
                        )
                        full_answer = collect_llm_answer(generator)
                        validation = validate_answer(full_answer, remaining_cards, "hotel_search", effective_location, t["code"])
                        final_display = validation["sanitized_answer"]

                        final_display = sanitize_before_render(final_display)
                        if final_display is None:
                            from cmu_rag_answer import safe_card_based_fallback_answer
                            final_display = safe_card_based_fallback_answer(
                                user_query=prompt,
                                hotel_cards=remaining_cards,
                                query_requirements=stored_requirements,
                                city=effective_location,
                                language=t["code"],
                            )

                        st.session_state["last_hotel_cards"] = remaining_cards
                        st.session_state["selected_hotel_index"] = 0
                        st.session_state["shown_hotel_count"] = shown_count + len(remaining_cards)

                        answer_container.markdown(final_display)
                        for card in remaining_cards:
                            render_hotel_card(card)

                        append_message({
                            "role": "assistant",
                            "content": final_display,
                            "hotels": remaining_cards,
                        })
                        st.rerun()

                else:
                    response_text, selected_idx = build_grounded_followup_answer(
                        prompt,
                        last_cards,
                        selected_index=selected_idx,
                        requested_hotel_name=route_res.get("requested_hotel_name"),
                    )
                    st.session_state.selected_hotel_index = selected_idx
                    answer_container.markdown(response_text)
                    append_message({"role": "assistant", "content": response_text})
                    st.rerun()

            else:
                results = search(
                    prompt,
                    location_filter=effective_location,
                    filters=None,
                    top_k_hotels=12,
                    requested_hotel_name=route_res.get("requested_hotel_name"),
                )

                if not results:
                    response_text = t["not_found"].format(effective_location or '')
                    st.markdown(response_text)
                    append_message({"role": "assistant", "content": response_text})
                    st.rerun()
                else:
                    cards_json = build_hotel_cards(
                        results=results,
                        user_query=prompt,
                        query_requirements=query_requirements,
                        requested_location=locals().get("effective_location"),
                        effective_scores=locals().get("effective_scores"),
                    )
                    full_sorted_cards = sorted(cards_json, key=lambda x: x.get("rank_score", 0.0), reverse=True)
                    sorted_cards = full_sorted_cards[:3]

                    answer_container = st.empty()

                    # Generate full answer silently with spinner
                    from cmu_rag_answer import build_hotel_context
                    hotel_context_str = ""
                    for i, result in enumerate(sorted_cards, start=1):
                        hotel_context_str += build_hotel_context(prompt, result, i, lang_code="en") + "\n\n"

                    generator = generate_llm_answer(
                        query=prompt,
                        hotel_context_str=hotel_context_str,
                        chat_history=st.session_state.messages[:-1],
                        location=effective_location,
                        lang_code=t["code"],
                        hotel_cards=sorted_cards,
                        query_requirements=query_requirements,
                    )
                    full_answer = collect_llm_answer(generator)
                    validation = validate_answer(full_answer, sorted_cards, "hotel_search", effective_location, t["code"])
                    final_display = validation["sanitized_answer"]

                    final_display = sanitize_before_render(final_display)
                    if final_display is None:
                        from cmu_rag_answer import safe_card_based_fallback_answer
                        final_display = safe_card_based_fallback_answer(
                            user_query=prompt,
                            hotel_cards=sorted_cards,
                            query_requirements=query_requirements,
                            city=effective_location,
                            language=t["code"]
                        )

                    # Update session state with the new context immediately
                    st.session_state["last_hotel_cards"] = sorted_cards
                    st.session_state["selected_hotel_index"] = 0
                    st.session_state["last_search_all_cards"] = full_sorted_cards
                    st.session_state["shown_hotel_count"] = len(sorted_cards)
                    st.session_state["last_search_query_requirements"] = query_requirements
                    if effective_location:
                        st.session_state["last_search_city"] = effective_location
                    if sorted_cards:
                        st.session_state["last_answered_hotel_name"] = sorted_cards[0].get("hotel_name", "UNKNOWN")

                    print(f"[STATE] last_hotel_cards_count={len(sorted_cards)} selected_hotel_index=0 last_search_city={effective_location}")

                    answer_container.markdown(final_display)
                    st.markdown("---")

                    for card in sorted_cards:
                        render_hotel_card(card)

                    st.session_state.last_hotel_cards = sorted_cards
                    st.session_state.selected_hotel_index = 0

                    append_message({
                        "role": "assistant",
                        "content": final_display,
                        "hotels": sorted_cards
                    })
                    st.rerun()


st.markdown(f"<div class='custom-footer'>{t['footer']}</div>", unsafe_allow_html=True)
