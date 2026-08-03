import re
import sys

def extract_final_answer(raw_text: str) -> str:
    """Return the user-facing text from a completed model stream.

    Foundry removes a matched stop sequence, so the normal successful shape
    is ``<answer>content`` without a closing tag. This mirrors
    ``cmu_rag_answer.extract_answer`` while keeping the validator import
    lightweight and independent from the model runtime.
    """
    if raw_text is None:
        return raw_text

    raw_text = str(raw_text).strip()
    if not raw_text:
        return raw_text

    opening_match = re.search(r"<answer\s*>", raw_text, re.IGNORECASE)
    if opening_match:
        answer_text = raw_text[opening_match.end():]
        closing_match = re.search(r"</answer\s*>", answer_text, re.IGNORECASE)
        if closing_match:
            answer_text = answer_text[:closing_match.start()]
        return answer_text.strip()

    raw_text = re.sub(
        r"\s*</answer\s*>\s*$", "", raw_text, flags=re.IGNORECASE
    )
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

    lines = raw_text.splitlines()
    first_nonempty = next(
        (index for index, line in enumerate(lines) if line.strip()), None
    )
    if first_nonempty is not None and meta_preamble.match(lines[first_nonempty]):
        remainder = "\n".join(lines[first_nonempty + 1:]).strip()
        if remainder:
            return remainder

    return raw_text


def _flatten_evidence_text(evidence_text) -> str:
    """Accept plain text as well as the review-result lists used by the UI."""
    if evidence_text is None:
        return ""
    if isinstance(evidence_text, str):
        return evidence_text
    if isinstance(evidence_text, dict):
        if "text" in evidence_text:
            return str(evidence_text.get("text") or "")
        return "\n".join(_flatten_evidence_text(value) for value in evidence_text.values())
    if isinstance(evidence_text, (list, tuple, set)):
        return "\n".join(_flatten_evidence_text(item) for item in evidence_text)
    return str(evidence_text)


def _normalize_claim_text(text: str) -> str:
    text = str(text or "").casefold()
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _claim_is_in_evidence(claim: str, evidence_text: str) -> bool:
    """Literal, punctuation-insensitive support check for a flagged claim."""
    normalized_claim = _normalize_claim_text(claim)
    normalized_evidence = _normalize_claim_text(evidence_text)
    return bool(
        normalized_claim
        and normalized_evidence
        and normalized_claim in normalized_evidence
    )


def _split_sentences(answer: str) -> list:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])(?:\s+|$)|\n+", str(answer or ""))
        if sentence.strip()
    ]

def detect_internal_analysis(answer: str) -> bool:
    forbidden = [
        "let me check", "düşünüyor...", "let's check",
        "i need to confirm", "okay, the user",
        "looking at the provided", "provided hotel cards",
        "after checking", "in conclusion, after checking",
        "<think>", "</think>",
        "okay, the user asked", "let me start", "i should mention",
        "i should also", "the user is asking", "i should make sure",
        "i must check", "i must answer", "i must follow",
        "i know the rules", "the user's question", "the user's request",
        "the answer must", "my answer must", "final answer must",
        "chain-of-thought", "hidden reasoning",
        "kullanıcı soruyor", "kullanıcının isteği", "bir değerlendireyim",
        "bir bakayım", "kısaca özetlemek gerekirse",
        "şimdi bilgileri kontrol ediyorum",
        "kullanıcı arıyor", "kullanıcı istiyor"
    ]
    ans_lower = answer.lower()
    return any(f in ans_lower for f in forbidden)

def detect_placeholders(answer: str) -> bool:
    placeholders = [
        "[insert", "[skor]", "[puan]", "[hotel name", "[otel adı",
        "travelmind is analyzing", "travelmind analiz ediyor",
        "chunk type:", "amenities_source:",
        "requirement_satisfaction:", "rank_score:",
    ]
    ans_lower = answer.lower()
    return any(p in ans_lower for p in placeholders)

def detect_score_overflow(answer: str) -> bool:
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*/\s*100', answer)
    for match in matches:
        try:
            if float(match) > 100.0:
                return True
        except ValueError:
            pass
    return False


def find_score_overflow_claims(answer: str) -> list:
    return [
        sentence
        for sentence in _split_sentences(answer)
        if detect_score_overflow(sentence)
    ]

def find_price_or_booking_claims(answer: str) -> list:
    price_keywords = [
        r"\$", "dollar", "euro", "£", "€", "₺", "tl",
        "per night", "nightly rate", "gecelik", "fiyatı",
        "rezervasyon yapabilirsiniz", "book now", "booking.com", "expedia",
        "reserve now", "available tonight", "available for booking",
        "rooms are available now", "müsait oda var"
    ]
    refusal_keywords = [
        "cannot provide live prices", "fiyat verisi güvenilir değil",
        "canlı fiyat", "fiyat bilgisi sunmaz",
        "does not provide live price", "not available",
        "does not provide", "sunmaz", "kapsamı dışında",
        "cannot confirm availability", "cannot confirm single-room availability"
    ]
    
    claims = []
    for sentence in _split_sentences(answer):
        lower_sent = sentence.lower()
        has_price_kw = False
        for kw in price_keywords:
            if kw in [r"\$", "£", "€", "₺"]:
                if kw.replace("\\", "") in lower_sent:
                    has_price_kw = True
                    break
            else:
                if re.search(r'\b' + re.escape(kw) + r'\b', lower_sent):
                    has_price_kw = True
                    break
        has_number = bool(re.search(r'\d+', sentence))
        is_refusal = any(kw in lower_sent for kw in refusal_keywords)
        
        if has_price_kw and not is_refusal:
            if (
                has_number
                or "book now" in lower_sent
                or "rezervasyon" in lower_sent
                or "reserve now" in lower_sent
                or "booking.com" in lower_sent
                or "available tonight" in lower_sent
                or "available for booking" in lower_sent
                or "rooms are available now" in lower_sent
                or "müsait oda var" in lower_sent
            ):
                claims.append(sentence)
    return claims


def detect_price_claims(answer: str) -> bool:
    return bool(find_price_or_booking_claims(answer))

def is_negative_context(sentence: str) -> bool:
    negatives = [
        r"\bnot\b", r"\bno\b", r"\bdoesn't\b", r"\bdon't\b", r"\bisn't\b", r"\baren't\b",
        r"\byok\b", r"\bbulunmuyor\b", r"mevcut değil", r"\bolmadığı\b", r"\bsunmuyor\b", r"\byoktur\b", r"\bbulunmamaktadır\b"
    ]
    sent_lower = sentence.lower()
    return any(re.search(neg, sent_lower) for neg in negatives)

def find_room_guarantees(answer: str, hotel_cards: list) -> list:
    live_availability_phrases = [
        "rooms are available now", "available for booking", 
        "available tonight", "müsait oda var", "rezervasyon yapabilirsiniz"
    ]
    single_room_phrases = [
        "single room is available", "single room available"
    ]
    safe_phrases = [
        "cannot confirm single-room availability",
        "does not provide live availability",
        "not available in the current records",
        "cannot be confirmed",
        "cannot confirm availability",
        "this information is not available",
        "appears in the current room data",
        "is confirmed in the current amenity data",
        "görünmektedir", "doğrulanıyor", "görünmüyor"
    ]
    claims = []
    for sentence in _split_sentences(answer):
        sent_lower = sentence.lower()
        if any(safe in sent_lower for safe in safe_phrases):
            continue
        if is_negative_context(sent_lower):
            continue
            
        if any(phrase in sent_lower for phrase in live_availability_phrases):
            claims.append(sentence)
            continue
            
        # A known room *type* is not proof of live bookability. Treat an
        # affirmative "is available" statement as a booking claim even when
        # the static card lists single rooms.
        if any(phrase in sent_lower for phrase in single_room_phrases):
            claims.append(sentence)
    return claims


def detect_room_guarantee(answer: str, hotel_cards: list) -> bool:
    return bool(find_room_guarantees(answer, hotel_cards))


def find_amenity_false_claims(answer: str, hotel_cards: list) -> list:
    breakfast_claims = [
        "has breakfast", "breakfast included", "breakfast is available",
        "breakfast service", "kahvaltısı var", "kahvaltı sunuyor",
        "kahvaltı hizmeti"
    ]
    claims = []
    for sentence in _split_sentences(answer):
        sent_lower = sentence.lower()
        if is_negative_context(sent_lower):
            continue
        for claim in breakfast_claims:
            if claim in sent_lower:
                any_breakfast = any(
                    card.get("amenities", {}).get("breakfast") == "YES"
                    for card in hotel_cards
                )
                if not any_breakfast:
                    claims.append(sentence)
                    break
    return claims


def detect_amenity_false_claim(answer: str, hotel_cards: list) -> bool:
    return bool(find_amenity_false_claims(answer, hotel_cards))

def detect_map_link_hallucination(answer: str, hotel_cards: list) -> bool:
    if "google.com/maps" in answer.lower():
        any_map = any(
            card.get("map_link_type") not in ("UNKNOWN", None)
            for card in hotel_cards
        )
        if not any_map:
            return True
    return False


_LINK_PATTERN = re.compile(
    r"(?:https?://|www\.)[^\s<>()\[\]{}]+"
    r"|\b(?:[a-z0-9-]+\.)+(?:com|org|net|io|co|travel|hotel)"
    r"/[^\s<>()\[\]{}]+",
    re.IGNORECASE,
)


def _canonical_link(link: str) -> str:
    link = str(link or "").strip().rstrip(".,;:!?\"'")
    link = re.sub(r"^https?://", "", link, flags=re.IGNORECASE)
    link = re.sub(r"^www\.", "", link, flags=re.IGNORECASE)
    return link.casefold().rstrip("/")


def find_fabricated_links(answer: str, hotel_cards: list, evidence_text: str = "") -> list:
    allowed_links = set()
    for card in hotel_cards or []:
        for key in ("map_link", "link", "url"):
            value = card.get(key)
            if value and str(value).upper() != "UNKNOWN":
                allowed_links.add(_canonical_link(value))

    evidence_lower = str(evidence_text or "").casefold()
    fabricated = []
    for match in _LINK_PATTERN.finditer(str(answer or "")):
        raw_link = match.group(0).rstrip(".,;:!?\"'")
        canonical = _canonical_link(raw_link)
        if not canonical:
            continue
        if canonical in allowed_links:
            continue
        if raw_link.casefold() in evidence_lower or canonical in evidence_lower:
            continue
        fabricated.append(raw_link)
    return fabricated

def build_safe_fallback_answer(target_language: str, intent: str, hotel_cards: list = None) -> str:
    # Just a wrapper if intent is hotel_search
    if intent == "hotel_search":
        from cmu_rag_answer import safe_card_based_fallback_answer
        return safe_card_based_fallback_answer(hotel_cards=hotel_cards or [], language=target_language)
    
    if intent == "price":
        if target_language.lower() in ["turkish", "tr"]:
            return "TravelMind sisteminde gerçek zamanlı rezervasyon verisi bulunmadığı için canlı fiyat veya müsaitlik bilgisi sağlanamamaktadır."
        else:
            return "TravelMind does not provide live pricing or availability because our system does not contain real-time booking data."
            
    if intent in ("follow_up", "specific_hotel_info", "review_question"):
        if target_language.lower() in ["turkish", "tr"]:
            return "İstediğiniz detayı doğrudan veri setimizde net olarak doğrulayamadım. Size başka nasıl yardımcı olabilirim?"
        else:
            return "I couldn't clearly verify that specific detail in our dataset. How else can I help you?"
            
    # Default fallback
    if target_language.lower() in ["turkish", "tr"]:
        return "TravelMind canlı fiyat veya rezervasyon bilgisi göstermez. Lütfen sadece konum belirterek arama yapınız."
    else:
        return "TravelMind does not provide live pricing or availability, only hotel recommendations based on historical reviews. Please try your search again using just a location."

def validate_answer(
    answer: str,
    hotel_cards: list,
    intent: str,
    requested_location: str,
    target_language: str,
    evidence_text=None,
    allowed_hotel_names=None,
) -> dict:
    """Validate an answer without discarding useful, evidence-based prose.

    ``passed`` remains a strict signal (no issues at all) for backward
    compatibility. ``needs_fallback`` is deliberately narrower: the answer is
    replaced only for leaked reasoning, unsupported price/booking claims, or
    fabricated links. All other findings are warnings and leave the original
    answer untouched.
    """
    answer = str(answer or "")
    hotel_cards = hotel_cards or []
    evidence = _flatten_evidence_text(evidence_text)
    issues = []

    def add_issue(issue_type, detail, blocks_output=False):
        issues.append(
            {
                "type": issue_type,
                "detail": detail,
                "blocks_output": bool(blocks_output),
            }
        )

    if detect_internal_analysis(answer):
        add_issue(
            "internal_analysis_leak",
            "Model leaked internal thoughts.",
            blocks_output=True,
        )

    if detect_placeholders(answer):
        add_issue("placeholder_leak", "Model copied template placeholders.")

    unsupported_score_claims = [
        claim
        for claim in find_score_overflow_claims(answer)
        if not _claim_is_in_evidence(claim, evidence)
    ]
    if unsupported_score_claims:
        add_issue("score_overflow", "Model hallucinated a score > 100.")

    unsupported_price_claims = [
        claim
        for claim in find_price_or_booking_claims(answer)
        if not _claim_is_in_evidence(claim, evidence)
    ]
    if unsupported_price_claims:
        add_issue(
            "price_booking_leak",
            f"Unsupported price/booking claim: {unsupported_price_claims[0]}",
            blocks_output=True,
        )

    allowed_hotels = [
        str(card.get("hotel_name"))
        for card in hotel_cards
        if card.get("hotel_name") not in (None, "", "UNKNOWN")
    ]
    if isinstance(allowed_hotel_names, str):
        allowed_hotels.append(allowed_hotel_names)
    elif allowed_hotel_names:
        allowed_hotels.extend(
            str(name)
            for name in allowed_hotel_names
            if name not in (None, "", "UNKNOWN")
        )

    potential_hotels = re.findall(
        r"([A-Z][a-zA-Z\s]+(?:Hotel|Resort|Inn|Suites|Motel))", answer
    )
    allowed_lower = [name.casefold().strip() for name in allowed_hotels]

    generic_terms = [
        "this hotel", "the hotel", "boutique hotel", "luxury hotel", "great hotel",
        "excellent hotel", "beautiful hotel", "nice hotel", "good hotel", "best hotel",
        "grand hotel", "spa resort", "family resort", "business hotel", "harika hotel",
        "mükemmel hotel", "güzel hotel", "iyi hotel", "a hotel", "an hotel",
    ]

    for potential_hotel in potential_hotels:
        hotel_name = potential_hotel.strip()
        hotel_name_lower = hotel_name.casefold()
        if (
            any(hotel_name_lower.endswith(generic) for generic in generic_terms)
            or hotel_name_lower in generic_terms
        ):
            continue

        is_allowed = any(
            hotel_name_lower == allowed
            or hotel_name_lower in allowed
            or allowed in hotel_name_lower
            for allowed in allowed_lower
        )
        if not is_allowed and not _claim_is_in_evidence(hotel_name, evidence):
            add_issue(
                "unknown_hotel_name",
                f"Mentioned unknown hotel: {hotel_name}",
            )

    unsupported_room_claims = [
        claim
        for claim in find_room_guarantees(answer, hotel_cards)
        if not _claim_is_in_evidence(claim, evidence)
    ]
    if unsupported_room_claims:
        # Live room availability is a booking claim, so it is one of the
        # three explicitly blocking categories even though the legacy issue
        # name is retained for callers that inspect it.
        add_issue(
            "single_room_hallucination",
            f"Unsupported room availability claim: {unsupported_room_claims[0]}",
            blocks_output=True,
        )

    unsupported_amenity_claims = [
        claim
        for claim in find_amenity_false_claims(answer, hotel_cards)
        if not _claim_is_in_evidence(claim, evidence)
    ]
    if unsupported_amenity_claims:
        add_issue(
            "breakfast_hallucination",
            f"Unverified breakfast claim: {unsupported_amenity_claims[0]}",
        )

    fabricated_links = find_fabricated_links(answer, hotel_cards, evidence)
    if fabricated_links:
        add_issue(
            "map_link_hallucination",
            f"Generated unsupported link: {fabricated_links[0]}",
            blocks_output=True,
        )

    blocking_issues = [issue for issue in issues if issue["blocks_output"]]
    warning_issues = [issue for issue in issues if not issue["blocks_output"]]
    for issue in warning_issues:
        print(
            f"[answer_validator] warning {issue['type']}: {issue['detail']}",
            file=sys.stderr,
        )

    sanitized = answer
    if blocking_issues:
        sanitized = build_safe_fallback_answer(target_language, intent, hotel_cards)

    return {
        "passed": len(issues) == 0,
        "issues": [issue["type"] for issue in issues],
        "blocking_issues": [issue["type"] for issue in blocking_issues],
        "warnings": [issue["type"] for issue in warning_issues],
        "sanitized_answer": sanitized,
        "needs_fallback": bool(blocking_issues),
    }
