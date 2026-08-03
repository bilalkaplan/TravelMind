"""Grounded, reusable hotel-feature wording for prompts, fallbacks, and UI.

The functions in this module deliberately ignore review-derived strengths and
boolean room guesses.  Public feature prose is built only from explicit hotel
metadata: confirmed amenities and named room types.
"""

from __future__ import annotations

import re
from typing import Iterable


CORE_AMENITY_ORDER = (
    "wifi",
    "pool",
    "breakfast",
    "parking",
    "wheelchair_accessible",
    "pet_friendly",
)

AMENITY_LABELS = {
    "wifi": "Wi-Fi",
    "pool": "pool",
    "breakfast": "breakfast",
    "parking": "parking",
    "wheelchair_accessible": "wheelchair access",
    "pet_friendly": "pet-friendly facilities",
    "gym": "gym/fitness facilities",
    "gym_fitness": "gym/fitness facilities",
    "restaurant_bar": "restaurant/bar",
}

OTHER_AMENITY_LABELS = {
    "gym fitness": "gym/fitness facilities",
    "fitness center": "gym/fitness facilities",
    "fitness centre": "gym/fitness facilities",
    "restaurant bar": "restaurant/bar",
    "business center": "business center",
    "business centre": "business center",
    "airport shuttle": "airport shuttle",
    "room service": "room service",
}

KNOWN_AMENITY_SIGNALS = {
    "wifi": (r"\bwi[\s-]?fi\b", r"\bwireless internet\b"),
    "pool": (r"\bpool\b", r"\bswimming\b"),
    "breakfast": (r"\bbreakfast\b",),
    "parking": (r"\bparking\b", r"\bvalet\b"),
    "wheelchair": (r"\bwheelchair\b", r"\baccessible\b"),
    "pet": (r"\bpet[\s-]?friendly\b", r"\bpets?\b"),
    "gym": (r"\bgym\b", r"\bfitness\b"),
    "restaurant": (r"\brestaurant\b",),
    "bar": (r"\bbar\b",),
    "spa": (r"\bspa\b",),
    "sauna": (r"\bsauna\b",),
    "shuttle": (r"\bshuttle\b",),
    "casino": (r"\bcasino\b",),
    "beach": (r"\bbeach\b",),
    "kitchen": (r"\bkitchen(?:ette)?\b",),
    "room service": (r"\broom service\b",),
    "laundry": (r"\blaundry\b",),
    "business center": (r"\bbusiness cent(?:er|re)\b",),
    "air conditioning": (r"\bair conditioning\b", r"\bair-conditioned\b"),
}

ROOM_TYPE_SIGNALS = (
    "single",
    "double",
    "twin",
    "king",
    "queen",
    "suite",
    "standard",
    "deluxe",
    "family",
    "studio",
    "apartment",
    "bungalow",
    "villa",
)


def _clean_fact(value) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"[<>{}\[\]|]", "", text).strip(" ,;:-")
    if text.casefold() in {"", "unknown", "none", "null", "nan"}:
        return ""
    return text[:100]


def _normalize(value) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _dedupe(items: Iterable[str]) -> list[str]:
    result = []
    seen = set()
    for item in items:
        cleaned = _clean_fact(item)
        key = _normalize(cleaned)
        if cleaned and key and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _amenity_label(key: str, raw_value=None) -> str:
    normalized_key = _normalize(key).replace(" ", "_")
    if normalized_key in AMENITY_LABELS:
        return AMENITY_LABELS[normalized_key]
    raw = _clean_fact(raw_value if raw_value is not None else key)
    normalized_raw = _normalize(raw)
    if normalized_raw in OTHER_AMENITY_LABELS:
        return OTHER_AMENITY_LABELS[normalized_raw]
    return raw


def get_verified_amenities(card: dict, limit: int | None = 6) -> list[str]:
    """Return only explicit, positively confirmed amenities in stable order."""
    amenities = (card or {}).get("amenities", {})
    labels = []

    if isinstance(amenities, dict):
        visited = set()
        for key in CORE_AMENITY_ORDER:
            if amenities.get(key) == "YES":
                labels.append(_amenity_label(key))
            visited.add(key)

        for key, value in amenities.items():
            if key in visited or key == "other":
                continue
            if value == "YES":
                labels.append(_amenity_label(key))

        other = amenities.get("other", [])
        if isinstance(other, str):
            other = [other]
        if isinstance(other, (list, tuple)):
            labels.extend(_amenity_label("other", item) for item in other)
    elif isinstance(amenities, (list, tuple)):
        labels.extend(_amenity_label("other", item) for item in amenities)

    result = _dedupe(labels)
    return result[:limit] if limit is not None else result


def get_recorded_room_types(card: dict, limit: int | None = 4) -> list[str]:
    """Return explicit named room types without inferring live availability."""
    card = card or {}
    room_info = card.get("room_info", {})
    metadata = card.get("metadata", {})
    sources = []

    if isinstance(room_info, dict):
        sources.extend(
            (room_info.get("room_types"), room_info.get("booking_room_types"))
        )
    sources.append(card.get("room_types"))
    if isinstance(metadata, dict):
        sources.extend(
            (metadata.get("room_types"), metadata.get("booking_room_types"))
        )

    room_types = []
    for source in sources:
        if isinstance(source, str):
            room_types.append(source)
        elif isinstance(source, (list, tuple)):
            room_types.extend(source)

    result = _dedupe(room_types)
    return result[:limit] if limit is not None else result


def join_english(items: Iterable[str]) -> str:
    values = [str(item) for item in items if str(item).strip()]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def get_required_cautions(card: dict, query_requirements: dict | None) -> list[str]:
    statuses = (card or {}).get("requirement_satisfaction", {}) or {}
    labels = {
        "breakfast": "breakfast",
        "pool": "pool",
        "wifi": "Wi-Fi",
        "wheelchair_accessible": "wheelchair access",
        "parking": "parking",
        "single_room": "a single room type",
        "double_room": "a double room type",
        "suite": "a suite room type",
    }
    required = [
        key
        for key, value in (query_requirements or {}).items()
        if value == "REQUIRED"
    ]
    cautions = []
    for key in required:
        label = labels.get(key, key.replace("_", " "))
        status = statuses.get(key)
        if status == "MISSING":
            cautions.append(f"requested {label} is not listed in its profile")
        elif status == "UNKNOWN":
            cautions.append(
                f"requested {label} could not be confirmed from its profile"
            )
    return cautions


def get_hotel_feature_facts(
    card: dict,
    query_requirements: dict | None = None,
    amenity_limit: int = 6,
    room_limit: int = 4,
) -> dict:
    raw_score = (card or {}).get(
        "travelmind_score", (card or {}).get("rank_score")
    )
    try:
        score = f"{float(raw_score):.1f}/100"
    except (TypeError, ValueError):
        score = ""
    return {
        "hotel_name": _clean_fact((card or {}).get("hotel_name")) or "This hotel",
        "location": _clean_fact((card or {}).get("location")),
        "score": score,
        "amenities": get_verified_amenities(card, limit=amenity_limit),
        "room_types": get_recorded_room_types(card, limit=room_limit),
        "cautions": get_required_cautions(card, query_requirements),
    }


def build_card_feature_sentence(
    card: dict,
    position: int = 0,
    city: str | None = None,
    query_requirements: dict | None = None,
) -> str:
    facts = get_hotel_feature_facts(card, query_requirements=query_requirements)
    name = facts["hotel_name"]
    score = facts["score"]
    location = _clean_fact(city) or facts["location"]

    if position == 0:
        place = f" for {location}" if location else ""
        lead = f"{name} leads the recommendations{place}"
    elif position == 1:
        lead = f"{name} is the next-ranked option"
    else:
        lead = f"{name} rounds out the displayed recommendations"
    if score:
        lead += f" at {score}"

    clauses = []
    if facts["amenities"]:
        clauses.append(
            "verified amenities include " + join_english(facts["amenities"])
        )
    if facts["room_types"]:
        clauses.append(
            "recorded room types include " + join_english(facts["room_types"])
        )
    clauses.extend(facts["cautions"])

    if not clauses:
        return lead + "."
    return lead + "; " + ", while ".join(clauses) + "."


def build_grounded_hotel_answer(
    hotel_cards: list[dict],
    city: str | None = None,
    query_requirements: dict | None = None,
) -> str:
    """Canonical answer: one fully grounded sentence per displayed card."""
    cards = list(hotel_cards or [])[:3]
    if not cards:
        return "Based on the current TravelMind records, I could not find matching hotel options."
    return " ".join(
        build_card_feature_sentence(
            card,
            position=index,
            city=city if index == 0 else None,
            query_requirements=query_requirements if index == 0 else None,
        )
        for index, card in enumerate(cards)
    )


def build_hotel_fact_lines(
    hotel_cards: list[dict], query_requirements: dict | None = None
) -> str:
    blocks = []
    for index, card in enumerate(list(hotel_cards or [])[:3], start=1):
        facts = get_hotel_feature_facts(
            card,
            query_requirements=query_requirements if index == 1 else None,
        )
        blocks.append(
            "\n".join(
                (
                    f"Hotel {index} name: {facts['hotel_name']}",
                    f"Exact score: {facts['score'] or 'not stated'}",
                    "Verified amenities: "
                    + (", ".join(facts["amenities"]) or "none supplied"),
                    "Recorded room types: "
                    + (", ".join(facts["room_types"]) or "none supplied"),
                    "Required caution: "
                    + ("; ".join(facts["cautions"]) or "none"),
                )
            )
        )
    return "\n\n".join(blocks)


def _fact_present(text: str, fact: str) -> bool:
    normalized_fact = _normalize(fact)
    return bool(normalized_fact and normalized_fact in _normalize(text))


def _amenity_concepts(labels: list[str]) -> set[str]:
    joined = " ".join(labels).casefold()
    concepts = set()
    for concept, patterns in KNOWN_AMENITY_SIGNALS.items():
        if any(re.search(pattern, joined, re.IGNORECASE) for pattern in patterns):
            concepts.add(concept)
    return concepts


def coalesce_hotel_rewrite_sentences(
    candidate: str, hotel_cards: list[dict]
) -> str:
    """Merge a model's fact-complete sub-sentences into one sentence per hotel.

    Small local models often turn one canonical hotel sentence into three
    sentences (score, amenities, room types).  This function changes only the
    punctuation between those clauses.  It refuses to operate unless every
    expected hotel name is present in the original ranked order; the fact gate
    still performs the substantive validation afterwards.
    """
    text = str(candidate or "").strip()
    cards = list(hotel_cards or [])[:3]
    if not text or not cards:
        return text

    positions = []
    cursor = 0
    for card in cards:
        name = get_hotel_feature_facts(card)["hotel_name"]
        match = re.search(re.escape(name), text[cursor:], re.IGNORECASE)
        if match is None:
            return text
        start = cursor + match.start()
        if not positions and text[:start].strip():
            return text
        positions.append((start, name, card))
        cursor = start + len(match.group(0))

    merged_segments = []
    for index, (start, _name, card) in enumerate(positions):
        end = positions[index + 1][0] if index + 1 < len(positions) else len(text)
        segment = re.sub(r"\s+", " ", text[start:end]).strip()

        # Protect fact values containing periods (for example "St. Regis")
        # so sentence splitting cannot alter the immutable text itself.
        facts = get_hotel_feature_facts(card)
        protected_values = (
            [facts["hotel_name"]] + facts["amenities"] + facts["room_types"]
        )
        replacements = {}
        for value in protected_values:
            if "." not in value:
                continue
            match = re.search(re.escape(value), segment, re.IGNORECASE)
            if match is None:
                continue
            token = f"FACTTOKEN{len(replacements)}"
            replacements[token] = match.group(0)
            segment = re.sub(
                re.escape(match.group(0)), token, segment, count=1,
                flags=re.IGNORECASE,
            )

        parts = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", segment)
            if part.strip()
        ]
        if len(parts) > 1:
            normalized_parts = []
            for part_index, part in enumerate(parts):
                clause = part.rstrip(".!? ")
                if part_index and clause.startswith("Verified "):
                    clause = "verified " + clause[len("Verified "):]
                elif part_index and clause.startswith("Recorded "):
                    clause = "recorded " + clause[len("Recorded "):]
                normalized_parts.append(clause)
            segment = "; ".join(normalized_parts) + "."

        for token, value in replacements.items():
            segment = segment.replace(token, value)
        merged_segments.append(segment)

    return " ".join(merged_segments)


def validate_hotel_feature_rewrite(
    candidate: str,
    hotel_cards: list[dict],
    query_requirements: dict | None = None,
) -> tuple[bool, list[str]]:
    """Strict local fact gate for Qwen's style-only rewrite."""
    text = str(candidate or "").strip()
    cards = list(hotel_cards or [])[:3]
    reasons = []
    if not text:
        return False, ["empty answer"]

    lower = text.casefold()
    forbidden_patterns = (
        r"<\/?(?:think|answer)\b",
        r"https?://|www\.",
        r"\b(?:book|booking|reserve|reservation|available|availability)\b",
        r"\b(?:price|nightly rate|per night|phone|telephone)\b",
        r"\b(?:free|included|complimentary)\b",
        r"\b(?:luxury|luxurious|spacious|comfortable|convenient|excellent|"
        r"ideal|perfect|premium|modern|cozy|relaxing)\b",
        r"\b(?:okay|let me|the user|i must|i should|i need to)\b",
    )
    for pattern in forbidden_patterns:
        if re.search(pattern, lower, re.IGNORECASE):
            reasons.append(f"forbidden language: {pattern}")

    # Preserve the original three-sentence answer contract: each displayed
    # card receives one sentence. Mask punctuation inside immutable fact names
    # (for example, "St. Regis") before counting terminal punctuation.
    sentence_text = text
    for card in cards:
        facts_for_masking = get_hotel_feature_facts(card)
        immutable_values = (
            [facts_for_masking["hotel_name"]]
            + facts_for_masking["amenities"]
            + facts_for_masking["room_types"]
        )
        for value in immutable_values:
            if "." not in value:
                continue
            sentence_text = re.sub(
                re.escape(value),
                value.replace(".", ""),
                sentence_text,
                flags=re.IGNORECASE,
            )
    sentence_count = len(re.findall(r"[.!?](?=\s|$)", sentence_text))
    if sentence_count != len(cards):
        reasons.append(
            f"expected {len(cards)} sentences, found {sentence_count}"
        )

    # Scores are the only numeric facts the rewrite may contain.
    allowed_numbers = set()
    for card in cards:
        facts = get_hotel_feature_facts(card)
        allowed_numbers.update(re.findall(r"\d+(?:\.\d+)?", facts["score"]))
    candidate_numbers = set(re.findall(r"\d+(?:\.\d+)?", text))
    if not candidate_numbers.issubset(allowed_numbers):
        reasons.append("unsupported number")

    name_positions = []
    for index, card in enumerate(cards):
        facts = get_hotel_feature_facts(
            card,
            query_requirements=query_requirements if index == 0 else None,
        )
        name = facts["hotel_name"]
        position = lower.find(name.casefold())
        if position < 0:
            reasons.append(f"missing hotel name: {name}")
        else:
            name_positions.append((position, index, facts))

    name_positions.sort()
    for order, (start, _card_index, facts) in enumerate(name_positions):
        end = name_positions[order + 1][0] if order + 1 < len(name_positions) else len(text)
        segment = text[start:end]
        signal_segment = re.sub(
            re.escape(facts["hotel_name"]), "", segment, flags=re.IGNORECASE
        )

        if facts["score"] and not re.search(
            re.escape(facts["score"]).replace(r"/", r"\s*/\s*"),
            segment,
        ):
            reasons.append(f"missing or changed score: {facts['hotel_name']}")

        for amenity in facts["amenities"]:
            if not _fact_present(segment, amenity):
                reasons.append(
                    f"missing amenity {amenity}: {facts['hotel_name']}"
                )
        for room_type in facts["room_types"]:
            if not _fact_present(segment, room_type):
                reasons.append(
                    f"missing room type {room_type}: {facts['hotel_name']}"
                )
        for caution in facts["cautions"]:
            if not _fact_present(segment, caution):
                reasons.append(f"missing required caution: {facts['hotel_name']}")

        # A required-but-missing feature may legitimately appear inside its
        # mandatory negative caution, so include cautions in the signal scope.
        allowed_concepts = _amenity_concepts(
            facts["amenities"] + facts["cautions"]
        )
        for concept, patterns in KNOWN_AMENITY_SIGNALS.items():
            if concept in allowed_concepts:
                continue
            if any(
                re.search(pattern, signal_segment, re.IGNORECASE)
                for pattern in patterns
            ):
                reasons.append(
                    f"unsupported amenity {concept}: {facts['hotel_name']}"
                )

        allowed_rooms = " ".join(
            facts["room_types"] + facts["cautions"]
        ).casefold()
        for room_signal in ROOM_TYPE_SIGNALS:
            if room_signal in allowed_rooms:
                continue
            if re.search(rf"\b{re.escape(room_signal)}\b", signal_segment, re.IGNORECASE):
                reasons.append(
                    f"unsupported room type {room_signal}: {facts['hotel_name']}"
                )

    return not reasons, reasons
