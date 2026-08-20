import json
import math
import re
from pathlib import Path


WEIGHTS = {
    "location_match": 15,
    "hotel_class": 10,
    "amenities_match": 15,
    "room_type_match": 12,
    "review_overall": 20,
    "review_service": 8,
    "review_rooms": 8,
    "review_cleanliness": 7,
    "review_volume": 5,
}

QUERY_REQUIREMENT_KEYS = (
    "breakfast",
    "single_room",
    "double_room",
    "suite",
    "pool",
    "wifi",
    "wheelchair_accessible",
    "parking",
)

_REQUIREMENT_ALIASES = {
    "single": "single_room",
    "double": "double_room",
    "king_room": "double_room",
    "queen_room": "double_room",
    "wheelchair": "wheelchair_accessible",
    "accessible": "wheelchair_accessible",
    "wi-fi": "wifi",
    "internet": "wifi",
}

_QUERY_FEATURE_PATTERNS = {
    "breakfast": (r"\bbreakfast\b", r"\bmorning meal\b"),
    "single_room": (
        r"\bsingle(?:\s+(?:room|bed))?\b",
    ),
    "double_room": (
        r"\bdouble(?:\s+(?:room|bed))?\b",
        r"\b(?:king|queen|twin|full)\s+(?:room|bed)\b",
    ),
    "suite": (r"\bsuite\b", r"\bpresidential(?:\s+suite)?\b"),
    "pool": (r"\bpool\b", r"\bswimming\s+pool\b"),
    "wifi": (r"\bwi[ -]?fi\b", r"\bwireless\b", r"\binternet\b"),
    "wheelchair_accessible": (
        r"\bwheelchair(?:\s+accessible)?\b",
        r"\baccessible\b",
        r"\bhandicap\b",
    ),
    "parking": (r"\bparking\b", r"\bvalet\b", r"\bgarage\b"),
}


def normalize_requirement_status(value):
    """Return one of the router's REQUIRED/OPTIONAL/NONE values."""

    if value is True:
        return "REQUIRED"
    if value in (False, None):
        return "NONE"
    normalized = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    if normalized in {"REQUIRED", "MUST", "YES", "TRUE"}:
        return "REQUIRED"
    if normalized in {"OPTIONAL", "PREFERRED", "PREFER", "NICE_TO_HAVE"}:
        return "OPTIONAL"
    return "NONE"


def normalize_query_requirements(requirements=None, query=None):
    """Canonicalize router/fast-router requirements and fill omitted signals.

    The LLM router normally returns all fields, including explicit ``NONE``.
    The fast router returns a sparse mapping, so feature words omitted from
    that mapping are recovered from the original query. Explicit ``NONE`` is
    always respected and is never re-inferred as a request.
    """

    raw = requirements if isinstance(requirements, dict) else {}
    if isinstance(raw.get("query_requirements"), dict):
        raw = raw["query_requirements"]

    normalized = {}
    explicit_keys = set()
    for raw_key, value in raw.items():
        key = str(raw_key).strip().casefold().replace("-", "_").replace(" ", "_")
        key = _REQUIREMENT_ALIASES.get(key, key)
        if key not in QUERY_REQUIREMENT_KEYS:
            continue
        normalized[key] = normalize_requirement_status(value)
        explicit_keys.add(key)

    query_text = normalize_text(query or "")
    optional_language = bool(
        re.search(
            r"\b(?:prefer|preferred|ideally|optional|nice to have|would be nice)\b",
            query_text,
        )
    )
    for key, patterns in _QUERY_FEATURE_PATTERNS.items():
        if key in explicit_keys or not any(re.search(pattern, query_text) for pattern in patterns):
            continue
        negative = any(
            re.search(prefix + pattern, query_text)
            for pattern in patterns
            for prefix in (
                r"\bno\s+",
                r"\bwithout\s+",
                r"\bdon t need\s+",
                r"\bdo not need\s+",
            )
        )
        if negative:
            normalized[key] = "NONE"
        else:
            normalized[key] = "OPTIONAL" if optional_language else "REQUIRED"

    return normalized

if sum(WEIGHTS.values()) != 100:
    raise RuntimeError("TravelMind scoring weights must add up to exactly 100.")

REVIEW_STATS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "hotel_review_stats.json"
)
REVIEW_RATING_FIELDS = ("overall", "service", "rooms", "cleanliness", "value")


def _load_review_stats():
    if not REVIEW_STATS_PATH.is_file():
        return {}
    try:
        with REVIEW_STATS_PATH.open("r", encoding="utf-8") as stats_file:
            data = json.load(stats_file)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(hotel_id): values
        for hotel_id, values in data.items()
        if isinstance(values, dict)
    }


HOTEL_REVIEW_STATS = _load_review_stats()


def _maximum_review_count(stats):
    counts = []
    for values in stats.values():
        try:
            count = int(values.get("review_count", 0))
        except (TypeError, ValueError):
            continue
        if count > 0:
            counts.append(count)
    return max(counts, default=0)


MAX_REVIEW_COUNT = _maximum_review_count(HOTEL_REVIEW_STATS)

CLEANLINESS_POSITIVE_KEYWORDS = [
    "clean",
    "cleanliness",
    "hygiene",
    "spotless",
    "tidy",
    "neat",
]

CLEANLINESS_NEGATIVE_KEYWORDS = [
    "dirty",
    "unclean",
    "filthy",
    "smelly",
    "dusty",
    "stained",
]

LOCATION_WORDS_TO_IGNORE = {
    "hotel", "clean", "location",
    "good", "double", "bed",
    "room", "looking", "want", "with",
    "and", "for", "the", "rating",
    "score", "find",
}

def normalize_text(text):
    text = str(text).lower()
    text = text.replace("'", " ").replace("’", " ")
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def to_float(value):
    try:
        if value is None: return None
        value = str(value).strip()
        if value == "" or value.lower() == "nan": return None
        return float(value)
    except ValueError:
        return None

def extract_possible_location_tokens(query):
    query_text = normalize_text(query)
    tokens = query_text.split()
    possible_tokens = []
    for token in tokens:
        if len(token) < 3 or token in LOCATION_WORDS_TO_IGNORE:
            continue
        possible_tokens.append(token)
    return possible_tokens

def score_location_match(query, location, requested_location=None):
    if requested_location:
        location_text = normalize_text(location)
        requested_text = normalize_text(requested_location)
        if not location_text:
            return None, "Location data not found in dataset."
        requested_city = requested_text.split(",", 1)[0].strip()
        if requested_text in location_text or (
            requested_city and requested_city in location_text
        ):
            return 100, f"Location matched: {requested_location}"
        return 0, f"Hotel is not in the requested location: {requested_location}"

    tokens = extract_possible_location_tokens(query)
    location_text = normalize_text(location)
    if not tokens:
        return None, "User did not specify a specific country/city/region."
    matched_tokens = [token for token in tokens if token in location_text]
    if matched_tokens:
        return 100, f"Location matched: {', '.join(matched_tokens)}"
    return 0, "The requested location was not found in this record's location field."

def score_room_match(query, room_types_list, query_requirements=None):
    query_text = normalize_text(query)

    if query_requirements is None:
        requested_types = []
        if any(kw in query_text for kw in ["suite", "presidential"]):
            requested_types.append("suite")
        if any(
            kw in query_text
            for kw in [
                "double bed",
                "double room",
                "queen bed",
                "king bed",
                "full bed",
            ]
        ):
            requested_types.append("double_room")
        if any(
            kw in query_text
            for kw in ["single room", "single bed"]
        ):
            requested_types.append("single_room")
    else:
        normalized_requirements = normalize_query_requirements(query_requirements)
        requested_types = [
            key
            for key in ("single_room", "double_room", "suite")
            if normalized_requirements.get(key) in {"REQUIRED", "OPTIONAL"}
        ]

    if not requested_types:
        return None, "User did not specify a room/bed type preference."

    if not room_types_list:
        return None, "Bed type data not found in dataset."

    room_text = normalize_text(" ".join(str(r) for r in room_types_list))
    matchers = {
        "suite": ("suite", "presidential"),
        "double_room": ("double", "twin", "king", "queen", "full"),
        "single_room": ("single", "twin"),
    }
    matches = sum(
        any(keyword in room_text for keyword in matchers[room_type])
        for room_type in requested_types
    )
    score = matches / len(requested_types) * 100
    if matches == len(requested_types):
        return score, "All requested room types matched the records."
    if matches:
        return score, "Requested room types partially matched."
    return 0, "Requested room types are not in the hotel's recorded list."


def score_amenities_match(query, amenities_list, query_requirements=None):
    amenities_known = bool(amenities_list)
    if isinstance(amenities_list, dict):
        amenities_known = bool(amenities_list)
        amenities_list = [
            key
            for key, value in amenities_list.items()
            if str(value).strip().upper() in {"YES", "TRUE", "1"}
        ]
    elif isinstance(amenities_list, str):
        amenities_list = [amenities_list] if amenities_list.strip() else []

    if not amenities_known or not isinstance(amenities_list, list):
        return None, "Amenity data not found in dataset."

    query_text = normalize_text(query)
    am_str = normalize_text(" ".join(str(amenity) for amenity in amenities_list))

    requests = []
    amenity_keywords = {
        "wifi": ("wifi", "wi fi", "internet", "wireless"),
        "pool": ("pool", "swimming"),
        "breakfast": ("breakfast", "morning meal"),
        "parking": ("parking", "valet", "garage"),
        "wheelchair_accessible": (
            "wheelchair",
            "accessible",
            "handicap",
            "disabled",
        ),
    }
    if query_requirements is not None:
        normalized_requirements = normalize_query_requirements(query_requirements)
        requests = [
            (key, amenity_keywords[key])
            for key in amenity_keywords
            if normalized_requirements.get(key) in {"REQUIRED", "OPTIONAL"}
        ]
    else:
        for key, keywords in amenity_keywords.items():
            if any(keyword in query_text for keyword in keywords):
                requests.append((key, keywords))

    if not requests:
        if query_requirements is not None:
            return None, "User did not specify a specific amenity preference."
        # Fallback to general amenity count
        core = ["wifi", "pool", "breakfast", "parking", "restaurant", "bar", "fitness"]
        matches = sum(1 for c in core if c in am_str)
        return min((matches / 3.0) * 100, 100), f"General amenity richness evaluated ({len(amenities_list)} amenities found)."
        
    matches = 0
    for _request_name, keywords in requests:
        if any(kw in am_str for kw in keywords):
            matches += 1
            
    score = (matches / len(requests)) * 100
    if score == 100:
        return score, "All user amenity requests were met."
    elif score > 0:
        return score, "User amenity requests were partially met."
    else:
        return 0, "User amenity requests were not met."

def score_hotel_class(hotel_class_str):
    value = to_float(hotel_class_str)
    if value is None:
        return None, "Hotel class (stars) data not found in dataset."
    value = max(0, min(value, 5))
    score = (value / 5.0) * 100
    return score, f"Hotel class sourced from dataset: {value} stars."

def normalize_review_rating(mean_rating):
    """Map an available mean on the 1-5 scale to a 0-100 score."""

    value = to_float(mean_rating)
    if value is None or not 1 <= value <= 5:
        return None
    return (value - 1) / 4 * 100


def score_review_rating(mean_rating, label):
    score = normalize_review_rating(mean_rating)
    if score is None:
        return None, f"{label} rating not found in dataset."
    return score, f"Average {label.lower()} rating: {float(mean_rating):.2f} / 5."


def score_review_volume(review_count, maximum_review_count=None):
    value = to_float(review_count)
    maximum = to_float(
        MAX_REVIEW_COUNT if maximum_review_count is None else maximum_review_count
    )
    if value is None or value <= 0:
        return None, "Review count data not found in dataset."
    if maximum is None or maximum <= 0:
        maximum = value
    score = min(math.log1p(value) / math.log1p(maximum) * 100, 100)
    return (
        score,
        "Review volume normalized against dataset maximum: "
        f"{int(value)} reviews.",
    )


def score_review_count(review_count):
    """Backward-compatible alias for the review-volume scorer."""

    return score_review_volume(review_count)


def _normalise_hotel_id(value):
    hotel_id = str(value or "").strip()
    if hotel_id.endswith(".0") and hotel_id[:-2].isdigit():
        return hotel_id[:-2]
    return hotel_id


def get_hotel_review_stats(metadata):
    """Resolve generated statistics, with metadata fallbacks for compatibility."""

    hotel_id = _normalise_hotel_id(metadata.get("hotel_id"))
    generated = HOTEL_REVIEW_STATS.get(hotel_id, {})
    stats = dict(generated) if isinstance(generated, dict) else {}

    embedded_stats = metadata.get("review_stats")
    if isinstance(embedded_stats, dict):
        for field in (*REVIEW_RATING_FIELDS, "review_count"):
            if stats.get(field) is None and embedded_stats.get(field) is not None:
                stats[field] = embedded_stats[field]

    for field in REVIEW_RATING_FIELDS:
        if stats.get(field) is not None:
            continue
        for key in (
            f"review_{field}",
            f"{field}_rating_mean",
            f"{field}_rating",
        ):
            if metadata.get(key) is not None:
                stats[field] = metadata[key]
                break

    if stats.get("review_count") is None:
        stats["review_count"] = metadata.get(
            "review_count_total", metadata.get("review_count_in_chunk")
        )
    return stats

def user_asked_cleanliness(query):
    query_text = normalize_text(query)
    terms = ["clean", "hygiene", "cleanliness"]
    return any(term in query_text for term in terms)

def score_cleanliness_comment(query, text):
    if not user_asked_cleanliness(query):
        return None, "User did not specify a preference for cleanliness."
    comment_text = normalize_text(text)
    has_positive = any(word in comment_text for word in CLEANLINESS_POSITIVE_KEYWORDS)
    has_negative = any(word in comment_text for word in CLEANLINESS_NEGATIVE_KEYWORDS)
    if has_negative:
        return 20, "Negative statement regarding cleanliness found in text."
    if has_positive:
        return 100, "Positive statement regarding cleanliness found in text."
    return 50, "No clear statement regarding cleanliness found."

def _merge_text_values(*collections):
    merged = []
    seen = set()
    for collection in collections:
        if isinstance(collection, str):
            collection = [collection]
        if not isinstance(collection, (list, tuple, set)):
            continue
        for value in collection:
            text = str(value).strip()
            key = normalize_text(text)
            if not text or not key or key in seen:
                continue
            seen.add(key)
            merged.append(text)
    return merged


def calculate_travelmind_score(
    query,
    result,
    query_requirements=None,
    requested_location=None,
):
    metadata = result.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    normalized_requirements = (
        normalize_query_requirements(query_requirements, query)
        if query_requirements is not None
        else None
    )
    location_score, location_reason = score_location_match(
        query,
        metadata.get("location", metadata.get("city", "")),
        requested_location=requested_location,
    )

    room_types = _merge_text_values(
        metadata.get("room_types", []),
        metadata.get("booking_room_types", []),
    )

    room_score, room_reason = score_room_match(
        query,
        room_types,
        query_requirements=normalized_requirements,
    )

    hotel_class_score, hotel_class_reason = score_hotel_class(metadata.get("hotel_class", ""))
    
    amenities = metadata.get("amenities", [])
    amenities_score, amenities_reason = score_amenities_match(
        query,
        amenities,
        query_requirements=normalized_requirements,
    )

    review_stats = get_hotel_review_stats(metadata)
    overall_score, overall_reason = score_review_rating(
        review_stats.get("overall"), "Overall"
    )
    service_score, service_reason = score_review_rating(
        review_stats.get("service"), "Service"
    )
    review_rooms_score, review_rooms_reason = score_review_rating(
        review_stats.get("rooms"), "Rooms"
    )
    review_cleanliness_score, review_cleanliness_reason = score_review_rating(
        review_stats.get("cleanliness"), "Cleanliness"
    )
    review_volume_score, review_volume_reason = score_review_volume(
        review_stats.get("review_count")
    )

    total_weight = 0
    weighted_sum = 0
    components = []

    component_values = (
        ("location_match", "Location Match", location_score, location_reason),
        ("hotel_class", "Hotel Class", hotel_class_score, hotel_class_reason),
        ("amenities_match", "Amenities Match", amenities_score, amenities_reason),
        ("room_type_match", "Room Type Match", room_score, room_reason),
        ("review_overall", "Overall Review Rating", overall_score, overall_reason),
        ("review_service", "Service Review Rating", service_score, service_reason),
        ("review_rooms", "Rooms Review Rating", review_rooms_score, review_rooms_reason),
        (
            "review_cleanliness",
            "Cleanliness Review Rating",
            review_cleanliness_score,
            review_cleanliness_reason,
        ),
        ("review_volume", "Review Volume", review_volume_score, review_volume_reason),
    )

    missing_signals = []
    for key, name, score, reason in component_values:
        if score is None:
            missing_signals.append(name)
            continue
        weight = WEIGHTS[key]
        weighted_sum += score * weight
        total_weight += weight
        components.append(
            {
                "key": key,
                "name": name,
                "score": score,
                "reason": reason,
                "weight": weight,
            }
        )

    if total_weight == 0:
        final_score = 0
    else:
        final_score = weighted_sum / total_weight

    return {
        "travelmind_score": final_score,
        "components": components,
        "missing_signals": missing_signals,
    }

def build_strengths(result):
    text = str(result.get("text", "")).lower()
    metadata = result.get("metadata", {})
    strengths = []
    
    hotel_class_str = metadata.get("hotel_class", "")
    if "4." in hotel_class_str or "5." in hotel_class_str:
        strengths.append("High-star, premium hotel classification.")
        
    amenities = metadata.get("amenities", [])
    if isinstance(amenities, dict):
        amenities = [k for k, v in amenities.items() if v == "YES"]
    if len(amenities) >= 4:
        strengths.append("Rich in hotel amenities.")
        
    try:
        review_count = int(metadata.get("review_count_total", 0))
        if review_count > 500:
            strengths.append(f"Highly reviewed and reliable ({review_count} reviews).")
    except (ValueError, TypeError):
        pass

    if any(k in text for k in CLEANLINESS_POSITIVE_KEYWORDS):
        strengths.append("Reviews contain distinctly positive statements about cleanliness.")
        
    if any(k in text for k in ["central", "midtown", "located", "metro", "subway"]):
        strengths.append("Reviews highlight a central and accessible location.")
        
    if any(k in text for k in ["staff", "friendly", "helpful", "service"]):
        strengths.append("Positive signals regarding staff or service in reviews.")
        
    if not strengths:
        strengths.append("Recommended due to the highest TravelMind suitability score for your query.")
        
    return strengths[:4]

def build_cautions(result):
    text = str(result.get("text", "")).lower()
    metadata = result.get("metadata", {})
    cautions = []
    
    hotel_class_str = metadata.get("hotel_class", "")
    if "1." in hotel_class_str or "2." in hotel_class_str:
        cautions.append("Low hotel class/star rating. May not offer a luxury experience.")

    if any(k in text for k in CLEANLINESS_NEGATIVE_KEYWORDS):
        cautions.append("Some reviews contain negative statements or complaints about cleanliness.")
        
    if any(k in text for k in ["small room", "extremely small", "tiny"]):
        cautions.append("Some reviews indicate that the rooms may be very small.")
        
    if any(k in text for k in ["noisy", "noise", "loud"]):
        cautions.append("Some reviews mention potential noise issues.")
        
    if any(k in text for k in ["over-booked", "overbooked"]):
        cautions.append("Some reviews mention overbooking issues.")
        
    if any(k in text for k in ["rude", "unhelpful", "horrible service"]):
        cautions.append("Some reviews contain negative statements regarding staff attitude.")
        
    return cautions[:3]
