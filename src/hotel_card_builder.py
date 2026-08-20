
from travelmind_scoring import (
    calculate_travelmind_score,
    normalize_query_requirements,
    normalize_text,
)


REQUIRED_MATCH_BOOST = 125.0
REQUIRED_MISSING_PENALTY = 125.0
REQUIRED_UNKNOWN_PENALTY = 60.0
OPTIONAL_MATCH_BOOST = 25.0

AMENITY_FEATURE_KEYWORDS = {
    "wifi": ("wifi", "wi fi", "internet", "wireless"),
    "breakfast": ("breakfast", "morning meal"),
    "pool": ("pool", "swimming"),
    "wheelchair_accessible": (
        "wheelchair",
        "accessible",
        "handicap",
        "disabled",
    ),
    "parking": ("parking", "park", "valet", "garage"),
    "pet_friendly": ("pet", "dog", "cat"),
}

ROOM_FEATURE_KEYWORDS = {
    "single_room": ("single",),
    "double_room": ("double", "twin", "king", "queen", "full"),
    "suite": ("suite", "presidential"),
}


def _has_metadata_value(value) -> bool:
    """Whether a retrieval value contains information worth overriding with."""
    if value is None:
        return False
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().casefold() not in {"", "unknown", "none", "null", "nan"}
    return True


def merge_hotel_metadata(full_metadata: dict, chunk_metadata: dict) -> dict:
    """Prefer retrieval fields only when they are actually populated.

    Some installations have an older vector DB whose rows contain empty
    ``amenities`` or room lists.  Those empty containers must not erase the
    newer enriched hotel profile loaded from JSON.
    """
    merged = dict(full_metadata or {})
    for key, value in (chunk_metadata or {}).items():
        if key in {"amenities", "room_types", "booking_room_types"}:
            existing = merged.get(key)
            if _has_metadata_value(existing) and _has_metadata_value(value):
                merged[key] = _merge_unique_text_values(existing, value)
                continue
        if key not in merged or _has_metadata_value(value):
            merged[key] = value
    return merged


def _merge_unique_text_values(*collections) -> list:
    merged = []
    seen = set()
    for collection in collections:
        if isinstance(collection, dict):
            collection = [
                key
                for key, value in collection.items()
                if str(value).strip().upper() in {"YES", "TRUE", "1"}
            ]
        elif isinstance(collection, str):
            collection = [collection]
        if not isinstance(collection, (list, tuple, set)):
            continue
        for value in collection:
            text = str(value).strip()
            normalized = normalize_text(text)
            if not text or not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(text)
    return merged


def get_combined_room_types(metadata: dict) -> list:
    return _merge_unique_text_values(
        metadata.get("room_types", []),
        metadata.get("booking_room_types", []),
    )


def _feature_status(values, keywords) -> str:
    """Return YES/NO/UNKNOWN while preserving explicit dictionary states."""

    if isinstance(values, dict):
        matching_states = []
        for key, value in values.items():
            normalized_key = normalize_text(key)
            if not any(keyword in normalized_key for keyword in keywords):
                continue
            normalized_value = str(value).strip().upper()
            if normalized_value in {"YES", "TRUE", "1"} or value is True:
                matching_states.append("YES")
            elif normalized_value in {"NO", "FALSE", "0"} or value is False:
                matching_states.append("NO")
            else:
                matching_states.append("UNKNOWN")
        if "YES" in matching_states:
            return "YES"
        if "NO" in matching_states:
            return "NO"
        return "UNKNOWN"

    if isinstance(values, str):
        values = [values] if values.strip() else []
    if not isinstance(values, (list, tuple, set)) or not values:
        return "UNKNOWN"
    normalized_values = normalize_text(" ".join(str(value) for value in values))
    return "YES" if any(keyword in normalized_values for keyword in keywords) else "NO"


def evaluate_hotel_features(metadata: dict) -> dict:
    amenities = metadata.get("amenities", [])
    room_types = get_combined_room_types(metadata)
    statuses = {
        key: _feature_status(amenities, keywords)
        for key, keywords in AMENITY_FEATURE_KEYWORDS.items()
    }
    statuses.update(
        {
            key: _feature_status(room_types, keywords)
            for key, keywords in ROOM_FEATURE_KEYWORDS.items()
        }
    )
    return statuses

def extract_boolean_feature(amenities: list, keywords: list) -> str:
    """
    Checks if any of the keywords exist in the amenities list.
    Returns 'YES' if found, 'NO' if amenities list exists but not found, 
    and 'UNKNOWN' if amenities list is completely missing or empty.
    """
    normalized_keywords = tuple(normalize_text(keyword) for keyword in keywords)
    return _feature_status(amenities, normalized_keywords)


def _locations_match(requested_location, actual_location) -> bool:
    requested = normalize_text(requested_location)
    actual = normalize_text(actual_location)
    if not requested or not actual:
        return True
    requested_city = requested.split(",", 1)[0].strip()
    return requested in actual or (requested_city and requested_city in actual)


def _hotel_names_match(requested_name, actual_name) -> bool:
    requested = normalize_text(requested_name)
    actual = normalize_text(actual_name)
    if not requested:
        return True
    return requested == actual or requested in actual or actual in requested


def build_hotel_cards(
    results: list, 
    user_query: str = None, 
    query_requirements: dict = None, 
    requested_location: str = None,
    effective_scores: dict = None,
    **kwargs
) -> list:
    """
    Transforms raw RAG results into structured hotel cards to prevent LLM hallucinations.
    """
    if query_requirements is None:
        query_requirements = {}
        
    if effective_scores is None:
        effective_scores = {}
        
    if user_query is None:
        user_query = kwargs.get("prompt") or kwargs.get("query") or ""
        
    if requested_location is None:
        requested_location = kwargs.get("location") or kwargs.get("location_filter")

    requested_hotel_name = kwargs.get("requested_hotel_name")
    normalized_requirements = normalize_query_requirements(
        query_requirements,
        user_query,
    )
        
    hotel_cards = []
    from cmu_retrieve import get_full_hotel_metadata
    
    for res in results:
        chunk_metadata = res.get("metadata", {})
        text = res.get("text", "")
        
        hotel_name = str(chunk_metadata.get("hotel_name", "UNKNOWN")).strip()
        
        # We don't want to show empty hotels
        if hotel_name == "UNKNOWN" or not hotel_name:
            continue
        if requested_hotel_name and not _hotel_names_match(
            requested_hotel_name, hotel_name
        ):
            continue

        # Merge full profile metadata so phone, amenities, and booking_room_types are never lost
        metadata_location = (
            chunk_metadata.get("location")
            or chunk_metadata.get("city")
            or requested_location
        )
        full_meta = get_full_hotel_metadata(
            hotel_name,
            location=metadata_location,
            hotel_id=chunk_metadata.get("hotel_id"),
        )
        metadata = merge_hotel_metadata(full_meta, chunk_metadata)
        actual_location = metadata.get("location") or metadata.get("city")
        if requested_location and actual_location and not _locations_match(
            requested_location, actual_location
        ):
            continue
        if not metadata.get("location") and actual_location:
            metadata["location"] = actual_location

        # Inject merged metadata back into res so scoring and explanations use it.
        res_for_scoring = res.copy()
        res_for_scoring["metadata"] = metadata

        amenities = metadata.get("amenities", [])
        feature_map = evaluate_hotel_features(metadata)
        wifi = feature_map["wifi"]
        breakfast = feature_map["breakfast"]
        pool = feature_map["pool"]
        wheelchair = feature_map["wheelchair_accessible"]
        parking = feature_map["parking"]
        pet_friendly = feature_map["pet_friendly"]

        # Calculate requirement satisfaction and ranking signals.
        requirement_satisfaction = {}
        matches = 0
        missing = 0
        unknowns = 0
        optional_matches = 0
        room_types = get_combined_room_types(metadata)
        has_single_room = feature_map["single_room"]
        has_double_room = feature_map["double_room"]
        has_suite = feature_map["suite"]

        for req, status in normalized_requirements.items():
            if status == "REQUIRED":
                val = feature_map.get(req, "UNKNOWN")
                if val == "YES":
                    requirement_satisfaction[req] = "MATCH"
                    matches += 1
                elif val == "NO":
                    requirement_satisfaction[req] = "MISSING"
                    missing += 1
                else:
                    requirement_satisfaction[req] = "UNKNOWN"
                    unknowns += 1
            elif status == "OPTIONAL":
                val = feature_map.get(req, "UNKNOWN")
                if val == "YES":
                    requirement_satisfaction[req] = "MATCH"
                    optional_matches += 1
                elif val == "NO":
                    requirement_satisfaction[req] = "MISSING"
                else:
                    requirement_satisfaction[req] = "UNKNOWN"
            else:
                requirement_satisfaction[req] = "NOT_REQUESTED"
                
        # Calculate scores
        travelmind_scoring_dict = calculate_travelmind_score(
            user_query,
            res_for_scoring,
            query_requirements=normalized_requirements,
            requested_location=requested_location,
        )
        true_tm_score = travelmind_scoring_dict.get("travelmind_score", 0)
        
        presented_score = true_tm_score
        rank_adjustment = (
            REQUIRED_MATCH_BOOST * matches
            - REQUIRED_MISSING_PENALTY * missing
            - REQUIRED_UNKNOWN_PENALTY * unknowns
            + OPTIONAL_MATCH_BOOST * optional_matches
        )
        rank_score = true_tm_score + rank_adjustment
        lat = metadata.get("latitude")
        lon = metadata.get("longitude")
        
        if lat and lon:
            map_link_type = "exact"
            map_link = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
        elif hotel_name != "UNKNOWN" and metadata.get("location"):
            map_link_type = "search"
            import urllib.parse
            encoded = urllib.parse.quote(f"{hotel_name} {metadata.get('location')}")
            map_link = f"https://www.google.com/maps/search/?api=1&query={encoded}"
        else:
            map_link_type = "UNKNOWN"
            map_link = "UNKNOWN"
        
        from travelmind_scoring import build_strengths, build_cautions
        card = {
            "hotel_name": hotel_name,
            "hotel_id": str(metadata.get("hotel_id", "UNKNOWN")),
            "location": str(metadata.get("location", "UNKNOWN")),
            "phone": str(metadata.get("phone", "UNKNOWN")),
            "chunk_type": str(res.get("chunk_type", "UNKNOWN")),
            "travelmind_score": presented_score,
            "rank_score": rank_score,
            "missing_signals": travelmind_scoring_dict.get("missing_signals", []),
            "requirement_satisfaction": requirement_satisfaction,
            "similarity_score": str(res.get("vector_score", "UNKNOWN")),
            "map_link_type": map_link_type,
            "map_link": map_link,
            "ratings": {
                "overall": "UNKNOWN",
                "cleanliness": "UNKNOWN", 
                "location": "UNKNOWN",
                "service": "UNKNOWN",
                "rooms": "UNKNOWN"
            },
            "review_count": metadata.get("review_count_total", "UNKNOWN"),
            "hotel_class": metadata.get("hotel_class", "UNKNOWN"),
            "chunk_text": text,
            "amenities": {
                "wifi": wifi,
                "breakfast": breakfast,
                "pool": pool,
                "wheelchair_accessible": wheelchair,
                "parking": parking,
                "pet_friendly": pet_friendly,
                "other": [a for a in amenities if isinstance(a, str) and not any(kw in a.lower() for kw in ["pool", "wifi", "internet", "breakfast", "parking", "wheelchair", "pet"])] [:10]
            },
            "amenities_source": "TripAdvisor amenities / enriched metadata" if amenities else "UNKNOWN",
            "room_info": {
                "single_room": has_single_room, 
                "double_room": has_double_room,
                "suite": has_suite,
                "room_types": room_types,
                "booking_room_types": metadata.get("booking_room_types", []),
                "source": "metadata_and_text"
            },
            "map_link": map_link,
            "map_link_type": map_link_type,
            "strengths": build_strengths(res_for_scoring),
            "cautions": build_cautions(res_for_scoring),
            "missing_information": "Review texts might not cover all specific amenities.",
            "metadata": metadata
        }
        
        card["_matches"] = matches
        card["_unknowns"] = unknowns
        card["_missing"] = missing
        card["_optional_matches"] = optional_matches
        card["_rank_adjustment"] = rank_adjustment

        hotel_cards.append(card)
        
    hotel_cards = sorted(
        hotel_cards,
        key=lambda card: card["rank_score"],
        reverse=True,
    )
        
    return hotel_cards
