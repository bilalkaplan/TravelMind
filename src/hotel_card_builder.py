
from travelmind_scoring import calculate_travelmind_score

def extract_boolean_feature(amenities: list, keywords: list) -> str:
    """
    Checks if any of the keywords exist in the amenities list.
    Returns 'YES' if found, 'NO' if amenities list exists but not found, 
    and 'UNKNOWN' if amenities list is completely missing or empty.
    """
    if not amenities:
        return "UNKNOWN"
    
    am_str = " ".join([str(a).lower() for a in amenities])
    for kw in keywords:
        if kw in am_str:
            return "YES"
    return "NO"


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
        
    hotel_cards = []
    from cmu_retrieve import get_full_hotel_metadata
    
    for res in results:
        chunk_metadata = res.get("metadata", {})
        text = res.get("text", "")
        
        hotel_name = str(chunk_metadata.get("hotel_name", "UNKNOWN")).strip()
        
        # We don't want to show empty hotels
        if hotel_name == "UNKNOWN" or not hotel_name:
            continue

        # Merge full profile metadata so phone, amenities, and booking_room_types are never lost
        full_meta = get_full_hotel_metadata(hotel_name)
        metadata = {**full_meta, **chunk_metadata}
        
        # Inject merged metadata back into res so calculate_travelmind_score uses it!
        res_for_scoring = res.copy()
        res_for_scoring["metadata"] = metadata
        
        # Get the real clamped score from the backend
        score_data = calculate_travelmind_score(user_query, res_for_scoring)
        base_score = score_data.get("travelmind_score", 0)
        
        amenities = metadata.get("amenities", [])
        
        # Safe extraction of amenities
        wifi = extract_boolean_feature(amenities, ["wifi", "wi-fi", "internet", "wireless"])
        breakfast = extract_boolean_feature(amenities, ["breakfast", "kahvaltı", "morning meal"])
        pool = extract_boolean_feature(amenities, ["pool", "havuz", "swimming"])
        wheelchair = extract_boolean_feature(amenities, ["wheelchair", "accessible", "handicap", "disabled", "engelli"])
        parking = extract_boolean_feature(amenities, ["parking", "park", "valet"])
        pet_friendly = extract_boolean_feature(amenities, ["pet", "dog", "cat", "evcil"])
        

        
        # Calculate requirement satisfaction and penalty
        requirement_satisfaction = {}
        penalty = 0
        matches = 0
        missing = 0
        unknowns = 0
        
        normalized_text = str(text).lower()
        room_types = metadata.get("room_types", metadata.get("booking_room_types", []))
        if isinstance(room_types, str):
            room_types = [room_types]
        
        room_types_lower = [str(r).lower() for r in room_types]
        
        has_single_room = "UNKNOWN"
        if any("single" in r or "tek" in r for r in room_types_lower):
            has_single_room = "YES"
        elif "single room" in normalized_text or "tek kişilik" in normalized_text:
            has_single_room = "YES"
            
        has_double_room = "UNKNOWN"
        if any("double" in r or "çift" in r or "twin" in r or "king" in r or "queen" in r for r in room_types_lower):
            has_double_room = "YES"
        elif "double room" in normalized_text or "çift kişilik" in normalized_text or "iki kişilik" in normalized_text:
            has_double_room = "YES"
        
        has_suite = "UNKNOWN"
        if any("suite" in r or "suit" in r or "kral" in r or "king suite" in r for r in room_types_lower):
            has_suite = "YES"
        elif "suite" in normalized_text or "suit oda" in normalized_text or "kral dairesi" in normalized_text:
            has_suite = "YES"

        # Map fields to extracted values
        feature_map = {
            "breakfast": breakfast,
            "pool": pool,
            "wifi": wifi,
            "wheelchair_accessible": wheelchair,
            "parking": parking,
            "single_room": has_single_room,
            "double_room": has_double_room,
            "suite": has_suite
        }
        
        for req, status in query_requirements.items():
            if status == "REQUIRED":
                val = feature_map.get(req, "UNKNOWN")
                if val == "YES":
                    requirement_satisfaction[req] = "MATCH"
                    matches += 1
                elif val == "NO":
                    requirement_satisfaction[req] = "MISSING"
                    missing += 1
                    penalty += 25
                else:
                    requirement_satisfaction[req] = "UNKNOWN"
                    unknowns += 1
                    penalty += 10
            elif status == "OPTIONAL":
                val = feature_map.get(req, "UNKNOWN")
                if val == "YES":
                    requirement_satisfaction[req] = "MATCH"
                elif val == "NO":
                    requirement_satisfaction[req] = "MISSING"
                else:
                    requirement_satisfaction[req] = "UNKNOWN"
            else:
                requirement_satisfaction[req] = "NOT_REQUESTED"
                
        # Calculate scores
        travelmind_scoring_dict = calculate_travelmind_score(user_query, res_for_scoring)
        true_tm_score = travelmind_scoring_dict.get("travelmind_score", base_score)
        
        presented_score = true_tm_score
        rank_score = true_tm_score + (25 * matches) - (35 * missing) - (10 * unknowns)
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
                "other": [a for a in amenities if isinstance(a, str)][:10]
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
            "strengths": build_strengths(res),
            "cautions": build_cautions(res),
            "missing_information": "Review texts might not cover all specific amenities.",
            "metadata": metadata
        }
        
        card["_matches"] = matches
        card["_unknowns"] = unknowns
        card["_missing"] = missing

        hotel_cards.append(card)
        
    hotel_cards = sorted(
        hotel_cards,
        key=lambda x: (x["_matches"], -x["_unknowns"], -x["_missing"], x["travelmind_score"]),
        reverse=True
    )
        
    return hotel_cards
