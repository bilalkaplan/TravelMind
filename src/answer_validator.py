import re

def detect_internal_analysis(answer: str) -> bool:
    forbidden = [
        "let me check", "düşünüyor...", "let's check",
        "i need to confirm", "okay, the user",
        "looking at the provided", "provided hotel cards",
        "after checking", "in conclusion, after checking",
        "<think>", "</think>",
        "okay, the user asked", "let me start", "i should mention",
        "i should also", "the user is asking", "i should make sure",
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

def detect_price_claims(answer: str) -> bool:
    price_keywords = [
        r"\$", "dollar", "euro", "£", "€", "₺", "tl",
        "per night", "nightly rate", "gecelik", "fiyatı",
        "rezervasyon yapabilirsiniz", "book now", "booking.com", "expedia",
        "reserve now", "available tonight", "müsait oda var"
    ]
    refusal_keywords = [
        "cannot provide live prices", "fiyat verisi güvenilir değil",
        "canlı fiyat", "fiyat bilgisi sunmaz",
        "does not provide live price", "not available",
        "does not provide", "sunmaz", "kapsamı dışında",
        "cannot confirm availability", "cannot confirm single-room availability"
    ]
    
    sentences = re.split(r'(?<=[.!?]) +', answer)
    for sentence in sentences:
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
            if has_number or "book now" in lower_sent or "rezervasyon" in lower_sent or "reserve now" in lower_sent or "booking.com" in lower_sent or "available tonight" in lower_sent:
                return True
    return False

def is_negative_context(sentence: str) -> bool:
    negatives = [
        r"\bnot\b", r"\bno\b", r"\bdoesn't\b", r"\bdon't\b", r"\bisn't\b", r"\baren't\b",
        r"\byok\b", r"\bbulunmuyor\b", r"mevcut değil", r"\bolmadığı\b", r"\bsunmuyor\b", r"\byoktur\b", r"\bbulunmamaktadır\b"
    ]
    sent_lower = sentence.lower()
    return any(re.search(neg, sent_lower) for neg in negatives)

def detect_room_guarantee(answer: str, hotel_cards: list) -> bool:
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
    sentences = re.split(r'(?<=[.!?]) +', answer)
    for sentence in sentences:
        sent_lower = sentence.lower()
        if any(safe in sent_lower for safe in safe_phrases):
            continue
        if is_negative_context(sent_lower):
            continue
            
        if any(phrase in sent_lower for phrase in live_availability_phrases):
            return True
            
        for phrase in single_room_phrases:
            if phrase in sent_lower:
                supported = False
                for card in hotel_cards:
                    r_info = card.get("room_info", {})
                    if r_info.get("single_room") == "YES":
                        supported = True
                        break
                    if any("single" in str(bt).lower() for bt in r_info.get("booking_room_types", [])):
                        supported = True
                        break
                if not supported:
                    return True
    return False

def detect_amenity_false_claim(answer: str, hotel_cards: list) -> bool:
    breakfast_claims = [
        "has breakfast", "breakfast included", "breakfast is available",
        "breakfast service", "kahvaltısı var", "kahvaltı sunuyor",
        "kahvaltı hizmeti"
    ]
    sentences = re.split(r'(?<=[.!?]) +', answer)
    for sentence in sentences:
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
                    return True
    return False

def detect_map_link_hallucination(answer: str, hotel_cards: list) -> bool:
    if "google.com/maps" in answer.lower():
        any_map = any(
            card.get("map_link_type") not in ("UNKNOWN", None)
            for card in hotel_cards
        )
        if not any_map:
            return True
    return False

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
            
    if intent in ("follow_up", "specific_hotel_info"):
        if target_language.lower() in ["turkish", "tr"]:
            return "İstediğiniz detayı doğrudan veri setimizde net olarak doğrulayamadım. Size başka nasıl yardımcı olabilirim?"
        else:
            return "I couldn't clearly verify that specific detail in our dataset. How else can I help you?"
            
    # Default fallback
    if target_language.lower() in ["turkish", "tr"]:
        return "TravelMind canlı fiyat veya rezervasyon bilgisi göstermez. Lütfen sadece konum belirterek arama yapınız."
    else:
        return "TravelMind does not provide live pricing or availability, only hotel recommendations based on historical reviews. Please try your search again using just a location."

def validate_answer(answer: str, hotel_cards: list, intent: str, requested_location: str, target_language: str) -> dict:
    issues = []
    hotel_cards = hotel_cards or []
    
    if detect_internal_analysis(answer):
        issues.append({"type": "internal_analysis_leak", "detail": "Model leaked internal thoughts."})
        
    if detect_placeholders(answer):
        issues.append({"type": "placeholder_leak", "detail": "Model copied template placeholders."})
        
    if detect_score_overflow(answer):
        issues.append({"type": "score_overflow", "detail": "Model hallucinated a score > 100."})
        
    if detect_price_claims(answer):
        issues.append({"type": "price_booking_leak", "detail": "Model tried to show price or booking links."})
    
    # Hotel name check
    allowed_hotels = [card.get("hotel_name") for card in hotel_cards if card.get("hotel_name") != "UNKNOWN"]
    potential_hotels = re.findall(r'([A-Z][a-zA-Z\s]+(?:Hotel|Resort|Inn|Suites|Motel))', answer)
    allowed_lower = [name.lower() for name in allowed_hotels]
    
    generic_terms = [
        "this hotel", "the hotel", "boutique hotel", "luxury hotel", "great hotel", 
        "excellent hotel", "beautiful hotel", "nice hotel", "good hotel", "best hotel", 
        "grand hotel", "spa resort", "family resort", "business hotel", "harika hotel", 
        "mükemmel hotel", "güzel hotel", "iyi hotel", "a hotel", "an hotel"
    ]
    
    for ph in potential_hotels:
        ph_clean = ph.strip().lower()
        # Ignore if it's just a generic descriptive term ending with "hotel"
        if any(ph_clean.endswith(g) for g in generic_terms) or ph_clean in generic_terms:
            continue
            
        if ph_clean not in allowed_lower:
            if not any(ph_clean in al for al in allowed_lower) and not any(al in ph_clean for al in allowed_lower):
                issues.append({"type": "unknown_hotel_name", "detail": f"Mentioned unknown hotel: {ph.strip()}"})

    # Room guarantee check
    if detect_room_guarantee(answer, hotel_cards):
        issues.append({"type": "single_room_hallucination", "detail": "Claimed single room availability but data is UNKNOWN."})
    
    # Amenity false claim check
    if detect_amenity_false_claim(answer, hotel_cards):
        issues.append({"type": "breakfast_hallucination", "detail": "Claimed breakfast but no hotel has it confirmed."})
    
    # Map link hallucination
    if detect_map_link_hallucination(answer, hotel_cards):
        issues.append({"type": "map_link_hallucination", "detail": "Generated map link when all map link types are UNKNOWN."})

    passed = len(issues) == 0
    
    # Generate fallback if needed
    sanitized = answer
    if not passed:
        sanitized = build_safe_fallback_answer(target_language, intent, hotel_cards)
        
    return {
        "passed": passed,
        "issues": [issue["type"] for issue in issues],
        "sanitized_answer": sanitized,
        "needs_fallback": not passed
    }
