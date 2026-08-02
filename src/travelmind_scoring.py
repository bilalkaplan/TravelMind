import math
import re

DOUBLE_BED_KEYWORDS = ["double", "queen", "king", "full"]

TWIN_BED_KEYWORDS = ["twin", "single"]

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
    "otel",
    "hotel",
    "hoteli",
    "oteli",
    "temiz",
    "clean",
    "konum",
    "location",
    "iyi",
    "good",
    "çift",
    "cift",
    "kişilik",
    "kisilik",
    "double",
    "bed",
    "yatak",
    "oda",
    "room",
    "arıyorum",
    "ariyorum",
    "istiyorum",
    "looking",
    "want",
    "with",
    "and",
    "for",
    "the",
    "bir",
    "ve",
    "ile",
    "için",
    "icin",
    "puan",
    "rating",
    "score",
    "skor",
}


def normalize_text(text):
    text = str(text).lower()
    text = text.replace("'", " ")
    text = text.replace("’", " ")
    text = re.sub(r"[^a-zA-Z0-9ğüşöçıİĞÜŞÖÇ\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def to_float(value):
    try:
        if value is None:
            return None

        value = str(value).strip()

        if value == "" or value.lower() == "nan":
            return None

        return float(value)

    except ValueError:
        return None


def user_asked_double_bed(query):
    query_text = normalize_text(query)

    double_terms = [
        "çift kişilik",
        "cift kisilik",
        "double bed",
        "double room",
        "queen bed",
        "king bed",
        "full bed",
    ]

    return any(term in query_text for term in double_terms)


def user_asked_cleanliness(query):
    query_text = normalize_text(query)

    terms = ["temiz", "clean", "hygiene", "cleanliness"]

    return any(term in query_text for term in terms)


def extract_possible_location_tokens(query):
    query_text = normalize_text(query)
    tokens = query_text.split()

    possible_tokens = []

    for token in tokens:
        if len(token) < 3:
            continue

        if token in LOCATION_WORDS_TO_IGNORE:
            continue

        possible_tokens.append(token)

    return possible_tokens


def score_location_match(query, location):
    tokens = extract_possible_location_tokens(query)
    location_text = normalize_text(location)

    if not tokens:
        return None, "Kullanıcı belirli ülke/şehir/bölge belirtmedi."

    matched_tokens = []

    for token in tokens:
        if token in location_text:
            matched_tokens.append(token)

    if matched_tokens:
        return 100, f"Konum eşleşti: {', '.join(matched_tokens)}"

    return 0, "Kullanıcının belirttiği konum bu kaydın location alanında bulunamadı."


def score_bed_match(query, room_types_list):
    if not user_asked_double_bed(query):
        return None, "Kullanıcı yatak tipi için özel bir tercih belirtmedi."

    if not room_types_list:
        return 50, "Yatak tipi bilgisi veri setinde bulunamadı."
        
    bed_text = normalize_text(" ".join(str(r) for r in room_types_list))

    if any(keyword in bed_text for keyword in DOUBLE_BED_KEYWORDS):
        return 100, "Çift kişilik yatak isteğiyle uyumlu görünüyor."

    if any(keyword in bed_text for keyword in TWIN_BED_KEYWORDS):
        return (
            30,
            "Kayıtta twin/single yatak geçiyor; çift kişilik yatak isteğiyle tam uyumlu değil.",
        )

    return 50, "Yatak tipi bilgisi belirsiz veya kısmen uyumlu."


def score_hotel_class(hotel_class_str):
    value = to_float(hotel_class_str)

    if value is None:
        return None, "Otel sınıfı (yıldız) bilgisi veri setinde bulunamadı."

    value = max(0, min(value, 5))
    score = (value / 5.0) * 100

    return score, f"Otel sınıfı veri setinden geldi: {value} yıldız."


def score_amenities(amenities_list):
    if not isinstance(amenities_list, list) or not amenities_list:
        return None, "Olanak (amenity) bilgisi veri setinde bulunamadı."
        
    core_amenities = ["wifi", "wi-fi", "pool", "havuz", "breakfast", "kahvaltı", "parking", "restaurant", "bar"]
    am_str = " ".join([str(a).lower() for a in amenities_list])
    
    matches = 0
    for kw in core_amenities:
        if kw in am_str:
            matches += 1
            
    # Max score if they have at least 3 core amenities
    score = min((matches / 3.0) * 100, 100)
    
    return score, f"Tesisin sunduğu zenginleştirilmiş olanaklar ({len(amenities_list)} adet) üzerinden değerlendirildi."


def score_review_count(review_count):
    value = to_float(review_count)

    if value is None or value <= 0:
        return None, "Yorum sayısı bilgisi veri setinde yok."

    # Using log scale for review count
    score = min(math.log10(value + 1) / 3 * 100, 100)

    return score, f"Yorum sayısı güven sinyali olarak kullanıldı: {int(value)} yorum."


def score_cleanliness_comment(query, text):
    if not user_asked_cleanliness(query):
        return None, "Kullanıcı temizlik için özel bir tercih belirtmedi."

    comment_text = normalize_text(text)

    has_positive = any(word in comment_text for word in CLEANLINESS_POSITIVE_KEYWORDS)
    has_negative = any(word in comment_text for word in CLEANLINESS_NEGATIVE_KEYWORDS)

    if has_negative:
        return 20, "Yorum/metin içinde temizlik açısından olumsuz ifade bulundu."

    if has_positive:
        return 100, "Yorum/metin içinde temizlik açısından olumlu ifade bulundu."

    return 50, "Temizlik hakkında açık bir ifade bulunamadı."


def calculate_travelmind_score(query, result):
    metadata = result.get("metadata", {})
    text = result.get("text", "")

    location_score, location_reason = score_location_match(
        query, metadata.get("location", "")
    )
    
    room_types = metadata.get("room_types", metadata.get("booking_room_types", []))
    if isinstance(room_types, str):
        room_types = [room_types]

    bed_score, bed_reason = score_bed_match(
        query, room_types
    )

    hotel_class_score, hotel_class_reason = score_hotel_class(
        metadata.get("hotel_class", "")
    )
    
    amenities = metadata.get("amenities", [])
    if isinstance(amenities, str):
        amenities = [amenities]
        
    amenities_score, amenities_reason = score_amenities(amenities)

    review_count_val = metadata.get("review_count_total", metadata.get("review_count_in_chunk", ""))
    review_count_score, review_count_reason = score_review_count(review_count_val)

    cleanliness_score, cleanliness_reason = score_cleanliness_comment(query, text)
    
    # NEW SCORING: Enriched Data (Phone, Map Data, Room Types)
    phone_data = metadata.get("phone", None)
    osm_tags = metadata.get("osm_tags", {})
    
    # Calculate phone score
    phone_score = 100 if phone_data else 0
    phone_reason = "Telefon verisi mevcut." if phone_data else "Telefon verisi bulunamadı."
    
    # Calculate map data score
    map_data_score = 100 if osm_tags and len(osm_tags) > 0 else 0
    map_data_reason = "Harita/OSM verisi mevcut." if map_data_score == 100 else "Harita verisi eksik."
    
    # Calculate room type richness score
    room_types_score = 100 if room_types and len(room_types) > 0 else 0
    room_types_reason = f"{len(room_types)} adet oda tipi verisi mevcut." if room_types_score == 100 else "Oda tipi verisi eksik."

    weights = {
        "location": 20,
        "class": 25,
        "amenities": 15,
        "bed": 10,
        "review_count": 5,
        "cleanliness": 5,
        "phone": 5,
        "map_data": 10,
        "room_types": 5
    }

    total_weight = 0
    weighted_sum = 0
    components = []
    
    # Location
    if location_score is not None:
        weighted_sum += location_score * weights["location"]
        total_weight += weights["location"]
        components.append({"name": "Konum Uyumu", "score": location_score, "reason": location_reason, "weight": weights["location"]})
    
    # Hotel Class
    if hotel_class_score is not None:
        weighted_sum += hotel_class_score * weights["class"]
        total_weight += weights["class"]
        components.append({"name": "Otel Sınıfı", "score": hotel_class_score, "reason": hotel_class_reason, "weight": weights["class"]})
        
    # Amenities
    if amenities_score is not None:
        weighted_sum += amenities_score * weights["amenities"]
        total_weight += weights["amenities"]
        components.append({"name": "Olanaklar", "score": amenities_score, "reason": amenities_reason, "weight": weights["amenities"]})
        
    # Bed
    if bed_score is not None:
        weighted_sum += bed_score * weights["bed"]
        total_weight += weights["bed"]
        components.append({"name": "Yatak Tipi", "score": bed_score, "reason": bed_reason, "weight": weights["bed"]})
        
    # Review Count
    if review_count_score is not None:
        weighted_sum += review_count_score * weights["review_count"]
        total_weight += weights["review_count"]
        components.append({"name": "Yorum Hacmi", "score": review_count_score, "reason": review_count_reason, "weight": weights["review_count"]})
        
    # Cleanliness
    if cleanliness_score is not None:
        weighted_sum += cleanliness_score * weights["cleanliness"]
        total_weight += weights["cleanliness"]
        components.append({"name": "Temizlik Hissiyatı", "score": cleanliness_score, "reason": cleanliness_reason, "weight": weights["cleanliness"]})

    # Add enriched data points
    weighted_sum += phone_score * weights["phone"]
    total_weight += weights["phone"]
    components.append({"name": "İletişim (Telefon) Verisi", "score": phone_score, "reason": phone_reason, "weight": weights["phone"]})
    
    weighted_sum += map_data_score * weights["map_data"]
    total_weight += weights["map_data"]
    components.append({"name": "Harita & Konum Verisi", "score": map_data_score, "reason": map_data_reason, "weight": weights["map_data"]})
    
    weighted_sum += room_types_score * weights["room_types"]
    total_weight += weights["room_types"]
    components.append({"name": "Oda Tipleri (Zenginlik)", "score": room_types_score, "reason": room_types_reason, "weight": weights["room_types"]})

    if total_weight == 0:
        final_score = 0
    else:
        final_score = weighted_sum / total_weight

    return {
        "travelmind_score": final_score,
        "components": components
    }

def build_strengths(result):
    text = str(result.get("text", "")).lower()
    metadata = result.get("metadata", {})
    strengths = []
    
    # Metadata-driven strengths
    hotel_class_str = metadata.get("hotel_class", "")
    if "4." in hotel_class_str or "5." in hotel_class_str:
        strengths.append("Yüksek yıldızlı, premium bir otel (Premium classification).")
        
    amenities = metadata.get("amenities", [])
    if isinstance(amenities, dict):
        amenities = [k for k, v in amenities.items() if v == "YES"]
    if len(amenities) >= 4:
        strengths.append("Otel olanakları (amenities) açısından oldukça zengin.")
        
    try:
        review_count = int(metadata.get("review_count_total", 0))
        if review_count > 500:
            strengths.append(f"Ziyaretçiler tarafından çok fazla ({review_count}) değerlendirilmiş, güvenilir.")
    except (ValueError, TypeError):
        pass

    # Keyword-driven strengths from text
    if any(k in text for k in CLEANLINESS_POSITIVE_KEYWORDS):
        strengths.append("Yorumlarda temizlikle ilgili belirgin olumlu ifadeler var.")
        
    if any(k in text for k in ["central", "midtown", "located", "metro", "subway", "merkezi"]):
        strengths.append("Yorumlarda merkezi ve ulaşıma elverişli konum vurgusu var.")
        
    if any(k in text for k in ["staff", "friendly", "helpful", "service", "personel"]):
        strengths.append("Yorumlarda personel veya servisle ilgili olumlu sinyaller var.")
        
    if not strengths:
        strengths.append("Bu otel, sorgunuza en uygun TravelMind uygunluk skorunu aldığı için önerildi.")
        
    return strengths[:4]


def build_cautions(result):
    text = str(result.get("text", "")).lower()
    metadata = result.get("metadata", {})
    cautions = []
    
    hotel_class_str = metadata.get("hotel_class", "")
    if "1." in hotel_class_str or "2." in hotel_class_str:
        cautions.append("Otelin sınıfı/yıldız değeri düşük. Lüks bir deneyim sunmayabilir.")

    if any(k in text for k in CLEANLINESS_NEGATIVE_KEYWORDS):
        cautions.append("Bazı yorumlarda temizlikle ilgili olumsuz ifadeler veya şikayetler yer almış.")
        
    if any(k in text for k in ["small room", "extremely small", "tiny"]):
        cautions.append("Bazı yorumlarda odaların çok küçük olabileceği belirtilmiş.")
        
    if any(k in text for k in ["noisy", "noise", "loud", "gürültü"]):
        cautions.append("Bazı yorumlarda dışarıdan veya içeriden gürültü problemi olabileceği belirtilmiş.")
        
    if any(k in text for k in ["over-booked", "overbooked"]):
        cautions.append("Bazı yorumlarda rezervasyon (overbooking) problemi geçtiği görülmüş.")
        
    if any(k in text for k in ["rude", "unhelpful", "horrible service", "kaba"]):
        cautions.append("Bazı yorumlarda personelin tavrıyla ilgili olumsuz ifadeler yer almış.")
        
    return cautions[:3]
