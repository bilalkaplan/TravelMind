import math
import re

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
    "otel", "hotel", "hoteli", "oteli", "temiz", "clean", "konum", "location",
    "iyi", "good", "çift", "cift", "kişilik", "kisilik", "double", "bed", "yatak",
    "oda", "room", "arıyorum", "ariyorum", "istiyorum", "looking", "want", "with",
    "and", "for", "the", "bir", "ve", "ile", "için", "icin", "puan", "rating",
    "score", "skor",
}

def normalize_text(text):
    text = str(text).lower()
    text = text.replace("'", " ").replace("’", " ")
    text = re.sub(r"[^a-zA-Z0-9ğüşöçıİĞÜŞÖÇ\s]", " ", text)
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

def score_location_match(query, location):
    tokens = extract_possible_location_tokens(query)
    location_text = normalize_text(location)
    if not tokens:
        return None, "Kullanıcı belirli ülke/şehir/bölge belirtmedi."
    matched_tokens = [token for token in tokens if token in location_text]
    if matched_tokens:
        return 100, f"Konum eşleşti: {', '.join(matched_tokens)}"
    return 0, "Kullanıcının belirttiği konum bu kaydın location alanında bulunamadı."

def score_room_match(query, room_types_list):
    query_text = normalize_text(query)
    
    # Parse what the user asked for
    wants_suite = any(kw in query_text for kw in ["suite", "suit oda", "kral dairesi"])
    wants_double = any(kw in query_text for kw in ["çift kişilik", "cift kisilik", "double bed", "double room", "iki kişilik", "queen bed", "king bed", "full bed"])
    wants_single = any(kw in query_text for kw in ["tek kişilik", "single room", "tek kisilik", "twin"])
    
    if not (wants_suite or wants_double or wants_single):
        return None, "Kullanıcı oda/yatak tipi için özel bir tercih belirtmedi."
        
    if not room_types_list:
        return 0, "Yatak tipi bilgisi veri setinde bulunamadı."
        
    room_text = normalize_text(" ".join(str(r) for r in room_types_list))
    
    if wants_suite:
        if any(kw in room_text for kw in ["suite", "suit", "kral", "king suite"]):
            return 100, "Kullanıcının suit oda isteğiyle tam eşleşti."
        return 0, "Suit oda isteği karşılanmıyor."
        
    if wants_double:
        if any(kw in room_text for kw in ["double", "çift", "twin", "king", "queen", "full"]):
            return 100, "Kullanıcının çift kişilik oda/yatak isteğiyle tam eşleşti."
        return 0, "Çift kişilik oda/yatak isteği karşılanmıyor."
        
    if wants_single:
        if any(kw in room_text for kw in ["single", "tek", "twin"]):
            return 100, "Kullanıcının tek kişilik oda/yatak isteğiyle tam eşleşti."
        return 0, "Tek kişilik oda/yatak isteği karşılanmıyor."
        
    return 50, "Oda tipi bilgisi belirsiz veya kısmen uyumlu."

def score_amenities_match(query, amenities_list):
    if isinstance(amenities_list, dict):
        amenities_list = [k for k, v in amenities_list.items() if str(v).upper() == "YES"]
        
    if not amenities_list or not isinstance(amenities_list, list):
        return None, "Olanak (amenity) bilgisi veri setinde bulunamadı."
        
    query_text = normalize_text(query)
    am_str = " ".join([str(a).lower() for a in amenities_list])
    
    # Parse what the user asked for
    requests = []
    if any(kw in query_text for kw in ["wifi", "wi-fi", "internet"]):
        requests.append(("wifi", ["wifi", "wi-fi", "internet", "wireless"]))
    if any(kw in query_text for kw in ["havuz", "pool", "yüzme"]):
        requests.append(("pool", ["pool", "havuz", "swimming"]))
    if any(kw in query_text for kw in ["kahvaltı", "breakfast", "sabah"]):
        requests.append(("breakfast", ["breakfast", "kahvaltı", "morning meal"]))
    if any(kw in query_text for kw in ["otopark", "park", "parking"]):
        requests.append(("parking", ["parking", "park", "valet"]))
    if any(kw in query_text for kw in ["evcil", "hayvan", "pet", "kedi", "köpek"]):
        requests.append(("pet", ["pet", "dog", "cat", "evcil"]))
        
    if not requests:
        # Fallback to general amenity count
        core = ["wifi", "pool", "breakfast", "parking", "restaurant", "bar", "fitness"]
        matches = sum(1 for c in core if c in am_str)
        return min((matches / 3.0) * 100, 100), f"Tesisin genel olanak zenginliği ({len(amenities_list)} adet) değerlendirildi."
        
    matches = 0
    for req_name, keywords in requests:
        if any(kw in am_str for kw in keywords):
            matches += 1
            
    score = (matches / len(requests)) * 100
    if score == 100:
        return score, "Kullanıcının tüm olanak istekleri karşılandı."
    elif score > 0:
        return score, "Kullanıcının olanak istekleri kısmen karşılandı."
    else:
        return 0, "Kullanıcının olanak istekleri karşılanmıyor."

def score_hotel_class(hotel_class_str):
    value = to_float(hotel_class_str)
    if value is None:
        return None, "Otel sınıfı (yıldız) bilgisi veri setinde bulunamadı."
    value = max(0, min(value, 5))
    score = (value / 5.0) * 100
    return score, f"Otel sınıfı veri setinden geldi: {value} yıldız."

def score_review_count(review_count):
    value = to_float(review_count)
    if value is None or value <= 0:
        return None, "Yorum sayısı bilgisi veri setinde yok."
    score = min(math.log10(value + 1) / 3 * 100, 100)
    return score, f"Yorum sayısı güven sinyali olarak kullanıldı: {int(value)} yorum."

def user_asked_cleanliness(query):
    query_text = normalize_text(query)
    terms = ["temiz", "clean", "hygiene", "cleanliness"]
    return any(term in query_text for term in terms)

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

    location_score, location_reason = score_location_match(query, metadata.get("location", ""))
    
    room_types = metadata.get("room_types", metadata.get("booking_room_types", []))
    if isinstance(room_types, str):
        room_types = [room_types]

    room_score, room_reason = score_room_match(query, room_types)

    hotel_class_score, hotel_class_reason = score_hotel_class(metadata.get("hotel_class", ""))
    
    amenities = metadata.get("amenities", [])
    if isinstance(amenities, str):
        amenities = [amenities]
        
    amenities_score, amenities_reason = score_amenities_match(query, amenities)

    review_count_val = metadata.get("review_count_total", metadata.get("review_count_in_chunk", ""))
    review_count_score, review_count_reason = score_review_count(review_count_val)

    cleanliness_score, cleanliness_reason = score_cleanliness_comment(query, text)
    
    phone_data = metadata.get("phone", None)
    osm_tags = metadata.get("osm_tags", {})
    
    phone_score = 100 if phone_data else 0
    phone_reason = "Telefon verisi mevcut." if phone_data else "Telefon verisi bulunamadı."
    
    map_data_score = 100 if osm_tags and len(osm_tags) > 0 else 0
    map_data_reason = "Harita/OSM verisi mevcut." if map_data_score == 100 else "Harita verisi eksik."
    
    room_types_score = 100 if room_types and len(room_types) > 0 else 0
    room_types_reason = f"{len(room_types)} adet oda tipi verisi mevcut." if room_types_score == 100 else "Oda tipi verisi eksik."

    weights = {
        "location": 20,
        "class": 25,
        "amenities": 25,
        "room_match": 25,
        "review_count": 5,
        "cleanliness": 5,
        "phone": 2,
        "map_data": 3,
        "room_types": 0
    }

    total_weight = 0
    weighted_sum = 0
    components = []
    
    if location_score is not None:
        weighted_sum += location_score * weights["location"]
        total_weight += weights["location"]
        components.append({"name": "Konum Uyumu", "score": location_score, "reason": location_reason, "weight": weights["location"]})
    
    if hotel_class_score is not None:
        weighted_sum += hotel_class_score * weights["class"]
        total_weight += weights["class"]
        components.append({"name": "Otel Sınıfı", "score": hotel_class_score, "reason": hotel_class_reason, "weight": weights["class"]})
        
    if amenities_score is not None:
        weighted_sum += amenities_score * weights["amenities"]
        total_weight += weights["amenities"]
        components.append({"name": "Olanaklar Eşleşmesi", "score": amenities_score, "reason": amenities_reason, "weight": weights["amenities"]})
        
    if room_score is not None:
        weighted_sum += room_score * weights["room_match"]
        total_weight += weights["room_match"]
        components.append({"name": "Oda Tipi Eşleşmesi", "score": room_score, "reason": room_reason, "weight": weights["room_match"]})
        
    if review_count_score is not None:
        weighted_sum += review_count_score * weights["review_count"]
        total_weight += weights["review_count"]
        components.append({"name": "Yorum Hacmi", "score": review_count_score, "reason": review_count_reason, "weight": weights["review_count"]})
        
    if cleanliness_score is not None:
        weighted_sum += cleanliness_score * weights["cleanliness"]
        total_weight += weights["cleanliness"]
        components.append({"name": "Temizlik Hissiyatı", "score": cleanliness_score, "reason": cleanliness_reason, "weight": weights["cleanliness"]})

    weighted_sum += phone_score * weights["phone"]
    total_weight += weights["phone"]
    components.append({"name": "İletişim Verisi", "score": phone_score, "reason": phone_reason, "weight": weights["phone"]})
    
    weighted_sum += map_data_score * weights["map_data"]
    total_weight += weights["map_data"]
    components.append({"name": "Harita Verisi", "score": map_data_score, "reason": map_data_reason, "weight": weights["map_data"]})

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
