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
    "breakfast": (r"\bbreakfast\b", r"\bkahvalt[ıi]\b"),
    "single_room": (
        r"\bsingle(?:\s+(?:room|bed))?\b",
        r"\btek\s+ki[sş]ilik\b",
    ),
    "double_room": (
        r"\bdouble(?:\s+(?:room|bed))?\b",
        r"\b(?:king|queen|twin|full)\s+(?:room|bed)\b",
        r"\b(?:çift|cift|iki)\s+ki[sş]ilik\b",
    ),
    "suite": (r"\bsuite\b", r"\bsuit(?:\s+oda)?\b", r"\bkral\s+dairesi\b"),
    "pool": (r"\bpool\b", r"\bswimming\s+pool\b", r"\bhavuz\b"),
    "wifi": (r"\bwi[ -]?fi\b", r"\bwireless\b", r"\binternet\b"),
    "wheelchair_accessible": (
        r"\bwheelchair(?:\s+accessible)?\b",
        r"\baccessible\b",
        r"\bengelli\b",
    ),
    "parking": (r"\bparking\b", r"\bvalet\b", r"\bgarage\b", r"\botopark\b"),
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

def score_location_match(query, location, requested_location=None):
    if requested_location:
        location_text = normalize_text(location)
        requested_text = normalize_text(requested_location)
        if not location_text:
            return None, "Otelin konum bilgisi veri setinde bulunamadı."
        requested_city = requested_text.split(",", 1)[0].strip()
        if requested_text in location_text or (
            requested_city and requested_city in location_text
        ):
            return 100, f"Konum eşleşti: {requested_location}"
        return 0, f"Otel istenen konumda değil: {requested_location}"

    tokens = extract_possible_location_tokens(query)
    location_text = normalize_text(location)
    if not tokens:
        return None, "Kullanıcı belirli ülke/şehir/bölge belirtmedi."
    matched_tokens = [token for token in tokens if token in location_text]
    if matched_tokens:
        return 100, f"Konum eşleşti: {', '.join(matched_tokens)}"
    return 0, "Kullanıcının belirttiği konum bu kaydın location alanında bulunamadı."

def score_room_match(query, room_types_list, query_requirements=None):
    query_text = normalize_text(query)

    if query_requirements is None:
        requested_types = []
        if any(kw in query_text for kw in ["suite", "suit oda", "kral dairesi"]):
            requested_types.append("suite")
        if any(
            kw in query_text
            for kw in [
                "çift kişilik",
                "cift kisilik",
                "double bed",
                "double room",
                "iki kişilik",
                "queen bed",
                "king bed",
                "full bed",
            ]
        ):
            requested_types.append("double_room")
        if any(
            kw in query_text
            for kw in ["tek kişilik", "single room", "tek kisilik"]
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
        return None, "Kullanıcı oda/yatak tipi için özel bir tercih belirtmedi."

    if not room_types_list:
        return None, "Yatak tipi bilgisi veri setinde bulunamadı."

    room_text = normalize_text(" ".join(str(r) for r in room_types_list))
    matchers = {
        "suite": ("suite", "suit", "kral", "presidential"),
        "double_room": ("double", "çift", "twin", "king", "queen", "full"),
        "single_room": ("single", "tek"),
    }
    matches = sum(
        any(keyword in room_text for keyword in matchers[room_type])
        for room_type in requested_types
    )
    score = matches / len(requested_types) * 100
    if matches == len(requested_types):
        return score, "Kullanıcının tüm oda tipi tercihleri kayıtlarla eşleşti."
    if matches:
        return score, "Kullanıcının oda tipi tercihleri kısmen eşleşti."
    return 0, "İstenen oda tipleri otelin kayıtlı oda listesinde bulunmuyor."


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
        return None, "Olanak (amenity) bilgisi veri setinde bulunamadı."

    query_text = normalize_text(query)
    am_str = normalize_text(" ".join(str(amenity) for amenity in amenities_list))

    requests = []
    amenity_keywords = {
        "wifi": ("wifi", "wi fi", "internet", "wireless"),
        "pool": ("pool", "havuz", "swimming"),
        "breakfast": ("breakfast", "kahvaltı", "morning meal"),
        "parking": ("parking", "park", "valet", "garage", "otopark"),
        "wheelchair_accessible": (
            "wheelchair",
            "accessible",
            "handicap",
            "disabled",
            "engelli",
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
            return None, "Kullanıcı belirli bir olanak tercihi belirtmedi."
        # Fallback to general amenity count
        core = ["wifi", "pool", "breakfast", "parking", "restaurant", "bar", "fitness"]
        matches = sum(1 for c in core if c in am_str)
        return min((matches / 3.0) * 100, 100), f"Tesisin genel olanak zenginliği ({len(amenities_list)} adet) değerlendirildi."
        
    matches = 0
    for _request_name, keywords in requests:
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

def normalize_review_rating(mean_rating):
    """Map an available mean on the 1-5 scale to a 0-100 score."""

    value = to_float(mean_rating)
    if value is None or not 1 <= value <= 5:
        return None
    return (value - 1) / 4 * 100


def score_review_rating(mean_rating, label):
    score = normalize_review_rating(mean_rating)
    if score is None:
        return None, f"{label} puanı veri setinde bulunamadı."
    return score, f"Ortalama {label.lower()} puanı: {float(mean_rating):.2f} / 5."


def score_review_volume(review_count, maximum_review_count=None):
    value = to_float(review_count)
    maximum = to_float(
        MAX_REVIEW_COUNT if maximum_review_count is None else maximum_review_count
    )
    if value is None or value <= 0:
        return None, "Yorum sayısı bilgisi veri setinde yok."
    if maximum is None or maximum <= 0:
        maximum = value
    score = min(math.log1p(value) / math.log1p(maximum) * 100, 100)
    return (
        score,
        "Yorum hacmi veri setindeki maksimuma göre normalize edildi: "
        f"{int(value)} yorum.",
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
        review_stats.get("overall"), "Genel yorum"
    )
    service_score, service_reason = score_review_rating(
        review_stats.get("service"), "Servis"
    )
    review_rooms_score, review_rooms_reason = score_review_rating(
        review_stats.get("rooms"), "Oda"
    )
    review_cleanliness_score, review_cleanliness_reason = score_review_rating(
        review_stats.get("cleanliness"), "Temizlik"
    )
    review_volume_score, review_volume_reason = score_review_volume(
        review_stats.get("review_count")
    )

    total_weight = 0
    weighted_sum = 0
    components = []

    component_values = (
        ("location_match", "Konum Uyumu", location_score, location_reason),
        ("hotel_class", "Otel Sınıfı", hotel_class_score, hotel_class_reason),
        ("amenities_match", "Olanaklar Eşleşmesi", amenities_score, amenities_reason),
        ("room_type_match", "Oda Tipi Eşleşmesi", room_score, room_reason),
        ("review_overall", "Genel Yorum Puanı", overall_score, overall_reason),
        ("review_service", "Servis Yorum Puanı", service_score, service_reason),
        ("review_rooms", "Oda Yorum Puanı", review_rooms_score, review_rooms_reason),
        (
            "review_cleanliness",
            "Temizlik Yorum Puanı",
            review_cleanliness_score,
            review_cleanliness_reason,
        ),
        ("review_volume", "Yorum Hacmi", review_volume_score, review_volume_reason),
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
