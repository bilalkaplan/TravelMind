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


def score_bed_match(query, bed_type, room_type):
    if not user_asked_double_bed(query):
        return None, "Kullanıcı yatak tipi için özel bir tercih belirtmedi."

    bed_text = normalize_text(f"{bed_type} {room_type}")

    if any(keyword in bed_text for keyword in DOUBLE_BED_KEYWORDS):
        return 100, "Çift kişilik yatak isteğiyle uyumlu görünüyor."

    if any(keyword in bed_text for keyword in TWIN_BED_KEYWORDS):
        return (
            30,
            "Kayıtta twin/single yatak geçiyor; çift kişilik yatak isteğiyle tam uyumlu değil.",
        )

    return 50, "Yatak tipi bilgisi belirsiz veya kısmen uyumlu."


def score_hotel_rating(hotel_rating):
    value = to_float(hotel_rating)

    if value is None:
        return None, "Hotel rating bilgisi veri setinde yok."

    value = max(0, min(value, 10))
    score = value * 10

    return score, f"Hotel rating veri setinden geldi: {value} / 10."


def score_room_score(room_score):
    value = to_float(room_score)

    if value is None:
        return None, "Room score bilgisi veri setinde yok."

    value = max(0, min(value, 10))
    score = value * 10

    return score, f"Room score veri setinden geldi: {value} / 10."


def score_review_count(review_count):
    value = to_float(review_count)

    if value is None or value <= 0:
        return None, "Review count bilgisi veri setinde yok."

    score = min(math.log10(value + 1) / 4 * 100, 100)

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
    metadata = result["metadata"]
    text = result["text"]

    location_score, location_reason = score_location_match(
        query, metadata.get("location", "")
    )

    bed_score, bed_reason = score_bed_match(
        query, metadata.get("bed_type", ""), metadata.get("room_type", "")
    )

    hotel_rating_score, hotel_rating_reason = score_hotel_rating(
        metadata.get("hotel_rating", "")
    )

    room_score, room_score_reason = score_room_score(metadata.get("room_score", ""))

    review_count_score, review_count_reason = score_review_count(
        metadata.get("review_count", "")
    )

    cleanliness_score, cleanliness_reason = score_cleanliness_comment(query, text)

    raw_components = [
        ("location_match", location_score, 25, location_reason),
        ("bed_match", bed_score, 20, bed_reason),
        ("hotel_rating", hotel_rating_score, 20, hotel_rating_reason),
        ("room_score", room_score, 15, room_score_reason),
        ("review_count", review_count_score, 10, review_count_reason),
        ("cleanliness_comment", cleanliness_score, 10, cleanliness_reason),
    ]

    total_weight = 0
    weighted_sum = 0
    components = []

    for name, score, weight, reason in raw_components:
        if score is None:
            components.append(
                {"name": name, "score": None, "weight": weight, "reason": reason}
            )
            continue

        weighted_sum += score * weight
        total_weight += weight

        components.append(
            {"name": name, "score": round(score, 2), "weight": weight, "reason": reason}
        )

    if total_weight == 0:
        final_score = 0
    else:
        final_score = weighted_sum / total_weight

    return {"travelmind_score": round(final_score, 2), "components": components}
