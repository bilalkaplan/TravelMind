import math
import re

from cmu_retrieve import search


def extract_ratings(text, rating_name):
    pattern = rf"{rating_name} rating:\s*([0-9.]+)\s*/\s*5"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)

    values = []

    for match in matches:
        try:
            values.append(float(match))
        except ValueError:
            pass

    return values


def average(values):
    if not values:
        return None

    return sum(values) / len(values)


def extract_total_review_count(metadata, text):
    value = metadata.get("review_count_total", "")

    if value:
        return value

    match = re.search(r"Total review count in CMU dataset:\s*([0-9]+)", text)

    if match:
        return match.group(1)

    return ""


def normalize_text(text):
    return str(text).lower()


def keyword_signal(text, keywords):
    lowered = normalize_text(text)
    count = 0

    for keyword in keywords:
        if keyword in lowered:
            count += 1

    return count


def calculate_recommendation_score(result):
    text = result["text"]
    metadata = result["metadata"]

    overall_avg = average(extract_ratings(text, "Overall"))
    cleanliness_avg = average(extract_ratings(text, "Cleanliness"))
    location_avg = average(extract_ratings(text, "Location"))
    service_avg = average(extract_ratings(text, "Service"))
    rooms_avg = average(extract_ratings(text, "Rooms"))

    total_review_count = extract_total_review_count(metadata, text)

    try:
        review_count = int(total_review_count)
    except ValueError:
        review_count = 0

    components = {}
    weights = {}

    retrieval_component = min(result["score"] / 1.3 * 100, 100)
    components["retrieval"] = retrieval_component
    weights["retrieval"] = 20

    if overall_avg is not None:
        components["overall"] = overall_avg / 5 * 100
        weights["overall"] = 20

    if cleanliness_avg is not None:
        components["cleanliness"] = cleanliness_avg / 5 * 100
        weights["cleanliness"] = 20

    if location_avg is not None:
        components["location"] = location_avg / 5 * 100
        weights["location"] = 15

    if service_avg is not None:
        components["service"] = service_avg / 5 * 100
        weights["service"] = 10

    if rooms_avg is not None:
        components["rooms"] = rooms_avg / 5 * 100
        weights["rooms"] = 10

    if review_count > 0:
        review_count_score = min(math.log10(review_count + 1) / 4 * 100, 100)
        components["review_count_confidence"] = review_count_score
        weights["review_count_confidence"] = 5

    weighted_sum = 0
    total_weight = 0

    for key in components:
        weighted_sum += components[key] * weights[key]
        total_weight += weights[key]

    if total_weight == 0:
        final_score = 0
    else:
        final_score = weighted_sum / total_weight

    return {
        "score": round(final_score, 2),
        "overall_avg": overall_avg,
        "cleanliness_avg": cleanliness_avg,
        "location_avg": location_avg,
        "service_avg": service_avg,
        "rooms_avg": rooms_avg,
        "review_count": review_count,
        "components": components,
        "weights": weights,
    }


def format_avg(value):
    if value is None:
        return "Bilgi Mevcut Değil"

    return f"{value:.2f} / 5"


def build_strengths(result, scoring):
    text = result["text"]
    strengths = []

    clean_hits = keyword_signal(
        text,
        [
            "clean",
            "cleanliness",
            "very clean",
            "spotless",
            "pulito",
            "propre",
            "schoon",
        ],
    )

    central_hits = keyword_signal(
        text,
        [
            "central",
            "midtown",
            "location",
            "located",
            "very well located",
            "bien situé",
            "centralissima",
            "metro",
            "subway",
        ],
    )

    service_hits = keyword_signal(
        text,
        [
            "staff",
            "service",
            "friendly",
            "helpful",
            "courteous",
            "pleasant",
            "écoute",
            "gentile",
        ],
    )

    comfort_hits = keyword_signal(
        text,
        [
            "comfortable",
            "comfy",
            "bed",
            "rooms",
            "room",
            "comode",
            "confort",
            "comfortable rooms",
        ],
    )

    if scoring["location_avg"] is not None and scoring["location_avg"] >= 4:
        strengths.append("Konum rating’i güçlü görünüyor.")

    if scoring["cleanliness_avg"] is not None and scoring["cleanliness_avg"] >= 4:
        strengths.append("Temizlik rating’i güçlü görünüyor.")

    if scoring["service_avg"] is not None and scoring["service_avg"] >= 4:
        strengths.append("Servis/personel rating’i olumlu görünüyor.")

    if clean_hits > 0:
        strengths.append("Yorumlarda temizlikle ilgili olumlu ifadeler yakalandı.")

    if central_hits > 0:
        strengths.append("Yorumlarda merkezi/ulaşımı kolay konum vurgusu var.")

    if service_hits > 0:
        strengths.append(
            "Yorumlarda personel veya servisle ilgili olumlu sinyaller var."
        )

    if comfort_hits > 0:
        strengths.append("Yorumlarda oda/konforla ilgili olumlu sinyaller var.")

    if not strengths:
        strengths.append(
            "Bu otel, sorguya en yakın retrieved review chunk üzerinden önerildi."
        )

    return strengths[:4]


def build_cautions(result):
    text = normalize_text(result["text"])
    cautions = []

    caution_keywords = {
        "small room": "Bazı yorumlarda odaların küçük olabileceği belirtilmiş.",
        "extremely small": "Bazı yorumlarda odaların çok küçük olduğu belirtilmiş.",
        "noisy": "Bazı yorumlarda gürültü problemi olabileceği belirtilmiş.",
        "noise": "Bazı yorumlarda gürültüye dair uyarı var.",
        "complaint": "Bazı yorumlarda şikayet içeren ifadeler var.",
        "horrible service": "Bazı yorumlarda servisle ilgili ciddi olumsuz ifade var.",
        "dirty": "Bazı yorumlarda temizlikle ilgili olumsuz ifade var.",
        "not clean": "Bazı yorumlarda temizlikle ilgili olumsuz ifade var.",
        "over-booked": "Bazı yorumlarda rezervasyon/overbooking problemi geçiyor.",
        "mediocre": "Bazı yorumlarda otelin ortalama/vasat olduğu ifade edilmiş.",
    }

    for keyword, warning in caution_keywords.items():
        if keyword in text:
            cautions.append(warning)

    unique_cautions = []

    for item in cautions:
        if item not in unique_cautions:
            unique_cautions.append(item)

    return unique_cautions[:3]


def print_recommendation(index, result):
    metadata = result["metadata"]
    text = result["text"]
    scoring = calculate_recommendation_score(result)

    hotel_name = metadata.get("hotel_name", "")
    location = metadata.get("location", "")
    hotel_class = metadata.get("hotel_class", "")
    source = metadata.get("source", "")
    total_review_count = extract_total_review_count(metadata, text)

    print(f"\n{index}. {hotel_name}")
    print("-" * 90)

    print(f"Konum: {location}")

    if hotel_class:
        print(f"Hotel class: {hotel_class}")

    print(f"CMU toplam review sayısı: {total_review_count}")
    print(f"Kanıt chunk ID: {result['chunk_id']}")
    print(f"Kaynak: {source}")

    print(f"\nTravelMind uygunluk skoru: {scoring['score']} / 100")

    print("\nRating özeti:")
    print(f"- Overall rating ortalaması: {format_avg(scoring['overall_avg'])}")
    print(f"- Cleanliness rating ortalaması: {format_avg(scoring['cleanliness_avg'])}")
    print(f"- Location rating ortalaması: {format_avg(scoring['location_avg'])}")
    print(f"- Service rating ortalaması: {format_avg(scoring['service_avg'])}")
    print(f"- Rooms rating ortalaması: {format_avg(scoring['rooms_avg'])}")

    print("\nNeden önerildi?")
    for strength in build_strengths(result, scoring):
        print(f"- {strength}")

    cautions = build_cautions(result)

    if cautions:
        print("\nDikkat edilmesi gerekenler:")
        for caution in cautions:
            print(f"- {caution}")

    print("\nKısa kanıt metni:")
    print(text[:900])
    print("-" * 90)


def main():
    print("TravelMind RAG - CMU Otel Öneri Modülü")
    print("-" * 50)

    query = input("Otel tercihini ülke/şehir/bölge dahil yaz: ").strip()

    if not query:
        print("Soru boş olamaz.")
        return

    results = search(query, top_k_hotels=5)

    if not results:
        print("Uygun sonuç bulunamadı.")
        return

    print("\nTravelMind otel önerileri:")
    print("=" * 90)

    results = sorted(
        results,
        key=lambda result: calculate_recommendation_score(result)["score"],
        reverse=True,
    )

    for index, result in enumerate(results, start=1):
        print_recommendation(index, result)

    print("\nNot:")
    print(
        "- Bu öneriler CMU TripAdvisor datasetindeki retrieved review chunk'lara dayanır."
    )
    print("- LLM burada skor uydurmaz; ratingler ve yorumlar veri setinden gelir.")
    print(
        "- TravelMind uygunluk skoru, retrieval skoru + rating ortalamaları + review sayısı sinyalinden hesaplanır."
    )


if __name__ == "__main__":
    main()
