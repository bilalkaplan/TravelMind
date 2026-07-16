from retrieve import search
from travelmind_scoring import calculate_travelmind_score


def main():
    print("TravelMind RAG - Controlled Hotel Recommendation")
    print("-" * 55)

    query = input("Otel tercihini ülke/şehir/bölge dahil yaz: ").strip()

    if not query:
        print("Soru boş olamaz.")
        return

    results = search(query, top_k=10)

    scored_results = []

    for result in results:
        scoring = calculate_travelmind_score(query, result)

        scored_results.append({"retrieval_result": result, "scoring": scoring})

    scored_results = sorted(
        scored_results,
        key=lambda item: item["scoring"]["travelmind_score"],
        reverse=True,
    )

    print("\nTravelMind kontrollü öneri sonuçları:")
    print("=" * 90)

    for i, item in enumerate(scored_results[:5], start=1):
        result = item["retrieval_result"]
        scoring = item["scoring"]
        metadata = result["metadata"]

        print(f"\n{i}. Öneri")
        print("-" * 90)

        print(f"Otel: {metadata.get('hotel_name', '')}")
        print(f"Konum: {metadata.get('location', '')}")
        print(f"Chunk ID: {result.get('chunk_id', '')}")

        hotel_rating = metadata.get("hotel_rating", "")
        room_score = metadata.get("room_score", "")

        if hotel_rating:
            print(f"Hotel rating: {hotel_rating} / 10")

        if room_score:
            print(f"Room score: {room_score} / 10")

        print(f"Review count: {metadata.get('review_count', '')}")
        print(f"Oda tipi: {metadata.get('room_type', '')}")
        print(f"Yatak tipi: {metadata.get('bed_type', '')}")
        print(f"Kaynak: {metadata.get('source', '')}")

        print(f"\nTravelMind suitability score: {scoring['travelmind_score']} / 100")

        print("\nSkor açıklaması:")
        for component in scoring["components"]:
            name = component["name"]
            score = component["score"]
            weight = component["weight"]
            reason = component["reason"]

            if score is None:
                print(f"- {name}: kullanılmadı | ağırlık: {weight} | {reason}")
            else:
                print(f"- {name}: {score}/100 | ağırlık: {weight} | {reason}")

        print("\nVeri seti metni:")
        print(result["text"][:700])

    print("\nNot:")
    print("Hotel rating ve room score LLM tarafından üretilmez; veri setinden gelir.")
    print(
        "TravelMind suitability score, kullanıcı tercihine göre açıklanabilir kurallarla hesaplanır."
    )
    print(
        "LLM daha sonra yalnızca yorumları anlamlandırma ve açıklama üretme aşamasında kullanılacaktır."
    )


if __name__ == "__main__":
    main()
