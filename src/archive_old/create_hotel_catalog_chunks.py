import json
import re
import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/hotel_catalog_reliable.csv")
OUTPUT_PATH = Path("data/processed/hotel_catalog_chunks.jsonl")

PRICE_KEYWORDS = [
    r"\bprice\b",
    r"\bprices\b",
    r"\bpriced\b",
    r"\bcost\b",
    r"\bcosts\b",
    r"\bcheap\b",
    r"\bexpensive\b",
    r"\brate\b",
    r"\brates\b",
    r"\bcurrency\b",
    r"\bbdt\b",
]


def safe_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def is_price_related(text):
    text = safe_value(text).lower()

    for pattern in PRICE_KEYWORDS:
        if re.search(pattern, text):
            return True

    return False


def build_chunk_text(row):
    parts = []

    hotel_name = safe_value(row.get("hotel_name"))
    location = safe_value(row.get("location"))
    hotel_rating = safe_value(row.get("hotel_rating"))
    room_score = safe_value(row.get("room_score"))
    review_count = safe_value(row.get("review_count"))
    room_type = safe_value(row.get("room_type"))
    bed_type = safe_value(row.get("bed_type"))
    room_comment = safe_value(row.get("room_comment"))
    source = safe_value(row.get("source"))

    if hotel_name:
        parts.append(f"Hotel name: {hotel_name}")

    if location:
        parts.append(f"Location: {location}")

    if hotel_rating:
        parts.append(f"Hotel rating: {hotel_rating} out of 10")

    if room_score:
        parts.append(f"Room score: {room_score} out of 10")

    if review_count:
        parts.append(f"Review count: {review_count}")

    if room_type:
        parts.append(f"Room type: {room_type}")

    if bed_type:
        parts.append(f"Bed type: {bed_type}")

    # Fiyatla ilgili yorumları LLM bağlamına koymuyoruz.
    if room_comment and not is_price_related(room_comment):
        parts.append(f"Room comment: {room_comment}")

    if source:
        parts.append(f"Source: {source}")

    return "\n".join(parts)


def main():
    df = pd.read_csv(INPUT_PATH)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            reliable_hotel_id = int(row["reliable_hotel_id"])
            original_hotel_id = int(row["hotel_id"])

            chunk = {
                "chunk_id": f"hotel_{reliable_hotel_id}",
                "text": build_chunk_text(row),
                "metadata": {
                    "reliable_hotel_id": reliable_hotel_id,
                    "original_hotel_id": original_hotel_id,
                    "hotel_name": safe_value(row.get("hotel_name")),
                    "location": safe_value(row.get("location")),
                    "hotel_rating": safe_value(row.get("hotel_rating")),
                    "room_score": safe_value(row.get("room_score")),
                    "review_count": safe_value(row.get("review_count")),
                    "room_type": safe_value(row.get("room_type")),
                    "bed_type": safe_value(row.get("bed_type")),
                    "source": safe_value(row.get("source")),
                },
            }

            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(
        "Hotel catalog chunk dosyası güvenilir veri setinden oluşturuldu:", OUTPUT_PATH
    )
    print("Toplam hotel chunk sayısı:", len(df))

    print("\nİlk chunk örneği:")
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        first_line = f.readline()
        print(json.dumps(json.loads(first_line), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
