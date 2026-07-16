from pathlib import Path
import json
import math

import pandas as pd

PROCESSED_DIR = Path("data/processed")

HOTELS_PATH = PROCESSED_DIR / "cmu_hotels_reliable.csv"
REVIEWS_MERGED_PATH = PROCESSED_DIR / "cmu_reviews_reliable_merged.csv"

OUTPUT_PATH = PROCESSED_DIR / "cmu_chunks.jsonl"
REPORT_PATH = PROCESSED_DIR / "cmu_chunks_report.txt"

REVIEWS_PER_CHUNK = 10


def safe_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def safe_rating(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    if value.lower() == "nan":
        return ""
    return value


def build_hotel_profile_chunk(row):
    parts = []

    hotel_name = safe_text(row.get("hotel_name", ""))
    location = safe_text(row.get("location", ""))
    hotel_class = safe_text(row.get("hotel_class", ""))
    review_count_total = safe_text(row.get("review_count_total", ""))
    review_count_used = safe_text(row.get("review_count_used", ""))

    parts.append("Chunk type: hotel_profile")

    if hotel_name:
        parts.append(f"Hotel name: {hotel_name}")

    if location:
        parts.append(f"Location: {location}")

    if hotel_class:
        parts.append(f"Hotel class: {hotel_class}")

    if review_count_total:
        parts.append(f"Total review count in CMU dataset: {review_count_total}")

    if review_count_used:
        parts.append(f"Used review count in TravelMind subset: {review_count_used}")

    parts.append("Source: CMU TripAdvisor offering dataset")

    return "\n".join(parts)


def build_review_group_chunk(hotel_row, review_group, group_index):
    parts = []

    hotel_name = safe_text(hotel_row.get("hotel_name", ""))
    location = safe_text(hotel_row.get("location", ""))
    hotel_class = safe_text(hotel_row.get("hotel_class", ""))
    review_count_total = safe_text(hotel_row.get("review_count_total", ""))

    parts.append("Chunk type: review_group")

    if hotel_name:
        parts.append(f"Hotel name: {hotel_name}")

    if location:
        parts.append(f"Location: {location}")

    if hotel_class:
        parts.append(f"Hotel class: {hotel_class}")

    if review_count_total:
        parts.append(f"Total review count in CMU dataset: {review_count_total}")

    parts.append(f"Review group number: {group_index}")

    review_texts = []

    for idx, (_, review) in enumerate(review_group.iterrows(), start=1):
        title = safe_text(review.get("review_title", ""))
        text = safe_text(review.get("review_text", ""))
        overall_rating = safe_rating(review.get("overall_rating", ""))
        value_rating = safe_rating(review.get("value_rating", ""))
        rooms_rating = safe_rating(review.get("rooms_rating", ""))
        location_rating = safe_rating(review.get("location_rating", ""))
        cleanliness_rating = safe_rating(review.get("cleanliness_rating", ""))
        service_rating = safe_rating(review.get("service_rating", ""))

        review_parts = []

        review_parts.append(f"Review {idx}:")

        if title:
            review_parts.append(f"Title: {title}")

        if overall_rating:
            review_parts.append(f"Overall rating: {overall_rating} / 5")

        if value_rating:
            review_parts.append(f"Value rating: {value_rating} / 5")

        if rooms_rating:
            review_parts.append(f"Rooms rating: {rooms_rating} / 5")

        if location_rating:
            review_parts.append(f"Location rating: {location_rating} / 5")

        if cleanliness_rating:
            review_parts.append(f"Cleanliness rating: {cleanliness_rating} / 5")

        if service_rating:
            review_parts.append(f"Service rating: {service_rating} / 5")

        if text:
            review_parts.append(f"Text: {text}")

        review_texts.append("\n".join(review_parts))

    parts.append("\n\n".join(review_texts))
    parts.append("Source: CMU TripAdvisor review dataset")

    return "\n".join(parts)


def main():
    if not HOTELS_PATH.exists():
        print("Hotels dosyası bulunamadı:", HOTELS_PATH)
        return

    if not REVIEWS_MERGED_PATH.exists():
        print("Merged reviews dosyası bulunamadı:", REVIEWS_MERGED_PATH)
        return

    print("CMU hotel subset okunuyor...")
    hotels_df = pd.read_csv(HOTELS_PATH)

    print("CMU merged review subset okunuyor...")
    reviews_df = pd.read_csv(REVIEWS_MERGED_PATH)

    hotels_df["hotel_id"] = hotels_df["hotel_id"].astype(str)
    reviews_df["hotel_id"] = reviews_df["hotel_id"].astype(str)

    hotel_lookup = {str(row["hotel_id"]): row for _, row in hotels_df.iterrows()}

    chunk_count = 0
    hotel_profile_count = 0
    review_group_count = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        print("Hotel profile chunk'ları oluşturuluyor...")

        for _, hotel in hotels_df.iterrows():
            hotel_id = str(hotel["hotel_id"])

            chunk = {
                "chunk_id": f"cmu_hotel_profile_{hotel_id}",
                "chunk_type": "cmu_hotel_profile",
                "text": build_hotel_profile_chunk(hotel),
                "metadata": {
                    "dataset": "cmu_tripadvisor",
                    "chunk_type": "hotel_profile",
                    "hotel_id": hotel_id,
                    "hotel_name": safe_text(hotel.get("hotel_name", "")),
                    "hotel_name_normalized": safe_text(
                        hotel.get("hotel_name_normalized", "")
                    ),
                    "location": safe_text(hotel.get("location", "")),
                    "hotel_class": safe_text(hotel.get("hotel_class", "")),
                    "review_count_total": safe_text(
                        hotel.get("review_count_total", "")
                    ),
                    "review_count_used": safe_text(hotel.get("review_count_used", "")),
                    "source": "cmu_tripadvisor_offering",
                },
            }

            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            chunk_count += 1
            hotel_profile_count += 1

        print("Review group chunk'ları oluşturuluyor...")

        for hotel_id, group in reviews_df.groupby("hotel_id"):
            hotel_row = hotel_lookup.get(str(hotel_id))

            if hotel_row is None:
                continue

            group = group.reset_index(drop=True)
            total_reviews = len(group)
            group_count = math.ceil(total_reviews / REVIEWS_PER_CHUNK)

            for group_index in range(group_count):
                start = group_index * REVIEWS_PER_CHUNK
                end = start + REVIEWS_PER_CHUNK
                review_group = group.iloc[start:end]

                chunk = {
                    "chunk_id": f"cmu_review_group_{hotel_id}_{group_index + 1}",
                    "chunk_type": "cmu_review_group",
                    "text": build_review_group_chunk(
                        hotel_row=hotel_row,
                        review_group=review_group,
                        group_index=group_index + 1,
                    ),
                    "metadata": {
                        "dataset": "cmu_tripadvisor",
                        "chunk_type": "review_group",
                        "hotel_id": str(hotel_id),
                        "hotel_name": safe_text(hotel_row.get("hotel_name", "")),
                        "hotel_name_normalized": safe_text(
                            hotel_row.get("hotel_name_normalized", "")
                        ),
                        "location": safe_text(hotel_row.get("location", "")),
                        "hotel_class": safe_text(hotel_row.get("hotel_class", "")),
                        "review_count_in_chunk": len(review_group),
                        "review_group_number": group_index + 1,
                        "source": "cmu_tripadvisor_review",
                    },
                }

                file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                chunk_count += 1
                review_group_count += 1

            if review_group_count % 1000 == 0:
                print(f"Review group chunk sayısı: {review_group_count}")

    report = f"""TravelMind RAG - CMU Chunk Report

Input files:
- Hotels: {HOTELS_PATH}
- Reviews merged: {REVIEWS_MERGED_PATH}

Chunk policy:
- 1 hotel profile chunk per hotel
- {REVIEWS_PER_CHUNK} reviews per review group chunk

Output:
- CMU chunks file: {OUTPUT_PATH}

Counts:
- Hotel profile chunks: {hotel_profile_count}
- Review group chunks: {review_group_count}
- Total chunks: {chunk_count}

Interpretation:
- Reviews are grouped instead of embedding each review separately.
- This keeps the vector database manageable.
- Hotel-level metadata and review evidence are both preserved.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    print("\nCMU chunk oluşturma tamamlandı.")
    print(f"Hotel profile chunks: {hotel_profile_count}")
    print(f"Review group chunks: {review_group_count}")
    print(f"Total chunks: {chunk_count}")
    print("Chunks file:", OUTPUT_PATH)
    print("Report:", REPORT_PATH)


if __name__ == "__main__":
    main()
