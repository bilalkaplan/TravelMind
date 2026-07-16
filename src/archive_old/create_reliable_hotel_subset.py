import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/hotel_catalog_clean.csv")
RELIABLE_OUTPUT_PATH = Path("data/processed/hotel_catalog_reliable.csv")
MISSING_LOCATION_OUTPUT_PATH = Path(
    "data/processed/hotels_missing_location_research_list.csv"
)
REPORT_PATH = Path("data/processed/reliable_dataset_report.txt")


def is_empty(series):
    return series.isna() | (series.astype(str).str.strip() == "")


def main():
    df = pd.read_csv(INPUT_PATH)

    original_rows = len(df)
    original_unique_hotels = df["hotel_name"].dropna().astype(str).str.strip().nunique()

    # Location boş olanları ayrı araştırma listesine al
    missing_location_df = df[is_empty(df["location"])].copy()

    research_columns = [
        "hotel_id",
        "hotel_name",
        "location",
        "hotel_rating",
        "room_score",
        "review_count",
        "room_type",
        "bed_type",
        "room_comment",
        "source",
    ]

    existing_research_columns = [
        col for col in research_columns if col in missing_location_df.columns
    ]

    research_list = missing_location_df[existing_research_columns].copy()
    research_list["verified_country"] = ""
    research_list["verified_city_or_area"] = ""
    research_list["verification_source_url"] = ""
    research_list["verification_status"] = "needs_manual_check"
    research_list["notes"] = ""

    research_list.to_csv(MISSING_LOCATION_OUTPUT_PATH, index=False, encoding="utf-8")

    # Ana güvenilir veri: hotel_name + source + location zorunlu
    reliable = df.copy()

    reliable = reliable[~is_empty(reliable["hotel_name"])]
    reliable = reliable[~is_empty(reliable["source"])]
    reliable = reliable[~is_empty(reliable["location"])]

    # Kullanılmayacak kolonları düşür
    drop_columns = []
    for col in ["room_price", "review_score"]:
        if col in reliable.columns:
            drop_columns.append(col)

    reliable = reliable.drop(columns=drop_columns)

    # Duplicate temizliği
    duplicate_before = reliable.duplicated(
        subset=[
            "hotel_name",
            "location",
            "room_type",
            "bed_type",
            "room_comment",
            "source",
        ]
    ).sum()

    reliable = reliable.drop_duplicates(
        subset=[
            "hotel_name",
            "location",
            "room_type",
            "bed_type",
            "room_comment",
            "source",
        ]
    )

    reliable = reliable.reset_index(drop=True)
    reliable["reliable_hotel_id"] = reliable.index + 1

    # Kolon sırası
    preferred_columns = [
        "reliable_hotel_id",
        "hotel_id",
        "hotel_name",
        "location",
        "hotel_rating",
        "room_score",
        "review_count",
        "room_type",
        "bed_type",
        "room_comment",
        "source",
    ]

    existing_columns = [col for col in preferred_columns if col in reliable.columns]
    reliable = reliable[existing_columns]

    reliable.to_csv(RELIABLE_OUTPUT_PATH, index=False, encoding="utf-8")

    reliable_rows = len(reliable)
    reliable_unique_hotels = (
        reliable["hotel_name"].dropna().astype(str).str.strip().nunique()
    )

    report = f"""
TravelMind RAG - Reliable Hotel Dataset Report

Original dataset:
- Total records: {original_rows}
- Unique hotel count: {original_unique_hotels}

Reliable subset:
- Total records: {reliable_rows}
- Unique hotel count: {reliable_unique_hotels}

Removed / excluded:
- Records with missing location were excluded from the main RAG dataset.
- Missing location records were exported for manual research.
- room_price was excluded because hotel prices vary by country, currency, season, date and room conditions.
- review_score was excluded because it is completely empty in the current processed dataset.

Score policy:
- hotel_rating is treated as a score out of 10.
- room_score is treated as a score out of 10.
- The assistant must explicitly mention that scores are interpreted as 0-10 scale values when using them.

Grounding policy:
- The LLM must only answer using retrieved dataset records.
- If location is missing, the system must not claim country/city.
- If a hotel is not found in the dataset, the assistant must say it is not found.
- If a score is missing, the assistant must not claim that the hotel is highly rated.
- The assistant must show source hotel IDs.
- The assistant must not invent hotel names, countries, locations, prices, amenities or scores.

Generated files:
- Reliable dataset: {RELIABLE_OUTPUT_PATH}
- Missing location research list: {MISSING_LOCATION_OUTPUT_PATH}

Duplicate rows removed from reliable subset: {duplicate_before}
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    print(report)
    print("Reliable dataset oluşturuldu:", RELIABLE_OUTPUT_PATH)
    print("Location araştırma listesi oluşturuldu:", MISSING_LOCATION_OUTPUT_PATH)
    print("Rapor oluşturuldu:", REPORT_PATH)


if __name__ == "__main__":
    main()
