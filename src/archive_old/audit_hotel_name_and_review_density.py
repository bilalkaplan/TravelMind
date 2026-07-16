from pathlib import Path
import re
import pandas as pd

CATALOG_PATH = Path("data/processed/hotel_catalog_clean.csv")
RELIABLE_PATH = Path("data/processed/hotel_catalog_reliable.csv")
ROOM_COMMENTS_PATH = Path("data/processed/hotel_catalog_clean.csv")

OUTPUT_DIR = Path("data/processed")

REPORT_PATH = OUTPUT_DIR / "hotel_name_and_review_density_report.txt"
NORMALIZED_HOTELS_PATH = OUTPUT_DIR / "normalized_hotel_name_summary.csv"
POSSIBLE_DUPLICATES_PATH = OUTPUT_DIR / "possible_duplicate_hotel_names.csv"
REVIEW_DENSITY_PATH = OUTPUT_DIR / "hotel_review_density_summary.csv"


def normalize_hotel_name(name):
    name = str(name).strip().lower()

    # Türkçe/İngilizce fark etmeksizin basit normalize
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"[^\w\s]", "", name)

    # gereksiz genel kelimeleri tamamen silmiyoruz, çünkü otel adı bozulabilir
    # sadece boşlukları düzenliyoruz
    return name.strip()


def is_empty(series):
    return series.isna() | (series.astype(str).str.strip() == "")


def safe_read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    return pd.read_csv(path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    catalog_df = safe_read_csv(CATALOG_PATH)
    reliable_df = safe_read_csv(RELIABLE_PATH)

    for df in [catalog_df, reliable_df]:
        df["hotel_name"] = df["hotel_name"].fillna("").astype(str)
        df["hotel_name_normalized"] = df["hotel_name"].apply(normalize_hotel_name)

    catalog_valid = catalog_df[catalog_df["hotel_name_normalized"] != ""].copy()
    reliable_valid = reliable_df[reliable_df["hotel_name_normalized"] != ""].copy()

    original_catalog_rows = len(catalog_df)
    original_reliable_rows = len(reliable_df)

    raw_unique_catalog = catalog_valid["hotel_name"].nunique()
    normalized_unique_catalog = catalog_valid["hotel_name_normalized"].nunique()

    raw_unique_reliable = reliable_valid["hotel_name"].nunique()
    normalized_unique_reliable = reliable_valid["hotel_name_normalized"].nunique()

    # Aynı normalized ada sahip farklı yazımlar
    duplicate_name_groups = (
        catalog_valid.groupby("hotel_name_normalized")
        .agg(
            raw_name_count=("hotel_name", "nunique"),
            record_count=("hotel_name", "count"),
            example_names=(
                "hotel_name",
                lambda x: " | ".join(sorted(set(x.astype(str)))[:10]),
            ),
            sources=(
                "source",
                lambda x: (
                    " | ".join(sorted(set(x.astype(str)))[:10])
                    if "source" in catalog_valid.columns
                    else ""
                ),
            ),
        )
        .reset_index()
    )

    possible_duplicates = duplicate_name_groups[
        duplicate_name_groups["raw_name_count"] > 1
    ].sort_values(["raw_name_count", "record_count"], ascending=False)

    possible_duplicates.to_csv(POSSIBLE_DUPLICATES_PATH, index=False, encoding="utf-8")

    normalized_summary = duplicate_name_groups.sort_values(
        "record_count", ascending=False
    )

    normalized_summary.to_csv(NORMALIZED_HOTELS_PATH, index=False, encoding="utf-8")

    # Otel başına comment/review yoğunluğu
    comment_df = catalog_valid.copy()

    if "room_comment" in comment_df.columns:
        comment_df["has_room_comment"] = ~is_empty(comment_df["room_comment"])
    else:
        comment_df["has_room_comment"] = False

    density = (
        comment_df.groupby("hotel_name_normalized")
        .agg(
            hotel_name_examples=(
                "hotel_name",
                lambda x: " | ".join(sorted(set(x.astype(str)))[:5]),
            ),
            total_records=("hotel_name", "count"),
            comment_count=("has_room_comment", "sum"),
            location_count=(
                "location",
                lambda x: (
                    (~is_empty(x)).sum() if "location" in comment_df.columns else 0
                ),
            ),
            source_count=(
                "source",
                lambda x: x.nunique() if "source" in comment_df.columns else 0,
            ),
            sources=(
                "source",
                lambda x: (
                    " | ".join(sorted(set(x.astype(str)))[:5])
                    if "source" in comment_df.columns
                    else ""
                ),
            ),
        )
        .reset_index()
        .sort_values(["comment_count", "total_records"], ascending=False)
    )

    density.to_csv(REVIEW_DENSITY_PATH, index=False, encoding="utf-8")

    total_hotels = len(density)
    hotels_with_zero_comment = (density["comment_count"] == 0).sum()
    hotels_with_one_comment = (density["comment_count"] == 1).sum()
    hotels_with_2_to_4_comments = (
        (density["comment_count"] >= 2) & (density["comment_count"] <= 4)
    ).sum()
    hotels_with_5_plus_comments = (density["comment_count"] >= 5).sum()
    hotels_with_10_plus_comments = (density["comment_count"] >= 10).sum()
    hotels_with_20_plus_comments = (density["comment_count"] >= 20).sum()

    avg_comments = density["comment_count"].mean() if total_hotels else 0
    median_comments = density["comment_count"].median() if total_hotels else 0
    max_comments = density["comment_count"].max() if total_hotels else 0

    # Veri setine göre eşik önerisi
    if median_comments < 2 and hotels_with_5_plus_comments < total_hotels * 0.10:
        recommended_policy = """
The current dataset is sparse at hotel level.
A strict minimum review threshold would remove most hotels.
Recommended policy for the current dataset:
- Use records with valid location for location-based recommendation.
- Use comment text only as supporting evidence when available.
- Do not claim strong hotel-level reliability if comment_count is low.
- For a stronger academic version, switch to a dataset with many reviews per hotel.
"""
    else:
        recommended_policy = """
The dataset has a usable number of comments for some hotels.
Recommended policy:
- Keep hotels with enough comments for review-based scoring.
- Use low-comment hotels only for metadata-based retrieval.
- Report comment_count in outputs as a reliability signal.
"""

    report = f"""
TravelMind RAG - Hotel Name Normalization and Review Density Report

1. Dataset Overview

Original catalog records: {original_catalog_rows}
Reliable dataset records: {original_reliable_rows}

2. Unique Hotel Name Audit

Catalog raw unique hotel names: {raw_unique_catalog}
Catalog normalized unique hotel names: {normalized_unique_catalog}

Reliable raw unique hotel names: {raw_unique_reliable}
Reliable normalized unique hotel names: {normalized_unique_reliable}

Meaning:
- Raw unique count is case-sensitive and formatting-sensitive.
- Normalized unique count lowercases hotel names and removes punctuation differences.
- If raw unique count is higher than normalized unique count, some hotels may have been counted multiple times due to naming differences.

Possible duplicate normalized-name groups: {len(possible_duplicates)}

3. Review / Comment Density Per Hotel

Total normalized hotels: {total_hotels}

Hotels with 0 comments: {hotels_with_zero_comment}
Hotels with exactly 1 comment: {hotels_with_one_comment}
Hotels with 2-4 comments: {hotels_with_2_to_4_comments}
Hotels with 5+ comments: {hotels_with_5_plus_comments}
Hotels with 10+ comments: {hotels_with_10_plus_comments}
Hotels with 20+ comments: {hotels_with_20_plus_comments}

Average comment count per hotel: {avg_comments:.2f}
Median comment count per hotel: {median_comments:.2f}
Maximum comment count for a hotel: {max_comments}

4. Recommended Dataset Policy

{recommended_policy}

5. Mentor Concern Response

The mentor's concern is valid:
- Hotel names must be normalized to avoid counting the same hotel multiple times.
- Case differences such as uppercase/lowercase should not create separate hotels.
- The number of comments per hotel must be measured before choosing a minimum review threshold.
- A fixed threshold should not be selected blindly. It should depend on the dataset distribution.
- The current dataset appears useful for a prototype, but may be too sparse for a strong academic hotel recommendation system.

6. Better Dataset Candidates

For a stronger version, the project can be extended with:
- HotelRec: TripAdvisor-based academic hotel recommendation dataset with about 50M reviews.
- Booking.com Accommodation Reviews: about 1.6M reviews from 40k accommodations.
- CMU TripAdvisor Hotel Reviews: 878,561 reviews from 4,333 hotels.

7. Generated Files

Normalized hotel summary:
{NORMALIZED_HOTELS_PATH}

Possible duplicate hotel names:
{POSSIBLE_DUPLICATES_PATH}

Hotel review density summary:
{REVIEW_DENSITY_PATH}
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    print(report)
    print("Rapor oluşturuldu:", REPORT_PATH)
    print("Normalized hotel summary:", NORMALIZED_HOTELS_PATH)
    print("Possible duplicate hotel names:", POSSIBLE_DUPLICATES_PATH)
    print("Review density summary:", REVIEW_DENSITY_PATH)


if __name__ == "__main__":
    main()
