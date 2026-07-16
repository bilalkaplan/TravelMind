import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/hotel_catalog_clean.csv")
REPORT_PATH = Path("data/processed/hotel_dataset_audit_report.txt")
MISSING_LOCATION_PATH = Path("data/processed/hotels_missing_location.csv")
UNIQUE_HOTELS_PATH = Path("data/processed/unique_hotels_summary.csv")


def empty_count(series):
    return series.isna().sum() + (series.astype(str).str.strip() == "").sum()


def valid_score_range(series, min_value=0, max_value=10):
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = numeric.dropna()[
        (numeric.dropna() < min_value) | (numeric.dropna() > max_value)
    ]
    return len(invalid)


def main():
    df = pd.read_csv(INPUT_PATH)

    total_rows = len(df)
    unique_hotels = df["hotel_name"].dropna().astype(str).str.strip().nunique()

    source_counts = df["source"].value_counts()

    missing_location = empty_count(df["location"])
    missing_hotel_rating = empty_count(df["hotel_rating"])
    missing_review_score = empty_count(df["review_score"])
    missing_room_score = empty_count(df["room_score"])
    missing_review_count = empty_count(df["review_count"])
    missing_room_type = empty_count(df["room_type"])
    missing_bed_type = empty_count(df["bed_type"])
    missing_room_comment = empty_count(df["room_comment"])

    invalid_hotel_rating = valid_score_range(df["hotel_rating"])
    invalid_review_score = valid_score_range(df["review_score"])
    invalid_room_score = valid_score_range(df["room_score"])

    duplicate_rows = df.duplicated(
        subset=[
            "hotel_name",
            "location",
            "room_type",
            "bed_type",
            "room_comment",
            "source",
        ]
    ).sum()

    missing_location_df = df[
        df["location"].isna() | (df["location"].astype(str).str.strip() == "")
    ].copy()

    missing_location_df = missing_location_df[
        [
            "hotel_id",
            "hotel_name",
            "location",
            "hotel_rating",
            "review_score",
            "room_score",
            "review_count",
            "room_type",
            "bed_type",
            "room_comment",
            "source",
        ]
    ]

    missing_location_df.to_csv(MISSING_LOCATION_PATH, index=False, encoding="utf-8")

    unique_summary = (
        df.groupby("hotel_name")
        .agg(
            record_count=("hotel_id", "count"),
            sources=("source", lambda x: ", ".join(sorted(set(x.astype(str))))),
            has_location=("location", lambda x: any(x.astype(str).str.strip() != "")),
            has_room_comment=(
                "room_comment",
                lambda x: any(x.astype(str).str.strip() != ""),
            ),
            has_room_type=("room_type", lambda x: any(x.astype(str).str.strip() != "")),
            has_bed_type=("bed_type", lambda x: any(x.astype(str).str.strip() != "")),
        )
        .reset_index()
        .sort_values("record_count", ascending=False)
    )

    unique_summary.to_csv(UNIQUE_HOTELS_PATH, index=False, encoding="utf-8")

    report = f"""
TravelMind RAG - Hotel Dataset Audit Report

1. Dataset Size

Total records / rows: {total_rows}
Unique hotel count: {unique_hotels}

This means the dataset contains {total_rows} hotel-related records for {unique_hotels} unique hotel names.
A high row count does not always mean many different hotels, because one hotel can appear in multiple rows.

2. Source Distribution

{source_counts.to_string()}

3. Missing Field Counts

Missing location: {missing_location}
Missing hotel_rating: {missing_hotel_rating}
Missing review_score: {missing_review_score}
Missing room_score: {missing_room_score}
Missing review_count: {missing_review_count}
Missing room_type: {missing_room_type}
Missing bed_type: {missing_bed_type}
Missing room_comment: {missing_room_comment}

4. Score Scale Policy

The score fields are treated as 0-10 scale values.

Used score fields:
- hotel_rating: interpreted as a hotel-level score out of 10
- review_score: interpreted as a review-level score out of 10
- room_score: interpreted as a room-level score out of 10

Invalid hotel_rating values outside 0-10: {invalid_hotel_rating}
Invalid review_score values outside 0-10: {invalid_review_score}
Invalid room_score values outside 0-10: {invalid_room_score}

5. Duplicate Check

Duplicate rows: {duplicate_rows}

6. Location Reliability

Rows with missing location were exported to:
{MISSING_LOCATION_PATH}

These hotels should be manually checked or enriched from a reliable external source before making location-based claims.

7. Price Policy

Room price is excluded from RAG chunks and LLM answer generation.

Reason:
Hotel prices vary by country, currency, season, date, room type and booking conditions.
The assistant will not recommend hotels based on price.

8. LLM Grounding Policy

The LLM must not answer from its own memory.

Rules:
- If a hotel exists in the retrieved dataset context, answer only using retrieved fields.
- If location is missing, do not claim a country or city.
- If score is missing, do not claim it is highly rated.
- If comments are missing, do not claim guests liked or disliked it.
- If the hotel is not found in the dataset, say it is not found.
- Always show source hotel_id or review_id.
- Do not invent hotel names, countries, prices, amenities or scores.

9. Academic Dataset Research Note

Current datasets are public Kaggle hotel datasets and are not generated by us.
For a stronger academic version, the project can be extended with:
- HotelRec: TripAdvisor-based large-scale hotel recommendation dataset with 50M reviews.
- Booking.com Accommodation Review Dataset: large-scale Booking.com review dataset with about 1.6M reviews from 40k accommodations.

10. Files Generated

Missing location file:
{MISSING_LOCATION_PATH}

Unique hotel summary:
{UNIQUE_HOTELS_PATH}
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    print(report)
    print("Audit report oluşturuldu:", REPORT_PATH)
    print("Location boş olan oteller:", MISSING_LOCATION_PATH)
    print("Unique hotel summary:", UNIQUE_HOTELS_PATH)


if __name__ == "__main__":
    main()
