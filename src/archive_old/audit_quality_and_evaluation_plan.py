import pandas as pd
from pathlib import Path

CLEAN_PATH = Path("data/processed/hotel_catalog_clean.csv")
RELIABLE_PATH = Path("data/processed/hotel_catalog_reliable.csv")

OUTPUT_DIR = Path("data/processed")
REPORT_PATH = OUTPUT_DIR / "quality_and_scoring_report.txt"
LOCATION_DROP_CSV = OUTPUT_DIR / "location_drop_report.csv"
SCORE_AUDIT_CSV = OUTPUT_DIR / "score_audit_summary.csv"
MANUAL_EVAL_SAMPLE_CSV = OUTPUT_DIR / "manual_comment_eval_sample.csv"


def is_empty(series):
    return series.isna() | (series.astype(str).str.strip() == "")


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def score_audit(df, score_column):
    if score_column not in df.columns:
        return {
            "score_field": score_column,
            "exists": False,
            "total_rows": len(df),
            "non_empty_count": 0,
            "missing_count": len(df),
            "numeric_count": 0,
            "non_numeric_count": 0,
            "min_value": "",
            "max_value": "",
            "mean_value": "",
            "invalid_outside_0_10": "",
            "sources_with_values": "",
        }

    empty = is_empty(df[score_column])
    non_empty_count = int((~empty).sum())
    missing_count = int(empty.sum())

    numeric = safe_numeric(df[score_column])
    numeric_count = int(numeric.notna().sum())
    non_numeric_count = int((~empty & numeric.isna()).sum())

    valid_numeric = numeric.dropna()

    if len(valid_numeric) > 0:
        min_value = float(valid_numeric.min())
        max_value = float(valid_numeric.max())
        mean_value = float(valid_numeric.mean())
        invalid_outside_0_10 = int(((valid_numeric < 0) | (valid_numeric > 10)).sum())
    else:
        min_value = ""
        max_value = ""
        mean_value = ""
        invalid_outside_0_10 = ""

    if "source" in df.columns:
        source_counts = df.loc[~empty, "source"].astype(str).value_counts().to_dict()
        sources_with_values = str(source_counts)
    else:
        sources_with_values = ""

    return {
        "score_field": score_column,
        "exists": True,
        "total_rows": len(df),
        "non_empty_count": non_empty_count,
        "missing_count": missing_count,
        "numeric_count": numeric_count,
        "non_numeric_count": non_numeric_count,
        "min_value": min_value,
        "max_value": max_value,
        "mean_value": mean_value,
        "invalid_outside_0_10": invalid_outside_0_10,
        "sources_with_values": sources_with_values,
    }


def create_location_drop_report(clean_df, reliable_df):
    total_rows = len(clean_df)

    missing_location_mask = is_empty(clean_df["location"])
    missing_location_count = int(missing_location_mask.sum())
    location_available_count = int((~missing_location_mask).sum())

    reliable_count = len(reliable_df)
    dropped_duplicate_or_other = location_available_count - reliable_count

    report_df = pd.DataFrame(
        [
            {"metric": "total_original_records", "value": total_rows},
            {"metric": "missing_location_records", "value": missing_location_count},
            {"metric": "location_available_records", "value": location_available_count},
            {"metric": "reliable_dataset_records", "value": reliable_count},
            {
                "metric": "dropped_after_location_filter_due_to_duplicate_or_other_cleaning",
                "value": dropped_duplicate_or_other,
            },
        ]
    )

    report_df.to_csv(LOCATION_DROP_CSV, index=False, encoding="utf-8")
    return report_df


def create_manual_eval_sample(clean_df, sample_size=80):
    """
    Bu dosya LLM'in yorumları doğru anlayıp anlamadığını ölçmek için hazırlanır.
    Burada manuel etiketleme yapılacak.

    manual_sentiment_label kolonuna daha sonra elle şunlardan biri yazılabilir:
    - positive
    - negative
    - neutral
    - unclear

    sarcasm_or_irony kolonuna:
    - yes
    - no
    - unclear
    yazılabilir.
    """

    if "room_comment" not in clean_df.columns:
        return pd.DataFrame()

    comment_df = clean_df[~is_empty(clean_df["room_comment"])].copy()

    wanted_columns = [
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

    existing_columns = [col for col in wanted_columns if col in comment_df.columns]
    comment_df = comment_df[existing_columns]

    if len(comment_df) > sample_size:
        comment_df = comment_df.sample(sample_size, random_state=42)

    comment_df["manual_sentiment_label"] = ""
    comment_df["sarcasm_or_irony"] = ""
    comment_df["notes"] = ""

    comment_df.to_csv(MANUAL_EVAL_SAMPLE_CSV, index=False, encoding="utf-8")
    return comment_df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {CLEAN_PATH}")

    if not RELIABLE_PATH.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {RELIABLE_PATH}")

    clean_df = pd.read_csv(CLEAN_PATH)
    reliable_df = pd.read_csv(RELIABLE_PATH)

    total_rows = len(clean_df)
    total_unique_hotels = (
        clean_df["hotel_name"].dropna().astype(str).str.strip().nunique()
    )

    reliable_rows = len(reliable_df)
    reliable_unique_hotels = (
        reliable_df["hotel_name"].dropna().astype(str).str.strip().nunique()
    )

    location_report_df = create_location_drop_report(clean_df, reliable_df)

    score_fields = ["hotel_rating", "room_score", "review_score"]

    score_rows = [score_audit(clean_df, field) for field in score_fields]
    score_audit_df = pd.DataFrame(score_rows)
    score_audit_df.to_csv(SCORE_AUDIT_CSV, index=False, encoding="utf-8")

    create_manual_eval_sample(clean_df)

    source_distribution = clean_df["source"].value_counts().to_string()

    report = f"""
TravelMind RAG - Quality, Location Drop and Scoring Report

1. Dataset Size

Original total records: {total_rows}
Original unique hotel count: {total_unique_hotels}

Reliable dataset records: {reliable_rows}
Reliable unique hotel count: {reliable_unique_hotels}

2. Source Distribution

{source_distribution}

3. Location Drop Analysis

{location_report_df.to_string(index=False)}

Explanation:
- missing_location_records are excluded from the main RAG hotel dataset.
- These records are not deleted from the project completely.
- They are kept separately for possible manual verification or future enrichment.
- The assistant must not make country/city/location claims for records with missing location.

4. Score Field Audit

{score_audit_df.to_string(index=False)}

Score interpretation:
- hotel_rating is treated as a score out of 10.
- room_score is treated as a score out of 10.
- review_score is not used if it is fully missing or unreliable.

Important:
The LLM must not calculate or invent hotel_rating or room_score.
These values must come only from the retrieved dataset metadata.

5. Are Accuracy and F1 Suitable for Score Validation?

Accuracy and F1 are not directly suitable for validating numeric hotel_rating or room_score values unless the task is converted into classification.

Example:
- High score: 8-10
- Medium score: 5-7.9
- Low score: 0-4.9

However, in this project, the safer approach is:
- Do not let the LLM generate numeric scores.
- Use numeric scores only if they exist in the dataset.
- If a score is missing, the assistant must say that the score is not available.

6. Where Accuracy and F1 Will Be Used

Accuracy and F1 can be used for LLM evaluation on comment understanding.

Example task:
Given a hotel comment, classify it as:
- positive
- negative
- neutral
- unclear

The manually labeled validation file has been created:
{MANUAL_EVAL_SAMPLE_CSV}

After manual labeling, LLM predictions can be compared against manual labels using:
- Accuracy
- Precision
- Recall
- Macro F1
- Weighted F1

7. Sarcasm / Irony Handling Policy

Some hotel comments may include sarcasm or unclear meaning.

Policy:
- If the comment is clearly positive, label it positive.
- If the comment is clearly negative, label it negative.
- If the comment is mixed, label it neutral or unclear.
- If sarcasm or irony is suspected, the LLM should not make a strong claim.
- Sarcastic or ambiguous comments should be treated as unclear unless the meaning is obvious from context.

8. LLM Grounding Policy

The LLM must follow these rules:
- Do not invent hotel names.
- Do not invent countries or cities.
- Do not invent scores.
- Do not invent prices.
- Do not claim that a hotel is good, clean, central or highly rated unless the retrieved dataset supports it.
- Always use retrieved hotel records as context.
- Always show source hotel_id / reliable_hotel_id where possible.
- If the dataset does not contain enough information, explicitly say that the information is insufficient.

9. Generated Files

Location drop report:
{LOCATION_DROP_CSV}

Score audit summary:
{SCORE_AUDIT_CSV}

Manual comment evaluation sample:
{MANUAL_EVAL_SAMPLE_CSV}
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    print(report)
    print("Rapor oluşturuldu:", REPORT_PATH)
    print("Location drop CSV:", LOCATION_DROP_CSV)
    print("Score audit CSV:", SCORE_AUDIT_CSV)
    print("Manual eval sample CSV:", MANUAL_EVAL_SAMPLE_CSV)


if __name__ == "__main__":
    main()
