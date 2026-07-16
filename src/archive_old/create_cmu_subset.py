from pathlib import Path
import ast
import json
import re

import pandas as pd

RAW_DIR = Path("data/raw/cmu_tripadvisor")
OFFERING_PATH = RAW_DIR / "offering.txt"
REVIEW_PATH = RAW_DIR / "review.txt"

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HOTELS_OUT = OUTPUT_DIR / "cmu_hotels_reliable.csv"
REVIEWS_OUT = OUTPUT_DIR / "cmu_reviews_reliable.csv"
MERGED_OUT = OUTPUT_DIR / "cmu_reviews_reliable_merged.csv"
REPORT_OUT = OUTPUT_DIR / "cmu_subset_report.txt"

MIN_REVIEW_COUNT = 100
MAX_REVIEWS_PER_HOTEL = 200
RANDOM_STATE = 42


def parse_record_line(line: str):
    line = line.strip()

    if not line:
        return None

    try:
        obj = json.loads(line)
        if isinstance(obj, dict):
            return obj
    except Exception: # pylint: disable=broad-exception-caught
        pass

    try:
        obj = ast.literal_eval(line)
        if isinstance(obj, dict):
            return obj
    except Exception: # pylint: disable=broad-exception-caught
        pass

    return None


def normalize_hotel_name(name):
    text = str(name).strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_get(dictionary, key, default=""):
    if isinstance(dictionary, dict):
        return dictionary.get(key, default)
    return default


def extract_location_text(address):
    if not isinstance(address, dict):
        return ""

    locality = address.get("locality", "")
    region = address.get("region", "")
    country = (
        address.get("country", "")
        or address.get("country_name", "")
        or address.get("country-name", "")
    )

    parts = []

    for value in [locality, region, country]:
        value = str(value).strip()
        if value:
            parts.append(value)

    return ", ".join(parts)


def load_offerings():
    rows = []
    skipped = 0

    with OFFERING_PATH.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            obj = parse_record_line(line)

            if obj is None:
                skipped += 1
                continue

            hotel_id = str(obj.get("id", "")).strip()
            hotel_name = str(obj.get("name", "")).strip()
            address = obj.get("address", {})
            details = obj.get("details", {})

            rows.append(
                {
                    "hotel_id": hotel_id,
                    "hotel_name": hotel_name,
                    "hotel_name_normalized": normalize_hotel_name(hotel_name),
                    "location": extract_location_text(address),
                    "hotel_class": obj.get("hotel_class", ""),
                    "region_id": obj.get("region_id", ""),
                    "hotel_type": obj.get("type", ""),
                    "phone": obj.get("phone", ""),
                    "url": obj.get("url", ""),
                    "raw_address_json": json.dumps(address, ensure_ascii=False),
                    "raw_details_json": json.dumps(details, ensure_ascii=False),
                }
            )

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df[df["hotel_id"].astype(str).str.strip() != ""].copy()
        df = df.drop_duplicates(subset=["hotel_id"]).reset_index(drop=True)

    return df, skipped


def load_reviews():
    rows = []
    skipped = 0

    with REVIEW_PATH.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            obj = parse_record_line(line)

            if obj is None:
                skipped += 1
                continue

            hotel_id = str(obj.get("offering_id", "")).strip()
            review_id = str(obj.get("id", "")).strip()
            review_text = str(obj.get("text", "")).strip()
            ratings = obj.get("ratings", {})

            if not isinstance(ratings, dict):
                ratings = {}

            rows.append(
                {
                    "review_id": review_id,
                    "hotel_id": hotel_id,
                    "author": str(obj.get("author", "")).strip(),
                    "review_title": str(obj.get("title", "")).strip(),
                    "review_text": review_text,
                    "date": str(obj.get("date", "")).strip(),
                    "date_stayed": str(obj.get("date_stayed", "")).strip(),
                    "via_mobile": str(obj.get("via_mobile", "")).strip(),
                    "num_helpful_votes": obj.get("num_helpful_votes", ""),
                    "overall_rating": safe_get(ratings, "overall", ""),
                    "value_rating": safe_get(ratings, "value", ""),
                    "rooms_rating": safe_get(ratings, "rooms", ""),
                    "location_rating": safe_get(ratings, "location", ""),
                    "cleanliness_rating": safe_get(ratings, "cleanliness", ""),
                    "checkin_frontdesk_rating": safe_get(
                        ratings, "check in / front desk", ""
                    ),
                    "service_rating": safe_get(ratings, "service", ""),
                    "business_service_rating": safe_get(
                        ratings, "business service", ""
                    ),
                }
            )

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df[df["hotel_id"].astype(str).str.strip() != ""].copy()
        df = df[df["review_text"].astype(str).str.strip() != ""].copy()
        df = df.drop_duplicates(subset=["review_id"]).reset_index(drop=True)

    return df, skipped


def create_balanced_review_subset(reviews_df, eligible_hotel_ids):
    reliable_reviews_df = reviews_df[
        reviews_df["hotel_id"].astype(str).isin(eligible_hotel_ids)
    ].copy()

    sampled_parts = []

    for _, group in reliable_reviews_df.groupby("hotel_id"):
        sample_size = min(len(group), MAX_REVIEWS_PER_HOTEL)

        sampled_group = group.sample(n=sample_size, random_state=RANDOM_STATE).copy()

        sampled_parts.append(sampled_group)

    if not sampled_parts:
        return pd.DataFrame(columns=reviews_df.columns)

    balanced_reviews_df = pd.concat(sampled_parts, ignore_index=True)

    return balanced_reviews_df


def main():
    if not OFFERING_PATH.exists():
        print("Offering dosyası bulunamadı:", OFFERING_PATH)
        return

    if not REVIEW_PATH.exists():
        print("Review dosyası bulunamadı:", REVIEW_PATH)
        return

    print("Offering verisi okunuyor...")
    hotels_df, skipped_offering = load_offerings()

    print("Review verisi okunuyor...")
    reviews_df, skipped_review = load_reviews()

    print(f"Offering kayıt sayısı: {len(hotels_df)}")
    print(f"Review kayıt sayısı: {len(reviews_df)}")
    print(f"Atlanan offering satırı: {skipped_offering}")
    print(f"Atlanan review satırı: {skipped_review}")

    if hotels_df.empty:
        print("Hotel tablosu boş geldi.")
        return

    if reviews_df.empty:
        print("Review tablosu boş geldi.")
        return

    review_counts = (
        reviews_df.groupby("hotel_id").size().reset_index(name="review_count_total")
    )

    eligible_counts = review_counts[
        review_counts["review_count_total"] >= MIN_REVIEW_COUNT
    ].copy()

    eligible_hotel_ids = set(eligible_counts["hotel_id"].astype(str))

    print("Minimum review eşiğini geçen otel sayısı:", len(eligible_hotel_ids))

    reliable_hotels_df = hotels_df[
        hotels_df["hotel_id"].astype(str).isin(eligible_hotel_ids)
    ].copy()

    balanced_reviews_df = create_balanced_review_subset(
        reviews_df=reviews_df, eligible_hotel_ids=eligible_hotel_ids
    )

    if balanced_reviews_df.empty:
        print("Balanced review subset boş oluştu.")
        return

    used_counts = (
        balanced_reviews_df.groupby("hotel_id")
        .size()
        .reset_index(name="review_count_used")
    )

    reliable_hotels_df = reliable_hotels_df.merge(
        eligible_counts, on="hotel_id", how="left"
    )

    reliable_hotels_df = reliable_hotels_df.merge(
        used_counts, on="hotel_id", how="left"
    )

    reliable_hotels_df["review_count_used"] = (
        reliable_hotels_df["review_count_used"].fillna(0).astype(int)
    )

    merged_df = balanced_reviews_df.merge(reliable_hotels_df, on="hotel_id", how="left")

    reliable_hotels_df = reliable_hotels_df.sort_values(
        by=["review_count_total", "hotel_name"], ascending=[False, True]
    ).reset_index(drop=True)

    balanced_reviews_df = balanced_reviews_df.sort_values(
        by=["hotel_id", "review_id"]
    ).reset_index(drop=True)

    merged_df = merged_df.sort_values(by=["hotel_id", "review_id"]).reset_index(
        drop=True
    )

    reliable_hotels_df.to_csv(HOTELS_OUT, index=False, encoding="utf-8")
    balanced_reviews_df.to_csv(REVIEWS_OUT, index=False, encoding="utf-8")
    merged_df.to_csv(MERGED_OUT, index=False, encoding="utf-8")

    hotel_count = len(reliable_hotels_df)
    total_used_reviews = len(balanced_reviews_df)

    avg_reviews_used = (
        reliable_hotels_df["review_count_used"].mean() if hotel_count > 0 else 0
    )

    median_reviews_used = (
        reliable_hotels_df["review_count_used"].median() if hotel_count > 0 else 0
    )

    possible_duplicate_groups = (
        reliable_hotels_df.groupby("hotel_name_normalized")["hotel_name"]
        .nunique()
        .reset_index(name="raw_name_count")
    )

    duplicate_group_count = (possible_duplicate_groups["raw_name_count"] > 1).sum()

    rating_available = int(
        balanced_reviews_df["overall_rating"]
        .astype(str)
        .str.strip()
        .replace("nan", "")
        .ne("")
        .sum()
    )

    report = f"""TravelMind RAG - CMU Reliable Subset Report

1. Threshold Policy

Minimum review count per hotel: {MIN_REVIEW_COUNT}
Maximum kept reviews per hotel: {MAX_REVIEWS_PER_HOTEL}

2. Raw Input

Offering records loaded: {len(hotels_df)}
Review records loaded: {len(reviews_df)}
Skipped offering lines: {skipped_offering}
Skipped review lines: {skipped_review}

3. Reliable Subset

Reliable hotel count: {hotel_count}
Balanced review count kept: {total_used_reviews}

Average kept reviews per hotel: {avg_reviews_used:.2f}
Median kept reviews per hotel: {median_reviews_used:.2f}

4. Hotel Name Normalization

Possible duplicate normalized hotel-name groups in selected subset: {duplicate_group_count}

5. Rating Availability

Reviews with extracted overall rating: {rating_available}
Reviews without extracted overall rating: {total_used_reviews - rating_available}

6. Interpretation

- Only hotels with at least {MIN_REVIEW_COUNT} total reviews were kept.
- At most {MAX_REVIEWS_PER_HOTEL} reviews per hotel were kept.
- This prevents very popular hotels from dominating the dataset.
- Hotel names were normalized with lowercase and punctuation removal.
- This subset is more suitable than the previous sparse datasets for RAG-based hotel recommendation.

7. Output Files

Hotels file:
{HOTELS_OUT}

Reviews file:
{REVIEWS_OUT}

Merged review+hotel file:
{MERGED_OUT}
"""

    REPORT_OUT.write_text(report, encoding="utf-8")

    print("\nSubset oluşturuldu.")
    print(f"Reliable hotel sayısı: {hotel_count}")
    print(f"Kullanılan review sayısı: {total_used_reviews}")
    print(f"Hotels dosyası: {HOTELS_OUT}")
    print(f"Reviews dosyası: {REVIEWS_OUT}")
    print(f"Merged dosya: {MERGED_OUT}")
    print(f"Rapor: {REPORT_OUT}")


if __name__ == "__main__":
    main()
