import json
from collections import Counter
from pathlib import Path

RAW_DIR = Path("data/raw/cmu_tripadvisor")
OUTPUT_DIR = Path("data/processed")

REPORT_PATH = OUTPUT_DIR / "cmu_dataset_initial_audit_report.txt"
REVIEW_DENSITY_PATH = OUTPUT_DIR / "cmu_review_density_summary.csv"
OFFERING_SAMPLE_PATH = OUTPUT_DIR / "cmu_offering_sample.json"
REVIEW_SAMPLE_PATH = OUTPUT_DIR / "cmu_review_sample.json"


def find_file(keyword):
    matches = list(RAW_DIR.glob(f"*{keyword}*.txt"))

    if not matches:
        matches = list(RAW_DIR.glob(f"*{keyword}*"))

    matches = [
        path for path in matches if path.is_file() and not path.name.endswith(".zip")
    ]

    if not matches:
        return None

    return matches[0]


def parse_json_line(line):
    line = line.strip()

    if not line:
        return None

    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def read_json_lines_sample(path, limit=5):
    samples = []

    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            item = parse_json_line(line)

            if item is not None:
                samples.append(item)

            if len(samples) >= limit:
                break

    return samples


def count_json_lines(path):
    count = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if parse_json_line(line) is not None:
                count += 1

    return count


def get_review_offering_id(review):
    possible_keys = ["offering_id", "offeringId", "hotel_id", "hotelId"]

    for key in possible_keys:
        if key in review:
            return str(review[key])

    return ""


def get_offering_id(offering):
    possible_keys = ["id", "offering_id", "offeringId", "hotel_id", "hotelId"]

    for key in possible_keys:
        if key in offering:
            return str(offering[key])

    return ""


def get_hotel_name(offering):
    possible_keys = ["name", "hotel_name", "hotelName", "title"]

    for key in possible_keys:
        if key in offering:
            return str(offering[key])

    return ""


def get_location_text(offering):
    possible_keys = ["address", "location", "city", "region", "state", "country"]

    values = []

    for key in possible_keys:
        value = offering.get(key, "")

        if isinstance(value, dict):
            values.append(json.dumps(value, ensure_ascii=False))
        elif str(value).strip():
            values.append(str(value).strip())

    return " | ".join(values)


def analyze_review_density(review_path):
    review_counter = Counter()
    total_reviews = 0
    missing_offering_id = 0

    with open(review_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            review = parse_json_line(line)

            if review is None:
                continue

            total_reviews += 1

            offering_id = get_review_offering_id(review)

            if offering_id:
                review_counter[offering_id] += 1
            else:
                missing_offering_id += 1

            if total_reviews % 100000 == 0:
                print(f"Review okundu: {total_reviews}")

    return total_reviews, missing_offering_id, review_counter


def write_review_density_csv(review_counter):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(REVIEW_DENSITY_PATH, "w", encoding="utf-8") as file:
        file.write("offering_id,review_count\n")

        for offering_id, count in review_counter.most_common():
            file.write(f"{offering_id},{count}\n")


def calculate_density_stats(review_counter):
    counts = list(review_counter.values())

    if not counts:
        return {
            "hotels_with_reviews": 0,
            "min_reviews": 0,
            "max_reviews": 0,
            "avg_reviews": 0,
            "median_reviews": 0,
            "hotels_5_plus": 0,
            "hotels_10_plus": 0,
            "hotels_25_plus": 0,
            "hotels_50_plus": 0,
            "hotels_100_plus": 0,
        }

    counts_sorted = sorted(counts)
    n = len(counts_sorted)

    if n % 2 == 1:
        median = counts_sorted[n // 2]
    else:
        median = (counts_sorted[n // 2 - 1] + counts_sorted[n // 2]) / 2

    return {
        "hotels_with_reviews": len(counts),
        "min_reviews": min(counts),
        "max_reviews": max(counts),
        "avg_reviews": sum(counts) / len(counts),
        "median_reviews": median,
        "hotels_5_plus": sum(1 for count in counts if count >= 5),
        "hotels_10_plus": sum(1 for count in counts if count >= 10),
        "hotels_25_plus": sum(1 for count in counts if count >= 25),
        "hotels_50_plus": sum(1 for count in counts if count >= 50),
        "hotels_100_plus": sum(1 for count in counts if count >= 100),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    offering_path = find_file("offering")
    review_path = find_file("review")

    if offering_path is None:
        print("offering dosyası bulunamadı.")
        print("Önce zipleri çıkarıp data/raw/cmu_tripadvisor içine koy.")
        return

    if review_path is None:
        print("review dosyası bulunamadı.")
        print("Önce zipleri çıkarıp data/raw/cmu_tripadvisor içine koy.")
        return

    print("Offering dosyası:", offering_path)
    print("Review dosyası:", review_path)

    print("\nÖrnek offering kayıtları okunuyor...")
    offering_samples = read_json_lines_sample(offering_path, limit=5)

    print("Örnek review kayıtları okunuyor...")
    review_samples = read_json_lines_sample(review_path, limit=5)

    OFFERING_SAMPLE_PATH.write_text(
        json.dumps(offering_samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    REVIEW_SAMPLE_PATH.write_text(
        json.dumps(review_samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nOffering toplam kayıt sayısı hesaplanıyor...")
    total_offerings = count_json_lines(offering_path)

    print("\nReview yoğunluğu hesaplanıyor...")
    total_reviews, missing_offering_id, review_counter = analyze_review_density(
        review_path
    )

    write_review_density_csv(review_counter)

    stats = calculate_density_stats(review_counter)

    offering_field_names = (
        sorted(set().union(*(item.keys() for item in offering_samples)))
        if offering_samples
        else []
    )
    review_field_names = (
        sorted(set().union(*(item.keys() for item in review_samples)))
        if review_samples
        else []
    )

    example_offering = offering_samples[0] if offering_samples else {}

    example_hotel_id = get_offering_id(example_offering)
    example_hotel_name = get_hotel_name(example_offering)
    example_location = get_location_text(example_offering)

    report = f"""
TravelMind RAG - CMU TripAdvisor Dataset Initial Audit

1. Files

Offering file:
{offering_path}

Review file:
{review_path}

2. Dataset Size

Total offering / hotel records: {total_offerings}
Total review records: {total_reviews}
Reviews missing offering_id / hotel_id: {missing_offering_id}

3. Field Structure

Offering fields:
{offering_field_names}

Review fields:
{review_field_names}

4. Example Offering

Hotel ID:
{example_hotel_id}

Hotel name:
{example_hotel_name}

Location text:
{example_location}

Raw example offering saved to:
{OFFERING_SAMPLE_PATH}

5. Example Review

Raw example review saved to:
{REVIEW_SAMPLE_PATH}

6. Review Density Per Hotel

Hotels with at least 1 review: {stats["hotels_with_reviews"]}
Minimum reviews per reviewed hotel: {stats["min_reviews"]}
Maximum reviews for one hotel: {stats["max_reviews"]}
Average reviews per reviewed hotel: {stats["avg_reviews"]:.2f}
Median reviews per reviewed hotel: {stats["median_reviews"]}

Hotels with 5+ reviews: {stats["hotels_5_plus"]}
Hotels with 10+ reviews: {stats["hotels_10_plus"]}
Hotels with 25+ reviews: {stats["hotels_25_plus"]}
Hotels with 50+ reviews: {stats["hotels_50_plus"]}
Hotels with 100+ reviews: {stats["hotels_100_plus"]}

7. Initial Interpretation

This audit is used before embedding.
No vector database is built at this stage.

The next step is to choose a practical subset based on review_count distribution.
The minimum review threshold should be selected after looking at this distribution, not guessed blindly.

8. Generated Files

Review density summary:
{REVIEW_DENSITY_PATH}

Offering sample:
{OFFERING_SAMPLE_PATH}

Review sample:
{REVIEW_SAMPLE_PATH}
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    print(report)
    print("Rapor oluşturuldu:", REPORT_PATH)


if __name__ == "__main__":
    main()
