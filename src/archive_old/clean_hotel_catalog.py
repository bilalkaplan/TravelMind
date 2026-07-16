import re
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/hotel_catalog")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "hotel_catalog_clean.csv"

BOOKING_PATH = RAW_DIR / "booking_hotel.csv"
TRIPADVISOR_PATH = RAW_DIR / "tripadvisor_room.csv"


def read_csv_safely(path):
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

    for enc in encodings:
        try:
            df = pd.read_csv(
                path, encoding=enc, sep=None, engine="python", on_bad_lines="skip"
            )
            df.columns = [str(col).strip() for col in df.columns]
            print(f"{path.name} okundu. Encoding: {enc}")
            return df
        except UnicodeDecodeError:
            continue

    raise Exception(f"Dosya okunamadı: {path}")


def normalize_col_name(name):
    name = str(name).strip().lower()
    name = re.sub(r"\s+", " ", name)
    return name


def get_column(df, possible_names, default_value=""):
    normalized_map = {normalize_col_name(col): col for col in df.columns}

    for name in possible_names:
        key = normalize_col_name(name)
        if key in normalized_map:
            return df[normalized_map[key]]

    return pd.Series([default_value] * len(df))


def clean_text(value):
    if pd.isna(value):
        return ""
    value = str(value)
    value = re.sub(r"<.*?>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def clean_price(value):
    if pd.isna(value):
        return None

    value = str(value)
    value = value.replace(",", "")
    value = re.sub(r"[^\d.]", "", value)

    if value == "":
        return None

    try:
        return float(value)
    except ValueError:
        return None


def clean_number(value):
    if pd.isna(value):
        return None

    value = str(value)
    value = value.replace(",", "")
    value = re.sub(r"[^\d.]", "", value)

    if value == "":
        return None

    try:
        return float(value)
    except ValueError:
        return None


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    booking = read_csv_safely(BOOKING_PATH)
    tripadvisor = read_csv_safely(TRIPADVISOR_PATH)

    print("Booking veri boyutu:", booking.shape)
    print("TripAdvisor veri boyutu:", tripadvisor.shape)

    print("\nBooking kolonları:")
    print(list(booking.columns))

    print("\nTripAdvisor kolonları:")
    print(list(tripadvisor.columns))

    booking_clean = pd.DataFrame()
    booking_clean["hotel_name"] = get_column(booking, ["Hotel Name"]).apply(clean_text)
    booking_clean["location"] = get_column(booking, ["Location"]).apply(clean_text)
    booking_clean["hotel_rating"] = get_column(booking, ["Rating"], None).apply(
        clean_number
    )
    booking_clean["review_score"] = get_column(booking, ["Review Score"], None).apply(
        clean_number
    )
    booking_clean["room_score"] = get_column(
        booking, ["Room Score", "Room    Score"], None
    ).apply(clean_number)
    booking_clean["review_count"] = get_column(booking, ["Number of"], None).apply(
        clean_number
    )
    booking_clean["room_type"] = get_column(booking, ["Room Type"]).apply(clean_text)
    booking_clean["bed_type"] = get_column(booking, ["Bed Type"]).apply(clean_text)
    booking_clean["room_price"] = get_column(
        booking, ["Room Price (in BDT or any other currency)"], None
    ).apply(clean_price)
    booking_clean["room_comment"] = ""
    booking_clean["source"] = "booking_hotel"

    trip_clean = pd.DataFrame()
    trip_clean["hotel_name"] = get_column(
        tripadvisor, ["property name", "property name "]
    ).apply(clean_text)
    trip_clean["location"] = ""
    trip_clean["hotel_rating"] = None
    trip_clean["review_score"] = None
    trip_clean["room_score"] = None
    trip_clean["review_count"] = get_column(tripadvisor, ["review_count"], None).apply(
        clean_number
    )
    trip_clean["room_type"] = ""
    trip_clean["bed_type"] = ""
    trip_clean["room_price"] = get_column(
        tripadvisor, ["Room Price (in BDT or any other currency)"], None
    ).apply(clean_price)
    trip_clean["room_comment"] = get_column(tripadvisor, ["Comment about room"]).apply(
        clean_text
    )
    trip_clean["source"] = "tripadvisor_room"

    df = pd.concat([booking_clean, trip_clean], ignore_index=True)

    df = df[df["hotel_name"].str.len() > 0]

    df = df.drop_duplicates(
        subset=[
            "hotel_name",
            "location",
            "room_type",
            "bed_type",
            "room_price",
            "room_comment",
            "source",
        ]
    )

    df = df.reset_index(drop=True)
    df.insert(0, "hotel_id", df.index + 1)

    df = df[
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
            "room_price",
            "room_comment",
            "source",
        ]
    ]

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("\nTemiz katalog veri boyutu:", df.shape)
    print("Kaydedilen dosya:", OUTPUT_PATH)

    print("\nKaynak dağılımı:")
    print(df["source"].value_counts())

    print("\nEksik değer sayısı:")
    print(df.isna().sum())

    print("\nİlk 5 kayıt:")
    print(df.head())


if __name__ == "__main__":
    main()
