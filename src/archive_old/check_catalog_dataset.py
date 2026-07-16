import pandas as pd
from pathlib import Path

CATALOG_DIR = Path("data/raw/hotel_catalog")

files = [CATALOG_DIR / "booking_hotel.csv", CATALOG_DIR / "tripadvisor_room.csv"]


def read_csv_safely(file_path):
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

    for enc in encodings:
        try:
            df = pd.read_csv(
                file_path, encoding=enc, sep=None, engine="python", on_bad_lines="skip"
            )
            print(f"Okuma başarılı. Kullanılan encoding: {enc}")
            return df
        except UnicodeDecodeError:
            print(f"Encoding başarısız: {enc}")
        except Exception as e:
            print(f"{enc} ile farklı hata oluştu: {e}")

    raise Exception(f"Dosya okunamadı: {file_path}")


for file in files:
    print("\n" + "=" * 80)
    print("Dosya:", file)

    df = read_csv_safely(file)

    print("Satır / sütun:", df.shape)

    print("\nKolonlar:")
    for col in df.columns:
        print("-", col)

    print("\nEksik değer sayısı:")
    print(df.isna().sum())

    print("\nİlk 3 satır:")
    print(df.head(3))
