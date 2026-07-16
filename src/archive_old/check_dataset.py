import pandas as pd
from pathlib import Path

csv_path = Path("data/raw/tripadvisor_hotel_reviews.csv")

df = pd.read_csv(csv_path)

print("Dosya okundu:", csv_path)
print("Satır / sütun:", df.shape)
print("Kolonlar:", list(df.columns))

print("\nEksik değer sayısı:")
print(df.isna().sum())

print("\nİlk 5 satır:")
print(df.head())
