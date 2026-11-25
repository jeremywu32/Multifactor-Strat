
import pandas as pd
from pathlib import Path

# 1. Load local Parquet file
file_path = Path("sp500_data/sp500_full_data.pq")
full_data = pd.read_parquet(file_path)

# 2. Basic information
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print(f"Shape: {full_data.shape}")
print(f"Number of stocks: {full_data['permno'].nunique():,}")
print(f"Date range: {full_data['dlycaldt'].min()} "
      f"to {full_data['dlycaldt'].max()}")

# 3. DataFrame schema
print("\nDATAFRAME SCHEMA")
full_data.info(show_counts=True, memory_usage="deep")

# 4. Column information
schema = pd.DataFrame({
    "column": full_data.columns,
    "dtype": full_data.dtypes.astype(str).values,
    "missing_count": full_data.isna().sum().values,
    "missing_pct": (
        full_data.isna().mean() * 100
    ).round(2).values
})

print("\nCOLUMN SUMMARY")
print(schema.to_string(index=False))

# 5. Preview data
print("\nFIRST 5 ROWS")
print(full_data.head().to_string(index=False))

# 6. Check duplicates
duplicates = full_data.duplicated(
    subset=["permno", "dlycaldt"]
).sum()

print(f"\nDuplicate stock-date observations: {duplicates}")


df = full_data.copy()

# 1. Rows with missing OHLC
missing = df[df["dlyopen"].isna()].copy()

print("Total missing OHLC rows:", len(missing))

# 2. Check whether all OHLC fields are missing together
ohlc_cols = ["dlyopen", "dlyhigh", "dlylow", "dlyclose"]

print(
    "\nAll four OHLC missing:",
    missing[ohlc_cols].isna().all(axis=1).sum()
)

# 3. Check delisting flags
print("\nDelisting flag distribution:")
print(missing["dlydelflg"].value_counts(dropna=False))

# 4. Check available price / return / volume
print("\nMissing OHLC but other fields available:")

for col in ["dlyprc", "dlyret", "dlyvol"]:
    count = missing[col].notna().sum()
    print(f"{col}: {count:,}")

# 5. Inspect missing observations
cols = [
    "permno",
    "ticker",
    "dlycaldt",
    "dlyopen",
    "dlyclose",
    "dlyprc",
    "dlyret",
    "dlyvol",
    "dlydelflg"
]

print("\nSample missing observations:")
print(missing[cols].head(30).to_string(index=False))

# 6. Stocks with the most missing observations
print("\nTop 15 stocks with missing OHLC:")
print(
    missing.groupby("permno")
    .size()
    .sort_values(ascending=False)
    .head(15)
)

