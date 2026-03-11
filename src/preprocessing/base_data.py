from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "cleaned"

DROP_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp"
]

def load_csv(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    #fix column names (CIC IDS has leading spaces)
    df.columns = df.columns.str.strip()
    print("Shape: ", df.shape)
    return df

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    #drop ID and timestamp columns
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    print("Dropped ID and timestamp columns")

    #replace inf values with nan
    df = df.replace([np.inf, -np.inf], np.nan)
    print("Replaced inf and -inf with NaN")

    # calculate missing values percentage
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isna().sum().sum()
    missing_percent = (total_missing / total_cells) * 100
    print(f"Missing values: {total_missing} ({missing_percent:.4f}% of dataset)")

    #drop rows with NaN
    rows_before = df.shape[0]
    df = df.dropna()
    rows_after = df.shape[0]
    removed_nan_rows = rows_before - rows_after
    removed_nan_percent = (removed_nan_rows / rows_before) * 100
    print(f"Removed rows with NaN: {removed_nan_rows} ({removed_nan_percent:.4f}% of rows)")

    #drop duplicates
    rows_before = df.shape[0]
    df = df.drop_duplicates()
    rows_after = df.shape[0]
    removed_duplicate_rows = rows_before - rows_after
    duplicate_percent = (removed_duplicate_rows / rows_before) * 100
    print(f"Removed duplicated rows: {removed_duplicate_rows} ({duplicate_percent:.4f}% of rows)")

    return df

def process_file(file_path: Path, main_df: pd.DataFrame):
    print(f"Processing {file_path.name}")
    df = load_csv(file_path)
    df = clean_dataframe(df)
    print("Shape after cleaning: ", df.shape)
    #merge with main df
    main_df = pd.concat([main_df,df], ignore_index=True)
    return main_df

def main():
    files = list(RAW_DATA_DIR.glob("*.csv"))
    main_df = pd.DataFrame()
    for file in files:
        main_df = process_file(file, main_df)

    constant_columns = [col for col in main_df.columns if main_df[col].nunique(dropna=False) <= 1]
    main_df = main_df.drop(columns=constant_columns)
    print(f"Dropped constant columns after merge: {constant_columns}")

    output_path = OUTPUT_DIR / "cicids2017_preprocessed.csv"
    main_df.to_csv(output_path, index=False)
    print("Final dataset shape:", main_df.shape)
    print(f"Saved merged dataset to {output_path}")

if __name__ == "__main__":
    main()