from pathlib import Path
import pandas as pd
import numpy as np
import logging

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "cleaned"
LOG_PATH = BASE_DIR / "logs" / "base_data_preprocessing.log"

DROP_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp"
]

def setup_logger():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler()
        ]
    )

def load_csv(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    #fix column names (CIC IDS has leading spaces)
    df.columns = df.columns.str.strip()
    logging.info(f"Loaded {file_path.name} with shape {df.shape}")
    return df

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    #drop ID and timestamp columns
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    logging.info("Dropped ID and timestamp columns")

    #replace inf values with nan
    df = df.replace([np.inf, -np.inf], np.nan)
    logging.info("Replaced inf and -inf with NaN")

    # calculate missing values percentage
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isna().sum().sum()
    missing_percent = (total_missing / total_cells) * 100
    logging.info(f"Missing values: {total_missing} ({missing_percent:.4f}% of dataset)")

    #drop rows with NaN
    rows_before = df.shape[0]
    df = df.dropna()
    rows_after = df.shape[0]
    removed_nan_rows = rows_before - rows_after
    removed_nan_percent = (removed_nan_rows / rows_before) * 100
    logging.info(f"Removed rows with NaN: {removed_nan_rows} ({removed_nan_percent:.4f}% of rows)")

    #drop duplicates
    rows_before = df.shape[0]
    df = df.drop_duplicates()
    rows_after = df.shape[0]
    removed_duplicate_rows = rows_before - rows_after
    duplicate_percent = (removed_duplicate_rows / rows_before) * 100
    logging.info(f"Removed duplicated rows: {removed_duplicate_rows} ({duplicate_percent:.4f}% of rows)")

    return df

def process_file(file_path: Path, dfs: list):
    logging.info(f"Processing file: {file_path.name}")
    df = load_csv(file_path)
    df = clean_dataframe(df)
    logging.info(f"Shape after cleaning: {df.shape}")
    dfs.append(df)
    return dfs

def main():
    setup_logger()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = list(RAW_DATA_DIR.glob("*.csv"))
    logging.info(f"Found {len(files)} files to process")
    dfs = []

    for file in files:
        dfs = process_file(file, dfs)
    main_df = pd.concat(dfs, ignore_index=True)

    constant_columns = [col for col in main_df.columns if main_df[col].nunique(dropna=False) <= 1]
    main_df = main_df.drop(columns=constant_columns)
    logging.info(f"Dropped constant columns after merge: {constant_columns}")

    output_path = OUTPUT_DIR / "cicids2017_preprocessed.csv"
    main_df.to_csv(output_path, index=False)
    logging.info(f"Final dataset shape: {main_df.shape}")
    logging.info(f"Saved merged dataset to {output_path}")

if __name__ == "__main__":
    main()