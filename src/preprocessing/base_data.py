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
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    print("Dropped ID and timestamp columns")
    constant_columns = [col for col in df.columns if df[col].nunique() <= 1]
    df = df.drop(columns=constant_columns)
    print("Dropped columns with constant values")
    df = df.replace([np.inf, -np.inf], np.nan)
    print("Replaced inf and -inf with NaN")
    df = df.dropna()
    print("Dropped NaN")
    df = df.drop_duplicates()
    print("Dropped duplicate rows")
    return df

def process_file(file_path: Path):
    print(f"Processing {file_path.name}")

    df = load_csv(file_path)
    df = clean_dataframe(df)
    print("Shape after cleaning: ", df.shape)
    output_path = OUTPUT_DIR / file_path.name
    df.to_csv(output_path, index=False)

    print(f"Saved cleaned file to {output_path}")

def main():
    files = list(RAW_DATA_DIR.glob("*.csv"))

    for file in files:
        process_file(file)

if __name__ == "__main__":
    main()