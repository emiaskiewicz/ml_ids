from pathlib import Path
import pandas as pd
from utils.logger import setup_logger

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "cleaned" / "cicids2017_preprocessed.csv"
OUTPUT_PATH = BASE_DIR / "data" / "cleaned" / "cicids2017_targets.csv"
LOG_PATH = BASE_DIR / "logs" / "target_data_preprocessing.log"

logger = setup_logger(LOG_PATH)

def load_csv(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, memory_map=True, low_memory=False)
    logger.info(f"Loaded {file_path.name} with shape {df.shape}")
    return df

def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    df["Target"] = (df["Label"] != "BENIGN").astype(int)
    logger.info("Created binary target column: BENIGN -> 0, ATTACK -> 1")
    return df

def log_binary_target_distribution(df: pd.DataFrame) -> None:
    logger.info("Binary target distribution (counts):")
    target_counts = df["Target"].value_counts(dropna=False).sort_index()

    for target_value, count in target_counts.items():
        label_name = "BENIGN" if target_value == 0 else "ATTACK"
        logger.info(f"{target_value} ({label_name}): {count}")

    logger.info("Binary target distribution (percentages):")
    target_percentages = df["Target"].value_counts(normalize=True, dropna=False).sort_index() * 100

    for target_value, percentage in target_percentages.items():
        label_name = "BENIGN" if target_value == 0 else "ATTACK"
        logger.info(f"{target_value} ({label_name}): {percentage:.4f}%")

def main():
    df = load_csv(INPUT_PATH)

    logger.info("Original label distribution (counts):")
    class_counts = df["Label"].value_counts(dropna=False)

    for label, count in class_counts.items():
        logger.info(f"{label}: {count}")

    df = create_binary_target(df)
    log_binary_target_distribution(df)
    logger.info(f"Final dataset shape: {df.shape}")

    df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Saved processed dataset to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()