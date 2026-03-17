from pathlib import Path
import pandas as pd
from utils.logger import setup_logger

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "cleaned" / "cicids2017_targets.csv"
OUTPUT_DIR = BASE_DIR / "data" / "cleaned"
EASY_OUTPUT = OUTPUT_DIR / "cicids2017_easy.csv"
MEDIUM_OUTPUT = OUTPUT_DIR / "cicids2017_medium.csv"
HARD_OUTPUT = OUTPUT_DIR / "cicids2017_hard.csv"
LOG_PATH = BASE_DIR / "logs" / "variants_split_preprocessing.log"

EASY_THRESHOLD = 1.0
MEDIUM_THRESHOLD = 0.1

logger = setup_logger(LOG_PATH)

def load_csv(file_path: Path) -> pd.DataFrame:
    logger.info(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path, memory_map=True, low_memory=False)
    logger.info(f"Loaded {file_path.name} with shape {df.shape}")
    return df

def compute_label_percentages(df: pd.DataFrame) -> pd.Series:
    logger.info("Computing label distribution (percentages)")
    percentages = df["Label"].value_counts(normalize=True) * 100
    for label, pct in percentages.items():
        logger.info(f"{label}: {pct:.4f}%")

    return percentages

def get_labels_by_threshold(percentages: pd.Series, threshold: float) -> list:
    labels = percentages[percentages > threshold].index.tolist()
    logger.info(f"Labels with percentage > {threshold}%: {labels}")
    return labels

def create_dataset_variant(df: pd.DataFrame, labels: list | None) -> pd.DataFrame:
    if labels is None:
        logger.info("Creating HARD dataset (all labels)")
        return df.copy()

    labels = set(labels)
    labels.add("BENIGN")
    logger.info(f"Creating dataset with labels: {sorted(labels)}")
    filtered_df = df[df["Label"].isin(labels)].copy()
    logger.info(f"Filtered dataset shape: {filtered_df.shape}")

    return filtered_df

def save_dataset(df: pd.DataFrame, output_path: Path, name: str) -> None:
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {name} dataset to {output_path} (shape: {df.shape})")

def generate_variants() -> None:
    df = load_csv(INPUT_PATH)

    percentages = compute_label_percentages(df)
    easy_labels = get_labels_by_threshold(percentages, EASY_THRESHOLD)
    medium_labels = get_labels_by_threshold(percentages, MEDIUM_THRESHOLD)

    df_easy = create_dataset_variant(df, easy_labels)
    save_dataset(df_easy, EASY_OUTPUT, "EASY")

    df_medium = create_dataset_variant(df, medium_labels)
    save_dataset(df_medium, MEDIUM_OUTPUT, "MEDIUM")

    df_hard = create_dataset_variant(df, None)
    save_dataset(df_hard, HARD_OUTPUT, "HARD")

def main():
    generate_variants()

if __name__ == "__main__":
    main()