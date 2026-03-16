from pathlib import Path
import pandas as pd
from utils.logger import setup_logger
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "cleaned" / "cicids2017_preprocessed.csv"
OUTPUT_DIR = BASE_DIR / "data" / "cleaned"
OUTPUT_PATH = OUTPUT_DIR / "cicids2017_targets.csv"
LOG_PATH = BASE_DIR / "logs" / "target_data_preprocessing.log"

TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42

logger = setup_logger(LOG_PATH)

def load_csv(file_path: Path) -> pd.DataFrame:
    logger.info(f"Loading dataset from: {file_path}")
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

def prepare_targets() -> pd.DataFrame:
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

    return df

def split_x_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=["Target", "Label"])
    y = df["Target"]
    logger.info(f"Prepared dataframes: X shape={X.shape}, y shape={y.shape}")

    return X, y

def split_data(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    logger.info(f"Starting train/validation/test split with parameters: test_size={TEST_SIZE}, val_size={VAL_SIZE}, random_state={RANDOM_STATE}")

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    val_relative_size = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_relative_size, stratify=y_temp, random_state=RANDOM_STATE)

    logger.info(f"Completed split: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}, y_train={y_train.shape}, y_val={y_val.shape}, y_test={y_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test

def log_split_target_distribution(y_train: pd.Series, y_val: pd.Series,y_test: pd.Series) -> None:
    split_targets = {
        "train": y_train,
        "validation": y_val,
        "test": y_test,
    }

    for split_name, split_target in split_targets.items():
        logger.info(f"Target distribution for {split_name} split (counts):")
        counts = split_target.value_counts(dropna=False).sort_index()

        for target_value, count in counts.items():
            label_name = "BENIGN" if target_value == 0 else "ATTACK"
            logger.info(f"  {target_value} ({label_name}): {count}")

        logger.info(f"Target distribution for {split_name} split (percentages):")
        percentages = split_target.value_counts(normalize=True, dropna=False).sort_index() * 100

        for target_value, percentage in percentages.items():
            label_name = "BENIGN" if target_value == 0 else "ATTACK"
            logger.info(f"  {target_value} ({label_name}): {percentage:.4f}%")

def save_split_datasets(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                        y_val: pd.Series, y_test: pd.Series):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    X_val.to_csv(OUTPUT_DIR / "X_val.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)

    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_val.to_csv(OUTPUT_DIR / "y_val.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

    logger.info(f"Saved split datasets to directory: {OUTPUT_DIR}")

def main():
    main_df = prepare_targets()
    X, y = split_x_y(main_df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    save_split_datasets(X_train, X_val, X_test, y_train, y_val, y_test)
    logger.info(f"Final split sizes: train={len(X_train)}, validation={len(X_val)}, test={len(X_test)}")

if __name__ == "__main__":
    main()