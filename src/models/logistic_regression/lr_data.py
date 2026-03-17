from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.logger import setup_logger

BASE_DIR = Path(__file__).resolve().parents[3]

DATASET_FILE_MAP = {
    "easy": "cicids2017_easy.csv",
    "medium": "cicids2017_medium.csv",
    "hard": "cicids2017_hard.csv",
}

def get_logger(config: dict):
    log_path = BASE_DIR / config["logging"]["log_path"]
    return setup_logger(log_path)

def get_dataset_path(config: dict, logger) -> Path:
    input_dir = BASE_DIR / config["data"]["input_dir"]
    dataset_variant = config["data"]["dataset_variant"]

    if dataset_variant not in DATASET_FILE_MAP:
        logger.critical(f"Invalid dataset variant: {dataset_variant}. Expected one of {list(DATASET_FILE_MAP.keys())}. Exiting")
        exit(1)

    return input_dir / DATASET_FILE_MAP[dataset_variant]

def load_dataset(dataset_path: Path, logger) -> pd.DataFrame:
    logger.info(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path, memory_map=True, low_memory=False)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    return df

def get_split_paths(config: dict) -> tuple[Path, Path, Path]:
    split_dir = BASE_DIR / config["split"]["split_dir"]
    return split_dir / "train.csv", split_dir / "val.csv", split_dir / "test.csv"

def split_dataset(df: pd.DataFrame, config: dict, logger) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_column = config["data"]["target_column"]
    random_state = config["experiment"]["random_state"]
    test_size = config["split"]["test_size"]
    val_size = config["split"]["val_size"]
    stratify_enabled = config["split"]["stratify"]

    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found in dataset")

    stratify_data = df[target_column] if stratify_enabled else None

    logger.info("Creating train/test split")
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state,
                                             stratify=stratify_data)
    logger.info(f"Train+Val shape: {train_val_df.shape}")
    logger.info(f"Test shape: {test_df.shape}")

    val_relative_size = val_size / (1 - test_size)
    stratify_train_val = train_val_df[target_column] if stratify_enabled else None

    logger.info("Creating train/validation split")
    train_df, val_df = train_test_split(train_val_df, test_size=val_relative_size, random_state=random_state,
                                        stratify=stratify_train_val)
    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Validation shape: {val_df.shape}")

    return train_df, val_df, test_df

def save_split_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, config: dict, logger) -> None:
    split_dir = BASE_DIR / config["split"]["split_dir"]
    train_path, val_path, test_path = get_split_paths(config)

    split_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_path, index=False)
    logger.info(f"Saved train split to: {train_path}")

    val_df.to_csv(val_path, index=False)
    logger.info(f"Saved validation split to: {val_path}")

    test_df.to_csv(test_path, index=False)
    logger.info(f"Saved test split to: {test_path}")

def prepare_lr_data(config: dict) -> None:
    logger = get_logger(config)

    dataset_path = get_dataset_path(config, logger)
    df = load_dataset(dataset_path, logger)
    train_df, val_df, test_df = split_dataset(df, config, logger)
    save_split_data(train_df, val_df, test_df, config, logger)