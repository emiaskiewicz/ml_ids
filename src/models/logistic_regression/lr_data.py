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
        #todo: dodac try/except w lr_model.py
        #raise ValueError(f"Invalid dataset variant: {dataset_variant}. Expected one of {list(DATASET_FILE_MAP.keys())}")
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
        #todo: dodac try/except w lr_model.py
        #raise ValueError(f"Target column {target_column} not found in dataset.")
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

def load_existing_split_data(config: dict, logger) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path, val_path, test_path = get_split_paths(config)

    logger.info("Loading existing split data")
    train_df = pd.read_csv(train_path, memory_map=True, low_memory=False)
    val_df = pd.read_csv(val_path, memory_map=True, low_memory=False)
    test_df = pd.read_csv(test_path, memory_map=True, low_memory=False)

    logger.info(f"Loaded train split shape: {train_df.shape}")
    logger.info(f"Loaded validation split shape: {val_df.shape}")
    logger.info(f"Loaded test split shape: {test_df.shape}")

    return train_df, val_df, test_df

def split_files_exist(config: dict) -> bool:
    train_path, val_path, test_path = get_split_paths(config)
    return train_path.exists() and val_path.exists() and test_path.exists()

def drop_feature_columns(df: pd.DataFrame, columns_to_drop: list[str], target_column: str, logger) -> pd.DataFrame:
    valid_columns_to_drop = [col for col in columns_to_drop if col in df.columns and col != target_column]

    if valid_columns_to_drop:
        logger.info(f"Dropping feature columns: {valid_columns_to_drop}")
        df = df.drop(columns=valid_columns_to_drop)
    else:
        logger.info("No feature columns were dropped")

    logger.info(f"Dataset shape after dropping columns: {df.shape}")
    return df

def separate_features_and_target(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                                 target_column: str,logger) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    X_val = val_df.drop(columns=[target_column])
    y_val = val_df[target_column]
    logger.info(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]
    logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test

def prepare_lr_data(config: dict):
    logger = get_logger(config)
    logger.info(f"Preparing data for experiment: {config['experiment']['name']}")
    logger.info(f"Config: {config}")

    target_column = config["data"]["target_column"]
    columns_to_drop = config["features"]["drop_columns"]
    load_existing_split = config["split"]["load_existing_split"]
    save_split = config["split"]["save_split"]
    force_regenerate_split = config["split"]["force_regenerate_split"]

    if force_regenerate_split:
        logger.info(f"force_regenerate_split: true, creating new split")
        dataset_path = get_dataset_path(config, logger)
        df = load_dataset(dataset_path, logger)
        df = drop_feature_columns(df, columns_to_drop, target_column, logger)
        train_df, val_df, test_df = split_dataset(df, config, logger)

        if save_split:
            save_split_data(train_df, val_df, test_df, config, logger)
    elif load_existing_split and split_files_exist(config):
        logger.info(f"Loading existing split data")
        train_df, val_df, test_df = load_existing_split(config, logger)
    else:
        logger.info(f"Creating new split")
        dataset_path = get_dataset_path(config, logger)
        df = load_dataset(dataset_path, logger)
        df = drop_feature_columns(df, columns_to_drop, target_column, logger)
        train_df, val_df, test_df = split_dataset(df, config, logger)

        if save_split:
            save_split_data(train_df, val_df, test_df, config, logger)

    return separate_features_and_target(train_df, val_df, test_df, target_column, logger)