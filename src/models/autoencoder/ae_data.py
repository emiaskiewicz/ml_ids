from pathlib import Path
import pandas as pd
import numpy as np
from utils.logger import setup_logger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

BASE_DIR = Path(__file__).resolve().parents[3]

def get_logger(config: dict):
    log_path = BASE_DIR / config["logging"]["log_path"]
    return setup_logger(log_path)

def get_split_paths(config: dict) -> tuple[Path, Path, Path]:
    split_dir = BASE_DIR / config["split"]["split_dir"]
    return split_dir / "train.csv", split_dir / "val.csv", split_dir / "test.csv"

def load_existing_split_data(config: dict, logger) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path, val_path, test_path = get_split_paths(config)

    logger.info("Loading existing split data")
    train_df = pd.read_csv(train_path, engine="pyarrow")
    val_df = pd.read_csv(val_path, engine="pyarrow")
    test_df = pd.read_csv(test_path, engine="pyarrow")

    logger.info(f"Loaded train split shape: {train_df.shape}")
    logger.info(f"Loaded validation split shape: {val_df.shape}")
    logger.info(f"Loaded test split shape: {test_df.shape}")

    return train_df, val_df, test_df

def split_files_exist(config: dict) -> bool:
    train_path, val_path, test_path = get_split_paths(config)
    return train_path.exists() and val_path.exists() and test_path.exists()

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

def compute_correlation_matrix(df: pd.DataFrame, logger) -> pd.DataFrame:
    logger.info("Computing correlation matrix")
    corr_matrix = df.corr(numeric_only=True)
    logger.info(f"Correlation matrix shape: {corr_matrix.shape}")
    return corr_matrix

def plot_correlation_matrix(corr_matrix: pd.DataFrame, output_dir: Path, filename: str, logger):
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
    path = BASE_DIR / output_dir / Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved correlation heatmap to {path}")

def remove_correlated_features(corr_matrix: pd.DataFrame, threshold: float, logger) -> list[str]:
    upper = corr_matrix.abs().where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    logger.info(f"Removing {len(to_drop)} correlated features")
    if to_drop:
        logger.info(f"Removed features: {to_drop}")

    return to_drop

def get_scaler(scaler_name: str, logger):
    scaler_name = scaler_name.lower()

    if scaler_name == "standard":
        logger.info("Using StandardScaler")
        return StandardScaler()
    elif scaler_name == "minmax":
        logger.info("Using MinMaxScaler")
        return MinMaxScaler()
    elif scaler_name == "robust":
        logger.info("Using RobustScaler")
        return RobustScaler()
    else:
        #todo: zmienic na raise pozniej
        logger.critical(f"Unsupported scaler: {scaler_name}")
        exit(1)

def scale_datasets(X_train_normal: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, scaler_name: str,
                   logger) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scaler = get_scaler(scaler_name, logger)

    logger.info("Fitting scaler on X_train_normal")
    X_train_scaled = scaler.fit_transform(X_train_normal)

    logger.info("Transforming X_val and X_test with fitted scaler")
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_normal.columns, index=X_train_normal.index)
    logger.info(f"Scaled X_train_normal shape: {X_train_scaled.shape}")
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    logger.info(f"Scaled X_val shape: {X_val_scaled.shape}")
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    logger.info(f"Scaled X_test shape: {X_test_scaled.shape}")

    return X_train_scaled, X_val_scaled, X_test_scaled

def validate_numeric_data(X_train_normal: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, logger) -> None:
    for name, X in [("X_train_normal", X_train_normal), ("X_val", X_val), ("X_test", X_test)]:
        non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()

        if non_numeric_columns:
            logger.critical(f"{name} contains non-numeric columns: {non_numeric_columns}")
            exit(1)

        if not np.isfinite(X.to_numpy()).all():
            logger.critical(f"{name} contains NaN, inf or -inf values")
            exit(1)

        logger.info(f"{name} numeric validation passed")

def filter_normal_training_data(X_train: pd.DataFrame, y_train: pd.Series, normal_label: int, logger) -> tuple[pd.DataFrame, pd.Series]:
    logger.info(f"Filtering normal samples for autoencoder training. Normal label: {normal_label}")
    logger.info(f"Original training class distribution:\n{y_train.value_counts().sort_index()}")

    normal_mask = y_train == normal_label
    X_train_normal = X_train.loc[normal_mask].copy()
    y_train_normal = y_train.loc[normal_mask].copy()

    if X_train_normal.empty:
        logger.critical(f"No normal samples found for normal_label={normal_label}")
        exit(1)

    logger.info(f"X_train_normal shape: {X_train_normal.shape}")
    logger.info(f"y_train_normal shape: {y_train_normal.shape}")
    logger.info(f"Autoencoder training class distribution:\n{y_train_normal.value_counts().sort_index()}")

    return X_train_normal, y_train_normal

def prepare_ae_data(config: dict):
    logger = get_logger(config)
    logger.info(f"Preparing data for experiment: {config['experiment']['name']}")
    logger.info(f"Config: {config}")

    data_cfg = config["data"]
    output_cfg = config["output"]
    features_cfg = config["features"]
    split_cfg = config["split"]

    if split_cfg["load_existing_split"] and split_files_exist(config):
        logger.info(f"Loading existing split data")
        train_df, val_df, test_df = load_existing_split_data(config, logger)
        X_train, X_val, X_test, y_train, y_val, y_test = separate_features_and_target(train_df, val_df, test_df, data_cfg["target_column"], logger)
    else:
        logger.error(f"Loading existing split data failed")
        exit(1)

    X_train_normal, y_train_normal = filter_normal_training_data(X_train, y_train, data_cfg["normal_label"], logger)

    logger.info(f"Validation class distribution:\n{y_val.value_counts().sort_index()}")
    logger.info(f"Test class distribution:\n{y_test.value_counts().sort_index()}")

    corr_matrix=compute_correlation_matrix(X_train_normal, logger)
    plot_correlation_matrix(corr_matrix, output_cfg["output_dir"], "base_corr.jpg" ,logger)

    if features_cfg["remove_correlated_features"]:
        logger.info(f"Removing correlated features")
        to_drop = remove_correlated_features(corr_matrix, features_cfg["correlation_threshold"], logger)
        if to_drop:
            X_train_normal = X_train_normal.drop(columns=to_drop)
            X_val = X_val.drop(columns=to_drop)
            X_test = X_test.drop(columns=to_drop)

            corr_matrix_new = compute_correlation_matrix(X_train_normal, logger)
            plot_correlation_matrix(corr_matrix_new, output_cfg["output_dir"], "corr_after_remove.jpg", logger)
    else:
        logger.info("Removing correlated features is disabled")

    if features_cfg["scaling"]:
        logger.info("Scaling is enabled")
        X_train_normal, X_val, X_test = scale_datasets(X_train_normal, X_val, X_test, features_cfg["scaler"], logger)
    else:
        logger.info("Scaling is disabled")

    validate_numeric_data(X_train_normal, X_val, X_test, logger)

    return X_train_normal, X_val, X_test, y_train_normal, y_val, y_test