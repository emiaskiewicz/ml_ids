from pathlib import Path
import pandas as pd
import numpy as np
from utils.logger import setup_logger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold

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

def get_log_transform_columns(X_train_normal: pd.DataFrame, feat_cfg, logger) -> list[str]:
    if feat_cfg == "auto":
        excluded_prefixes = ("FIN_", "SYN_", "RST_", "PSH_", "ACK_", "URG_", "CWE_", "ECE_")
        excluded_columns = {"Protocol", "Source_Port", "Destination_Port"}
        selected_columns = []

        for column in X_train_normal.columns:
            if column in excluded_columns or column.endswith("_Flag_Count") or column.startswith(excluded_prefixes):
                continue
            series = X_train_normal[column]
            if not pd.api.types.is_numeric_dtype(series) or (series < 0).any() or series.nunique(dropna=False) <= 2:
                continue
            skewness = series.skew()
            if pd.notna(skewness) and abs(skewness) >= 1.0:
                selected_columns.append(column)

        logger.info(f"Auto-selected {len(selected_columns)} columns for log1p transform")
        logger.info(f"Log1p columns: {selected_columns}")
        return selected_columns

    if isinstance(feat_cfg, list):
        return feat_cfg

    logger.critical("log_transform_columns must be a list or 'auto'")
    exit(1)

def apply_log1p_transform(X_train: pd.DataFrame, X_train_normal: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, feat_cfg, 
                          logger) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log_columns = get_log_transform_columns(X_train_normal, feat_cfg, logger)

    for column in log_columns:
        for dataset_name, X in [("X_train", X_train), ("X_train_normal", X_train_normal), ("X_val", X_val), ("X_test", X_test)]:
            if column not in X.columns:
                logger.critical(f"Column {column} not found in {dataset_name}")
                exit(1)
            if (X[column] < 0).any():
                logger.critical(f"Column {column} in {dataset_name} contains negative values. Cannot apply log1p.")
                exit(1)

            X[column] = np.log1p(X[column])

    logger.info(f"Applied log1p transform to {len(log_columns)} columns")
    return X_train, X_train_normal, X_val, X_test

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

def build_selected_dataframe(selected_array, selected_columns: list[str], original_index, dataset_name: str, logger) -> pd.DataFrame:
    selected_df = pd.DataFrame(selected_array, columns=selected_columns, index=original_index)
    logger.info(f"{dataset_name} shape after feature selection: {selected_df.shape}")

    return selected_df

def apply_feature_selection(X_train: pd.DataFrame, X_train_normal: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                            y_train: pd.Series, method: str, k_features: int | None, variance_threshold: float,
                            logger) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    method = method.lower()

    if method == "select_k_best":
        if k_features is None or k_features <= 0 or k_features > X_train.shape[1]:
            logger.critical(f"selected_k_features is wrong, SelectKBest is enabled")
            exit(1)
        logger.info(f"Applying supervised SelectKBest with k={k_features}")
        selector = SelectKBest(score_func=f_classif, k=k_features)
        selector.fit(X_train, y_train)
    elif method == "variance_threshold":
        logger.info(f"Applying unsupervised VarianceThreshold with threshold={variance_threshold}")
        selector = VarianceThreshold(threshold=variance_threshold)
        selector.fit(X_train_normal)
    else:
        logger.critical(f"Unsupported feature selection method: {method}")
        exit(1)

    selected_columns = X_train_normal.columns[selector.get_support()].tolist()

    if not selected_columns:
        logger.critical("Feature selection removed all features")
        exit(1)

    logger.info(f"Selected {len(selected_columns)} features")
    logger.info(f"Selected features: {selected_columns}")

    X_train_normal_selected = selector.transform(X_train_normal)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)

    X_train_normal_selected = build_selected_dataframe(X_train_normal_selected, selected_columns, X_train_normal.index,
                                                       "X_train_normal", logger)
    X_val_selected = build_selected_dataframe(X_val_selected, selected_columns, X_val.index, "X_val", logger)
    X_test_selected = build_selected_dataframe(X_test_selected, selected_columns, X_test.index, "X_test", logger)

    return X_train_normal_selected, X_val_selected, X_test_selected

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

    if features_cfg["log_transform"]:
        logger.info("Log1p transform is enabled")
        X_train, X_train_normal, X_val, X_test = apply_log1p_transform(X_train, X_train_normal, X_val, X_test, features_cfg["log_transform_columns"], logger)
    else:
        logger.info("Log1p transform is disabled")

    corr_matrix=compute_correlation_matrix(X_train_normal, logger)
    plot_correlation_matrix(corr_matrix, output_cfg["output_dir"], "base_corr.jpg" ,logger)

    if features_cfg["remove_correlated_features"]:
        logger.info(f"Removing correlated features")
        to_drop = remove_correlated_features(corr_matrix, features_cfg["correlation_threshold"], logger)
        if to_drop:
            X_train = X_train.drop(columns=to_drop)
            X_train_normal = X_train_normal.drop(columns=to_drop)
            X_val = X_val.drop(columns=to_drop)
            X_test = X_test.drop(columns=to_drop)

            corr_matrix_new = compute_correlation_matrix(X_train_normal, logger)
            plot_correlation_matrix(corr_matrix_new, output_cfg["output_dir"], "corr_after_remove.jpg", logger)
    else:
        logger.info("Removing correlated features is disabled")

    if features_cfg["use_feature_selection"]:
        logger.info("Feature selection is enabled")

        X_train_normal, X_val, X_test = apply_feature_selection(X_train=X_train, X_train_normal=X_train_normal, X_val=X_val,
                    X_test=X_test, y_train=y_train, method=features_cfg["feature_selection_method"], k_features=features_cfg["selected_k_features"],
                    variance_threshold=features_cfg["variance_threshold"], logger=logger)
    else:
        logger.info("Feature selection is disabled")

    if features_cfg["scaling"]:
        logger.info("Scaling is enabled")
        X_train_normal, X_val, X_test = scale_datasets(X_train_normal, X_val, X_test, features_cfg["scaler"], logger)
    else:
        logger.info("Scaling is disabled")

    validate_numeric_data(X_train_normal, X_val, X_test, logger)

    return X_train_normal, X_val, X_test, y_train_normal, y_val, y_test