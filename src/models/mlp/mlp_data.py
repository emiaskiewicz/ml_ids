from pathlib import Path
import pandas as pd
import numpy as np
from utils.logger import setup_logger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

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

def remove_correlated_features(df: pd.DataFrame, threshold: float, logger):
    corr_matrix = df.corr(numeric_only=True).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    logger.info(f"Removing {len(to_drop)} correlated features")
    logger.info(f"Removed features: {to_drop}")
    df_reduced = df.drop(columns=to_drop)

    return df_reduced, to_drop

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

def scale_datasets(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, scaler_name: str,
                   logger) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scaler = get_scaler(scaler_name, logger)

    logger.info("Fitting scaler on X_train")
    X_train_scaled = scaler.fit_transform(X_train)

    logger.info("Transforming X_val and X_test with fitted scaler")
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    logger.info(f"Scaled X_train shape: {X_train_scaled.shape}")
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    logger.info(f"Scaled X_val shape: {X_val_scaled.shape}")
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    logger.info(f"Scaled X_test shape: {X_test_scaled.shape}")

    return X_train_scaled, X_val_scaled, X_test_scaled

def apply_feature_selection(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                            method: str, k_features: int, logger) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #todo: zminic na raise
    if k_features is None:
        logger.critical("selected_k_features must be provided when feature selection is enabled")
        exit(1)
    if k_features <= 0:
        logger.critical("selected_k_features must be greater than 0")
        exit(1)
    if k_features > X_train.shape[1]:
        logger.critical(f"selected_k_features={k_features} is greater than the number of available features={X_train.shape[1]}")
        exit(1)

    method = method.lower()

    if method == "select_k_best":
        logger.info(f"Applying SelectKBest with {k_features} selected features")
        selector = SelectKBest(score_func=f_classif, k=k_features)
    else:
        #todo: tez zmienic
        logger.critical(f"Unsupported feature selection method: {method}")
        exit(1)

    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)

    selected_columns = X_train.columns[selector.get_support()].tolist()
    logger.info(f"Selected {len(selected_columns)} features")
    logger.info(f"Selected features: {selected_columns}")

    X_train_selected = pd.DataFrame(X_train_selected, columns=selected_columns, index=X_train.index)
    logger.info(f"X_train shape after feature selection: {X_train_selected.shape}")
    X_val_selected = pd.DataFrame(X_val_selected, columns=selected_columns, index=X_val.index)
    logger.info(f"X_val shape after feature selection: {X_val_selected.shape}")
    X_test_selected = pd.DataFrame(X_test_selected, columns=selected_columns, index=X_test.index)
    logger.info(f"X_test shape after feature selection: {X_test_selected.shape}")

    return X_train_selected, X_val_selected, X_test_selected

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int, logger):
    logger.info("Applying SMOTE to training data")
    logger.info(f"Before SMOTE: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_resampled = pd.Series(y_resampled, name=y_train.name)

    logger.info(f"After SMOTE: X_train shape: {X_resampled.shape}, y_train shape: {y_resampled.shape}")
    return X_resampled, y_resampled

def validate_numeric_data(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, logger) -> None:
    for name, X in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()

        if non_numeric_columns:
            logger.critical(f"{name} contains non-numeric columns: {non_numeric_columns}")
            exit(1)

        if not np.isfinite(X.to_numpy()).all():
            logger.critical(f"{name} contains NaN, inf or -inf values")
            exit(1)

        logger.info(f"{name} numeric validation passed")

def prepare_mlp_data(config: dict):
    logger = get_logger(config)
    logger.info(f"Preparing data for experiment: {config['experiment']['name']}")
    logger.info(f"Config: {config}")

    exp_cfg = config["experiment"]
    data_cfg = config["data"]
    output_cfg = config["output"]
    features_cfg = config["features"]
    split_cfg = config["split"]
    prep_cfg = config["preprocessing"]

    if split_cfg["load_existing_split"] and split_files_exist(config):
        logger.info(f"Loading existing split data")
        train_df, val_df, test_df = load_existing_split_data(config, logger)
        X_train, X_val, X_test, y_train, y_val, y_test = separate_features_and_target(train_df, val_df, test_df, data_cfg["target_column"], logger)
    else:
        logger.error(f"Loading existing split data failed")
        exit(1)

    corr_matrix=compute_correlation_matrix(X_train, logger)
    plot_correlation_matrix(corr_matrix, output_cfg["output_dir"], "base_corr.jpg" ,logger)
    if features_cfg["remove_correlated_features"]:
        logger.info(f"Removing correlated features")
        X_train, to_drop = remove_correlated_features(X_train, features_cfg["correlation_threshold"], logger)
        X_val.drop(columns=to_drop, inplace=True)
        X_test.drop(columns=to_drop, inplace=True)
        corr_matrix = compute_correlation_matrix(X_train, logger)
        plot_correlation_matrix(corr_matrix, output_cfg["output_dir"], "corr_after_remove.jpg", logger)

    if prep_cfg["scaling"]:
        logger.info("Scaling is enabled")
        X_train, X_val, X_test = scale_datasets(X_train, X_val, X_test, prep_cfg["scaler"], logger)
    else:
        logger.info("Scaling is disabled")

    if features_cfg["use_feature_selection"]:
        logger.info("Feature selection is enabled")
        X_train, X_val, X_test = apply_feature_selection(X_train, X_val, X_test, y_train, features_cfg["feature_selection_method"],
                                                         features_cfg["selected_k_features"], logger)
    else:
        logger.info("Feature selection is disabled")

    if prep_cfg["smote"]:
        logger.info("SMOTE is enabled")
        logger.info(f"Class distribution before SMOTE:\n{y_train.value_counts()}")
        X_train, y_train = apply_smote(X_train, y_train, exp_cfg["random_state"], logger)
        logger.info(f"Class distribution after SMOTE:\n{y_train.value_counts()}")

    validate_numeric_data(X_train, X_val, X_test, logger)

    return X_train, X_val, X_test, y_train, y_val, y_test