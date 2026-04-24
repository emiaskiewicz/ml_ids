from pathlib import Path
import pandas as pd
import numpy as np
from utils.logger import setup_logger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
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

def apply_feature_selection(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                            method: str, k_features: int, random_state: int, logger) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    if method == "rfe":
        logger.info(f"Applying RFE with {k_features} selected features")
        estimator = DecisionTreeClassifier(random_state=random_state)
        selector = RFE(estimator=estimator, n_features_to_select=k_features)
    elif method == "select_k_best":
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

    logger.info(f"After SMOTE: X_train shape: {X_resampled.shape}, y_train shape: {y_resampled.shape}")
    return X_resampled, y_resampled

def prepare_dt_data(config: dict):
    logger = get_logger(config)
    logger.info(f"Preparing data for experiment: {config['experiment']['name']}")
    logger.info(f"Config: {config}")

    exp_cfg = config["experiment"]
    data_cfg = config["data"]
    output_cfg = config["output"]
    features_cfg = config["features"]
    split_cfg = config["split"]

    if split_cfg["load_existing_split"] and split_files_exist(config):
        logger.info(f"Loading existing split data")
        train_df, val_df, test_df = load_existing_split_data(config, logger)
        train_df = drop_feature_columns(train_df, features_cfg["drop_columns"], data_cfg["target_column"], logger)
        val_df = drop_feature_columns(val_df, features_cfg["drop_columns"], data_cfg["target_column"], logger)
        test_df = drop_feature_columns(test_df, features_cfg["drop_columns"], data_cfg["target_column"], logger)
        X_train, X_val, X_test, y_train, y_val, y_test = separate_features_and_target(train_df, val_df, test_df, data_cfg["target_column"], logger)
    else:
        logger.error("Could not load existing split data")
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

    if features_cfg["use_feature_selection"]:
        logger.info("Feature selection is enabled")
        X_train, X_val, X_test = apply_feature_selection(X_train, X_val, X_test, y_train, features_cfg["feature_selection_method"],
                                                         features_cfg["selected_k_features"], config["experiment"]["random_state"], logger)
    else:
        logger.info("Feature selection is disabled")

    if features_cfg["smote"]:
        logger.info("SMOTE is enabled")
        logger.info(f"Class distribution before SMOTE:\n{y_train.value_counts()}")
        X_train, y_train = apply_smote(X_train, y_train, exp_cfg["random_state"], logger)
        logger.info(f"Class distribution after SMOTE:\n{y_train.value_counts()}")

    return X_train, X_val, X_test, y_train, y_val, y_test