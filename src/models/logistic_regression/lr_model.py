import logging
from pathlib import Path
import yaml
from lr_data import prepare_lr_data, get_logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = BASE_DIR / "config" / "logistic_regression.yaml"

def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def log_config(config: dict, logger) -> None:
    config_text = yaml.safe_dump(
        config,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False
    )
    logger.info(f"Loaded configuration:\n{config_text}")

def build_model(config: dict, logger) -> LogisticRegression:
    model_cfg = config["model"]
    random_state = config["experiment"]["random_state"]
    logger.info("Building Logistic Regression model")
    logger.info(f"Model parameters: max_iter={model_cfg['max_iter']}, solver={model_cfg['solver']},"
                f"class_weight={model_cfg['class_weight']}, random_state={random_state}")

    model = LogisticRegression(
        max_iter=model_cfg["max_iter"],
        solver=model_cfg["solver"],
        class_weight=model_cfg["class_weight"],
        random_state=random_state
    )

    return model

def train_model(model: LogisticRegression, X_train, y_train, logger) -> LogisticRegression:
    logger.info("Training Logistic Regression model")
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    return model

def evaluate_model(model, X, y, split_name: str, logger) -> dict:
    logger.info(f"Evaluating model on {split_name} set")

    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        y_proba = None

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    if y_proba is not None:
        roc_auc = roc_auc_score(y, y_proba)
    else:
        roc_auc = None

    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, zero_division=0)

    logger.info(f"{split_name} Accuracy: {accuracy:.4f}")
    logger.info(f"{split_name} Precision: {precision:.4f}")
    logger.info(f"{split_name} Recall: {recall:.4f}")
    logger.info(f"{split_name} F1-score: {f1:.4f}")

    if roc_auc is not None:
        logger.info(f"{split_name} ROC-AUC: {roc_auc:.4f}")

    logger.info(f"{split_name} Confusion Matrix:\n{cm}")
    logger.info(f"{split_name} Classification Report:\n{report}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": report
    }

def save_metrics(config: dict, logger: logging.Logger) -> dict:
    pass

def save_model(config: dict, logger: logging.Logger) -> dict:
    pass

def main() -> None:
    config = load_config(CONFIG_PATH)
    logger = get_logger(config)
    log_config(config, logger)

    logger.info("Start experiment")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_lr_data(config)
    logger.info("Data prepared successfully")

    model = build_model(config, logger)

    model = train_model(model, X_train, y_train, logger)

    val_metrics = evaluate_model(model, X_val, y_val, "Validation", logger)

    print(val_metrics)

if __name__ == "__main__":
    main()