import logging
from pathlib import Path
import yaml
from lr_data import prepare_lr_data, get_logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report, average_precision_score,
                             ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay)
import matplotlib.pyplot as plt
import json


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
    roc_auc = roc_auc_score(y, y_proba) if y_proba is not None else None
    avg_precision = average_precision_score(y, y_proba) if y_proba is not None else None
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, zero_division=0)

    logger.info(f"{split_name} Accuracy: {accuracy:.4f}")
    logger.info(f"{split_name} Precision: {precision:.4f}")
    logger.info(f"{split_name} Recall: {recall:.4f}")
    logger.info(f"{split_name} F1-score: {f1:.4f}")

    if roc_auc is not None:
        logger.info(f"{split_name} ROC-AUC: {roc_auc:.4f}")
    if avg_precision is not None:
        logger.info(f"{split_name} Average Precision: {avg_precision:.4f}")

    logger.info(f"{split_name} Confusion Matrix:\n{cm}")
    logger.info(f"{split_name} Classification Report:\n{report}")

    return {
        "split_name": split_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_true": y.tolist() if hasattr(y, "tolist") else list(y),
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist() if y_proba is not None else None,
    }

def save_to_txt(metrics: dict, exp_name: str, path: Path):
    txt_content = [
        f"Experiment: {exp_name}",
        f"Split: {metrics['split_name']}",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall: {metrics['recall']:.4f}",
        f"F1-score: {metrics['f1']:.4f}",
        f"ROC-AUC: {metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "ROC-AUC: None",
        f"Average Precision: {metrics['average_precision']:.4f}" if metrics["average_precision"] is not None else "Average Precision: None",
        "",
        "Confusion Matrix:", str(metrics["confusion_matrix"]),
        "",
        "Classification Report:",metrics["classification_report"],
    ]

    with path.open("w", encoding="utf-8") as file:
        file.write("\n".join(txt_content))

def save_to_json(metrics: dict, exp_name: str, path: Path):
    json_ready = {
        "experiment_name": exp_name,
        "split_name": metrics["split_name"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "average_precision": metrics["average_precision"],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "classification_report": metrics["classification_report"],
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(json_ready, file, indent=4, ensure_ascii=False)

def save_metrics(metrics: dict, config: dict, logger: logging.Logger) -> None:
    path = BASE_DIR / config["output"]["output_dir"]
    path.mkdir(parents=True, exist_ok=True)
    split_name = metrics["split_name"].lower()

    txt_path = path / f"{split_name}_metrics.txt"
    json_path = path / f"{split_name}_metrics.json"

    save_to_txt(metrics, config["experiment"]["name"], txt_path)
    logger.info(f"Saved metrics to: {txt_path}")

    save_to_json(metrics, config["experiment"]["name"], json_path)
    logger.info(f"Saved metrics JSON to: {json_path}")

def plot_confusion_matrix(metrics: dict, config: dict, logger: logging.Logger) -> None:
    split_name = metrics["split_name"].lower()
    save_path = BASE_DIR / config["output"]["output_dir"] / f"{split_name}_confusion_matrix.jpg"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true=metrics["y_true"],
        y_pred=metrics["y_pred"],
        display_labels=["BENIGN", "ATTACK"],
        cmap="Blues",
        ax=ax
    )
    ax.set_title(f"{metrics['split_name']} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved confusion matrix plot to: {save_path}")

def plot_roc_curve(metrics: dict, config: dict, logger: logging.Logger) -> None:
    if metrics["y_proba"] is None:
        logger.warning("Skipping ROC curve: probabilities not available")
        return

    split_name = metrics["split_name"].lower()
    save_path = BASE_DIR / config["output"]["output_dir"] / f"{split_name}_roc_curve.jpg"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        y_true=metrics["y_true"],
        y_score=metrics["y_proba"],
        ax=ax,
        name=f"{config['experiment']['name']} ({metrics['split_name']})"
    )
    ax.set_title(f"{metrics['split_name']} - ROC Curve")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved ROC curve plot to: {save_path}")

def plot_precision_recall_curve(metrics: dict, config: dict, logger: logging.Logger) -> None:
    if metrics["y_proba"] is None:
        logger.warning("Skipping Precision-Recall curve: probabilities not available")
        return

    split_name = metrics["split_name"].lower()
    save_path = BASE_DIR / config["output"]["output_dir"] / f"{split_name}_pr_curve.jpg"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(
        y_true=metrics["y_true"],
        y_score=metrics["y_proba"],
        ax=ax,
        name=f"{config['experiment']['name']} ({metrics['split_name']})"
    )
    ax.set_title(f"{metrics['split_name']} - Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved Precision-Recall curve plot to: {save_path}")

def save_visualizations(metrics: dict, config: dict, logger: logging.Logger) -> None:
    plot_confusion_matrix(metrics, config, logger)
    plot_roc_curve(metrics, config, logger)
    plot_precision_recall_curve(metrics, config, logger)

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

    if config["output"]["save_metrics"]:
        save_metrics(val_metrics, config, logger)

    if config["output"]["save_plots"]:
        save_visualizations(val_metrics, config, logger)

if __name__ == "__main__":
    main()