import logging
import csv
import winsound
from pathlib import Path
import yaml
from mlp_data import prepare_mlp_data, get_logger
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report, average_precision_score,
                             ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay)
import matplotlib.pyplot as plt
import json
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = BASE_DIR / "config" / "mlp.yaml"

RESULTS_COLUMNS = ["experiment", "dataset_variant", "split", "accuracy", "precision", "recall", "f1", "roc_auc",
                   "average_precision", "threshold", "scaling", "scaler", "feature_selection", "feature_selection_method",
                   "selected_k_features", "smote", "hidden_layers", "dropout", "learning_rate", "batch_size", "epochs",
                   "weight_decay", "device", "tuning_stage_1", "tuning_stage_2"]

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

class MLPNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int], dropout: float):
        super().__init__()

        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(1)

def build_model(input_dim: int, config: dict, overrides: dict, logger) -> MLPNetwork:
    model_cfg = config["model"].copy()

    if overrides:
        model_cfg.update(overrides)

    logger.info("Building MLP model")
    logger.info(f"Model parameters: input_dim={input_dim}, hidden_layers={model_cfg['hidden_layers']}, dropout={model_cfg['dropout']}")

    model = MLPNetwork(
        input_dim=input_dim,
        hidden_layers=model_cfg["hidden_layers"],
        dropout=model_cfg["dropout"]
    )

    return model

def create_dataloader(X, y, batch_size: int, shuffle: bool) -> DataLoader:
    X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)
    y_tensor = torch.tensor(y.to_numpy(), dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def calculate_loss(model, data_loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(data_loader.dataset)

def train_model(model, train_loader, val_loader, config: dict, device, logger):
    model_cfg = config["model"]

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(
        model.parameters(),
        lr=model_cfg["learning_rate"],
        weight_decay=model_cfg["weight_decay"]
    )

    epochs = model_cfg["epochs"]
    history = []

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        val_loss = calculate_loss(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        logger.info(f"Epoch {epoch}/{epochs} - train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    logger.info("Model training completed")
    return model, pd.DataFrame(history)

def predict_proba(model, data_loader, device) -> tuple[list[int], list[float]]:
    model.eval()

    y_true = []
    y_proba = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)

            logits = model(X_batch)
            probabilities = torch.sigmoid(logits)

            y_true.extend(y_batch.cpu().numpy().astype(int).tolist())
            y_proba.extend(probabilities.cpu().numpy().tolist())

    return y_true, y_proba

def get_device(config: dict, logger):
    requested_device = config["model"].get("device", "auto")

    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)

    logger.info(f"Using device: {device}")
    return device

def set_seed(random_state: int, logger) -> None:
    torch.manual_seed(random_state)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    logger.info(f"PyTorch random seed set to: {random_state}")

def evaluate_model(model, data_loader, split_name: str, threshold: float, device, logger) -> dict:
    logger.info(f"Evaluating model on {split_name} set")

    y_true, y_proba = predict_proba(model, data_loader, device)
    y_pred = apply_threshold(y_proba, threshold)

    metrics = calculate_binary_metrics(y_true, y_pred, y_proba)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)

    logger.info(f"{split_name} Threshold used: {threshold}")
    logger.info(f"{split_name} Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"{split_name} Precision: {metrics['precision']:.4f}")
    logger.info(f"{split_name} Recall: {metrics['recall']:.4f}")
    logger.info(f"{split_name} F1-score: {metrics['f1']:.4f}")
    logger.info(f"{split_name} ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"{split_name} Average Precision: {metrics['average_precision']:.4f}")
    logger.info(f"{split_name} Confusion Matrix:\n{cm}")
    logger.info(f"{split_name} Classification Report:\n{report}")

    return {
        "split_name": split_name,
        "threshold_used": threshold,
        **metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba
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
        "Classification Report:",metrics["classification_report"]
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
        "classification_report": metrics["classification_report"]
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

def append_results_to_csv(results: dict, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()
    file_is_empty = file_exists and csv_path.stat().st_size == 0

    row = {column: results.get(column, None) for column in RESULTS_COLUMNS}

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULTS_COLUMNS)

        if not file_exists or file_is_empty:
            writer.writeheader()

        writer.writerow(row)

def build_results_summary_row(metrics: dict, config: dict, model_params: dict | None) -> dict:
    if model_params is None:
        model_params = config["model"]

    features_cfg = config["features"]
    prep_cfg = config["preprocessing"]
    model_cfg = config["model"]

    return {
        "experiment": config["experiment"]["name"],
        "dataset_variant": config["data"]["dataset_variant"],
        "split": metrics["split_name"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "average_precision": metrics["average_precision"],
        "threshold": metrics["threshold_used"],

        "scaling": prep_cfg.get("scaling", False),
        "scaler": prep_cfg.get("scaler", None),

        "feature_selection": features_cfg.get("use_feature_selection", False),
        "feature_selection_method": features_cfg.get("feature_selection_method", None),
        "selected_k_features": features_cfg.get("selected_k_features", None),
        "smote": prep_cfg.get("smote", False),

        "hidden_layers": model_params.get("hidden_layers", model_cfg.get("hidden_layers")),
        "dropout": model_params.get("dropout", model_cfg.get("dropout")),
        "learning_rate": model_params.get("learning_rate", model_cfg.get("learning_rate")),
        "batch_size": model_params.get("batch_size", model_cfg.get("batch_size")),
        "epochs": model_params.get("epochs", model_cfg.get("epochs")),
        "weight_decay": model_params.get("weight_decay", model_cfg.get("weight_decay")),
        "device": model_params.get("device", model_cfg.get("device")),

        "tuning_stage_1": config.get("tuning_stage_1", {}).get("enabled", False),
        "tuning_stage_2": config.get("tuning_stage_2", {}).get("enabled", False),
    }

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
        values_format="d",
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
    ax.grid(True, alpha=0.3)
    #ax.set_xlim(0.0, 0.02)
    #ax.set_ylim(0.98, 1.0)
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
    #ax.set_xlim(0.98, 1.0)
    #ax.set_ylim(0.98, 1.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved Precision-Recall curve plot to: {save_path}")

def save_visualizations(metrics: dict, config: dict, logger: logging.Logger) -> None:
    plot_confusion_matrix(metrics, config, logger)
    plot_roc_curve(metrics, config, logger)
    plot_precision_recall_curve(metrics, config, logger)

def apply_threshold(y_proba: pd.Series | list, threshold: float) -> list:
    return [1 if prob >= threshold else 0 for prob in y_proba]

def calculate_binary_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if y_proba is not None else None,
        "average_precision": average_precision_score(y_true, y_proba) if y_proba is not None else None
    }

def plot_training_history(history_df: pd.DataFrame, config: dict, logger) -> None:
    output_dir = BASE_DIR / config["output"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "training_loss_curve.jpg"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    ax.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("MLP training history")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved training history plot to: {save_path}")

def save_training_history(history_df: pd.DataFrame, config: dict, logger) -> None:
    output_dir = BASE_DIR / config["output"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "training_history.csv"

    history_df.to_csv(save_path, index=False)
    logger.info(f"Saved training history to: {save_path}")

def main() -> None:
    config = load_config(CONFIG_PATH)
    logger = get_logger(config)
    log_config(config, logger)

    logger.info("Start experiment")
    set_seed(config["experiment"]["random_state"], logger)
    device = get_device(config, logger)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_mlp_data(config)
    logger.info("Data prepared successfully")

    batch_size = config["model"]["batch_size"]

    train_loader = create_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = create_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)

    model = build_model(X_train.shape[1], config, {}, logger)

    model, history_df = train_model(model, train_loader, val_loader, config, device, logger)

    save_training_history(history_df, config, logger)
    threshold = config["model"].get("decision_threshold", 0.5)
    val_metrics = evaluate_model(model, val_loader, "Validation", threshold, device, logger)

    summary_row = build_results_summary_row(
        metrics=val_metrics,
        config=config,
        model_params=config["model"]
    )

    summary_csv_path = BASE_DIR / config["output"]["summary_path"]
    append_results_to_csv(summary_row, summary_csv_path)
    logger.info(f"Added experiment results to summary CSV: {summary_csv_path}")

    if config["output"]["save_metrics"]:
        save_metrics(val_metrics, config, logger)

    if config["output"]["save_plots"]:
        save_visualizations(val_metrics, config, logger)
        plot_training_history(history_df, config, logger)

    winsound.Beep(2500,1000)

if __name__ == "__main__":
    main()