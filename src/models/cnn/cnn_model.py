import logging
import csv
from pathlib import Path
import yaml
from cnn_data import prepare_cnn_data, get_logger
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
from itertools import product
import numpy as np
import os

BASE_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = BASE_DIR / "config" / "cnn.yaml"

RESULTS_COLUMNS = ["experiment", "dataset_variant", "split", "accuracy", "precision", "recall", "f1", "roc_auc",
                   "average_precision", "threshold", "scaling", "scaler","use_network_features", "drop_original_port_columns", "feature_selection", "feature_selection_method",
                   "selected_k_features", "smote", "use_pos_weight", "pos_weight_mode", "pos_weight_value", "conv_channels", "kernel_size", "fc_layers",
                   "activation", "batch_norm", "dropout", "learning_rate", "batch_size", "epochs","gradient_clip_norm", "global_pooling",
                   "input_dropout", "input_noise_std", "scheduler_enabled", "scheduler_factor", "scheduler_patience", "scheduler_min_lr", "weight_decay", "device", 
                   "early_stopping", "patience", "min_delta", "actual_epochs", "best_epoch", "best_val_loss", "tuning_stage_1", "tuning_stage_2"]

IGNORED_TUNING_PARAMS_RULES = [
    {"when": {"scheduler_enabled": False}, "ignore": ["scheduler_factor", "scheduler_patience", "scheduler_min_lr"]},
]

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

def get_activation_layer(activation_name: str) -> nn.Module:
    activation_name = activation_name.lower()

    if activation_name == "relu":
        return nn.ReLU()
    elif activation_name == "leaky_relu":
        return nn.LeakyReLU()
    elif activation_name == "tanh":
        return nn.Tanh()
    elif activation_name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")

class InputNoise(nn.Module):
    def __init__(self, noise_std: float):
        super().__init__()
        self.noise_std = noise_std

    def forward(self, x):
        if self.training and self.noise_std > 0:
            return x + torch.randn_like(x) * self.noise_std
        return x

class CNNNetwork(nn.Module):
    def __init__(self, input_dim: int, conv_channels: list[int], kernel_size: int, fc_layers: list[int], dropout: float, 
                 activation: str, batch_norm: bool, global_pooling: bool, input_dropout: float, input_noise_std: float):
        super().__init__()

        self.input_noise = InputNoise(input_noise_std)
        self.input_dropout = nn.Dropout(input_dropout)

        conv_layers = []
        in_channels = 1
        padding = 0 if kernel_size % 2 == 0 else kernel_size // 2
        conv_output_dim = input_dim if padding > 0 else input_dim - len(conv_channels) * (kernel_size - 1)

        if conv_output_dim <= 0:
            raise ValueError(f"kernel_size={kernel_size} with {len(conv_channels)} convolutional layers reduces input_dim={input_dim} to non-positive feature length")

        for out_channels in conv_channels:
            conv_layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding))
            if batch_norm:
                conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(get_activation_layer(activation))
            if dropout > 0:
                conv_layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.features = nn.Sequential(*conv_layers)
        self.global_pooling = global_pooling
        self.pool = nn.AdaptiveAvgPool1d(1) if global_pooling else None

        classifier_layers = []
        previous_dim = conv_channels[-1] if global_pooling else conv_channels[-1] * conv_output_dim

        for hidden_dim in fc_layers:
            classifier_layers.append(nn.Linear(previous_dim, hidden_dim))
            classifier_layers.append(get_activation_layer(activation))
            if dropout > 0:
                classifier_layers.append(nn.Dropout(dropout))

            previous_dim = hidden_dim

        classifier_layers.append(nn.Linear(previous_dim, 1))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        x= self.input_noise(x)
        x = self.input_dropout(x)
        x = self.features(x)

        if self.global_pooling:
            x = self.pool(x)

        x = torch.flatten(x, start_dim=1)
        return self.classifier(x).squeeze(1)

class EarlyStopping:
    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_state_dict = None
        self.epochs_without_improvement = 0
        self.should_stop = False

    def update(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        improved = val_loss < self.best_loss - self.min_delta

        if improved:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True

        return improved

    def restore_best_weights(self, model: nn.Module, device, logger) -> None:
        if self.best_state_dict is None:
            logger.warning("EarlyStopping has no best state to restore")
            return

        model.load_state_dict({
            key: value.to(device)
            for key, value in self.best_state_dict.items()
        })

        logger.info(f"Restored best model weights from epoch {self.best_epoch} with val_loss={self.best_loss:.6f}")

def build_model(input_dim: int, config: dict, overrides: dict, logger) -> CNNNetwork:
    model_cfg = config["model"].copy()

    if overrides:
        model_cfg.update(overrides)

    logger.info("Building CNN model")
    logger.info(f"Model parameters: input_dim={input_dim}, conv_channels={model_cfg['conv_channels']}, kernel_size={model_cfg['kernel_size']},"
                f" fc_layers={model_cfg['fc_layers']}, dropout={model_cfg['dropout']}, activation={model_cfg['activation']},"
                f" batch_norm={model_cfg['batch_norm']}, global_pooling={model_cfg['global_pooling']}, input_dropout={model_cfg['input_dropout']}, input_noise_std={model_cfg['input_noise_std']}")

    model = CNNNetwork(
        input_dim=input_dim,
        conv_channels=model_cfg["conv_channels"],
        kernel_size=model_cfg["kernel_size"],
        fc_layers=model_cfg["fc_layers"],
        dropout=model_cfg["dropout"],
        activation=model_cfg.get("activation", "relu"),
        batch_norm=model_cfg["batch_norm"],
        global_pooling=model_cfg["global_pooling"],
        input_dropout=model_cfg.get("input_dropout", 0.0),
        input_noise_std=model_cfg.get("input_noise_std", 0.0)
    )

    return model

def create_dataloader(X, y, batch_size: int, shuffle: bool) -> DataLoader:
    X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y.to_numpy(), dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)

    use_cuda = torch.cuda.is_available()
    workers = min(4, os.cpu_count() or 1)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=use_cuda,
                      persistent_workers=True if workers > 0 else False)

def calculate_loss(model, data_loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)

        return total_loss / total_samples

def calculate_pos_weight(y_train: pd.Series, device, logger: logging.Logger) -> tuple[torch.Tensor, float]:
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    if negative_count == 0:
        logger.critical("Cannot calculate pos_weight: no negative class samples found in y_train")
        exit(1)

    if positive_count == 0:
        logger.critical("Cannot calculate pos_weight: no positive class samples found in y_train")
        exit(1)

    pos_weight_value = negative_count / positive_count
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    logger.info(f"Calculated pos_weight={pos_weight_value:.6f} from y_train: negative_count={negative_count}, "
                f"positive_count={positive_count}")

    return pos_weight, pos_weight_value

def build_loss_function(y_train: pd.Series, config: dict, device, logger: logging.Logger) -> tuple[nn.Module, dict]:
    model_cfg = config["model"]

    loss_info = {
        "use_pos_weight": model_cfg["use_pos_weight"],
        "pos_weight_mode": model_cfg.get("pos_weight_mode", "auto"),
        "pos_weight_value": None
    }

    if not model_cfg["use_pos_weight"]:
        logger.info("Using BCEWithLogitsLoss without pos_weight")
        return nn.BCEWithLogitsLoss(), loss_info

    if config["preprocessing"]["smote"]:
        logger.warning("SMOTE and pos_weight are both enabled.")

    pos_weight_mode = model_cfg.get("pos_weight_mode", "auto")

    if pos_weight_mode == "auto":
        pos_weight, pos_weight_value = calculate_pos_weight(y_train, device, logger)
    elif pos_weight_mode == "manual":
        pos_weight_value = float(model_cfg["pos_weight_value"])

        if pos_weight_value <= 0:
            logger.critical("pos_weight_value must be greater than 0 when pos_weight_mode is manual")
            exit(1)

        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
        logger.info(f"Using manual pos_weight={pos_weight_value:.6f}")
    else:
        logger.critical(f"Unsupported pos_weight_mode: {pos_weight_mode}. Use 'auto' or 'manual'")
        exit(1)

    loss_info["pos_weight_value"] = pos_weight_value

    logger.info(f"Using BCEWithLogitsLoss with pos_weight={pos_weight_value:.6f}, mode={pos_weight_mode}")

    return nn.BCEWithLogitsLoss(pos_weight=pos_weight), loss_info

def build_scheduler(optimizer, config: dict, logger: logging.Logger):
    model_cfg = config["model"]

    if not model_cfg["scheduler_enabled"]:
        logger.info("Learning rate scheduler disabled")
        return None

    logger.info(f"Using ReduceLROnPlateau scheduler: factor={model_cfg['scheduler_factor']}, patience={model_cfg['scheduler_patience']}, "
                f"min_lr={model_cfg['scheduler_min_lr']}")

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=model_cfg["scheduler_factor"],
                                  patience=model_cfg["scheduler_patience"], min_lr=model_cfg["scheduler_min_lr"])

    return scheduler

def get_current_learning_rate(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]

def train_model(model, train_loader, val_loader, y_train: pd.Series, config: dict, device, logger):
    model_cfg = config["model"]

    criterion, loss_info = build_loss_function(y_train=y_train, config=config, device=device, logger=logger)
    optimizer = Adam(
        model.parameters(),
        lr=model_cfg["learning_rate"],
        weight_decay=model_cfg["weight_decay"]
    )
    scheduler = build_scheduler(optimizer, config, logger)

    epochs = model_cfg["epochs"]
    history = []
    early_stopping_enabled = model_cfg.get("early_stopping", False)
    early_stopping = None

    if early_stopping_enabled:
        early_stopping = EarlyStopping(
            patience=model_cfg.get("patience", 5),
            min_delta=model_cfg.get("min_delta", 0.0)
        )
        logger.info(f"Early stopping enabled: patience={early_stopping.patience}, min_delta={early_stopping.min_delta}")
    else:
        logger.info("Early stopping disabled")

    model.to(device)
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        total_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()

            gradient_clip_norm = model_cfg["gradient_clip_norm"]

            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)

        train_loss = train_loss / total_samples
        val_loss = calculate_loss(model, val_loader, criterion, device)

        if scheduler is not None:
            old_lr = get_current_learning_rate(optimizer)
            scheduler.step(val_loss)
            new_lr = get_current_learning_rate(optimizer)

            if new_lr < old_lr:
                logger.info(f"Learning rate reduced from {old_lr:.8f} to {new_lr:.8f}")

        if early_stopping_enabled:
            improved = early_stopping.update(val_loss, model, epoch)
            best_val_loss = early_stopping.best_loss
            best_epoch = early_stopping.best_epoch
            epochs_without_improvement = early_stopping.epochs_without_improvement
        else:
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch
            epochs_without_improvement = None

        current_lr = get_current_learning_rate(optimizer)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "improved": improved,
            "learning_rate": current_lr
        })

        logger.info(f"Epoch {epoch}/{epochs} - train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, best_val_loss={best_val_loss:.6f}, "
                    f"best_epoch={best_epoch}, epochs_without_improvement={epochs_without_improvement}, learning_rate={current_lr:.8f}")

        if early_stopping_enabled and early_stopping.should_stop:
            logger.info(f"Early stopping triggered at epoch {epoch}. Best epoch: {early_stopping.best_epoch}, best_val_loss={early_stopping.best_loss:.6f}")
            break

    if early_stopping_enabled and early_stopping is not None:
        early_stopping.restore_best_weights(model, device, logger)

    logger.info("Model training completed")
    return model, pd.DataFrame(history), loss_info

def predict_proba(model, data_loader, device) -> tuple[list[int], list[float]]:
    model.eval()
    y_true_list = []
    y_proba_list = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probabilities = torch.sigmoid(logits)

            y_true_list.append(y_batch.detach().cpu())
            y_proba_list.append(probabilities.detach().cpu())

    y_true_tensor = torch.cat(y_true_list)
    y_proba_tensor = torch.cat(y_proba_list)

    y_true = y_true_tensor.to(torch.int32).tolist()
    y_proba = y_proba_tensor.view(-1).tolist()

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

def get_training_summary(history_df: pd.DataFrame) -> dict:
    if history_df.empty:
        return {
            "actual_epochs": None,
            "best_epoch": None,
            "best_val_loss": None
        }

    last_row = history_df.iloc[-1]

    return {
        "actual_epochs": int(history_df["epoch"].max()),
        "best_epoch": int(last_row["best_epoch"]),
        "best_val_loss": float(last_row["best_val_loss"])
    }

def evaluate_model(model, data_loader, split_name: str, threshold: float, device, logger) -> dict:
    logger.info(f"Evaluating model on {split_name} set")

    y_true, y_proba = predict_proba(model, data_loader, device)
    y_pred = (np.array(y_proba) >= threshold).astype(int).tolist()

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

def save_metrics(metrics: dict, config: dict, logger: logging.Logger) -> None:
    path = BASE_DIR / config["output"]["output_dir"]
    path.mkdir(parents=True, exist_ok=True)

    split_name = metrics["split_name"].lower()
    exp_name = config["experiment"]["name"]

    json_path = path / f"{split_name}_metrics.json"
    with json_path.open("w", encoding="utf-8") as file:
        json_data = {**metrics, "confusion_matrix": metrics["confusion_matrix"].tolist()}
        json.dump(json_data, file, indent=4, ensure_ascii=False)
    logger.info(f"Saved metrics JSON to: {json_path}")

    txt_path = path / f"{split_name}_metrics.txt"
    with txt_path.open("w", encoding="utf-8") as file:
        file.write(f"Experiment: {exp_name}\nSplit: {metrics['split_name']}\n")
        file.write(f"Accuracy: {metrics['accuracy']:.4f}\nPrecision: {metrics['precision']:.4f}\n")
        file.write(f"Recall: {metrics['recall']:.4f}\nF1-score: {metrics['f1']:.4f}\n")
        roc_auc = metrics.get('roc_auc')
        file.write(f"ROC-AUC: {roc_auc:.4f}\n" if roc_auc is not None else "ROC-AUC: None\n")
        file.write(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}\n\n")
        file.write(f"Classification Report:\n{metrics['classification_report']}\n")
    logger.info(f"Saved metrics to: {txt_path}")

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

def build_results_summary_row(metrics: dict, config: dict, model_params: dict | None, training_summary: dict | None) -> dict:
    if model_params is None:
        model_params = config["model"]
    if training_summary is None:
        training_summary = {}

    model_cfg = config["model"].copy()
    model_cfg.update(model_params)

    training_cfg = {
        "actual_epochs": None,
        "best_epoch": None,
        "best_val_loss": None,
        "pos_weight_value": None
    }
    training_cfg.update(training_summary)
    features_cfg = config["features"]
    prep_cfg = config["preprocessing"]

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
        "use_network_features": features_cfg.get("use_network_features", False),
        "drop_original_port_columns": features_cfg.get("drop_original_port_columns", False),
        "feature_selection": features_cfg.get("use_feature_selection", False),
        "feature_selection_method": features_cfg.get("feature_selection_method", None),
        "selected_k_features": features_cfg.get("selected_k_features", None),
        "smote": prep_cfg.get("smote", False),
        "use_pos_weight": model_params["use_pos_weight"],
        "pos_weight_mode": model_params.get("pos_weight_mode", "auto"),
        "pos_weight_value": training_summary["pos_weight_value"],

        "input_noise_std": model_params.get("input_noise_std", model_cfg.get("input_noise_std")),
        "input_dropout": model_params.get("input_dropout", model_cfg.get("input_dropout")),
        "global_pooling": model_params.get("global_pooling", model_cfg.get("global_pooling")),
        "gradient_clip_norm": model_params.get("gradient_clip_norm", model_cfg.get("gradient_clip_norm")),
        "conv_channels": model_params.get("conv_channels", model_cfg.get("conv_channels")),
        "kernel_size": model_params.get("kernel_size", model_cfg.get("kernel_size")),
        "fc_layers": model_params.get("fc_layers", model_cfg.get("fc_layers")),
        "activation": model_params.get("activation", model_cfg.get("activation")),
        "dropout": model_params.get("dropout", model_cfg.get("dropout")),
        "learning_rate": model_params.get("learning_rate", model_cfg.get("learning_rate")),
        "batch_size": model_params.get("batch_size", model_cfg.get("batch_size")),
        "epochs": model_params.get("epochs", model_cfg.get("epochs")),
        "weight_decay": model_params.get("weight_decay", model_cfg.get("weight_decay")),
        "device": model_params.get("device", model_cfg.get("device")),
        "scheduler_enabled": model_params.get("scheduler_enabled", model_cfg["scheduler_enabled"]),
        "scheduler_factor": model_params.get("scheduler_factor", model_cfg["scheduler_factor"]),
        "scheduler_patience": model_params.get("scheduler_patience", model_cfg["scheduler_patience"]),
        "scheduler_min_lr": model_params.get("scheduler_min_lr", model_cfg["scheduler_min_lr"]),

        "early_stopping": model_params.get("early_stopping", model_cfg.get("early_stopping")),
        "patience": model_params.get("patience", model_cfg.get("patience")),
        "min_delta": model_params.get("min_delta", model_cfg.get("min_delta")),
        "actual_epochs": training_summary.get("actual_epochs"),
        "best_epoch": training_summary.get("best_epoch"),
        "best_val_loss": training_summary.get("best_val_loss"),

        "tuning_stage_1": config.get("tuning_stage_1", {}).get("enabled", False),
        "tuning_stage_2": config.get("tuning_stage_2", {}).get("enabled", False),
    }

def plot_confusion_matrix(metrics: dict, config: dict, logger: logging.Logger) -> None:
    split_name = metrics["split_name"].lower()
    save_path = BASE_DIR / config["output"]["output_dir"] / f"{split_name}_confusion_matrix.jpg"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        metrics["y_true"],
        metrics["y_pred"],
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
    y_true = metrics["y_true"]
    y_proba = metrics["y_proba"]

    RocCurveDisplay.from_predictions(y_true, y_proba, name=f"CNN (AUC = {metrics['roc_auc']:.4f})", ax=ax,
                                     color='darkorange', linewidth=2)
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
    y_true = metrics["y_true"]
    y_proba = metrics["y_proba"]

    PrecisionRecallDisplay.from_predictions(y_true, y_proba, name=f"CNN (AP = {metrics['average_precision']:.4f})", ax=ax,
                                            color='purple', linewidth=2)
    ax.set_title(f"{metrics['split_name']} - Precision-Recall Curve")
    #ax.set_xlim(0.98, 1.0)
    #ax.set_ylim(0.98, 1.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved Precision-Recall curve plot to: {save_path}")

def save_visualizations(metrics: dict, config: dict, logger: logging.Logger, history_df: pd.DataFrame | None = None) -> None:
    plot_confusion_matrix(metrics, config, logger)
    plot_roc_curve(metrics, config, logger)
    plot_precision_recall_curve(metrics, config, logger)

    if history_df is not None:
        plot_training_history(history_df, config, logger)
        if config["model"].get("scheduler_enabled", False):
            plot_learning_rate_curve(history_df, config, logger)

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

    early_stopping_enabled = config["model"].get("early_stopping", False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    ax.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")

    if early_stopping_enabled and not history_df.empty:
        best_epoch = int(history_df.iloc[-1]["best_epoch"])
        best_val_loss = float(history_df.iloc[-1]["best_val_loss"])

        ax.axvline(best_epoch, linestyle="--", alpha=0.7, label=f"restored epoch={best_epoch}")
        ax.scatter([best_epoch], [best_val_loss])
        ax.set_title("CNN training history with early stopping")
    else:
        ax.set_title("CNN training history")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
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

def plot_learning_rate_curve(history_df: pd.DataFrame, config: dict, logger) -> None:
    output_dir = BASE_DIR / config["output"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "learning_rate_curve.jpg"

    if "learning_rate" not in history_df.columns:
        logger.warning("Skipping learning rate plot: 'learning_rate' column not found in training history")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history_df["epoch"], history_df["learning_rate"], marker="o", label="learning_rate")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("CNN learning rate history")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved learning rate curve to: {save_path}")

def evaluate_and_save_split(model, data_loader, split_name: str, threshold: float, device, config: dict, logger: logging.Logger,
                            training_summary: dict | None = None, history_df: pd.DataFrame | None = None, model_params: dict | None = None) -> dict:
    if model_params is None:
        model_params = config["model"]

    metrics = evaluate_model(
        model=model,
        data_loader=data_loader,
        split_name=split_name,
        threshold=threshold,
        device=device,
        logger=logger
    )

    summary_row = build_results_summary_row(
        metrics=metrics,
        config=config,
        model_params=model_params,
        training_summary=training_summary
    )

    summary_csv_path = BASE_DIR / config["output"]["summary_path"]
    append_results_to_csv(summary_row, summary_csv_path)
    logger.info(f"Added {split_name} results to summary CSV: {summary_csv_path}")

    if config["output"]["save_metrics"]:
        save_metrics(metrics, config, logger)

    if config["output"]["save_plots"]:
        save_visualizations(metrics, config, logger, history_df)

    if config["output"]["save_predictions"]:
        save_predictions(metrics, config, logger)

    return metrics

def make_hashable_tuning_value(value):
    if isinstance(value, list):
        return tuple(make_hashable_tuning_value(item) for item in value)
    if isinstance(value, dict):
        return tuple((key, make_hashable_tuning_value(value[key])) for key in sorted(value))
    return value

def tuning_rule_matches(params: dict, condition: dict) -> bool:
    return all(params.get(param_name) == expected_value for param_name, expected_value in condition.items())

def get_effective_tuning_key(params: dict) -> tuple:
    effective_params = params.copy()

    for rule in IGNORED_TUNING_PARAMS_RULES:
        if tuning_rule_matches(params, rule["when"]):
            for param_name in rule["ignore"]:
                if param_name in effective_params:
                    effective_params[param_name] = None

    return tuple((param_name, make_hashable_tuning_value(value)) for param_name, value in effective_params.items())

def get_tuning_param_grid(config: dict, logger: logging.Logger) -> list[dict]:
    param_grid_cfg = config["tuning_stage_1"]["param_grid"]

    for param_name, values in param_grid_cfg.items():
        if not isinstance(values, list) or len(values) == 0:
            logger.critical(f"Invalid tuning values for parameter '{param_name}'. Expected non-empty list")
            exit(1)

    param_names = list(param_grid_cfg.keys())
    param_values = [param_grid_cfg[param_name] for param_name in param_names]

    raw_param_grid = [
        dict(zip(param_names, combination))
        for combination in product(*param_values)
    ]

    seen_effective_configs = set()
    param_grid = []

    for params in raw_param_grid:
        effective_key = get_effective_tuning_key(params)

        if effective_key in seen_effective_configs:
            continue

        seen_effective_configs.add(effective_key)
        param_grid.append(params)

    logger.info(f"Generated {len(param_grid)} Autoencoder tuning combinations")
    if len(param_grid) < len(raw_param_grid):
        logger.info(f"Skipped {len(raw_param_grid) - len(param_grid)} duplicate effective tuning combinations")
    logger.info(f"Tuned parameters: {param_names}")

def run_single_cnn_experiment(params: dict, X_train, X_val, y_train, y_val, config: dict, device, logger: logging.Logger) -> tuple[dict, nn.Module, pd.DataFrame, dict]:
    experiment_config = copy.deepcopy(config)
    experiment_config["model"].update(params)

    logger.info(f"Running CNN tuning experiment with params: {params}")

    set_seed(experiment_config["experiment"]["random_state"], logger)
    batch_size = experiment_config["model"]["batch_size"]

    train_loader = create_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)

    val_loader = create_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)

    model = build_model(input_dim=X_train.shape[1], config=experiment_config, overrides={}, logger=logger)

    model, history_df, loss_info = train_model(model=model, train_loader=train_loader, val_loader=val_loader, y_train=y_train,
                                               config=experiment_config, device=device, logger=logger)

    training_summary = get_training_summary(history_df)
    training_summary.update(loss_info)

    threshold = experiment_config["model"]["decision_threshold"]

    val_metrics = evaluate_model(model=model, data_loader=val_loader, split_name="Validation", threshold=threshold,
                                 device=device, logger=logger)

    result_row = build_results_summary_row(metrics=val_metrics, config=config, model_params=params, training_summary=training_summary)

    return result_row, model, history_df, training_summary

def tuning_stage_1(X_train, y_train, X_val, y_val, config: dict, device, logger: logging.Logger) -> tuple[dict, nn.Module, pd.DataFrame, pd.DataFrame, dict]:
    logger.info("Starting CNN tuning stage 1")

    metric_name = config["tuning_stage_1"]["metric"]
    param_grid = get_tuning_param_grid(config, logger)
    results = []
    best_metric_value = -1.0
    best_params = None
    best_model = None
    best_history_df = None
    best_training_summary = None

    for trial_number, params in enumerate(param_grid, start=1):
        logger.info(f"Starting CNN tuning trial {trial_number}/{len(param_grid)}")

        result_row, model, history_df, training_summary = run_single_cnn_experiment(params=params, X_train=X_train,
                                    X_val=X_val, y_train=y_train, y_val=y_val, config=config, device=device, logger=logger)

        selected_metric_value = result_row[metric_name]

        result_row["trial"] = trial_number
        result_row["selected_metric"] = selected_metric_value
        results.append(result_row)

        logger.info(f"Tuning trial {trial_number}/{len(param_grid)} completed - {metric_name}={selected_metric_value:.4f}, params={params}")

        if selected_metric_value > best_metric_value:
            best_metric_value = selected_metric_value
            best_params = params
            best_model = model
            best_history_df = history_df
            best_training_summary = training_summary

            logger.info(f"New best CNN tuning result: {metric_name}={best_metric_value:.4f}, params={best_params}")

    results_df = pd.DataFrame(results)

    logger.info(f"CNN tuning stage 1 completed. Best {metric_name}={best_metric_value:.4f}, best_params={best_params}")

    return best_params, best_model, results_df, best_history_df, best_training_summary

def plot_tuning_stage_1(results_df: pd.DataFrame, config: dict, logger: logging.Logger) -> None:
    output_dir = BASE_DIR / config["output"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_name = config["tuning_stage_1"]["metric"]
    save_path = output_dir / "tuning_stage1_metric_curve.jpg"

    best_idx = results_df[metric_name].idxmax()
    best_trial = results_df.loc[best_idx, "trial"]
    best_metric_value = results_df.loc[best_idx, metric_name]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(results_df["trial"], results_df[metric_name], marker="o", label=metric_name)
    ax.axvline(best_trial, linestyle="--", alpha=0.7, label=f"best trial={int(best_trial)}")
    ax.scatter([best_trial], [best_metric_value])

    ax.set_xlabel("Tuning trial")
    ax.set_ylabel(metric_name)
    ax.set_title(f"CNN tuning stage 1 - {metric_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved tuning stage 1 plot to: {save_path}")

def save_stage_results(results_df: pd.DataFrame, best_params: dict, output_dir: Path, stage: str, logger) -> None:
    path = BASE_DIR / output_dir
    path.mkdir(parents=True, exist_ok=True)
    csv_path = path / f"tuning_stage{stage}_results.csv"
    json_path = path / f"tuning_stage{stage}_params.json"

    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved tuning stage {stage} results to: {csv_path}")

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(best_params, file, indent=4, ensure_ascii=False)

    logger.info(f"Saved tuning stage {stage} best params to: {json_path}")

def tune_decision_threshold(y_true: list[int], y_proba: list[float], metric_name: str, start: float, stop: float,
                            step: float, logger: logging.Logger) -> tuple[float, pd.DataFrame]:
    metric_name = metric_name.lower()

    if metric_name not in ["accuracy", "precision", "recall", "f1"]:
        logger.critical(f"Unsupported threshold tuning metric: {metric_name}. Use one of: accuracy, precision, recall, f1")
        exit(1)
    if step <= 0:
        logger.critical("threshold_step must be greater than 0")
        exit(1)
    if start >= stop:
        logger.critical("threshold_start must be lower than threshold_stop")
        exit(1)

    logger.info(f"Starting decision threshold tuning: metric={metric_name}, start={start}, stop={stop}, step={step}")

    results = []
    best_threshold = start
    best_metric_value = -1.0
    threshold = start

    while threshold <= stop + 1e-9:
        threshold = round(threshold, 6)

        y_pred = (np.array(y_proba) >= threshold).astype(int).tolist()
        metrics = calculate_binary_metrics(y_true, y_pred, y_proba)
        selected_metric_value = metrics[metric_name]

        results.append({
            "threshold": threshold,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "average_precision": metrics["average_precision"],
            "selected_metric": selected_metric_value
        })

        logger.info(f"Threshold={threshold:.4f} - accuracy={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, "
                    f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}, {metric_name}={selected_metric_value:.4f}")

        if selected_metric_value > best_metric_value:
            best_metric_value = selected_metric_value
            best_threshold = threshold

        threshold += step

    results_df = pd.DataFrame(results)

    logger.info(f"Best decision threshold: {best_threshold:.4f} with {metric_name}={best_metric_value:.4f}")

    return best_threshold, results_df

def save_threshold_tuning_results(results_df: pd.DataFrame, best_threshold: float, config: dict, logger: logging.Logger) -> None:
    output_dir = BASE_DIR / config["output"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "threshold_tuning_results.csv"
    json_path = output_dir / "threshold_tuning_best_params.json"

    results_df.to_csv(csv_path, index=False)

    best_params = {
        "decision_threshold": best_threshold,
        "metric": config.get("tuning_stage_2", {}).get("metric", "f1")
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(best_params, file, indent=4, ensure_ascii=False)

    logger.info(f"Saved threshold tuning results to: {csv_path}")
    logger.info(f"Saved best threshold params to: {json_path}")

def plot_threshold_tuning_results(results_df: pd.DataFrame, metric_name: str, config: dict, logger: logging.Logger) -> None:
    output_dir = BASE_DIR / config["output"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_name = metric_name.lower()
    save_path = output_dir / "threshold_tuning_curve.jpg"

    if metric_name not in results_df.columns:
        logger.warning(f"Skipping threshold tuning plot: metric {metric_name} not found in results")
        return

    best_idx = results_df[metric_name].idxmax()
    best_threshold = results_df.loc[best_idx, "threshold"]
    best_metric_value = results_df.loc[best_idx, metric_name]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(results_df["threshold"], results_df[metric_name], marker="o", label=metric_name)
    ax.axvline(best_threshold, linestyle="--", alpha=0.7, label=f"best threshold={best_threshold:.2f}")
    ax.scatter([best_threshold], [best_metric_value])

    ax.set_xlabel("Decision threshold")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Decision threshold tuning - {metric_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved threshold tuning plot to: {save_path}")

def tuning_stage_2(model, val_loader, config: dict, device, logger: logging.Logger) -> tuple[dict, pd.DataFrame]:
    logger.info("Starting CNN tuning stage 2.")

    stage_2_cfg = config["tuning_stage_2"]
    metric_name = stage_2_cfg["metric"]

    y_true, y_proba = predict_proba(model, val_loader, device)

    best_threshold, threshold_results_df = tune_decision_threshold(
        y_true=y_true,
        y_proba=y_proba,
        metric_name=metric_name,
        start=stage_2_cfg["threshold_start"],
        stop=stage_2_cfg["threshold_stop"],
        step=stage_2_cfg["threshold_step"],
        logger=logger
    )

    best_params = {
        "decision_threshold": best_threshold,
        "metric": metric_name
    }

    save_threshold_tuning_results(results_df=threshold_results_df, best_threshold=best_threshold, config=config, logger=logger)
    plot_threshold_tuning_results(results_df=threshold_results_df, metric_name=metric_name, config=config, logger=logger)

    logger.info(f"CNN tuning stage 2 completed. Best threshold={best_threshold:.4f}, metric={metric_name}")

    return best_params, threshold_results_df

def save_predictions(metrics: dict, config: dict, logger: logging.Logger) -> None:
    output_dir = BASE_DIR / config["output"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    split_name = metrics["split_name"].lower()
    save_path = output_dir / f"{split_name}_predictions.csv"

    predictions_df = pd.DataFrame({
        "y_true": metrics["y_true"],
        "y_proba": metrics["y_proba"],
        "y_pred": metrics["y_pred"],
        "threshold": metrics["threshold_used"]
    })

    predictions_df.to_csv(save_path, index=False)

    logger.info(f"Saved {metrics['split_name']} predictions to: {save_path}")

def train_final_model(model: nn.Module, train_loader: DataLoader, y_train_final: pd.Series, config: dict,
                      device: torch.device, logger: logging.Logger, fixed_epochs: int) -> tuple[nn.Module, dict]:
    model_cfg = config["model"]
    criterion, loss_info = build_loss_function(y_train=y_train_final, config=config, device=device, logger=logger)
    optimizer = Adam(
        model.parameters(),
        lr=model_cfg["learning_rate"],
        weight_decay=model_cfg["weight_decay"]
    )

    model.to(device)
    logger.info(f"Retraining final model on Train+Val for EXACTLY {fixed_epochs} epochs")

    for epoch in range(1, fixed_epochs + 1):
        model.train()
        train_loss = 0.0
        total_samples = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()

            gradient_clip_norm = model_cfg["gradient_clip_norm"]

            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)

        train_loss = train_loss / total_samples

        if epoch % 10 == 0 or epoch == fixed_epochs:
            logger.info(f"Final Retrain - Epoch {epoch}/{fixed_epochs} - train_loss={train_loss:.6f}")

    return model, loss_info

def save_model(model: nn.Module, config: dict, logger: logging.Logger, input_dim: int, feature_columns: list[str] | None = None,
               model_params: dict | None = None, threshold: float | None = None, training_summary: dict | None = None) -> None:
    output_dir = BASE_DIR / config["output"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / "model.pt"
    model_config = config["model"].copy()

    if model_params is not None:
        model_config.update(model_params)

    checkpoint = {
        "experiment": config["experiment"]["name"],
        "dataset_variant": config["data"]["dataset_variant"],
        "model_type": "cnn",
        "model_class": model.__class__.__name__,
        "input_dim": input_dim,
        "input_shape": [1, input_dim],
        "feature_columns": feature_columns,
        "model_config": model_config,
        "decision_threshold": threshold,
        "training_summary": training_summary,
        "state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "full_config": config
    }

    torch.save(checkpoint, save_path)

    logger.info(f"Saved CNN model checkpoint to: {save_path}")

def main() -> None:
    config = load_config(CONFIG_PATH)
    logger = get_logger(config)
    log_config(config, logger)

    logger.info("Start experiment")
    set_seed(config["experiment"]["random_state"], logger)
    device = get_device(config, logger)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_cnn_data(config)
    logger.info("Data prepared successfully")

    stage_1_enabled = config.get("tuning_stage_1", {}).get("enabled", False)
    stage_2_enabled = config.get("tuning_stage_2", {}).get("enabled", False)

    if stage_2_enabled and not stage_1_enabled:
        logger.critical("Tuning stage 2 cannot be enabled without tuning stage 1")
        exit(1)

    if stage_1_enabled:
        logger.info("Tuning mode enabled")

        best_stage_1_params, _, stage_1_results_df, best_history_df, best_training_summary = tuning_stage_1(
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, config=config, device=device, logger=logger)

        if config["output"]["save_tuning_results"]:
            save_stage_results(results_df=stage_1_results_df, best_params=best_stage_1_params, output_dir=config["output"]["output_dir"], stage="1", logger=logger)

        if config["output"]["save_plots"]:
            plot_tuning_stage_1(stage_1_results_df, config, logger)

        save_training_history(best_history_df, config, logger)

        best_config = copy.deepcopy(config)
        best_config["model"].update(best_stage_1_params)

        logger.info("Preparing final train+val dataset for retraining")
        X_train_final = pd.concat([X_train, X_val], axis=0)
        y_train_final = pd.concat([y_train, y_val], axis=0)
        logger.info(f"Final train+val shapes: X={X_train_final.shape}, y={y_train_final.shape}")

        batch_size = best_config["model"]["batch_size"]
        train_final_loader = create_dataloader(X_train_final, y_train_final, batch_size=batch_size, shuffle=True)
        test_loader = create_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)

        final_model = build_model(input_dim=X_train_final.shape[1], config=best_config, overrides={}, logger=logger)
        best_epoch = int(best_training_summary["best_epoch"])

        final_model, final_loss_info = train_final_model(model=final_model, train_loader=train_final_loader, y_train_final=y_train_final,
                                                         config=best_config, device=device, logger=logger, fixed_epochs=best_epoch)

        best_training_summary.update(final_loss_info)
        best_threshold = best_config["model"].get("decision_threshold", 0.5)

        if stage_2_enabled:
            logger.info("Tuning stage 2 is enabled")
            val_final_loader = create_dataloader(X_train_final, y_train_final, batch_size=batch_size, shuffle=False)

            best_stage_2_params, _ = tuning_stage_2(model=final_model, val_loader=val_final_loader,
                config=config, device=device, logger=logger)

            best_threshold = best_stage_2_params["decision_threshold"]

        evaluate_and_save_split(model=final_model, data_loader=test_loader, split_name="Test", threshold=best_threshold,
                                device=device, config=config, logger=logger, training_summary=best_training_summary,
                                history_df=best_history_df, model_params=best_stage_1_params)
        
        if config["output"].get("save_model", False):
            save_model(model=final_model, config=best_config, logger=logger, input_dim=X_train_final.shape[1], feature_columns=X_train_final.columns.tolist(),
                model_params=best_stage_1_params, threshold=best_threshold, training_summary=best_training_summary)

    else:
        logger.info("Standard run mode")

        batch_size = config["model"]["batch_size"]
        train_loader = create_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)

        val_loader = create_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)

        model = build_model(input_dim=X_train.shape[1], config=config, overrides={}, logger=logger)

        model, history_df, loss_info = train_model(model=model, train_loader=train_loader, val_loader=val_loader, y_train=y_train,
                                                   config=config, device=device, logger=logger)

        training_summary = get_training_summary(history_df)
        training_summary.update(loss_info)
        save_training_history(history_df, config, logger)

        threshold = config["model"].get("decision_threshold", 0.5)

        evaluate_and_save_split(model=model, data_loader=val_loader, split_name="Validation", threshold=threshold, device=device,
                                config=config, logger=logger, training_summary=training_summary, history_df=history_df)

if __name__ == "__main__":
    main()
