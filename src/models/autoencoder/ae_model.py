import logging
import csv
#import winsound
from pathlib import Path
import yaml
from ae_data import prepare_ae_data, get_logger
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
CONFIG_PATH = BASE_DIR / "config" / "autoencoder.yaml"

RESULTS_COLUMNS = ["experiment", "dataset_variant", "split", "accuracy", "precision", "recall", "f1", "roc_auc",
    "average_precision", "reconstruction_threshold", "use_network_features", "drop_original_port_columns", "log_transform",
    "log_transform_columns", "scaling", "scaler", "feature_selection", "feature_selection_method", "selected_k_features",
    "remove_correlated_features", "correlation_threshold", "encoder_layers", "latent_dim", "activation", "output_activation", 
    "batch_norm", "dropout", "loss_function", "learning_rate", "batch_size", "epochs", "scheduler_enabled", "scheduler_factor", 
    "scheduler_patience", "scheduler_min_lr", "denoising", "noise_std", "weight_decay", "device",  "early_stopping", "patience", 
    "min_delta", "actual_epochs", "best_epoch", "best_val_loss", "threshold_metric", "threshold_candidates", 
    "tuning_stage_1", "tuning_stage_2"]

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

def add_hidden_layer(layers: list, input_dim: int, output_dim: int, activation: str, dropout: float, batch_norm: bool) -> None:
    layers.append(nn.Linear(input_dim, output_dim))

    if batch_norm:
        layers.append(nn.BatchNorm1d(output_dim))

    layers.append(get_activation_layer(activation))

    if dropout > 0:
        layers.append(nn.Dropout(dropout))

class AutoencoderNetwork(nn.Module):
    def __init__(self, input_dim: int, encoder_layers: list[int], latent_dim: int, dropout: float, activation: str,
                 batch_norm: bool, output_activation: str):
        super().__init__()

        encoder = []
        previous_dim = input_dim

        for hidden_dim in encoder_layers:
            add_hidden_layer(layers=encoder, input_dim=previous_dim, output_dim=hidden_dim, activation=activation, dropout=dropout,
                             batch_norm=batch_norm)
            previous_dim = hidden_dim

        encoder.append(nn.Linear(previous_dim, latent_dim))
        encoder.append(get_activation_layer(activation))

        decoder = []
        previous_dim = latent_dim

        for hidden_dim in reversed(encoder_layers):
            add_hidden_layer(layers=decoder, input_dim=previous_dim, output_dim=hidden_dim, activation=activation, dropout=dropout,
                             batch_norm=batch_norm)
            previous_dim = hidden_dim

        decoder.append(nn.Linear(previous_dim, input_dim))

        if output_activation != "none":
            decoder.append(get_activation_layer(output_activation))

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

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

def build_model(input_dim: int, config: dict, overrides: dict, logger) -> AutoencoderNetwork:
    model_cfg = config["model"].copy()

    if overrides:
        model_cfg.update(overrides)

    logger.info("Building Autoencoder model")
    logger.info(f"Model parameters: input_dim={input_dim}, encoder_layers={model_cfg['encoder_layers']}, latent_dim={model_cfg['latent_dim']}, "
                f"dropout={model_cfg['dropout']}, activation={model_cfg['activation']}, output_activation={model_cfg['output_activation']}, "
                f"batch_norm={model_cfg['batch_norm']}")

    model = AutoencoderNetwork(input_dim=input_dim, encoder_layers=model_cfg["encoder_layers"], latent_dim=model_cfg["latent_dim"],
                               dropout=model_cfg["dropout"], activation=model_cfg["activation"], batch_norm=model_cfg["batch_norm"],
                               output_activation=model_cfg["output_activation"])

    return model

def create_reconstruction_dataloader(X, batch_size: int, shuffle: bool) -> DataLoader:
    X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)
    dataset = TensorDataset(X_tensor)

    use_cuda = torch.cuda.is_available()
    workers = min(4, os.cpu_count() or 1)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=use_cuda,
                      persistent_workers=True if workers > 0 else False)

def create_labeled_dataloader(X, y, batch_size: int, shuffle: bool) -> DataLoader:
    X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)
    y_tensor = torch.tensor(y.to_numpy(), dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)

    use_cuda = torch.cuda.is_available()
    workers = min(4, os.cpu_count() or 1)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=use_cuda,
                      persistent_workers=True if workers > 0 else False)

def calculate_reconstruction_loss(model, data_loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for (X_batch,) in data_loader:
            X_batch = X_batch.to(device)

            reconstructed = model(X_batch)
            loss = criterion(reconstructed, X_batch)

            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)

    return total_loss / total_samples

def build_reconstruction_loss(config: dict, logger: logging.Logger) -> tuple[nn.Module, dict]:
    model_cfg = config["model"]
    loss_name = model_cfg["loss_function"].lower()
    loss_info = {"loss_function": loss_name}

    if loss_name == "mse":
        logger.info("Using MSELoss for reconstruction")
        return nn.MSELoss(), loss_info
    if loss_name == "mae":
        logger.info("Using L1Loss for reconstruction")
        return nn.L1Loss(), loss_info
    if loss_name == "smooth_l1":
        logger.info("Using SmoothL1Loss for reconstruction")
        return nn.SmoothL1Loss(), loss_info

    logger.critical(f"Unsupported reconstruction loss function: {loss_name}")
    exit(1)

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

def add_denoising_noise(X_batch: torch.Tensor, noise_std: float) -> torch.Tensor:
    if noise_std <= 0:
        return X_batch
    noise = torch.randn_like(X_batch) * noise_std
    return X_batch + noise

def train_model(model, train_loader, val_loader, config: dict, device, logger):
    model_cfg = config["model"]

    criterion, loss_info = build_reconstruction_loss(config=config, logger=logger)
    optimizer = Adam(model.parameters(), lr=model_cfg["learning_rate"], weight_decay=model_cfg["weight_decay"])

    scheduler = build_scheduler(optimizer, config, logger)
    epochs = model_cfg["epochs"]
    history = []

    early_stopping_enabled = model_cfg["early_stopping"]
    early_stopping = None

    if early_stopping_enabled:
        early_stopping = EarlyStopping(patience=model_cfg["patience"], min_delta=model_cfg["min_delta"])
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

        for (X_batch,) in train_loader:
            X_batch = X_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            if model_cfg["denoising"]:
                model_input = add_denoising_noise(X_batch, model_cfg["noise_std"])
            else:
                model_input = X_batch
            reconstructed = model(model_input)
            loss = criterion(reconstructed, X_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)

        train_loss = train_loss / total_samples
        val_loss = calculate_reconstruction_loss(model, val_loader, criterion, device)

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

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "best_val_loss": best_val_loss,
                        "best_epoch": best_epoch, "epochs_without_improvement": epochs_without_improvement, "improved": improved,
                        "learning_rate": current_lr})

        logger.info(f"Epoch {epoch}/{epochs} - train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, best_val_loss={best_val_loss:.6f}, "
                    f"best_epoch={best_epoch}, epochs_without_improvement={epochs_without_improvement}, learning_rate={current_lr:.8f}")

        if early_stopping_enabled and early_stopping.should_stop:
            logger.info(f"Early stopping triggered at epoch {epoch}. Best epoch: {early_stopping.best_epoch}, best_val_loss="
                        f"{early_stopping.best_loss:.6f}")
            break

    if early_stopping_enabled and early_stopping is not None:
        early_stopping.restore_best_weights(model, device, logger)

    logger.info("Autoencoder training completed")
    return model, pd.DataFrame(history), loss_info

def calculate_reconstruction_errors_per_sample(reconstructed: torch.Tensor, X_batch: torch.Tensor, error_metric: str, logger) -> torch.Tensor:
    error_metric = error_metric.lower()

    if error_metric == "mse":
        return torch.mean((reconstructed - X_batch) ** 2, dim=1)
    if error_metric == "mae":
        return torch.mean(torch.abs(reconstructed - X_batch), dim=1)
    if error_metric == "smooth_l1":
        return nn.functional.smooth_l1_loss(reconstructed, X_batch, reduction="none").mean(dim=1)

    logger.critical(f"Unsupported reconstruction error metric: {error_metric}")
    exit(1)

def compute_reconstruction_errors(model, data_loader, device, config: dict, logger) -> tuple[list[int], list[float]]:
    model.eval()
    model_cfg = config["model"]

    y_true_list = []
    error_list = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)

            reconstructed = model(X_batch)
            errors = calculate_reconstruction_errors_per_sample(reconstructed=reconstructed, X_batch=X_batch,
                                                                error_metric=model_cfg["loss_function"], logger=logger)

            y_true_list.append(y_batch.detach().cpu())
            error_list.append(errors.detach().cpu())

    y_true_tensor = torch.cat(y_true_list)
    error_tensor = torch.cat(error_list)

    y_true = y_true_tensor.to(torch.int32).tolist()
    reconstruction_errors = error_tensor.tolist()

    return y_true, reconstruction_errors

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

def evaluate_model(model, data_loader, split_name: str, threshold: float, device, config: dict, logger) -> dict:
    logger.info(f"Evaluating autoencoder on {split_name} set")

    y_true, reconstruction_errors = compute_reconstruction_errors(model=model, data_loader=data_loader, device=device,
                                                                  config=config, logger=logger)
    y_pred = (np.array(reconstruction_errors) >= threshold).astype(int).tolist()

    metrics = calculate_binary_metrics(y_true, y_pred, reconstruction_errors)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)

    logger.info(f"{split_name} Reconstruction threshold used: {threshold:.6f}")
    logger.info(f"{split_name} Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"{split_name} Precision: {metrics['precision']:.4f}")
    logger.info(f"{split_name} Recall: {metrics['recall']:.4f}")
    logger.info(f"{split_name} F1-score: {metrics['f1']:.4f}")
    if metrics["roc_auc"] is not None:
        logger.info(f"{split_name} ROC-AUC: {metrics['roc_auc']:.4f}")
    else:
        logger.info(f"{split_name} ROC-AUC: None")
    if metrics["average_precision"] is not None:
        logger.info(f"{split_name} Average Precision: {metrics['average_precision']:.4f}")
    else:
        logger.info(f"{split_name} Average Precision: None")
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
        "reconstruction_errors": reconstruction_errors,
        "anomaly_scores": reconstruction_errors
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
        file.write(f"Reconstruction threshold: {metrics['threshold_used']:.6f}\n")
        file.write(f"Accuracy: {metrics['accuracy']:.4f}\nPrecision: {metrics['precision']:.4f}\n")
        file.write(f"Recall: {metrics['recall']:.4f}\nF1-score: {metrics['f1']:.4f}\n")
        roc_auc = metrics.get("roc_auc")
        average_precision = metrics.get("average_precision")
        file.write(f"ROC-AUC: {roc_auc:.4f}\n" if roc_auc is not None else "ROC-AUC: None\n")
        file.write(f"Average Precision: {average_precision:.4f}\n" if average_precision is not None else "Average Precision: None\n")
        file.write(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}\n\n")
        file.write(f"Classification Report:\n{metrics['classification_report']}\n")

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

    features_cfg = config["features"]
    stage_1_cfg = config["tuning_stage_1"]
    stage_2_cfg = config["tuning_stage_2"]

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
        "reconstruction_threshold": metrics["threshold_used"],

        "use_network_features": features_cfg.get("use_network_features", False),
        "drop_original_port_columns": features_cfg.get("drop_original_port_columns", False),
        "log_transform": features_cfg.get("log_transform", False),
        "log_transform_columns": features_cfg.get("log_transform_columns", None),
        "scaling": features_cfg.get("scaling", False),
        "scaler": features_cfg.get("scaler", None),
        "feature_selection": features_cfg["use_feature_selection"],
        "feature_selection_method": features_cfg["feature_selection_method"],
        "selected_k_features": features_cfg["selected_k_features"],
        "remove_correlated_features": features_cfg["remove_correlated_features"],
        "correlation_threshold": features_cfg["correlation_threshold"],

        "encoder_layers": model_cfg["encoder_layers"],
        "latent_dim": model_cfg["latent_dim"],
        "activation": model_cfg["activation"],
        "output_activation": model_cfg["output_activation"],
        "batch_norm": model_cfg["batch_norm"],
        "dropout": model_cfg["dropout"],
        "loss_function": model_cfg["loss_function"],

        "learning_rate": model_cfg["learning_rate"],
        "batch_size": model_cfg["batch_size"],
        "epochs": model_cfg["epochs"],
        "scheduler_enabled": model_cfg["scheduler_enabled"],
        "scheduler_factor": model_cfg["scheduler_factor"],
        "scheduler_patience": model_cfg["scheduler_patience"],
        "scheduler_min_lr": model_cfg["scheduler_min_lr"],
        "denoising": model_cfg["denoising"],
        "noise_std": model_cfg["noise_std"],
        "weight_decay": model_cfg["weight_decay"],
        "device": model_cfg["device"],
        "early_stopping": model_cfg["early_stopping"],
        "patience": model_cfg["patience"],
        "min_delta": model_cfg["min_delta"],
        "actual_epochs": training_summary.get("actual_epochs"),
        "best_epoch": training_summary.get("best_epoch"),
        "best_val_loss": training_summary.get("best_val_loss"),

        "threshold_metric": stage_2_cfg["metric"],
        "threshold_candidates": stage_2_cfg["threshold_candidates"],
        "tuning_stage_1": stage_1_cfg["enabled"],
        "tuning_stage_2": stage_2_cfg["enabled"]
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
    if metrics["anomaly_scores"] is None or metrics["roc_auc"] is None:
        logger.warning("Skipping ROC curve.")
        return

    split_name = metrics["split_name"].lower()
    save_path = BASE_DIR / config["output"]["output_dir"] / f"{split_name}_roc_curve.jpg"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    y_true = metrics["y_true"]
    anomaly_scores = metrics["anomaly_scores"]

    RocCurveDisplay.from_predictions(y_true, anomaly_scores, name=f"Autoencoder (AUC = {metrics['roc_auc']:.4f})", ax=ax,
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
    if metrics["anomaly_scores"] is None or metrics["average_precision"] is None:
        logger.warning("Skipping Precision-Recall curve.")
        return

    split_name = metrics["split_name"].lower()
    save_path = BASE_DIR / config["output"]["output_dir"] / f"{split_name}_pr_curve.jpg"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    y_true = metrics["y_true"]
    anomaly_scores = metrics["anomaly_scores"]

    PrecisionRecallDisplay.from_predictions(y_true, anomaly_scores, name=f"Autoencoder (AP = {metrics['average_precision']:.4f})", ax=ax,
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
    plot_reconstruction_error_histogram(metrics, config, logger)

    if history_df is not None:
        plot_training_history(history_df, config, logger)
        plot_learning_rate_curve(history_df, config, logger)

def calculate_binary_metrics(y_true, y_pred, anomaly_scores) -> dict:
    try:
        roc_auc = roc_auc_score(y_true, anomaly_scores)
    except ValueError:
        roc_auc = None
    try:
        average_precision = average_precision_score(y_true, anomaly_scores)
    except ValueError:
        average_precision = None

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "average_precision": average_precision
    }

def plot_reconstruction_error_histogram(metrics: dict, config: dict, logger: logging.Logger) -> None:
    split_name = metrics["split_name"].lower()
    save_path = BASE_DIR / config["output"]["output_dir"] / f"{split_name}_reconstruction_error_histogram.jpg"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame({"y_true": metrics["y_true"], "reconstruction_error": metrics["reconstruction_errors"]})

    benign_errors = results_df.loc[results_df["y_true"] == 0, "reconstruction_error"]
    attack_errors = results_df.loc[results_df["y_true"] == 1, "reconstruction_error"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(benign_errors, bins=50, alpha=0.6, label="BENIGN")
    ax.hist(attack_errors, bins=50, alpha=0.6, label="ATTACK")
    ax.axvline(metrics["threshold_used"], linestyle="--", label=f"threshold={metrics['threshold_used']:.6f}")

    ax.set_xlabel("Reconstruction error")
    ax.set_ylabel("Count")
    ax.set_title(f"{metrics['split_name']} - Reconstruction error distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved reconstruction error histogram to: {save_path}")

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
        ax.set_title("Autoencoder training history with early stopping")
    else:
        ax.set_title("Autoencoder training history")

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
    ax.set_title("Autoencoder learning rate history")
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

    metrics = evaluate_model(model=model, data_loader=data_loader, split_name=split_name, threshold=threshold, device=device,
                             config=config, logger=logger)

    summary_row = build_results_summary_row(metrics=metrics, config=config, model_params=model_params, training_summary=training_summary)

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

def get_tuning_param_grid(config: dict, logger: logging.Logger) -> list[dict]:
    param_grid_cfg = config["tuning_stage_1"]["param_grid"]

    for param_name, values in param_grid_cfg.items():
        if not isinstance(values, list) or len(values) == 0:
            logger.critical(f"Invalid tuning values for parameter '{param_name}'. Expected non-empty list")
            exit(1)

    param_names = list(param_grid_cfg.keys())
    param_values = [param_grid_cfg[param_name] for param_name in param_names]

    param_grid = [
        dict(zip(param_names, combination))
        for combination in product(*param_values)
    ]

    logger.info(f"Generated {len(param_grid)} Autoencoder tuning combinations")
    logger.info(f"Tuned parameters: {param_names}")

    return param_grid

def get_stage_1_metric_value(result_row: dict, metric_name: str) -> float:
    metric_name = metric_name.lower()
    if metric_name in ["val_loss", "best_val_loss"]:
        return result_row["best_val_loss"]
    return result_row[metric_name]

def is_better_stage_1_result(metric_name: str, selected_value: float, best_value: float) -> bool:
    metric_name = metric_name.lower()
    if metric_name in ["val_loss", "best_val_loss"]:
        return selected_value < best_value
    return selected_value > best_value

def get_initial_best_metric_value(metric_name: str) -> float:
    metric_name = metric_name.lower()
    if metric_name in ["val_loss", "best_val_loss"]:
        return float("inf")
    return -1.0

def run_single_ae_experiment(params: dict, X_train_normal, X_val_normal, X_val, y_val, config: dict, device,
                             logger: logging.Logger) -> tuple[dict, nn.Module, pd.DataFrame, dict, float]:
    experiment_config = copy.deepcopy(config)
    experiment_config["model"].update(params)

    logger.info(f"Running Autoencoder tuning experiment with params: {params}")

    set_seed(experiment_config["experiment"]["random_state"], logger)

    batch_size = experiment_config["model"]["batch_size"]

    train_loader = create_reconstruction_dataloader(X_train_normal, batch_size=batch_size, shuffle=True)

    val_reconstruction_loader = create_reconstruction_dataloader(X_val_normal, batch_size=batch_size, shuffle=False)

    val_labeled_loader = create_labeled_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)

    model = build_model(input_dim=X_train_normal.shape[1], config=experiment_config, overrides={}, logger=logger)

    model, history_df, loss_info = train_model(model=model, train_loader=train_loader,
        val_loader=val_reconstruction_loader, config=experiment_config, device=device, logger=logger)

    training_summary = get_training_summary(history_df)
    training_summary.update(loss_info)

    stage_1_metric = experiment_config["tuning_stage_1"]["metric"].lower()
    stage_2_metric = experiment_config["tuning_stage_2"]["metric"].lower()

    if stage_1_metric in ["accuracy", "precision", "recall", "f1"]:
        threshold_metric = stage_1_metric
    else:
        threshold_metric = stage_2_metric

    if experiment_config["tuning_stage_2"]["enabled"] or stage_1_metric in ["accuracy", "precision", "recall", "f1"]:
        y_true, anomaly_scores = compute_reconstruction_errors(model=model, data_loader=val_labeled_loader,
            device=device, config=experiment_config, logger=logger)

        best_threshold, _ = tune_reconstruction_threshold(y_true=y_true, anomaly_scores=anomaly_scores,
            metric_name=threshold_metric,
            threshold_candidates=experiment_config["tuning_stage_2"]["threshold_candidates"], logger=logger)
    else:
        best_threshold = experiment_config["model"]["reconstruction_threshold"]

    val_metrics = evaluate_model(model=model, data_loader=val_labeled_loader, split_name="Validation",
        threshold=best_threshold, device=device, config=experiment_config, logger=logger)

    result_row = build_results_summary_row(metrics=val_metrics, config=experiment_config, model_params=params,
        training_summary=training_summary)

    result_row["trial_reconstruction_threshold"] = best_threshold
    result_row["selected_metric"] = get_stage_1_metric_value(result_row, stage_1_metric)

    return result_row, model, history_df, training_summary, best_threshold

def tuning_stage_1(X_train_normal, X_val_normal, X_val, y_val, config: dict, device, logger: logging.Logger) -> tuple[dict, nn.Module, pd.DataFrame, pd.DataFrame, dict, float]:
    logger.info("Starting Autoencoder tuning stage 1")

    metric_name = config["tuning_stage_1"]["metric"].lower()
    param_grid = get_tuning_param_grid(config, logger)

    results = []
    best_metric_value = get_initial_best_metric_value(metric_name)
    best_params = None
    best_model = None
    best_history_df = None
    best_training_summary = None
    best_threshold = None

    for trial_number, params in enumerate(param_grid, start=1):
        logger.info(f"Starting Autoencoder tuning trial {trial_number}/{len(param_grid)}")

        result_row, model, history_df, training_summary, threshold = run_single_ae_experiment(
            params=params,
            X_train_normal=X_train_normal,
            X_val_normal=X_val_normal,
            X_val=X_val,
            y_val=y_val,
            config=config,
            device=device,
            logger=logger
        )

        selected_metric_value = get_stage_1_metric_value(result_row, metric_name)

        result_row["trial"] = trial_number
        result_row["selected_metric"] = selected_metric_value
        results.append(result_row)

        logger.info(f"Tuning trial {trial_number}/{len(param_grid)} completed - {metric_name}={selected_metric_value:.6f}, params={params}")

        if is_better_stage_1_result(metric_name, selected_metric_value, best_metric_value):
            best_metric_value = selected_metric_value
            best_params = params
            best_model = model
            best_history_df = history_df
            best_training_summary = training_summary
            best_threshold = threshold

            logger.info(f"New best Autoencoder tuning result: {metric_name}={best_metric_value:.6f}, threshold="
                        f"{best_threshold:.6f}, params={best_params}")

    results_df = pd.DataFrame(results)

    logger.info(f"Autoencoder tuning stage 1 completed. Best {metric_name}={best_metric_value:.6f}, best_threshold="
                f"{best_threshold:.6f}, best_params={best_params}")

    return best_params, best_model, results_df, best_history_df, best_training_summary, best_threshold

def plot_tuning_stage_1(results_df: pd.DataFrame, config: dict, logger: logging.Logger) -> None:
    output_dir = BASE_DIR / config["output"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_name = config["tuning_stage_1"]["metric"].lower()
    save_path = output_dir / "tuning_stage1_metric_curve.jpg"

    if metric_name in ["val_loss", "best_val_loss"]:
        y_column = "best_val_loss"
        best_idx = results_df[y_column].idxmin()
    else:
        y_column = metric_name
        best_idx = results_df[y_column].idxmax()

    best_trial = results_df.loc[best_idx, "trial"]
    best_metric_value = results_df.loc[best_idx, y_column]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(results_df["trial"], results_df[y_column], marker="o", label=y_column)
    ax.axvline(best_trial, linestyle="--", alpha=0.7, label=f"best trial={int(best_trial)}")
    ax.scatter([best_trial], [best_metric_value])

    ax.set_xlabel("Tuning trial")
    ax.set_ylabel(y_column)
    ax.set_title(f"Autoencoder tuning stage 1 - {y_column}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved tuning stage 1 plot to: {save_path}")

def tune_reconstruction_threshold(y_true: list[int], anomaly_scores: list[float], metric_name: str, threshold_candidates: int,
                                  logger: logging.Logger) -> tuple[float, pd.DataFrame]:
    metric_name = metric_name.lower()

    if metric_name not in ["accuracy", "precision", "recall", "f1"]:
        logger.critical(f"Unsupported threshold tuning metric: {metric_name}. Use one of: accuracy, precision, recall, f1")
        exit(1)
    if threshold_candidates <= 1:
        logger.critical("threshold_candidates must be greater than 1")
        exit(1)

    scores_array = np.array(anomaly_scores)
    thresholds = np.unique(np.quantile(scores_array, np.linspace(0.0, 1.0, threshold_candidates)))

    logger.info(f"Starting reconstruction threshold tuning: metric={metric_name}, threshold_candidates={len(thresholds)}")

    results = []
    best_threshold = float(thresholds[0])
    best_metric_value = -1.0

    for threshold in thresholds:
        threshold = float(threshold)
        y_pred = (scores_array >= threshold).astype(int).tolist()
        metrics = calculate_binary_metrics(y_true, y_pred, anomaly_scores)
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

        if selected_metric_value > best_metric_value:
            best_metric_value = selected_metric_value
            best_threshold = threshold

    results_df = pd.DataFrame(results)
    logger.info(f"Best reconstruction threshold: {best_threshold:.6f} with {metric_name}={best_metric_value:.4f}")

    return best_threshold, results_df

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

def save_threshold_tuning_results(results_df: pd.DataFrame, best_threshold: float, config: dict, logger: logging.Logger) -> None:
    output_dir = BASE_DIR / config["output"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "threshold_tuning_results.csv"
    json_path = output_dir / "threshold_tuning_best_params.json"

    results_df.to_csv(csv_path, index=False)

    best_params = {
        "reconstruction_threshold": best_threshold,
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
    ax.axvline(best_threshold, linestyle="--", alpha=0.7, label=f"best threshold={best_threshold:.6f}")
    ax.scatter([best_threshold], [best_metric_value])
    ax.set_xlabel("Reconstruction threshold")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Reconstruction threshold tuning - {metric_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved threshold tuning plot to: {save_path}")

def tuning_stage_2(model, val_loader, config: dict, device, logger: logging.Logger) -> tuple[dict, pd.DataFrame]:
    logger.info("Starting Autoencoder tuning stage 2.")

    stage_2_cfg = config["tuning_stage_2"]
    metric_name = stage_2_cfg["metric"]

    y_true, anomaly_scores = compute_reconstruction_errors(model, val_loader, device, config, logger)

    best_threshold, threshold_results_df = tune_reconstruction_threshold(y_true=y_true, anomaly_scores=anomaly_scores,
                                           metric_name=metric_name, threshold_candidates=stage_2_cfg["threshold_candidates"], logger=logger)
    best_params = {
        "reconstruction_threshold": best_threshold,
        "metric": metric_name
    }

    save_threshold_tuning_results(results_df=threshold_results_df, best_threshold=best_threshold, config=config, logger=logger)
    plot_threshold_tuning_results(results_df=threshold_results_df, metric_name=metric_name, config=config, logger=logger)

    logger.info(f"Autoencoder tuning stage 2 completed. Best threshold={best_threshold:.4f}, metric={metric_name}")

    return best_params, threshold_results_df

def save_predictions(metrics: dict, config: dict, logger: logging.Logger) -> None:
    output_dir = BASE_DIR / config["output"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    split_name = metrics["split_name"].lower()
    save_path = output_dir / f"{split_name}_predictions.csv"

    predictions_df = pd.DataFrame({
        "y_true": metrics["y_true"],
        "reconstruction_error": metrics["reconstruction_errors"],
        "y_pred": metrics["y_pred"],
        "reconstruction_threshold": metrics["threshold_used"]
    })

    predictions_df.to_csv(save_path, index=False)

    logger.info(f"Saved {metrics['split_name']} predictions to: {save_path}")

def save_model(model: nn.Module, config: dict, logger: logging.Logger, model_params: dict | None = None, threshold: float | None = None,
               training_summary: dict | None = None) -> None:
    output_dir = BASE_DIR / config["output"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "model.pt"
    model_config = config["model"].copy()

    if model_params is not None:
        model_config.update(model_params)

    first_encoder_layer = model.encoder[0]
    input_dim = first_encoder_layer.in_features if isinstance(first_encoder_layer, nn.Linear) else None

    checkpoint = {
        "experiment": config["experiment"]["name"],
        "dataset_variant": config["data"]["dataset_variant"],
        "model_type": "autoencoder",
        "model_class": model.__class__.__name__,
        "input_dim": input_dim,
        "model_config": model_config,
        "reconstruction_threshold": threshold,
        "training_summary": training_summary,
        "state_dict": model.state_dict(),
        "full_config": config
    }

    torch.save(checkpoint, save_path)

    logger.info(f"Saved autoencoder model checkpoint to: {save_path}")

def main() -> None:
    config = load_config(CONFIG_PATH)
    logger = get_logger(config)
    log_config(config, logger)

    logger.info("Start autoencoder experiment")
    set_seed(config["experiment"]["random_state"], logger)
    device = get_device(config, logger)

    X_train_normal, X_val, X_test, y_train_normal, y_val, y_test = prepare_ae_data(config)
    logger.info("Data prepared successfully")

    normal_label = config["data"]["normal_label"]
    X_val_normal = X_val.loc[y_val == normal_label].copy()

    if X_val_normal.empty:
        logger.critical(f"No normal validation samples found for normal_label={normal_label}")
        exit(1)

    logger.info(f"X_train_normal shape: {X_train_normal.shape}\nX_val_normal shape: {X_val_normal.shape}")
    logger.info(f"X_val full shape: {X_val.shape}\nX_test shape: {X_test.shape}")
    logger.info(f"y_train_normal shape: {y_train_normal.shape}\ny_val shape: {y_val.shape}")
    logger.info(f"y_test shape: {y_test.shape}")

    stage_1_enabled = config["tuning_stage_1"]["enabled"]
    stage_2_enabled = config["tuning_stage_2"]["enabled"]

    batch_size = config["model"]["batch_size"]

    val_labeled_loader = create_labeled_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)

    test_loader = create_labeled_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)

    if stage_1_enabled:
        logger.info("Tuning stage 1 enabled")

        best_stage_1_params, best_model, stage_1_results_df, best_history_df, best_training_summary, best_threshold = tuning_stage_1(
            X_train_normal=X_train_normal, X_val_normal=X_val_normal, X_val=X_val, y_val=y_val, config=config,
            device=device, logger=logger)

        if config["output"]["save_tuning_results"]:
            save_stage_results(results_df=stage_1_results_df,
                best_params={**best_stage_1_params, "reconstruction_threshold": best_threshold},
                output_dir=config["output"]["output_dir"], stage="1", logger=logger)

        if config["output"]["save_plots"]:
            plot_tuning_stage_1(stage_1_results_df, config, logger)

        save_training_history(best_history_df, config, logger)

        best_config = copy.deepcopy(config)
        best_config["model"].update(best_stage_1_params)

        final_threshold = best_threshold

        if stage_2_enabled:
            logger.info("Tuning stage 2 enabled for best stage 1 model")
            best_stage_2_params, stage_2_results_df = tuning_stage_2(model=best_model, val_loader=val_labeled_loader,
                config=best_config, device=device, logger=logger)

            final_threshold = best_stage_2_params["reconstruction_threshold"]

            if config["output"]["save_tuning_results"]:
                save_stage_results(results_df=stage_2_results_df, best_params=best_stage_2_params,
                    output_dir=config["output"]["output_dir"], stage="2", logger=logger)

        if config["output"]["save_model"]:
            save_model(model=best_model, config=best_config, logger=logger, model_params=best_stage_1_params,
                       threshold=final_threshold, training_summary=best_training_summary)

        evaluate_and_save_split(model=best_model, data_loader=val_labeled_loader, split_name="Validation",
            threshold=final_threshold, device=device, config=best_config, logger=logger,
            training_summary=best_training_summary, history_df=best_history_df, model_params=best_stage_1_params)

        evaluate_and_save_split(model=best_model, data_loader=test_loader, split_name="Test", threshold=final_threshold,
            device=device, config=best_config, logger=logger, training_summary=best_training_summary, history_df=None,
            model_params=best_stage_1_params)

    else:
        logger.info("Standard run mode")

        train_loader = create_reconstruction_dataloader(X_train_normal, batch_size=batch_size, shuffle=True)

        val_reconstruction_loader = create_reconstruction_dataloader(X_val_normal, batch_size=batch_size, shuffle=False)

        model = build_model(input_dim=X_train_normal.shape[1], config=config, overrides={}, logger=logger)

        model, history_df, loss_info = train_model(model=model, train_loader=train_loader,
            val_loader=val_reconstruction_loader, config=config, device=device, logger=logger)

        training_summary = get_training_summary(history_df)
        training_summary.update(loss_info)

        save_training_history(history_df, config, logger)

        threshold = config["model"]["reconstruction_threshold"]
        logger.info(f"Threshold tuning disabled. Using reconstruction_threshold={threshold:.6f}")

        if config["output"]["save_model"]:
            save_model(model=model, config=config, logger=logger, threshold=threshold,
                training_summary=training_summary)

        evaluate_and_save_split(model=model, data_loader=val_labeled_loader, split_name="Validation",
            threshold=threshold, device=device, config=config, logger=logger, training_summary=training_summary,
            history_df=history_df)

        evaluate_and_save_split(model=model, data_loader=test_loader, split_name="Test", threshold=threshold,
            device=device, config=config, logger=logger, training_summary=training_summary, history_df=None)

    logger.info("Autoencoder experiment completed")
    # winsound.Beep(2500, 1000)

if __name__ == "__main__":
    main()