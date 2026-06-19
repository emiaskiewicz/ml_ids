from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))
sys.path.append(str(SRC_DIR / "models" / "mlp"))
sys.path.append(str(SRC_DIR / "models" / "cnn"))
sys.path.append(str(SRC_DIR / "models" / "autoencoder"))
TARGET_COLUMN = "Target"
DEFAULT_SAMPLE_SIZE = 1000

from mlp_model import MLPNetwork
from cnn_model import CNNNetwork
from ae_model import AutoencoderNetwork

DIFFICULTIES = {
    "1": ("easy", "Easy"),
    "2": ("medium", "Medium"),
    "3": ("hard", "Hard"),
}

MODELS = {
    "1": {
        "name": "Logistic Regression",
        "type": "sklearn",
        "folders": {"easy": "LR-01e-final", "medium": "LR-01m-final", "hard": "LR-01h-final"}
    },
    "2": {
        "name": "Decision Tree",
        "type": "sklearn",
        "folders": {"easy": "DT-03e-final", "medium": "DT-03m-final", "hard": "DT-03h-final"}
    },
    "3": {
        "name": "SVM",
        "type": "sklearn",
        "folders": {"easy": "SVM-01e-final", "medium": "SVM-01m-final", "hard": "SVM-01h-final"}
    },
    "4": {
        "name": "MLP",
        "type": "torch",
        "folders": {"easy": "MLP-04e-final", "medium": "MLP-04m-final", "hard": "MLP-04h-final"}
    },
        "5": {
        "name": "Autoencoder",
        "type": "torch",
        "folders": {"easy": "AE-21e-final", "medium": "AE-21m-final", "hard": "AE-24h-final"}
    },
    "6": {
        "name": "CNN",
        "type": "torch",
        "folders": {"easy": "CNN-20e-final", "medium": "CNN-20m-final", "hard": "CNN-20h-final"}
    }
}

def choose_from_menu(title: str, options: dict[str, Any]) -> str | None:
    while True:
        print(f"\n=== {title} ===")
        for key, value in options.items():
            label = value["name"] if isinstance(value, dict) else value[1]
            print(f"{key}. {label}")
        print("0. Wyjscie")

        choice = input("\nWybor: ").strip()
        if choice == "0":
            return None
        if choice in options:
            return choice

        print("Niepoprawny wybor. Sprobuj ponownie.")

def ask_sample_size() -> int | None:
    value = input(f"\nIlosc rekordow do przetestowania ('all' dla calego test.csv): ").strip().lower()

    if value == "":
        return DEFAULT_SAMPLE_SIZE
    if value == "all":
        return None

    try:
        sample_size = int(value)
    except ValueError:
        print(f"Niepoprawna liczba. Uzywam {DEFAULT_SAMPLE_SIZE}.")
        return DEFAULT_SAMPLE_SIZE

    if sample_size <= 0:
        print(f"Liczba musi byc dodatnia. Uzywam {DEFAULT_SAMPLE_SIZE}.")
        return DEFAULT_SAMPLE_SIZE

    return sample_size

def resolve_paths(model_info: dict[str, Any], difficulty: str) -> tuple[Path, Path]:
    folder_name = model_info["folders"][difficulty]
    model_dir = BASE_DIR / "final_models" / folder_name / "model"
    data_path = BASE_DIR / "data" / "split" / difficulty / "test.csv"

    if not model_dir.exists():
        raise FileNotFoundError(f"Nie znaleziono folderu modelu: {model_dir}")
    if not data_path.exists():
        raise FileNotFoundError(f"Nie znaleziono danych testowych: {data_path}")
    return model_dir, data_path

def load_test_data(data_path: Path, sample_size: int | None) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_path, low_memory=False)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Brak kolumny celu '{TARGET_COLUMN}' w pliku {data_path}")

    if sample_size > len(df):
        print(f"Uwaga: zmienna sample_size ({sample_size}) jest wieksza niz liczba rekordow w zbiorze danych testowych ({len(df)}). Uzywam calego zbioru testowego.")
        sample_size = None

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)

    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=[TARGET_COLUMN])
    return X, y

def add_network_port_protocol_features(X: pd.DataFrame, drop_original_columns: bool) -> pd.DataFrame:
    required_columns = ["Source_Port", "Destination_Port", "Protocol"]
    missing_columns = [column for column in required_columns if column not in X.columns]

    if missing_columns:
        raise ValueError(f"Brak kolumn wymaganych do cech sieciowych: {missing_columns}")

    X = X.copy()
    src_port = X["Source_Port"]
    dst_port = X["Destination_Port"]
    protocol = X["Protocol"]

    X["protocol_tcp"] = (protocol == 6).astype(int)
    X["protocol_udp"] = (protocol == 17).astype(int)
    X["protocol_icmp"] = (protocol == 1).astype(int)
    X["src_port_is_well_known"] = src_port.between(0, 1023).astype(int)
    X["src_port_is_registered"] = src_port.between(1024, 49151).astype(int)
    X["src_port_is_ephemeral"] = src_port.between(49152, 65535).astype(int)
    X["dst_port_is_well_known"] = dst_port.between(0, 1023).astype(int)
    X["dst_port_is_registered"] = dst_port.between(1024, 49151).astype(int)
    X["dst_port_is_ephemeral"] = dst_port.between(49152, 65535).astype(int)
    X["dst_port_is_dns"] = (dst_port == 53).astype(int)
    X["dst_port_is_http"] = (dst_port == 80).astype(int)
    X["dst_port_is_https"] = (dst_port == 443).astype(int)
    X["dst_port_is_ssh"] = (dst_port == 22).astype(int)
    X["dst_port_is_smtp"] = (dst_port == 25).astype(int)
    X["dst_port_is_ntp"] = (dst_port == 123).astype(int)
    X["dst_port_is_ftp"] = dst_port.isin([20, 21]).astype(int)
    X["dst_port_is_rdp"] = (dst_port == 3389).astype(int)
    X["same_src_dst_port"] = (src_port == dst_port).astype(int)
    X["src_port_zero"] = (src_port == 0).astype(int)
    X["dst_port_zero"] = (dst_port == 0).astype(int)

    if drop_original_columns:
        X = X.drop(columns=required_columns)
    return X

def apply_log1p_transform(X: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    X = X.copy()
    for column in columns:
        if column in X.columns:
            X[column] = np.log1p(X[column])
    return X

def align_columns(X: pd.DataFrame, columns: list[str] | None, step_name: str) -> pd.DataFrame:
    if not columns:
        return X

    missing_columns = [column for column in columns if column not in X.columns]
    if missing_columns:
        raise ValueError(f"Brak kolumn dla etapu '{step_name}': {missing_columns}")

    return X.loc[:, columns]

def get_feature_names(estimator: Any) -> list[str] | None:
    names = getattr(estimator, "feature_names_in_", None)
    if names is None:
        return None
    return list(names)

def transform_with_estimator(X: pd.DataFrame, estimator: Any, output_columns: list[str] | None) -> pd.DataFrame:
    X = align_columns(X, get_feature_names(estimator), estimator.__class__.__name__)
    transformed = estimator.transform(X)
    columns = output_columns or get_feature_names(estimator)

    if columns is None or len(columns) != transformed.shape[1]:
        columns = [f"feature_{idx}" for idx in range(transformed.shape[1])]

    return pd.DataFrame(transformed, columns=columns, index=X.index)

def get_preprocessing_artifacts(artifact: dict[str, Any]) -> dict[str, Any]:
    nested = artifact.get("preprocessing_artifacts")
    if isinstance(nested, dict):
        return nested
    return artifact

def preprocess_sklearn_features(X, artifact):
    scaler = artifact.get("scaler")
    selector = artifact.get("selector")
    selected_features = artifact.get("selected_features")
    model = artifact["model"]

    if scaler is not None:
        X_scaled = scaler.transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    if selector is not None:
        X_selected = selector.transform(X)
        X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    elif selected_features is not None:
        X = X[selected_features]

    if hasattr(model, "feature_names_in_"):
        X = X[list(model.feature_names_in_)]

    return X

def preprocess_torch_features(X: pd.DataFrame, checkpoint: dict[str, Any]) -> pd.DataFrame:
    artifacts = get_preprocessing_artifacts(checkpoint)
    model_type = checkpoint.get("model_type", "").lower()

    if artifacts.get("network_features_enabled"):
        X = add_network_port_protocol_features(X, artifacts.get("drop_original_port_columns", False))

    log_columns = artifacts.get("log_transform_columns") or []
    if log_columns:
        X = apply_log1p_transform(X, log_columns)

    dropped_columns = artifacts.get("dropped_correlated_features") or []
    existing_dropped_columns = [column for column in dropped_columns if column in X.columns]
    if existing_dropped_columns:
        X = X.drop(columns=existing_dropped_columns)

    selector = artifacts.get("selector") or checkpoint.get("selector")
    scaler = artifacts.get("scaler") or checkpoint.get("scaler")
    selected_features = artifacts.get("selected_features") or checkpoint.get("selected_features")
    final_columns = artifacts.get("feature_columns") or checkpoint.get("feature_columns") or selected_features

    if model_type == "autoencoder":
        if selector is not None:
            X = transform_with_estimator(X, selector, selected_features)
        elif selected_features:
            X = align_columns(X, selected_features, "selected_features")

        if scaler is not None:
            X = transform_with_estimator(X, scaler, final_columns)
    else:
        if scaler is not None:
            X = transform_with_estimator(X, scaler, get_feature_names(scaler))

        if selector is not None:
            X = transform_with_estimator(X, selector, selected_features)
        elif selected_features:
            X = align_columns(X, selected_features, "selected_features")

    return align_columns(X, final_columns, "final_columns")

def load_torch_checkpoint(model_path: Path) -> dict[str, Any]:
    try:
        return torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location="cpu")

def build_torch_model(checkpoint):
    model_type = checkpoint.get("model_type", "").lower()
    model_config = checkpoint["model_config"]
    input_dim = int(checkpoint["input_dim"])

    if model_type == "mlp":
        return MLPNetwork(
            input_dim=input_dim,
            hidden_layers=model_config["hidden_layers"],
            dropout=model_config["dropout"],
            activation=model_config.get("activation", "relu"),
            batch_norm=model_config["batch_norm"],
        )

    if model_type == "cnn":
        return CNNNetwork(
            input_dim=input_dim,
            conv_channels=model_config["conv_channels"],
            kernel_size=model_config["kernel_size"],
            fc_layers=model_config["fc_layers"],
            dropout=model_config["dropout"],
            activation=model_config.get("activation", "relu"),
            batch_norm=model_config["batch_norm"],
            global_pooling=model_config["global_pooling"],
            input_dropout=model_config.get("input_dropout", 0.0),
            input_noise_std=model_config.get("input_noise_std", 0.0),
        )

    if model_type == "autoencoder":
        return AutoencoderNetwork(
            input_dim=input_dim,
            encoder_layers=model_config["encoder_layers"],
            latent_dim=model_config["latent_dim"],
            dropout=model_config["dropout"],
            activation=model_config["activation"],
            batch_norm=model_config["batch_norm"],
            output_activation=model_config["output_activation"],
        )

    raise ValueError(f"Nieznany typ modelu PyTorch: {model_type}")

def predict_sklearn(artifact: dict[str, Any], X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None, float | None]:
    model = artifact["model"]
    threshold = artifact.get("decision_threshold")

    if threshold is not None:
        threshold = float(threshold)

        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X)
        else:
            scores = model.predict(X)

        y_pred = (scores >= threshold).astype(int)
        return y_pred, np.asarray(scores), threshold

    y_pred = model.predict(X).astype(int)
    scores = None

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)

    return y_pred, None if scores is None else np.asarray(scores), None

def calculate_reconstruction_errors(reconstructed: torch.Tensor, original: torch.Tensor, metric: str) -> torch.Tensor:
    metric = metric.lower()

    if metric == "mse":
        return torch.mean((reconstructed - original) ** 2, dim=1)
    if metric == "mae":
        return torch.mean(torch.abs(reconstructed - original), dim=1)
    if metric == "smooth_l1":
        return nn.functional.smooth_l1_loss(reconstructed, original, reduction="none").mean(dim=1)

    raise ValueError(f"Nieobslugiwana metryka bledu rekonstrukcji: {metric}")

def predict_torch(checkpoint: dict[str, Any], X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    model_type = checkpoint.get("model_type", "").lower()
    model = build_torch_model(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X_tensor = torch.tensor(X.to_numpy(dtype=np.float32), dtype=torch.float32)

    if model_type == "cnn":
        X_tensor = X_tensor.unsqueeze(1)

    threshold = checkpoint.get("reconstruction_threshold")
    if threshold is None:
        threshold = checkpoint.get("decision_threshold", 0.5)
    threshold = float(threshold)

    with torch.no_grad():
        if model_type == "autoencoder":
            reconstructed = model(X_tensor)
            scores = calculate_reconstruction_errors(
                reconstructed=reconstructed,
                original=X_tensor,
                metric=checkpoint["model_config"].get("loss_function", "mse"),
            ).cpu().numpy()
        else:
            logits = model(X_tensor)
            scores = torch.sigmoid(logits).cpu().numpy()

    scores = np.asarray(scores).ravel()
    y_pred = (scores >= threshold).astype(int)
    return y_pred, scores, threshold

def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, scores: np.ndarray | None) -> dict[str, Any]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }

    if scores is not None and len(set(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, scores)
        metrics["average_precision"] = average_precision_score(y_true, scores)
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    return metrics

def print_results(model_name: str, difficulty_label: str, sample_count: int, threshold: float | None, metrics: dict[str, Any]) -> None:
    print(f"\n=== Wyniki: {model_name} / {difficulty_label} ===")
    print(f"Liczba testowanych rekordow: {sample_count}")

    print(f"Accuracy          : {metrics['accuracy']:.4f}")
    print(f"Precision         : {metrics['precision']:.4f}")
    print(f"Recall            : {metrics['recall']:.4f}")
    print(f"F1-score          : {metrics['f1']:.4f}")

    if metrics["roc_auc"] is not None:
        print(f"ROC-AUC           : {metrics['roc_auc']:.4f}")
    if metrics["average_precision"] is not None:
        print(f"Average precision : {metrics['average_precision']:.4f}")

    print("\nMacierz pomylek:")
    print(metrics["confusion_matrix"])

    print("\nRaport klasyfikacji:")
    print(metrics["classification_report"])

def run_single_evaluation() -> bool:
    model_choice = choose_from_menu("Wybierz model", MODELS)
    if model_choice is None:
        return False

    difficulty_choice = choose_from_menu("Wybierz trudnosc danych", DIFFICULTIES)
    if difficulty_choice is None:
        return False

    sample_size = ask_sample_size()
    model_info = MODELS[model_choice]
    difficulty, difficulty_label = DIFFICULTIES[difficulty_choice]
    model_dir, data_path = resolve_paths(model_info, difficulty)

    print(f"\nModel: {model_info['name']}")
    print(f"Trudnosc: {difficulty_label}")
    print(f"Dane: {data_path}")
    print(f"Folder modelu: {model_dir}")

    X_raw, y = load_test_data(data_path, sample_size)

    if model_info["type"] == "sklearn":
        artifact_path = model_dir / "model.joblib"
        artifact = joblib.load(artifact_path)
        X = preprocess_sklearn_features(X_raw, artifact)
        y_pred, scores, threshold = predict_sklearn(artifact, X)
    else:
        checkpoint_path = model_dir / "model.pt"
        checkpoint = load_torch_checkpoint(checkpoint_path)
        X = preprocess_torch_features(X_raw, checkpoint)
        y_pred, scores, threshold = predict_torch(checkpoint, X)

    metrics = evaluate_predictions(y, y_pred, scores)
    print_results(model_info["name"], difficulty_label, len(y), threshold, metrics)
    return True

def main() -> None:
    print("Aplikacja CLI do testowania finalnych modeli")

    while True:
        try:
            should_continue = run_single_evaluation()
        except Exception as e:
            print(f"\nError: {e}")
            should_continue = True

        if not should_continue:
            print("Koniec programu.")
            break

        again = input("\nCzy chcesz wykonac kolejny test? [t/n]: ").strip().lower()
        if again != "t":
            print("Koniec programu.")
            break

if __name__ == "__main__":
    main()
