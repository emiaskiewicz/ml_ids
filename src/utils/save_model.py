from pathlib import Path
import joblib
import json
import torch


def save_sklearn_model(model, output_dir: str | Path, config: dict, logger, scaler=None, selector=None, selected_features=None, decision_threshold: float | None = None,
                       extra_artifacts: dict | None = None) -> None:
    if not config["output"].get("save_model", False):
        logger.info("Model saving disabled in config.")
        return

    model_dir = Path(output_dir) / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "scaler": scaler,
        "selector": selector,
        "selected_features": selected_features,
        "decision_threshold": decision_threshold,
        "config": config,
    }

    if extra_artifacts:
        artifact.update(extra_artifacts)

    model_path = model_dir / "model.joblib"
    joblib.dump(artifact, model_path)

    metadata_path = model_dir / "model_metadata.json"
    metadata = {
        "experiment_name": config["experiment"]["name"],
        "dataset_variant": config["data"]["dataset_variant"],
        "decision_threshold": decision_threshold,
        "selected_features_count": len(selected_features) if selected_features is not None else None,
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    logger.info(f"Saved sklearn model artifact to: {model_path}")

def save_torch_model(model, output_dir: str | Path, config: dict, logger, input_dim: int, selected_features=None,
                     decision_threshold: float | None = None, scaler=None, selector=None, feature_columns=None,
                     model_config: dict | None = None, training_summary: dict | None = None,
                     preprocessing_artifacts: dict | None = None, threshold_key: str = "decision_threshold",
                     extra_artifacts: dict | None = None) -> None:
    if not config["output"].get("save_model", False):
        logger.info("Model saving disabled in config.")
        return

    model_dir = Path(output_dir) / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    effective_model_config = model_config if model_config is not None else config["model"]

    checkpoint = {
        "model_state_dict": state_dict,
        "state_dict": state_dict,
        "model_config": effective_model_config,
        "input_dim": input_dim,
        "feature_columns": feature_columns,
        "selected_features": selected_features,
        threshold_key: decision_threshold,
        "training_summary": training_summary,
        "preprocessing_artifacts": preprocessing_artifacts,
        "config": config,
        "full_config": config,
    }

    if extra_artifacts:
        checkpoint.update(extra_artifacts)

    model_path = model_dir / "model.pt"
    torch.save(checkpoint, model_path)

    preprocessing_path = model_dir / "preprocessing.joblib"
    joblib.dump(
        {
            "scaler": scaler,
            "selector": selector,
            "selected_features": selected_features,
            "feature_columns": feature_columns,
            "preprocessing_artifacts": preprocessing_artifacts,
        },
        preprocessing_path,
    )

    metadata_path = model_dir / "model_metadata.json"
    metadata = {
        "experiment_name": config["experiment"]["name"],
        "dataset_variant": config["data"]["dataset_variant"],
        "input_dim": input_dim,
        "decision_threshold": decision_threshold,
        "selected_features_count": len(selected_features) if selected_features is not None else None,
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    logger.info(f"Saved torch model checkpoint to: {model_path}")
    logger.info(f"Saved preprocessing artifacts to: {preprocessing_path}")

def load_sklearn_model(model_path: str | Path) -> dict:
    return joblib.load(model_path)

def load_torch_model(model_path: str | Path, map_location=None) -> dict:
    return torch.load(model_path, map_location=map_location)
