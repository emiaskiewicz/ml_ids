import logging
import subprocess
import csv
import winsound
from pathlib import Path
import yaml
from svm_data import prepare_svm_data, get_logger
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report, average_precision_score,
                             ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay)
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
from itertools import product

os.environ["LOKY_MAX_CPU_COUNT"] = "8"

BASE_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = BASE_DIR / "config" / "svm.yaml"

RESULTS_COLUMNS = [ "experiment", "dataset_variant", "split", "accuracy", "precision", "recall", "f1", "roc_auc", "average_precision",
                    "threshold", "scaling", "scaler","feature_selection", "feature_selection_method", "selected_k_features", "smote",
                    "C", "loss", "penalty", "dual", "tol", "max_iter", "class_weight", "tuning_stage_1", "tuning_stage_2"]

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

def build_model(config: dict, overrides: dict, logger) -> LinearSVC:
    model_cfg = config["model"].copy()
    if overrides:
        model_cfg.update(overrides)

    random_state = config["experiment"]["random_state"]
    logger.info("Building SVM model")
    logger.info(f"Model parameters: C={model_cfg['C']}, loss={model_cfg['loss']}, penalty={model_cfg['penalty']},"
                f" dual={model_cfg['dual']}, tol={model_cfg['tol']}, max_iter={model_cfg['max_iter']},"
                f" class_weight={model_cfg['class_weight']}, random_state={random_state}")

    model = LinearSVC(
        C=model_cfg["C"],
        loss=model_cfg["loss"],
        penalty=model_cfg["penalty"],
        dual=model_cfg["dual"],
        tol=model_cfg["tol"],
        max_iter=model_cfg["max_iter"],
        class_weight=model_cfg["class_weight"],
        random_state=random_state
    )

    return model

def train_model(model: LinearSVC, X_train, y_train, logger) -> LinearSVC:
    logger.info("Training SVM model")
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    return model

def evaluate_model(model, X, y, split_name: str, threshold: float, logger) -> dict:
    logger.info(f"Evaluating model on {split_name} set")
    if hasattr(model, "decision_function"):
        y_score = model.decision_function(X)
        y_pred = apply_threshold(y_score, threshold)
        threshold_used = threshold
    else:
        y_score = None
        y_pred = model.predict(X)
        threshold_used = None
        logger.warning("Model does not support predict_proba. Falling back to model.predict(); threshold is ignored.")

    metrics = calculate_binary_metrics(y, y_pred, y_score)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, zero_division=0)

    logger.info(f"{split_name} Threshold used: {threshold_used}")
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
        "threshold_used": threshold_used,
        **metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_true": y.tolist() if hasattr(y, "tolist") else list(y),
        "y_pred": y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred),
        "y_score": y_score.tolist() if y_score is not None else None
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

        "C": model_params.get("C", model_cfg.get("C")),
        "loss": model_params.get("loss", model_cfg.get("loss")),
        "penalty": model_params.get("penalty", model_cfg.get("penalty")),
        "dual": model_params.get("dual", model_cfg.get("dual")),
        "tol": model_params.get("tol", model_cfg.get("tol")),
        "max_iter": model_params.get("max_iter", model_cfg.get("max_iter")),
        "class_weight": model_params.get("class_weight", model_cfg.get("class_weight")),

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
    if metrics["y_score"] is None:
        logger.warning("Skipping ROC curve: probabilities not available")
        return

    split_name = metrics["split_name"].lower()
    save_path = BASE_DIR / config["output"]["output_dir"] / f"{split_name}_roc_curve.jpg"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        y_true=metrics["y_true"],
        y_score=metrics["y_score"],
        ax=ax,
        name=f"{config['experiment']['name']} ({metrics['split_name']})"
    )
    ax.set_title(f"{metrics['split_name']} - ROC Curve")
    ax.set_xlim(0.0, 0.02)
    ax.set_ylim(0.98, 1.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved ROC curve plot to: {save_path}")

def plot_precision_recall_curve(metrics: dict, config: dict, logger: logging.Logger) -> None:
    if metrics["y_score"] is None:
        logger.warning("Skipping Precision-Recall curve: probabilities not available")
        return

    split_name = metrics["split_name"].lower()
    save_path = BASE_DIR / config["output"]["output_dir"] / f"{split_name}_pr_curve.jpg"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(
        y_true=metrics["y_true"],
        y_score=metrics["y_score"],
        ax=ax,
        name=f"{config['experiment']['name']} ({metrics['split_name']})"
    )
    ax.set_title(f"{metrics['split_name']} - Precision-Recall Curve")
    ax.set_xlim(0.98, 1.0)
    ax.set_ylim(0.98, 1.0)
    ax.grid(True, alpha=0.3)
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

def apply_threshold(y_score: pd.Series | list, threshold: float) -> list:
    return [1 if prob >= threshold else 0 for prob in y_score]

def calculate_binary_metrics(y_true, y_pred, y_score) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score) if y_score is not None else None,
        "average_precision": average_precision_score(y_true, y_score) if y_score is not None else None
    }

def tuning_stage_1(X_train, y_train, X_val, y_val, config: dict, logger) -> tuple[dict, LinearSVC, pd.DataFrame]:
    tuning_cfg = config["tuning_stage_1"]
    model_cfg = config["model"]

    metric_name = tuning_cfg["metric"]
    threshold = model_cfg.get("decision_threshold", 0.0)
    C_values = tuning_cfg.get("C_values", [model_cfg["C"]])
    loss_values = tuning_cfg.get("loss_values", [model_cfg["loss"]])
    penalty_values = tuning_cfg.get("penalty_values", [model_cfg["penalty"]])
    dual_values = tuning_cfg.get("dual_values", [model_cfg["dual"]])
    tol_values = tuning_cfg.get("tol_values", [model_cfg["tol"]])
    class_weight_values = tuning_cfg.get("class_weight_values", [model_cfg["class_weight"]])

    results = []
    best_score = float("-inf")
    best_params = None
    best_model = None

    total_combinations = (len(C_values) * len(loss_values) * len(penalty_values) * len(dual_values) *
                          len(tol_values) * len(class_weight_values))

    logger.info("Starting tuning stage 1")
    logger.info(f"Stage 1 metric: {metric_name}")
    logger.info(f"Stage 1 total combinations: {total_combinations}")

    for (C, loss, penalty, dual, tol, class_weight) in product(C_values, loss_values, penalty_values,
                                                               dual_values, tol_values, class_weight_values):

        current_params = {
            "C": C,
            "loss": loss,
            "penalty": penalty,
            "dual": dual,
            "tol": tol,
            "class_weight": class_weight,
        }

        logger.info(f"Training stage 1 candidate: C={C}, loss={loss}, penalty={penalty}, dual={dual}, "
                    f"tol={tol}, class_weight={class_weight}")

        try:
            model = build_model(config, current_params, logger)
            model.fit(X_train, y_train)

            val_score = model.decision_function(X_val)
            val_pred = apply_threshold(val_score, threshold)

            metrics = calculate_binary_metrics(y_true=y_val, y_pred=val_pred, y_score=val_score)

            row = {**current_params, "decision_threshold": threshold, **metrics}
            results.append(row)

            current_score = metrics[metric_name]

            logger.info(f"Stage 1 result: {metric_name}={current_score:.4f}, f1={metrics['f1']:.4f}, "
                        f"recall={metrics['recall']:.4f}, precision={metrics['precision']:.4f}, "
                        f"roc_auc={metrics['roc_auc']:.4f}")

            if current_score > best_score:
                best_score = current_score
                best_params = current_params.copy()
                best_model = model
        except Exception as e:
            logger.error(f"Stage 1 canditeate failed/illegal: {e}")

    if best_params is None or best_model is None:
        raise RuntimeError("Tuning stage 1 failed to produce a best model.")

    results_df = pd.DataFrame(results).sort_values(
        by=[metric_name, "f1", "roc_auc", "recall", "precision"],
        ascending=False,
    ).reset_index(drop=True)

    logger.info(f"Stage 1 best params: {best_params}")
    logger.info(f"Stage 1 best {metric_name}: {best_score:.4f}")

    return best_params, best_model, results_df

def save_stage_results(results_df: pd.DataFrame, best_params: dict, output_dir: Path, stage: str, logger) -> None:
    path = BASE_DIR / output_dir
    path.mkdir(parents=True, exist_ok=True)
    file_csv = "tuning_stage" + str(stage) + "_results.csv"
    file_json = "tuning_stage" + str(stage) + "_params.json"

    csv_path = path / file_csv
    json_path = path / file_json

    results_df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved stage {stage} results to: {csv_path}")
    logger.info(f"Saved stage {stage} best params to: {json_path}")

def tuning_stage_2(model, X_val, y_val, config: dict, logger) -> tuple[dict, pd.DataFrame]:
    tuning_cfg = config["tuning_stage_2"]
    metric_name = tuning_cfg["metric"]
    threshold_values = tuning_cfg["threshold_values"]

    logger.info("Starting tuning stage 2")
    logger.info(f"Stage 2 metric: {metric_name}, threshold values: {threshold_values}")

    val_score = model.decision_function(X_val)
    results = []
    best_result = None
    best_score = float("-inf")

    for threshold in threshold_values:
        logger.info(f"Evaluating threshold candidate: {threshold:.3f}")
        val_pred = apply_threshold(val_score, threshold)

        metrics = calculate_binary_metrics(y_true=y_val, y_pred=val_pred, y_score=val_score)
        row = {"decision_threshold": threshold, **metrics }
        results.append(row)

        current_score = metrics[metric_name]

        logger.info(
            f"Stage 2 result: threshold={threshold:.3f}, "
            f"{metric_name}={current_score:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"precision={metrics['precision']:.4f}"
        )

        if current_score > best_score:
            best_score = current_score
            best_result = row

    if best_result is None:
        raise RuntimeError("Tuning stage 2 failed to produce a best threshold.")

    results_df = pd.DataFrame(results).sort_values(
        by=[metric_name, "f1", "roc_auc", "recall", "precision"],
        ascending=False,
    ).reset_index(drop=True)

    logger.info(f"Stage 2 best result: {best_result}")

    return best_result, results_df

def plot_tuning_stage_1(results_df: pd.DataFrame, config: dict, logger) -> None:
    metric_name = config["tuning_stage_1"]["metric"]
    output_dir = BASE_DIR / config["output"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "stage_1_top_configs.jpg"

    plot_df = results_df.head(15).copy()
    plot_df["label"] = [
        f"{i + 1}. C={row['C']}, loss={row['loss']}, penalty={row['penalty']}, cw={row['class_weight']}"
        for i, (_, row) in enumerate(plot_df.iterrows())
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df["label"], plot_df[metric_name])
    ax.invert_yaxis()
    ax.set_xlabel(metric_name)
    ax.set_xlim(0.90, 1.0)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_title(f"Stage 1 - top configurations by {metric_name.upper()}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved stage 1 tuning plot to: {save_path}")

def plot_tuning_stage_2(results_df: pd.DataFrame, config: dict, logger) -> None:
    metric_name = config["tuning_stage_2"]["metric"]
    output_dir = BASE_DIR / config["output"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "stage_2_threshold_curves.jpg"

    plot_df = results_df.sort_values("decision_threshold")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(plot_df["decision_threshold"], plot_df["precision"], label="precision")
    ax.plot(plot_df["decision_threshold"], plot_df["recall"], label="recall")
    ax.plot(plot_df["decision_threshold"], plot_df["f1"], label="f1")

    if metric_name not in {"precision", "recall", "f1"} and metric_name in plot_df.columns:
        ax.plot(plot_df["decision_threshold"], plot_df[metric_name], label=metric_name)

    best_idx = plot_df[metric_name].idxmax()
    best_threshold = plot_df.loc[best_idx, "decision_threshold"]
    best_score = plot_df.loc[best_idx, metric_name]

    ax.axvline(best_threshold, linestyle="--", alpha=0.7)
    ax.text(best_threshold, best_score, f" best={best_threshold:.2f}", va="bottom")

    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0.98, 1.0)
    ax.set_title(f"Stage 2 - metrics vs threshold ({metric_name.upper()})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved stage 2 tuning plot to: {save_path}")

def main() -> None:
    config = load_config(CONFIG_PATH)
    logger = get_logger(config)
    log_config(config, logger)

    logger.info("Start experiment")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_svm_data(config)
    logger.info("Data prepared successfully")

    stage_1_enabled = config.get("tuning_stage_1", {}).get("enabled", False)
    stage_2_enabled = config.get("tuning_stage_2", {}).get("enabled", False)

    if stage_1_enabled:
        logger.info("Tuning mode enabled")

        best_stage_1_params, best_stage_1_model, stage_1_results_df = tuning_stage_1(X_train, y_train, X_val, y_val, config, logger)
        save_stage_results(stage_1_results_df, best_stage_1_params, config["output"]["output_dir"], "1", logger)
        plot_tuning_stage_1(stage_1_results_df, config, logger)
        best_threshold = config["model"].get("decision_threshold", 0.5)

        if stage_2_enabled:
            best_stage_2_params, stage_2_results_df = tuning_stage_2(best_stage_1_model, X_val, y_val, config,logger)
            save_stage_results(stage_2_results_df, best_stage_2_params, config["output"]["output_dir"], "2",logger)
            plot_tuning_stage_2(stage_2_results_df, config, logger)
            best_threshold = best_stage_2_params["decision_threshold"]

        logger.info("Preparing final train+val dataset")
        X_train_final = pd.concat([X_train, X_val], axis=0)
        y_train_final = pd.concat([y_train, y_val], axis=0)

        logger.info(f"Final train+val shapes: X={X_train_final.shape}, y={y_train_final.shape}")

        final_model = build_model(config, best_stage_1_params, logger)
        final_model = train_model(final_model, X_train_final, y_train_final, logger)

        test_metrics = evaluate_model(final_model,X_test,y_test,"Test", best_threshold, logger)

        summary_row = build_results_summary_row(metrics=test_metrics, config=config, model_params=best_stage_1_params)
        summary_csv_path = BASE_DIR / config["output"]["summary_path"]
        append_results_to_csv(summary_row, summary_csv_path)
        logger.info(f"Added experiment results to summary CSV: {summary_csv_path}")

        if config["output"]["save_metrics"]:
            save_metrics(test_metrics, config, logger)

        if config["output"]["save_plots"]:
            save_visualizations(test_metrics, config, logger)

    else:
        logger.info("Standard run mode")
        model = build_model(config, {}, logger)

        model = train_model(model, X_train, y_train, logger)

        threshold = config["model"].get("decision_threshold", 0.0)
        val_metrics = evaluate_model(model, X_val, y_val, "Validation", threshold, logger)

        summary_row = build_results_summary_row(metrics=val_metrics, config=config, model_params=config["model"])
        summary_csv_path = BASE_DIR / config["output"]["summary_path"]
        append_results_to_csv(summary_row, summary_csv_path)
        logger.info(f"Added experiment results to summary CSV: {summary_csv_path}")

        if config["output"]["save_metrics"]:
            save_metrics(val_metrics, config, logger)

        if config["output"]["save_plots"]:
            save_visualizations(val_metrics, config, logger)

    winsound.Beep(2500,1000)
    #subprocess.run(["paplay", "/usr/share/sounds/freedesktop/stereo/complete.oga"], check=False)

if __name__ == "__main__":
    main()