import os
import sys
import time
import subprocess
from pathlib import Path
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_DIR / "config" / "autoencoder.yaml"
SCRIPT_PATH = PROJECT_DIR / "src" / "models" / "autoencoder" / "ae_model.py"
DATASET_VARIANT = "easy"
LOG_PATH = PROJECT_DIR / "logs" / "autoencoder" / DATASET_VARIANT / "runner.log"
STOP_ON_ERROR = True
SKIP_FINISHED = True

experiments = [
    {"id": "AE-01e", "scaling": True, "correlation": False, "feature_selection": False, "log_transform": False,
     "network_features": False, "drop_original_ports": False, "stage_1": False, "stage_2": False},
    {"id": "AE-02e", "scaling": True, "correlation": True, "feature_selection": False, "log_transform": False,
     "network_features": False, "drop_original_ports": False, "stage_1": False, "stage_2": False},
    {"id": "AE-03e", "scaling": True, "correlation": False, "feature_selection": True, "log_transform": False,
     "network_features": False, "drop_original_ports": False, "stage_1": False, "stage_2": False},
    {"id": "AE-04e", "scaling": True, "correlation": True, "feature_selection": True, "log_transform": False,
     "network_features": False, "drop_original_ports": False, "stage_1": False, "stage_2": False},
    {"id": "AE-05e", "scaling": True, "correlation": False, "feature_selection": False, "log_transform": True,
     "network_features": False, "drop_original_ports": False, "stage_1": False, "stage_2": False},
    {"id": "AE-06e", "scaling": True, "correlation": True, "feature_selection": False, "log_transform": True,
     "network_features": False, "drop_original_ports": False, "stage_1": False, "stage_2": False},
    {"id": "AE-07e", "scaling": True, "correlation": False, "feature_selection": True, "log_transform": True,
     "network_features": False, "drop_original_ports": False, "stage_1": False, "stage_2": False},
    {"id": "AE-08e", "scaling": True, "correlation": True, "feature_selection": True, "log_transform": True,
     "network_features": False, "drop_original_ports": False, "stage_1": False, "stage_2": False},
    {"id": "AE-09e", "scaling": True, "correlation": False, "feature_selection": False, "log_transform": False,
     "network_features": True, "drop_original_ports": False, "stage_1": False, "stage_2": False},
    {"id": "AE-10e", "scaling": True, "correlation": True, "feature_selection": False, "log_transform": False,
     "network_features": True, "drop_original_ports": False, "stage_1": False, "stage_2": False},
    {"id": "AE-11e", "scaling": True, "correlation": False, "feature_selection": True, "log_transform": False,
     "network_features": True, "drop_original_ports": False, "stage_1": False, "stage_2": False},
    {"id": "AE-12e", "scaling": True, "correlation": True, "feature_selection": True, "log_transform": False,
     "network_features": True, "drop_original_ports": False, "stage_1": False, "stage_2": False},
    {"id": "AE-13e", "scaling": True, "correlation": False, "feature_selection": False, "log_transform": True,
     "network_features": True, "drop_original_ports": False, "stage_1": False, "stage_2": False},
    {"id": "AE-14e", "scaling": True, "correlation": True, "feature_selection": False, "log_transform": True,
     "network_features": True, "drop_original_ports": False, "stage_1": False, "stage_2": False},
    {"id": "AE-15e", "scaling": True, "correlation": False, "feature_selection": True, "log_transform": True,
     "network_features": True, "drop_original_ports": False, "stage_1": False, "stage_2": False},
    {"id": "AE-16e", "scaling": True, "correlation": True, "feature_selection": True, "log_transform": True,
     "network_features": True, "drop_original_ports": False, "stage_1": False, "stage_2": False},
    {"id": "AE-17e", "scaling": True, "correlation": False, "feature_selection": False, "log_transform": False,
     "network_features": True, "drop_original_ports": True, "stage_1": False, "stage_2": False},
    {"id": "AE-18e", "scaling": True, "correlation": True, "feature_selection": False, "log_transform": False,
     "network_features": True, "drop_original_ports": True, "stage_1": False, "stage_2": False},
    {"id": "AE-19e", "scaling": True, "correlation": False, "feature_selection": True, "log_transform": False,
     "network_features": True, "drop_original_ports": True, "stage_1": False, "stage_2": False},
    {"id": "AE-20e", "scaling": True, "correlation": True, "feature_selection": True, "log_transform": False,
     "network_features": True, "drop_original_ports": True, "stage_1": False, "stage_2": False},
    {"id": "AE-21e", "scaling": True, "correlation": False, "feature_selection": False, "log_transform": True,
     "network_features": True, "drop_original_ports": True, "stage_1": False, "stage_2": False},
    {"id": "AE-22e", "scaling": True, "correlation": True, "feature_selection": False, "log_transform": True,
     "network_features": True, "drop_original_ports": True, "stage_1": False, "stage_2": False},
    {"id": "AE-23e", "scaling": True, "correlation": False, "feature_selection": True, "log_transform": True,
     "network_features": True, "drop_original_ports": True, "stage_1": False, "stage_2": False},
    {"id": "AE-24e", "scaling": True, "correlation": True, "feature_selection": True, "log_transform": True,
     "network_features": True, "drop_original_ports": True, "stage_1": False, "stage_2": False},
]

def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}min {secs}s"
    if minutes > 0:
        return f"{minutes}min {secs}s"

    return f"{secs}s"

def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def apply_experiment_settings(config: dict, exp: dict) -> dict:
    exp_id = exp["id"]

    config["experiment"]["name"] = exp_id
    config["data"]["dataset_variant"] = DATASET_VARIANT
    config["split"]["split_dir"] = f"data/split/{DATASET_VARIANT}/"
    config["split"]["load_existing_split"] = True

    config["features"]["scaling"] = exp["scaling"]
    config["features"]["remove_correlated_features"] = exp["correlation"]
    config["features"]["use_feature_selection"] = exp["feature_selection"]
    config["features"]["log_transform"] = exp["log_transform"]
    config["features"]["use_network_features"] = exp["network_features"]
    config["features"]["drop_original_port_columns"] = exp["drop_original_ports"]

    config["tuning_stage_1"]["enabled"] = exp["stage_1"]
    config["tuning_stage_2"]["enabled"] = exp["stage_2"]

    config["output"]["output_dir"] = f"outputs/autoencoder/{DATASET_VARIANT}/{exp_id}"
    config["logging"]["log_path"] = f"logs/autoencoder/{DATASET_VARIANT}/{exp_id}.log"
    config["output"]["summary_path"] = f"outputs/autoencoder/{DATASET_VARIANT}/ae_{DATASET_VARIANT}_results_sum.csv"

    return config

def experiment_finished(config: dict) -> bool:
    output_dir = PROJECT_DIR / config["output"]["output_dir"]
    return (output_dir / "test_metrics.json").exists()

def run_experiment(exp_id: str) -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_DIR}:{PROJECT_DIR / 'src'}:{env.get('PYTHONPATH', '')}"

    command = [sys.executable, "-u", str(SCRIPT_PATH)]

    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"Running experiment: {exp_id}\n")
        log_file.flush()
        result = subprocess.run(command, cwd=PROJECT_DIR, env=env, stdout=log_file, stderr=subprocess.STDOUT, text=True)

    return result.returncode

def save_config(config: dict) -> None:
    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)

def main() -> None:
    queue_start = time.time()
    print(f"Project dir: {PROJECT_DIR}")
    print(f"Config path: {CONFIG_PATH}")
    print(f"Dataset variant: {DATASET_VARIANT}")
    print(f"Experiments in queue: {len(experiments)}")
    print(f"Runner log: {LOG_PATH}")

    for index, exp in enumerate(experiments, start=1):
        exp_id = exp["id"]
        config = load_config()
        config = apply_experiment_settings(config, exp)

        if SKIP_FINISHED and experiment_finished(config):
            print(f"[{index}/{len(experiments)}] SKIP: {exp_id}")
            print("Reason: test_metrics.json already exists")
            continue

        print(f"[{index}/{len(experiments)}] Running: {exp_id}")
        print(f"Dataset: {DATASET_VARIANT}")
        print(f"Scaling={exp['scaling']}, Corr={exp['correlation']}, FS={exp['feature_selection']}, "
              f"Log={exp['log_transform']}, NetFeat={exp['network_features']}, DropPorts={exp['drop_original_ports']},"
              f"Stage1={exp['stage_1']}, Stage2={exp['stage_2']}")
        print(f"Output: {config['output']['output_dir']}")
        print(f"Log: {config['logging']['log_path']}")

        save_config(config)
        exp_start = time.time()
        return_code = run_experiment(exp_id)
        exp_duration = time.time() - exp_start

        if return_code != 0:
            print(f"ERROR: {exp_id} failed after {format_duration(exp_duration)}")
            print(f"Runner log: {LOG_PATH}")
            if STOP_ON_ERROR:
                print("Experiments stopped.")
                break
            print("Continuing with next experiment.")
        else:
            print(f"FINISHED: {exp_id} in {format_duration(exp_duration)}")

    queue_duration = time.time() - queue_start
    print(f"\nExperiments finished in {format_duration(queue_duration)}")

if __name__ == "__main__":
    main()
