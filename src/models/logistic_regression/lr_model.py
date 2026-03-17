from pathlib import Path
import yaml
from lr_data import prepare_lr_data
from utils.logger import setup_logger

BASE_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = BASE_DIR / "config" / "logistic_regression.yaml"

def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def run_experiment() -> None:
    config = load_config(CONFIG_PATH)
    print(config)
    prepare_lr_data(config)
    print("Done")

def main():
    run_experiment()

if __name__ == "__main__":
    main()