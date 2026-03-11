import logging
from pathlib import Path

def setup_logger(log_file: Path, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger()

    if logger.handlers:
        return logger

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }

    logger.setLevel(level_map.get(level.upper(), logging.INFO))

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    #console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    #file
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger