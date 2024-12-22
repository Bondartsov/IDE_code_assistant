# core/logger.py

import logging
import logging.config
import yaml
from typing import Any, Dict

def setup_logging(
    default_path: str = 'logging.yaml', default_level: int = logging.INFO
) -> None:
    """
    Настраивает логирование из файла конфигурации.

    Args:
        default_path (str): Путь к файлу конфигурации логирования.
        default_level (int): Уровень логирования по умолчанию.
    """
    try:
        with open(default_path, 'rt') as f:
            config: Dict[str, Any] = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    except Exception as e:
        print(f"Error in logging configuration. Using default configs. {e}")
        logging.basicConfig(level=default_level)

setup_logging()
logger = logging.getLogger(__name__)