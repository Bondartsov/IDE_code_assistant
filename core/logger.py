# Файл: core/logger.py

import logging

def setup_logging():
    """
    Настройка логирования для приложения.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    # Отключение логов Uvicorn сервера, если необходимо
    logging.getLogger("uvicorn.access").handlers = []

logger = logging.getLogger("IDE_Code_Assistant")