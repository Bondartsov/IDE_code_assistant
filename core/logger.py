# core/logger.py

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Настройка обработчика и форматирования
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)