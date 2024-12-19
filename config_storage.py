# Файл: config_storage.py

import os
import secrets
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Загружаем переменные окружения из файла .env
load_dotenv()

class ConfigManager:
    """
    Класс для управления конфигурацией приложения, включая управление API-ключами.
    """

    def __init__(self):
        # Словарь для хранения валидных API-ключей и времени их истечения
        self.valid_keys = {}

    def generate_api_key(self):
        """
        Генерирует новый API-ключ и сохраняет его со временем истечения.

        Возвращает:
            str: Сгенерированный API-ключ
        """
        new_key = secrets.token_hex(16)
        expiry_time = datetime.now() + timedelta(hours=24)
        self.valid_keys[new_key] = expiry_time
        return new_key

    def validate_api_key(self, key):
        """
        Проверяет валидность предоставленного API-ключа.

        Аргументы:
            key (str): API-ключ для проверки

        Возвращает:
            bool: True, если ключ валиден, иначе False
        """
        self.cleanup_expired_keys()
        return key in self.valid_keys
    
    def cleanup_expired_keys(self):
        """
        Удаляет устаревшие API-ключи из списка валидных ключей.
        """
        now = datetime.now()
        self.valid_keys = {k: v for k, v in self.valid_keys.items() if v > now}

    def get_openai_api_key(self):
        """
        Получает API-ключ OpenAI из переменных окружения.

        Возвращает:
            str: API-ключ OpenAI

        Вызывает:
            ValueError: Если API-ключ OpenAI не найден
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API ключ не найден. Добавьте его в .env файл с ключом OPENAI_API_KEY.")
        return api_key
