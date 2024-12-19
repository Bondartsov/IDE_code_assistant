# test_openai_key.py

from config_storage import ConfigManager
import openai

# Инициализируем ConfigManager и устанавливаем API-ключ OpenAI
config_manager = ConfigManager()
openai_api_key = config_manager.get_openai_api_key()
openai.api_key = openai_api_key

try:
    # Тестовый вызов API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, world!"}]
    )
    print("API-ключ OpenAI действителен.")
    print("Ответ:", response.choices[0].message.content)
except openai.error.AuthenticationError as e:
    print("Ошибка аутентификации OpenAI:", e)
except Exception as e:
    print("Произошла непредвиденная ошибка:", e)