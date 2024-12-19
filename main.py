# Файл: main.py

from config_storage import ConfigManager
import openai

# Инициализация менеджера конфигурации
config_manager = ConfigManager()

# Проверка на наличие API-ключа OpenAI
api_key = config_manager.get_openai_api_key()
if not api_key:
    raise ValueError("API ключ не найден. Добавьте его в файл .env с ключом OPENAI_API_KEY.")

# Установка API-ключа OpenAI
openai.api_key = api_key

# Меню выбора действий
print("Выберите действие:")
print("1. Отправить новый запрос к OpenAI API")

choice = input("Введите номер действия: ")

if choice == "1":
    # Отправка нового запроса
    prompt = input("Введите текст запроса: ")
    try:
        # Отправка запроса к OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        result = response.choices[0].message.content
        print("\nРезультат запроса:")
        print(result)
    except openai.error.OpenAIError as e:
        print(f"Ошибка OpenAI API: {e}")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")
else:
    print("Неверный выбор действия.")