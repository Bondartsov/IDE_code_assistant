IDE Code Assistant
Описание
IDE Code Assistant — это приложение, предоставляющее интерфейс для взаимодействия с OpenAI API через REST API, реализованный с помощью FastAPI. Приложение позволяет генерировать внутренние API-ключи для доступа к своим эндпоинтам, выполнять запросы к моделям OpenAI и включает механизмы валидации и управления ключами доступа.

Основные возможности:

Генерация и управление внутренними API-ключами для доступа к API вашего приложения.
Интеграция с OpenAI API для выполнения запросов к моделям GPT (например, gpt-3.5-turbo).
REST API с использованием FastAPI для обработки запросов и ответа клиентам.
Валидация API-ключей и очистка устаревших ключей.
Тестирование с использованием Pytest для обеспечения надежности и стабильности кода.
Содержание
Требования
Установка
Клонирование репозитория
Создание и активация виртуального окружения
Для Windows (CMD)
Для Windows (PowerShell)
Для Linux/MacOS
Установка зависимостей
Конфигурация
Настройка переменных окружения
Запуск приложения
Запуск сервера FastAPI
Использование API эндпоинтов
Доступ к документации API
Генерация нового API-ключа
Получение списка моделей
Выполнение запроса к OpenAI
Запуск тестов
Структура проекта
Безопасность
Лицензия
Контакты
Требования
Python 3.10 или выше
pip — менеджер пакетов Python
virtualenv — для создания виртуального окружения (опционально, но рекомендуется)
Аккаунт OpenAI и действительный API-ключ
Установка
Клонирование репозитория
Клонируйте репозиторий проекта на ваш локальный компьютер:

git clone https://github.com/yourusername/ide-code-assistant.git
cd ide-code-assistant
Создание и активация виртуального окружения
Использование виртуального окружения рекомендуется для изоляции зависимостей проекта.

Для Windows (CMD)
python -m venv venv
venv\Scripts\activate
Для Windows (PowerShell)
Важно: В PowerShell активация виртуального окружения немного отличается.

python -m venv venv
.\venv\Scripts\Activate.ps1
Если при активации возникает ошибка, связанная с политикой выполнения скриптов, выполните следующую команду для установки политики исполнения скриптов:

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
После этого снова попробуйте активировать виртуальное окружение:

.\venv\Scripts\Activate.ps1
Примечание: Возвращайте политику выполнения скриптов к предыдущему значению после активации, если это необходимо для вашей безопасности.

Для Linux/MacOS
python3 -m venv venv
source venv/bin/activate
Установка зависимостей
Установите необходимые пакеты из файла requirements.txt:

pip install -r requirements.txt
Конфигурация
Настройка переменных окружения
Создайте файл .env в корневой директории проекта и добавьте в него следующие строки:

# .env
API_KEY=test_api_key
OPENAI_API_KEY=your_openai_api_key_here
Замените your_openai_api_key_here на ваш реальный API-ключ от OpenAI.
Важно: Никогда не публикуйте ваш API-ключ публично и не добавляйте файл .env в систему контроля версий.
Запуск приложения
Запуск сервера FastAPI
Используйте uvicorn для запуска сервера FastAPI:

uvicorn api:app --reload
Если вы используете PowerShell: Убедитесь, что вы находитесь в корневой директории проекта перед запуском команды.

Сервер будет запущен на http://127.0.0.1:8000.
Параметр --reload автоматически перезапускает сервер при изменении кода (удобно для разработки).
Использование API эндпоинтов
Вы можете использовать такие инструменты, как curl, Postman или встроенную документацию Swagger для взаимодействия с API.

Доступ к документации API
После запуска сервера перейдите по адресу:

Swagger UI: http://127.0.0.1:8000/docs
Redoc: http://127.0.0.1:8000/redoc
Генерация нового API-ключа
Эндпоинт: POST /api/generate_key/

Описание: Генерирует новый внутренний API-ключ для доступа к API вашего приложения.

Пример запроса с использованием PowerShell:

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/generate_key/"
Или с использованием curl в PowerShell:

curl -Method Post "http://127.0.0.1:8000/api/generate_key/"
Ответ:

{
  "api_key": "generated_api_key_here"
}
Получение списка моделей
Эндпоинт: GET /api/models/

Описание: Возвращает список доступных моделей OpenAI.

Заголовки:

api-key: Ваш внутренний API-ключ, сгенерированный ранее.
Пример запроса в PowerShell:

$headers = @{ "api-key" = "your_generated_api_key" }
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/api/models/" -Headers $headers
Или с использованием curl:

curl -Method Get "http://127.0.0.1:8000/api/models/" -Headers @{ "api-key" = "your_generated_api_key" }
Ответ:

{
  "models": [
    "gpt-3.5-turbo",
    "...",
    "text-davinci-003"
  ]
}
Выполнение запроса к OpenAI
Эндпоинт: POST /api/openai/

Описание: Отправляет запрос к модели OpenAI и возвращает ответ.

Заголовки:

api-key: Ваш внутренний API-ключ.
Content-Type: application/json
Тело запроса (JSON):

{
  "prompt": "Your prompt here"
}
Пример запроса в PowerShell:

$headers = @{
  "api-key" = "your_generated_api_key";
  "Content-Type" = "application/json"
}
$body = @{
  "prompt" = "Write a haiku about AI"
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/openai/" -Headers $headers -Body $body
Или с использованием curl:

curl -Method Post "http://127.0.0.1:8000/api/openai/" -Headers @{ "api-key" = "your_generated_api_key"; "Content-Type" = "application/json" } -Body '{"prompt": "Write a haiku about AI"}'
Ответ:

{
  "response": "Constructed response from OpenAI"
}
Запуск тестов
Тесты написаны с использованием Pytest и находятся в файле test_api.py.

Запуск всех тестов
Убедитесь, что виртуальное окружение активировано, и выполните команду:

pytest
В PowerShell:

pytest
Результаты тестов
Если все настроено корректно, вы должны увидеть отчет об успешном прохождении тестов:

==================================== test session starts ====================================
platform win32 -- Python 3.10.x, pytest-8.x.x, pluggy-1.x.x
collected 4 items

test_api.py ....                                                                      [100%]

==================================== 4 passed in 2.34s =====================================
Структура проекта
ide-code-assistant/
├── api.py                 # Основной файл приложения FastAPI
├── config_storage.py      # Управление конфигурацией и API-ключами
├── main.py                # Консольное приложение для взаимодействия с OpenAI API
├── requirements.txt       # Список зависимостей проекта
├── test_api.py            # Тесты для приложения
├── .env                   # Файл с переменными окружения (не допускать в репозиторий)
├── .gitignore             # Список файлов и папок, игнорируемых Git
├── pytest.ini             # Конфигурация Pytest
├── README.md              # Документация проекта
└── venv/                  # Виртуальное окружение (не включать в репозиторий)
Безопасность
API-ключ OpenAI: Держите ваш API-ключ OpenAI в секрете. Никогда не публикуйте его и не добавляйте в систему контроля версий. Используйте файл .env для хранения ключа.
Файл .env: Добавьте файл .env в ваш .gitignore, чтобы предотвратить его попадание в репозиторий.
API-ключи приложения: Ваше приложение генерирует внутренние API-ключи для контролируемого доступа к вашим эндпоинтам. Обращайтесь с ними бережно и при необходимости реализуйте дополнительные меры безопасности.
Логирование: Избегайте логирования конфиденциальной информации, такой как API-ключи или персональные данные пользователей.
Лицензия
[Укажите лицензию вашего проекта, например, MIT, Apache 2.0 или GNU GPL.]

MIT License

Copyright (c) ...

Permission is hereby granted, free of charge, to any person obtaining a copy...
Контакты
Автор: [Ваше имя]
Email: [ваш.email@example.com]
GitHub: https://github.com/yourusername