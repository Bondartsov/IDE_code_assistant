# IDE Code Assistant

## Обзор

[IDE Code Assistant](https://github.com/Bondartsov/IDE_code_assistant) — это Python-приложение, которое предоставляет интерфейс для взаимодействия с моделями GPT от OpenAI или локальными моделями через LMStudio. Приложение работает как посредник между плагинами для интегрированных сред разработки (IDE) и языковыми моделями, обогащая пользовательские запросы релевантным контекстом из локальной базы знаний с использованием подхода Retrieval-Augmented Generation (RAG).

---

## Содержание

1. [Особенности](#особенности)
2. [Требования](#требования)
3. [Установка](#установка)
4. [Настройка](#настройка)
5. [Запуск](#запуск)
6. [Использование](#использование)
   - [Генерация API-ключа](#генерация-api-ключа)
   - [Загрузка данных в базу знаний](#загрузка-данных-в-базу-знаний)
   - [Поиск по базе знаний](#поиск-по-базе-знаний)
   - [Генерация ответа с использованием RAG](#генерация-ответа-с-использованием-rag)
7. [Структура проекта](#структура-проекта)
8. [Вклад в проект](#вклад-в-проект)
9. [Лицензия](#лицензия)

---

## Особенности

- **Интеграция с OpenAI и LMStudio**: Возможность использования моделей GPT от OpenAI или локально развернутых моделей через LMStudio.
- **Подход RAG (Retrieval-Augmented Generation)**: Обогащение запросов пользователя контекстом из локальной базы знаний.
- **База знаний**: Поддержка загрузки и обработки файлов различных форматов (PDF, DOCX, Markdown) для создания локальной базы знаний.
- **Поиск по базе знаний**: Использование FAISS для быстрого поиска по эмбеддингам документов.
- **API на базе FastAPI**: RESTful API для взаимодействия с приложением.
- **Управление API-ключами**: Генерация и проверка ключей доступа для безопасного взаимодействия с API.
- **Тестирование**: Модульные тесты с использованием Pytest для обеспечения стабильности кода.
- **Логирование**: Настраиваемое логирование для отслеживания работы приложения.

---

## Требования

- **Python**: версия 3.10 или выше
- **Операционная система**: Windows, macOS или Linux
- **Виртуальное окружение**: Рекомендуется использовать (`venv`, `conda`, `pipenv`)
- **Аккаунт OpenAI**: Если используете OpenAI API (необходимо для получения API-ключа)
- **LMStudio**: Если используете локальную модель через LMStudio
- **FAISS**: Библиотека для поиска по векторным эмбеддингам (устанавливается через `pip`)

---

## Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/Bondartsov/IDE_code_assistant.git
cd IDE_code_assistant
```

### 2. Создание и активация виртуального окружения

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

---

## Настройка

### 1. Конфигурация переменных окружения

Создайте файл `.env` в корневом каталоге проекта и заполните его следующим образом:

```ini
# Тип провайдера API: 'openai' или 'lmstudio'
API_PROVIDER=openai

# Ваш API-ключ OpenAI (если используете OpenAI)
OPENAI_API_KEY=ваш_ключ_от_OpenAI

# URL LMStudio API (если используете LMStudio)
LMSTUDIO_API_URL=http://localhost:8080/v1/completions

# Имя модели (по умолчанию 'chatgpt-4o-latest')
MODEL_NAME=chatgpt-4o-latest

# URL базы данных
DATABASE_URL=sqlite:///./app.db

# Настройки логирования
LOGGING_CONFIG=logging.yaml
```

**Важно:** Никогда не добавляйте файл `.env` в систему контроля версий. Убедитесь, что он перечислен в файле `.gitignore`.

### 2. Инициализация базы данных

Примените миграции для настройки базы данных:

```bash
alembic upgrade head
```

---

## Запуск

### Запуск сервера

Запустите приложение с помощью Uvicorn:

```bash
uvicorn main:app --reload
```

Сервер будет доступен по адресу: `http://127.0.0.1:8000`

### Документация API

- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`

---

## Использование

### Генерация API-ключа

Перед использованием API сгенерируйте API-ключ:

#### Запрос

```bash
POST /api/generate_key/
Content-Type: application/json

{
  "name": "Название_вашего_приложения"
}
```

#### Ответ

```json
{
  "api_key": "сгенерированный_api_ключ",
  "name": "Название_вашего_приложения",
  "created_at": "2023-12-24T12:00:00"
}
```

### Загрузка данных в базу знаний

Вы можете загружать текстовые данные или файлы (PDF, DOCX, Markdown) в базу знаний.

#### Запрос

```bash
POST /api/knowledge_base/
Content-Type: multipart/form-data
api-key: ваш_api_ключ

Поля формы:
- files: список файлов для загрузки
- text_data: текстовые данные (опционально)
```

#### Пример использования cURL

```bash
curl -X POST "http://127.0.0.1:8000/api/knowledge_base/" \
 -H "api-key: ваш_api_ключ" \
 -F "files=@path/to/your/document.pdf"
```

### Поиск по базе знаний

Ищите релевантные документы в базе знаний по текстовому запросу.

#### Запрос

```bash
POST /api/search/
Content-Type: application/json
api-key: ваш_api_ключ

{
  "query": "Ваш поисковой запрос",
  "top_k": 5  # Количество возвращаемых результатов (опционально)
}
```

#### Ответ

```json
{
  "results": [
    {
      "id": 1,
      "title": "Название документа",
      "content": "Релевантный фрагмент текста...",
      "score": 0.85
    },
    ...
  ]
}
```

### Генерация ответа с использованием RAG

Получите ответ от модели, обогащенный контекстом из базы знаний.

#### Запрос

```bash
POST /api/rag/
Content-Type: application/json
api-key: ваш_api_ключ

{
  "user_query": "Ваш вопрос"
}
```

#### Ответ

```json
{
  "response": "Сгенерированный ответ модели с учетом вашего контекста"
}
```

---

## Структура проекта

```plaintext
IDE_code_assistant/
├── main.py                # Главный файл приложения FastAPI
├── requirements.txt       # Список зависимостей для установки через pip
├── .env                   # Переменные окружения (НЕ добавляйте в Git)
├── .gitignore             # Файлы и папки, игнорируемые Git
├── README.md              # Документация проекта
├── alembic.ini            # Конфигурация для Alembic (миграции)
├── alembic/               # Папка с миграциями базы данных
│   ├── versions/          # Конкретные миграции
│   └── env.py             # Скрипт окружения для Alembic
├── api/                   # Пакет с API-эндпоинтами
│   ├── __init__.py
│   ├── endpoints.py       # Определение маршрутов API
│   └── models.py          # Pydantic-модели
├── core/                  # Основные компоненты приложения
│   ├── __init__.py
│   ├── config.py          # Настройки приложения, загрузка переменных окружения
│   ├── database.py        # Настройки базы данных и модели SQLAlchemy
│   ├── security.py        # Управление API-ключами и безопасность
│   ├── logger.py          # Настройка логирования
│   └── logging.yaml       # Конфигурация логирования
├── services/              # Сервисы для работы приложения
│   ├── __init__.py
│   ├── openai_service.py  # Работа с OpenAI API
│   ├── lmstudio_service.py# Работа с LMStudio API
│   ├── embedding_service.py# Генерация эмбеддингов
│   ├── indexing_service.py# Управление индексом FAISS
│   └── file_service.py    # Обработка файлов и извлечение текста
├── tests/                 # Модульные тесты
│   ├── __init__.py
│   ├── conftest.py        # Конфигурация тестов и фикстуры
│   ├── test_api.py        # Тесты API-эндпоинтов
│   ├── test_embedding_service.py    # Тесты для embedding_service
│   ├── test_file_service.py         # Тесты для file_service
│   ├── test_indexing_service.py     # Тесты для indexing_service
│   ├── test_openai_service.py       # Тесты для openai_service
│   └── test_security.py             # Тесты для security
├── utils/                 # Вспомогательные функции и утилиты
│   ├── __init__.py
│   └── helpers.py         # Различные вспомогательные функции
├── clear_documents.py     # Скрипт для очистки базы данных и индексов
└── logging.yaml           # Конфигурация логирования
```

---

## Вклад в проект

Мы приветствуем вклад сообщества! Если вы хотите помочь развитию проекта:

1. **Форкните репозиторий**

   ```bash
   git clone https://github.com/ваш_пользователь/IDE_code_assistant.git
   ```

2. **Создайте новую ветку**

   ```bash
   git checkout -b feature/название-функции
   ```

3. **Внесите изменения**

   Внесите свои улучшения или новые функции.

4. **Сделайте коммит**

   ```bash
   git add .
   git commit -m "Добавлена новая функциональность"
   ```

5. **Отправьте изменения на GitHub**

   ```bash
   git push origin feature/название-функции
   ```

6. **Создайте Pull Request**

   Перейдите в ваш форк на GitHub и создайте Pull Request обратно в оригинальный репозиторий.

---

## Лицензия

Этот проект распространяется под лицензией [MIT](https://github.com/Bondartsov/IDE_code_assistant/blob/main/LICENSE).

---

**Если у вас есть вопросы или предложения, пожалуйста, создайте issue или Pull Request в [репозитории](https://github.com/Bondartsov/IDE_code_assistant).**

---

**Спасибо, что используете IDE Code Assistant!**

---

## Дополнительная информация

### Скрипт очистки базы данных

Добавлен скрипт `clear_documents.py` для полного удаления данных из базы знаний. Он удаляет:

- Все записи из таблицы `documents` в базе данных.
- Файлы индекса FAISS: `index.faiss` и `id_to_idx.pkl`.
- Файл `documents.pkl`, содержащий сериализованные данные документов.

#### Запуск скрипта

```bash
python clear_documents.py
```

### Тестирование

- Все тесты находятся в папке `tests/` и запускаются с помощью Pytest.
- Для запуска всех тестов:

  ```bash
  pytest -v
  ```

- Тесты настроены на использование отдельной тестовой базы данных `test_app.db`, чтобы не влиять на основную базу данных приложения.

### Логирование

- Конфигурация логирования находится в файле `logging.yaml`.
- Логи помогают отслеживать работу приложения и быстро находить ошибки.

### Интеграция с LMStudio

- Приложение поддерживает интеграцию с LMStudio для использования локальных моделей.
- Для переключения между OpenAI и LMStudio измените значение `API_PROVIDER` в файле `.env`.

---

**Продолжайте пользоваться и улучшать IDE Code Assistant!**