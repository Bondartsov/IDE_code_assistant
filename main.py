# Файл: main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.endpoints import router as api_router
from core.logger import logger, setup_logging

# Настройка логирования
setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Действия при запуске приложения
    logger.info("Приложение запущено")
    yield
    # Действия при остановке приложения
    logger.info("Приложение остановлено")

# Инициализация приложения FastAPI с использованием lifespan
app = FastAPI(title="IDE Code Assistant", version="1.0.0", lifespan=lifespan)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Разрешаем запросы с любых источников
    allow_credentials=True,
    allow_methods=["*"],       # Разрешаем все методы (GET, POST и т.д.)
    allow_headers=["*"],       # Разрешаем все заголовки
)

# Подключение маршрутов из пакета api
app.include_router(api_router)