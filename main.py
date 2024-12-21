# Файл: main.py

from fastapi import FastAPI
from api.endpoints import router as api_router
# Инициализация приложения FastAPI
app = FastAPI()
# Подключение маршрутов из пакета api с префиксом "/api"
app.include_router(api_router, prefix="/api")
