# main.py

from fastapi import FastAPI
from api.endpoints import router as api_router

# Инициализация приложения FastAPI
app = FastAPI(
    title="IDE Code Assistant API",
    description="API для взаимодействия с IDE Code Assistant",
    version="1.0.0"
)

# Подключение маршрутов из пакета api с префиксом "/api"
app.include_router(api_router, prefix="/api")