# main.py

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from api.endpoints import router as api_router
from fastapi.security import APIKeyHeader

# Инициализация приложения FastAPI
app = FastAPI(
    title="IDE Code Assistant API",
    description="API для взаимодействия с IDE Code Assistant",
    version="1.0.0"
)

# Определение схемы безопасности API-ключа
API_KEY_NAME = "api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Подключение маршрутов из пакета api с префиксом "/api"
app.include_router(api_router, prefix="/api")

# Настройка схемы OpenAPI с включением безопасности
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="IDE Code Assistant API",
        version="1.0.0",
        description="API для взаимодействия с IDE Code Assistant",
        routes=app.routes,
    )
    # Определяем схему безопасности API-ключа
    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyHeader": {
            "type": "apiKey",
            "in": "header",
            "name": API_KEY_NAME
        }
    }
    # Добавляем информацию о безопасности для всех путей
    for path in openapi_schema["paths"].values():
        for operation in path.values():
            operation["security"] = [{"APIKeyHeader": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi