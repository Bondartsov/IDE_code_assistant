# File: alembic/env.py

import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# Добавьте путь к корневой директории вашего проекта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импортируйте ваши модели
from config_storage import Base  # Замените на путь к вашим моделям, если они в другом модуле

# Настройка метаданных для Alembic
target_metadata = Base.metadata

# Это объект конфигурации Alembic, который предоставляет доступ к значениям внутри файла .ini
config = context.config

# Интерпретируем файл конфигурации для логирования Python.
# Эта строка настраивает логирование.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Другие значения из конфигурации, определяемые потребностями env.py,
# могут быть получены здесь:
# my_important_option = config.get_main_option("my_important_option")
# ... и так далее.

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()