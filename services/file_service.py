# services/file_service.py

"""
Модуль для обработки файлов и извлечения текста.
"""

from typing import Union
from fastapi import UploadFile
from io import BytesIO
import pypdf
import docx

async def extract_text_from_pdf(file: UploadFile) -> str:
    """
    Извлекает текст из PDF-файла.

    Args:
        file (UploadFile): Загруженный файл.

    Returns:
        str: Извлеченный текст.
    """
    contents = await file.read()
    reader = pypdf.PdfReader(BytesIO(contents))
    text = "".join(page.extract_text() for page in reader.pages)
    return text

async def extract_text_from_docx(file: UploadFile) -> str:
    """
    Извлекает текст из DOCX-файла.

    Args:
        file (UploadFile): Загруженный файл.

    Returns:
        str: Извлеченный текст.
    """
    contents = await file.read()
    document = docx.Document(BytesIO(contents))
    text = "\n".join(para.text for para in document.paragraphs)
    return text

async def extract_text_from_markdown(file: UploadFile) -> str:
    """
    Извлекает текст из Markdown-файла.

    Args:
        file (UploadFile): Загруженный файл.

    Returns:
        str: Извлеченный текст.
    """
    contents = await file.read()
    text = contents.decode('utf-8')
    return text

async def extract_text(file: UploadFile) -> str:
    """
    Определяет тип файла и извлекает текст соответствующим образом.

    Args:
        file (UploadFile): Загруженный файл.

    Returns:
        str: Извлеченный текст.

    Raises:
        ValueError: Если тип файла не поддерживается.
    """
    if file.content_type == "application/pdf":
        return await extract_text_from_pdf(file)
    elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return await extract_text_from_docx(file)
    elif file.content_type == "text/markdown":
        return await extract_text_from_markdown(file)
    else:
        raise ValueError(f"Unsupported file type: {file.content_type}")