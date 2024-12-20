# Файл: services/file_service.py

from fastapi import UploadFile
from io import BytesIO
import pypdf
import docx

async def extract_text_from_pdf(file: UploadFile) -> str:
    """
    Извлечение текста из PDF-файла.
    """
    contents = await file.read()
    reader = pypdf.PdfReader(BytesIO(contents))
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

async def extract_text_from_docx(file: UploadFile) -> str:
    """
    Извлечение текста из DOCX-файла.
    """
    contents = await file.read()
    document = docx.Document(BytesIO(contents))
    text = "\n".join([para.text for para in document.paragraphs])
    return text