# services/file_service.py

from typing import List, Optional
from fastapi import UploadFile
from io import BytesIO
import pypdf
import docx

async def extract_text(files: Optional[List[UploadFile]], texts: Optional[List[str]]) -> str:
    """
    Обработка файлов и текстовых данных, извлечение и объединение текста.

    Args:
        files (Optional[List[UploadFile]]): Список загруженных файлов.
        texts (Optional[List[str]]): Список текстовых данных.

    Returns:
        str: Объединённый текст из файлов и текстовых данных.
    """
    combined_text = ""

    if files:
        for file in files:
            if file.content_type == "text/plain":
                text = await extract_text_from_txt(file)
            elif file.content_type == "application/pdf":
                text = await extract_text_from_pdf(file)
            elif file.content_type in [
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword",
            ]:
                text = await extract_text_from_docx(file)
            elif file.content_type == "text/markdown":
                text = await extract_text_from_markdown(file)
            else:
                text = ""
            combined_text += text + "\n"

    if texts:
        for text in texts:
            combined_text += text + "\n"

    return combined_text.strip()

async def extract_text_from_txt(file: UploadFile) -> str:
    contents = await file.read()
    return contents.decode('utf-8')

async def extract_text_from_pdf(file: UploadFile) -> str:
    contents = await file.read()
    pdf_reader = pypdf.PdfReader(BytesIO(contents))
    text = ''
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

async def extract_text_from_docx(file: UploadFile) -> str:
    contents = await file.read()
    document = docx.Document(BytesIO(contents))
    text = '\n'.join([para.text for para in document.paragraphs])
    return text

async def extract_text_from_markdown(file: UploadFile) -> str:
    contents = await file.read()
    return contents.decode('utf-8')