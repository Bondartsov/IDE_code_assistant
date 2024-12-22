# services/file_service.py

import io
from typing import List, Optional
from fastapi import UploadFile
from PyPDF2 import PdfReader
import docx2txt
import markdown
from bs4 import BeautifulSoup

print("file_service.py загружен")

async def extract_text(files: Optional[List[UploadFile]], texts: Optional[List[str]]) -> str:
    """
    Обрабатывает загруженные файлы и текстовые данные, извлекает текст из них и объединяет.

    :param files: Список загруженных файлов.
    :param texts: Список текстовых данных.
    :return: Объединённый текст из файлов и текстовых данных.
    """
    combined_text = ""

    if files:
        file_texts = await process_files(files)
        combined_text += "\n".join(file_texts) + "\n"
    if texts:
        processed_texts = process_text_data(texts)
        combined_text += "\n".join(processed_texts)
    return combined_text.strip()

async def process_files(files: List[UploadFile]) -> List[str]:
    """
    Обрабатывает загруженные файлы и извлекает текст из них.

    :param files: Список загруженных файлов.
    :return: Список извлечённых текстовых данных.
    """
    texts = []
    for file in files:
        content = await file.read()
        filename = file.filename.lower()

        if filename.endswith(".txt"):
            text = content.decode('utf-8')
            texts.append(text)

        elif filename.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(content))
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            if text:
                texts.append(text)

        elif filename.endswith(".docx"):
            # docx2txt ожидает путь к файлу, поэтому сохраняем его временно
            with open("temp.docx", "wb") as temp_docx:
                temp_docx.write(content)
            text = docx2txt.process("temp.docx")
            texts.append(text)
            import os
            os.remove("temp.docx")

        elif filename.endswith((".md", ".markdown")):
            html = markdown.markdown(content.decode('utf-8'))
            soup = BeautifulSoup(html, features="html.parser")
            text = soup.get_text()
            texts.append(text)

        else:
            # Если формат файла не поддерживается, пропускаем его
            continue

    return texts

def process_text_data(text_data: List[str]) -> List[str]:
    """
    Обрабатывает переданные текстовые данные.

    :param text_data: Список строк с текстом.
    :return: Список обработанных текстовых данных.
    """
    # Здесь можно добавить дополнительную обработку текста, если необходимо
    return text_data