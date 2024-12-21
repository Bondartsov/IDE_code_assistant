# tests/test_file_service.py

import pytest
from services.file_service import extract_text_from_pdf, extract_text_from_docx
from fastapi import UploadFile
from io import BytesIO
from unittest import mock

class MockUploadFile(UploadFile):
    def __init__(self, filename: str, file, content_type: str):
        super().__init__(filename=filename, file=file)
        self._content_type = content_type  # Устанавливаем через защищённое свойство

    @property
    def content_type(self):
        return self._content_type

@pytest.mark.asyncio
async def test_extract_text_from_pdf(mocker):
    # Мокируем содержимое PDF-файла
    mocker.patch('pypdf.PdfReader', return_value=mocker.Mock(
        pages=[mocker.Mock(extract_text=lambda: "Page 1 text"),
               mocker.Mock(extract_text=lambda: "Page 2 text")]
    ))

    file = MockUploadFile(filename="test.pdf", file=BytesIO(b''), content_type="application/pdf")
    text = await extract_text_from_pdf(file)
    assert text == "Page 1 textPage 2 text"

@pytest.mark.asyncio
async def test_extract_text_from_docx(mocker):
    # Мокируем содержимое DOCX-файла
    mocker.patch('docx.Document', return_value=mocker.Mock(
        paragraphs=[mocker.Mock(text="Paragraph 1"),
                    mocker.Mock(text="Paragraph 2")]
    ))

    file = MockUploadFile(
        filename="test.docx",
        file=BytesIO(b''),
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    text = await extract_text_from_docx(file)
    assert text == "Paragraph 1\nParagraph 2"