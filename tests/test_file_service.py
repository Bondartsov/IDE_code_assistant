# tests/test_file_service.py

import pytest
from services.file_service import extract_text_from_pdf, extract_text_from_docx, extract_text_from_markdown
from fastapi import UploadFile
from io import BytesIO
from unittest import mock

class MockUploadFile(UploadFile):
    def __init__(self, filename: str, file, content_type: str):
        super().__init__(filename=filename, file=file)
        self._content_type = content_type

    @property
    def content_type(self):
        return self._content_type

@pytest.mark.asyncio
async def test_extract_text_from_pdf():
    """
    Тест извлечения текста из PDF-файла.
    """
    with mock.patch('pypdf.PdfReader') as mock_reader:
        mock_reader.return_value.pages = [
            mock.Mock(extract_text=lambda: "Page 1 text"),
            mock.Mock(extract_text=lambda: "Page 2 text")
        ]

        file = MockUploadFile(filename="test.pdf", file=BytesIO(b''), content_type="application/pdf")
        text = await extract_text_from_pdf(file)
        assert text == "Page 1 textPage 2 text"

@pytest.mark.asyncio
async def test_extract_text_from_docx():
    """
    Тест извлечения текста из DOCX-файла.
    """
    with mock.patch('docx.Document') as mock_document:
        mock_document.return_value.paragraphs = [
            mock.Mock(text="Paragraph 1"),
            mock.Mock(text="Paragraph 2")
        ]

        file = MockUploadFile(
            filename="test.docx",
            file=BytesIO(b''),
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        text = await extract_text_from_docx(file)
        assert text == "Paragraph 1\nParagraph 2"

@pytest.mark.asyncio
async def test_extract_text_from_markdown():
    """
    Тест извлечения текста из Markdown-файла.
    """
    file_content = b"# Heading\nThis is a test."
    file = MockUploadFile(
        filename="test.md",
        file=BytesIO(file_content),
        content_type="text/markdown"
    )
    text = await extract_text_from_markdown(file)
    assert text == "# Heading\nThis is a test."