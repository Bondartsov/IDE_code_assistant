# tests/test_file_service.py

import pytest
from services.file_service import process_files
from fastapi import UploadFile
from io import BytesIO
from unittest import mock

class MockUploadFile(UploadFile):
    def __init__(self, filename: str, content: bytes, content_type: str):
        super().__init__(filename=filename, file=BytesIO(content))
        self._content_type = content_type

    @property
    def content_type(self):
        return self._content_type

@pytest.mark.asyncio
async def test_process_files():
    """
    Тест обработки файлов с извлечением текста.
    """
    txt_content = b"Sample text content."
    pdf_content = b"%PDF-1.4..."  # Simplified PDF content
    docx_content = b"PK..."  # Simplified DOCX content
    md_content = b"# Heading\nThis is a test."

    txt_file = MockUploadFile(
        filename="test.txt",
        content=txt_content,
        content_type="text/plain"
    )

    pdf_file = MockUploadFile(
        filename="test.pdf",
        content=pdf_content,
        content_type="application/pdf"
    )

    docx_file = MockUploadFile(
        filename="test.docx",
        content=docx_content,
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

    md_file = MockUploadFile(
        filename="test.md",
        content=md_content,
        content_type="text/markdown"
    )

    files = [txt_file, pdf_file, docx_file, md_file]

    with mock.patch('services.file_service.PdfReader') as mock_pdf_reader, \
         mock.patch('services.file_service.docx2txt.process') as mock_docx_process, \
         mock.patch('services.file_service.markdown.markdown') as mock_markdown, \
         mock.patch('services.file_service.BeautifulSoup') as mock_bs:

        # Мокаем PdfReader
        mock_pdf = mock.Mock()
        mock_pdf.pages = [
            mock.Mock(extract_text=lambda: "PDF Page 1 text"),
            mock.Mock(extract_text=lambda: "PDF Page 2 text")
        ]
        mock_pdf_reader.return_value = mock_pdf

        # Мокаем docx2txt.process
        mock_docx_process.return_value = "DOCX Paragraph 1\nDOCX Paragraph 2"

        # Мокаем markdown.markdown и BeautifulSoup
        mock_markdown.return_value = "<h1>Heading</h1><p>This is a test.</p>"
        mock_soup = mock.Mock()
        mock_soup.get_text.return_value = "Heading\nThis is a test."
        mock_bs.return_value = mock_soup

        texts = await process_files(files)

        assert len(texts) == 4
        assert "Sample text content." in texts

        # Проверяем содержимое PDF-файла
        assert "PDF Page 1 text" in texts[1]
        assert "PDF Page 2 text" in texts[1]

        # Проверяем содержимое DOCX-файла
        assert "DOCX Paragraph 1" in texts[2]
        assert "DOCX Paragraph 2" in texts[2]

        # Проверяем содержимое Markdown-файла
        assert "Heading\nThis is a test." in texts[3]