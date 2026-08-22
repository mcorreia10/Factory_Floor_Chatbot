import io
from pathlib import Path

from pypdf import PdfReader, PdfWriter


def extract_page_pdf(manual_dir: Path, source_file: str, page_number: int) -> bytes:
    reader = PdfReader(str(Path(manual_dir) / source_file))
    writer = PdfWriter()
    writer.add_page(reader.pages[page_number - 1])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()
