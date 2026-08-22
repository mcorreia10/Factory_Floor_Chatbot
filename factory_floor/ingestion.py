import csv
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from factory_floor.config import SOURCES_CSV


def load_equipment_types(sources_csv: Path = SOURCES_CSV) -> dict:
    with Path(sources_csv).open(encoding="utf-8") as f:
        return {row["filename"]: row["equipment_type"] for row in csv.DictReader(f)}


def load_manuals(manual_dir: Path):
    equipment_types = load_equipment_types()
    pdf_files = sorted(Path(manual_dir).glob("*.pdf"))
    documents = []
    for pdf_path in pdf_files:
        pages = PyPDFLoader(str(pdf_path)).load()
        for page in pages:
            page.metadata["source_file"] = pdf_path.name
            page.metadata["manufacturer"] = "Siemens" if "Siemens" in pdf_path.name else "Unknown"
            page.metadata["equipment_type"] = equipment_types.get(pdf_path.name, "unknown")
        documents.extend(pages)
    return documents
