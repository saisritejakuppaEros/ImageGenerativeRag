"""Convert PDF files in papers/ to markdown using PyMuPDF."""
from pathlib import Path

import pymupdf

PAPERS_DIR = Path(__file__).parent / "papers"


def pdf_to_md(pdf_path: Path) -> Path:
    doc = pymupdf.open(pdf_path)
    title = pdf_path.stem.replace("_", " ")
    parts = [f"# {title}\n"]

    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            parts.append(f"## Page {i}\n\n{text}")

    doc.close()

    md_path = pdf_path.with_suffix(".md")
    md_path.write_text("\n\n---\n\n".join(parts), encoding="utf-8")
    return md_path


def main() -> None:
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        print("No PDF files found in papers/")
        return

    for pdf in pdfs:
        md_path = pdf_to_md(pdf)
        print(f"Converted: {pdf.name} -> {md_path.name}")


if __name__ == "__main__":
    main()
