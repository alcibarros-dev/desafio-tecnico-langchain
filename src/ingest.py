import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import SetupError, get_vector_store

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = os.getenv("PDF_PATH") or str(PROJECT_ROOT / "document.pdf")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def ingest_pdf():
    pdf_path = Path(PDF_PATH)
    if not pdf_path.is_absolute() and not pdf_path.exists():
        pdf_path = PROJECT_ROOT / pdf_path
    if not pdf_path.exists():
        print(f"Arquivo PDF não encontrado: {PDF_PATH}")
        sys.exit(1)

    print(f"Carregando PDF: {pdf_path}")
    documents = PyPDFLoader(str(pdf_path)).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"{len(documents)} páginas divididas em {len(chunks)} chunks.")

    try:
        # Recria a collection a cada execução para não duplicar chunks.
        store = get_vector_store(pre_delete_collection=True)
        ids = [f"{pdf_path.stem}-{i}" for i in range(len(chunks))]
        store.add_documents(chunks, ids=ids)
    except SetupError as e:
        print(f"Erro: {e}")
        sys.exit(1)

    print(f"Ingestão concluída: {len(chunks)} chunks salvos no PostgreSQL.")


if __name__ == "__main__":
    ingest_pdf()
