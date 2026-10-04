"""Ingestion: load documents, chunk them, embed and store them in ChromaDB."""

import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.retrieval import get_vector_store

SUPPORTED = {".pdf", ".md", ".txt"}  # file types we know how to read
SOURCES = [
    (settings.SAMPLE_DIR, "sample"),
    (settings.DOCS_DIR, "upload"),
]  # (folder, origin tag)


def clean_department(name: str) -> str:
    """Normalise a department name to a safe folder name, e.g. 'Customer Support' -> 'customer_support'."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_") or "general"


def load_file(path: Path, department: str, origin: str) -> list[Document]:
    """Read one file and tag every page with its department, source file name and origin."""
    is_pdf = path.suffix.lower() == ".pdf"
    loader = (
        PyPDFLoader(str(path)) if is_pdf else TextLoader(str(path), encoding="utf-8")
    )
    docs = loader.load()
    for d in docs:
        d.metadata.update(source=path.name, department=department, origin=origin)
    return docs


def load_documents() -> list[Document]:
    """Read every supported file from the sample and upload folders (department = subfolder name)."""
    docs = []
    for root, origin in SOURCES:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() in SUPPORTED:
                rel = path.relative_to(root)
                department = rel.parts[0] if len(rel.parts) > 1 else "general"
                docs.extend(load_file(path, department, origin))
    return docs


def chunk_documents(
    docs: list[Document],
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> list[Document]:
    """Split into overlapping chunks; IDs are numbered per file so re-uploads replace cleanly."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    counters: dict[str, int] = {}
    for c in chunks:
        m = c.metadata
        key = f"{m['origin']}-{m['department']}-{m['source']}"
        counters[key] = counters.get(key, 0) + 1
        c.metadata["chunk_id"] = f"{key}-{counters[key]}"
    return chunks


def remove_from_index(source: str, department: str, origin: str = "upload") -> None:
    """Delete all stored chunks of one document."""
    store = get_vector_store()
    where = {
        "$and": [{"source": source}, {"department": department}, {"origin": origin}]
    }
    ids = store.get(where=where)["ids"]
    if ids:
        store.delete(ids=ids)


def index_file(path: Path, department: str, origin: str = "upload") -> int:
    """Index one file, replacing any older version of it. Returns the number of chunks added."""
    chunks = chunk_documents(load_file(path, department, origin))
    if not chunks:  # e.g. a scanned PDF with no extractable text
        return 0
    remove_from_index(path.name, department, origin)
    get_vector_store().add_documents(
        chunks, ids=[c.metadata["chunk_id"] for c in chunks]
    )
    return len(chunks)


def save_upload(file_name: str, data: bytes, department: str) -> int:
    """Save an uploaded file under data/documents/<department>/ and index it."""
    safe_name = Path(file_name).name  # strips any folder parts from the name
    if Path(safe_name).suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {safe_name}")

    dept = clean_department(department)
    folder = settings.DOCS_DIR / dept
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / safe_name
    path.write_bytes(data)
    return index_file(path, dept, "upload")


def delete_document(source: str, department: str) -> None:
    """Remove an uploaded document from the index and disk (sample documents are protected)."""
    remove_from_index(source, department, "upload")
    path = settings.DOCS_DIR / department / source
    if not path.exists():  # files placed directly in the documents folder
        path = settings.DOCS_DIR / source
    path.unlink(missing_ok=True)


def list_documents() -> list[dict]:
    """List indexed documents with origin and chunk counts, read straight from ChromaDB."""
    counts: dict[tuple[str, str, str], int] = {}
    for meta in get_vector_store().get(include=["metadatas"])["metadatas"]:
        key = (meta["origin"], meta["department"], meta["source"])
        counts[key] = counts.get(key, 0) + 1
    return [
        {"origin": o, "department": d, "source": s, "chunks": n}
        for (o, d, s), n in sorted(counts.items())
    ]


def build_index(reset: bool = True) -> int:
    """Re-index everything (samples and uploads); reset=True wipes the old index first."""
    if reset:
        get_vector_store().delete_collection()
        get_vector_store.cache_clear()  # forces a fresh, empty collection

    chunks = chunk_documents(load_documents())
    if not chunks:
        return 0
    get_vector_store().add_documents(
        chunks, ids=[c.metadata["chunk_id"] for c in chunks]
    )
    print(f"Indexed {len(chunks)} chunks")
    return len(chunks)


def ensure_index() -> int:
    """Build the index only if the database is empty, so nobody has to run ingestion manually."""
    if get_vector_store().get(limit=1)["ids"]:
        return 0  # already indexed
    return build_index(reset=False)


if __name__ == "__main__":
    build_index()  # manual rebuild: python -m src.ingestion
