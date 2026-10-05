"""Retrieval: ChromaDB search, metadata filtering, relevance grading and citations."""

import re
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import get_llm, settings
from src.prompts import GRADER_PROMPT

STOP_WORDS = {
    "about",
    "after",
    "and",
    "are",
    "can",
    "does",
    "for",
    "from",
    "have",
    "how",
    "i",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "please",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "with",
    "you",
    "your",
}


def _keywords(text: str) -> set[str]:
    """Small, dependency-free lexical signal for obvious policy matches."""
    return {
        word.rstrip("s")
        for word in re.findall(r"[a-z]{3,}", text.lower())
        if word not in STOP_WORDS
    }


# Free local embeddings, cached so the model loads only once
@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


# Open (or create) the persistent ChromaDB collection
@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=settings.COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.CHROMA_DIR),
    )


# Combine department and origin conditions into one Chroma metadata filter
def build_filter(department: str | None, include_samples: bool) -> dict | None:
    conditions = []
    if department:
        conditions.append({"department": department})
    if not include_samples:
        conditions.append({"origin": "upload"})  # hide the sample documents
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


# Vector search, optionally limited to one department and/or uploaded documents only
def retrieve(
    query: str,
    department: str | None = None,
    k: int | None = None,
    include_samples: bool = True,
) -> list[Document]:
    flt = build_filter(department, include_samples)
    return get_vector_store().similarity_search(
        query, k=k or settings.TOP_K, filter=flt
    )


# Relevance grader: judge each chunk separately (easier for small models), pass if any one is relevant
def grade_documents(question: str, docs: list[Document]) -> bool:
    if not docs:
        return False  # nothing retrieved means nothing relevant
    question_terms = _keywords(question)
    for doc in docs:
        # Do not let an inconsistent LLM discard an obvious match such as
        # "How long do refunds take?" against the refund policy.
        if question_terms & _keywords(doc.page_content):
            return True
    chain = GRADER_PROMPT | get_llm() | StrOutputParser()
    for doc in docs:
        try:
            verdict = chain.invoke({"question": question, "context": doc.page_content})
        except Exception:
            continue
        # Look at the first few words only, so "Yes.", "**Yes**" and "Yes, it does" all count as yes
        first_words = re.findall(r"[a-z]+", verdict.lower())[:3]
        if "yes" in first_words:
            return True  # one relevant chunk is enough, so stop early
    return False


# Join chunks into one prompt string, each labelled with its source file
def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
    )


# Build unique citations, with page numbers for PDFs
def format_sources(docs: list[Document]) -> list[str]:
    out = set()
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        out.add(f"{src} (p.{page + 1})" if page is not None else src)
    return sorted(out)
