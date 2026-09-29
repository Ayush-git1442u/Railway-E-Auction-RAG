"""FAISS vector store loading and similarity search."""
from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from utils.logger import get_logger

logger = get_logger(__name__)

_REQUIRED_INDEX_FILES = ("index.faiss", "index.pkl")


class VectorStoreError(RuntimeError):
    """Raised for FAISS loading or search failures."""


class VectorStoreManager:
    """Loads a persisted FAISS index and exposes similarity search."""

    def __init__(
        self,
        faiss_path: Path,
        embeddings: Embeddings,
        k: int = 3,
        allow_dangerous_deserialization: bool = True,
    ) -> None:
        self._faiss_path = Path(faiss_path)
        self._embeddings = embeddings
        self._k = k
        self._allow_dangerous = allow_dangerous_deserialization
        self._db: FAISS | None = None

    @property
    def is_loaded(self) -> bool:
        return self._db is not None

    def load(self) -> None:
        """Load the FAISS index from disk.

        Raises:
            VectorStoreError: If files are missing or the index cannot be loaded.
        """
        if not self._faiss_path.is_dir():
            raise VectorStoreError(f"FAISS directory not found: {self._faiss_path}")

        missing = [f for f in _REQUIRED_INDEX_FILES if not (self._faiss_path / f).is_file()]
        if missing:
            raise VectorStoreError(
                f"FAISS directory {self._faiss_path} is missing required file(s): {', '.join(missing)}"
            )

        logger.info("Loading FAISS index from %s", self._faiss_path)
        try:
            self._db = FAISS.load_local(
                str(self._faiss_path),
                embeddings=self._embeddings,
                allow_dangerous_deserialization=self._allow_dangerous,
            )
        except Exception as exc:
            logger.exception("Failed to load FAISS index")
            raise VectorStoreError(f"Could not load FAISS index: {exc}") from exc
        logger.info("FAISS index loaded successfully.")

    def similarity_search(self, query: str, k: int | None = None) -> list[Document]:
        """Return the top-``k`` most similar documents for ``query``.

        Raises:
            VectorStoreError: If the index isn't loaded or the search fails.
        """
        if self._db is None:
            raise VectorStoreError("Vector store not loaded. Call load() first.")

        top_k = k or self._k
        try:
            docs = self._db.similarity_search(query, k=top_k)
        except Exception as exc:
            logger.exception("Similarity search failed")
            raise VectorStoreError(f"Similarity search failed: {exc}") from exc

        logger.debug("Retrieved %d document(s) for query.", len(docs))
        return docs
