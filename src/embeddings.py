"""HuggingFace embeddings loading and initialisation."""
from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingsError(RuntimeError):
    """Raised when the embedding model cannot be loaded."""


def load_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Load a HuggingFace sentence-embedding model.

    Args:
        model_name: HuggingFace model id, e.g. ``sentence-transformers/all-MiniLM-L6-v2``.

    Raises:
        EmbeddingsError: If the model cannot be downloaded or initialised.
    """
    logger.info("Loading embedding model: %s", model_name)
    try:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
    except Exception as exc:  # third-party libs raise many exception types
        logger.exception("Failed to load embedding model '%s'", model_name)
        raise EmbeddingsError(f"Could not load embedding model '{model_name}': {exc}") from exc
    logger.info("Embedding model loaded successfully.")
    return embeddings
