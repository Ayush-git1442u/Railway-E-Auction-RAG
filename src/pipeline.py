"""RAG orchestration: retrieval -> prompt -> LLM."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.prompts import PromptTemplate

from config.settings import Settings
from src.embeddings import EmbeddingsError, load_embeddings
from src.llm_client import LLMService, LLMServiceError
from src.prompt_templates import (
    NO_CONTEXT_ANSWER,
    SYSTEM_PROMPT,
    build_rag_prompt,
    format_context,
)
from src.vector_store import VectorStoreError, VectorStoreManager
from utils.logger import get_logger

logger = get_logger(__name__)


class RAGPipelineError(RuntimeError):
    """Raised for any failure inside the RAG pipeline."""


@dataclass(frozen=True)
class RAGResult:
    """Structured output of a pipeline run."""

    query: str
    answer: str
    context: str
    sources: list[dict[str, Any]] = field(default_factory=list)


class RAGPipeline:
    """Connects vector retrieval, prompt construction and LLM generation."""

    def __init__(
        self,
        vector_store: VectorStoreManager,
        llm_service: LLMService,
        prompt_template: PromptTemplate,
        system_prompt: str = SYSTEM_PROMPT,
        chat_history_limit: int = 50,
    ) -> None:
        self._vector_store = vector_store
        self._llm = llm_service
        self._prompt = prompt_template
        self._system_prompt = system_prompt
        self._history_limit = chat_history_limit
        self._chat_history: list[tuple[str, str]] = []

    @classmethod
    def from_settings(cls, settings: Settings) -> "RAGPipeline":
        """Build a fully wired pipeline from application settings.

        Raises:
            RAGPipelineError: If any component fails to initialise.
        """
        try:
            embeddings = load_embeddings(settings.embedding_model_name)
            vector_store = VectorStoreManager(
                faiss_path=settings.faiss_path,
                embeddings=embeddings,
                k=settings.retrieval_k,
                allow_dangerous_deserialization=settings.faiss_allow_dangerous_deserialization,
            )
            vector_store.load()
            llm_service = LLMService(settings)
        except (EmbeddingsError, VectorStoreError, LLMServiceError) as exc:
            raise RAGPipelineError(f"Pipeline initialisation failed: {exc}") from exc

        return cls(
            vector_store=vector_store,
            llm_service=llm_service,
            prompt_template=build_rag_prompt(),
            chat_history_limit=settings.chat_history_limit,
        )

    @property
    def chat_history(self) -> list[tuple[str, str]]:
        """A copy of the (query, answer) history."""
        return list(self._chat_history)

    def run_query(self, user_query: str) -> RAGResult:
        """Answer ``user_query`` using retrieved context.

        Raises:
            ValueError: If the query is empty.
            RAGPipelineError: If retrieval or generation fails.
        """
        query = (user_query or "").strip()
        if not query:
            raise ValueError("Query must not be empty.")

        logger.info("Running query: %s", query)
        try:
            documents = self._vector_store.similarity_search(query)
        except VectorStoreError as exc:
            raise RAGPipelineError(f"Retrieval failed: {exc}") from exc

        sources = [dict(doc.metadata) for doc in documents]

        if not documents:
            logger.warning("No documents retrieved; skipping LLM call.")
            result = RAGResult(query=query, answer=NO_CONTEXT_ANSWER, context="", sources=[])
            self._remember(query, result.answer)
            return result

        context = format_context(documents)
        prompt = self._prompt.format(context=context, question=query)

        try:
            answer = self._llm.generate(user_prompt=prompt, system_prompt=self._system_prompt)
        except LLMServiceError as exc:
            raise RAGPipelineError(f"Generation failed: {exc}") from exc

        self._remember(query, answer)
        return RAGResult(query=query, answer=answer, context=context, sources=sources)

    def _remember(self, query: str, answer: str) -> None:
        self._chat_history.append((query, answer))
        if self._history_limit and len(self._chat_history) > self._history_limit:
            self._chat_history = self._chat_history[-self._history_limit :]
