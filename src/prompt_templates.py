"""Prompt strings and builders."""
from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

SYSTEM_PROMPT: str = "You are a helpful assistant."

RAG_PROMPT_TEMPLATE: str = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

NO_CONTEXT_ANSWER: str = "I don't know. No relevant information was found in the knowledge base."


def build_rag_prompt() -> PromptTemplate:
    """Build the RAG :class:`PromptTemplate`."""
    return PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )


def format_context(documents: Sequence[Document]) -> str:
    """Join retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in documents)
