"""Groq chat-completion wrapper."""
from __future__ import annotations

import groq
from groq import Groq

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


class LLMServiceError(RuntimeError):
    """Raised when an LLM request fails or returns unusable output."""


class LLMService:
    """Thin service around the Groq chat-completions API."""

    def __init__(self, settings: Settings, client: Groq | None = None) -> None:
        self._settings = settings
        self._client = client or Groq(
            api_key=settings.groq_api_key.get_secret_value(),
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
        logger.info("Groq client initialised (model=%s).", settings.llm_model)

    def generate(self, user_prompt: str, system_prompt: str) -> str:
        """Send a chat completion request and return the message text.

        Raises:
            LLMServiceError: On connection, rate-limit, API or empty-response errors.
        """
        s = self._settings
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            completion = self._client.chat.completions.create(
                model=s.llm_model,
                messages=messages,
                temperature=s.llm_temperature,
                max_completion_tokens=s.llm_max_completion_tokens,
                top_p=s.llm_top_p,
                reasoning_effort=s.llm_reasoning_effort,
                stream=False,
                stop=None,
            )
        except groq.APIConnectionError as exc:
            logger.error("Groq connection error: %s", exc)
            raise LLMServiceError("Could not connect to the Groq API.") from exc
        except groq.RateLimitError as exc:
            logger.warning("Groq rate limit hit: %s", exc)
            raise LLMServiceError("Groq rate limit exceeded. Please retry shortly.") from exc
        except groq.APIStatusError as exc:
            logger.error("Groq API error (status=%s): %s", exc.status_code, exc)
            raise LLMServiceError(f"Groq API returned an error (status {exc.status_code}).") from exc
        except Exception as exc:
            logger.exception("Unexpected error calling Groq")
            raise LLMServiceError(f"Unexpected LLM error: {exc}") from exc

        try:
            content = completion.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            logger.error("Malformed completion response: %r", completion)
            raise LLMServiceError("Malformed response from Groq API.") from exc

        if not content or not content.strip():
            logger.error("Groq returned an empty completion.")
            raise LLMServiceError("Groq API returned an empty response.")

        return content
