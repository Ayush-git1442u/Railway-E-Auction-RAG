"""CLI entry point for the Railway e-Auction RAG assistant."""
from __future__ import annotations

import sys

from pydantic import ValidationError

from config.settings import get_settings
from src.pipeline import RAGPipeline, RAGPipelineError
from utils.logger import get_logger, setup_logging

EXIT_COMMANDS = {"exit", "quit"}


def main() -> int:
    try:
        settings = get_settings()
    except ValidationError as exc:
        print(f"Configuration error (check your .env):\n{exc}", file=sys.stderr)
        return 2

    setup_logging(settings.log_level)
    logger = get_logger("app")

    try:
        pipeline = RAGPipeline.from_settings(settings)
    except RAGPipelineError as exc:
        logger.error("%s", exc)
        return 1

    while True:
        try:
            user_query = input("\nEnter your query (or 'exit' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_query.lower() in EXIT_COMMANDS:
            break
        if not user_query:
            continue

        try:
            result = pipeline.run_query(user_query)
        except RAGPipelineError as exc:
            logger.error("Query failed: %s", exc)
            print(f"\nSorry, something went wrong: {exc}")
            continue

        print("\nRESULT:\n", result.answer)
        print("\nSOURCE DOCUMENTS:\n", result.sources)

    return 0


if __name__ == "__main__":
    sys.exit(main())
