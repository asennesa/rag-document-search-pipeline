"""Shared command-line entry point, so every script fails the same clear way."""

import sys
from collections.abc import Callable

from pydantic import ValidationError

from rag_pipeline.errors import PipelineError


def run_cli(main: Callable[[], int]) -> int:
    """Turn expected failures into a short message and exit code 2.

    Configuration errors name the settings only. Values are left out, since a rejected
    value could be a secret.
    """
    try:
        return main()
    except ValidationError as error:
        names = sorted(
            {
                ".".join(str(part) for part in item["loc"]) or "settings"
                for item in error.errors(include_input=False, include_url=False)
            }
        )
        print(f"error: Invalid configuration for: {', '.join(names)}", file=sys.stderr)
    except PipelineError as error:
        print(f"error: {error}", file=sys.stderr)
    return 2
