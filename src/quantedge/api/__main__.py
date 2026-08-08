"""CLI entry point for the QuantEdge API server.

Usage:
    python -m quantedge.api
    quantedge-api            # via pyproject.toml [project.scripts]
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    """Launch the FastAPI application via uvicorn."""
    import uvicorn

    from quantedge.config import get_settings

    settings = get_settings()

    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    reload = settings.app_env == "development"

    uvicorn.run(
        "quantedge.api.app:app",
        host=host,
        port=port,
        reload=reload,
        reload_dirs=["src"] if reload else None,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    # Ensure src/ is on the path when invoked directly
    src = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    src = os.path.abspath(src)
    if src not in sys.path:
        sys.path.insert(0, src)

    main()
