"""CLI entry point for the MCP stdio server.

Usage:
    python -m quantedge.mcp_server
    quantedge-mcp              # via pyproject.toml [project.scripts]
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    """Launch the MCP stdio server."""
    from quantedge.mcp_server.server import main as _serve

    _serve()


if __name__ == "__main__":
    src = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    src = os.path.abspath(src)
    if src not in sys.path:
        sys.path.insert(0, src)

    main()
