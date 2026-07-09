"""CLI entry point: python -m hyppo.mcp [--transport stdio|http] ..."""

from __future__ import annotations

import argparse
import asyncio
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hyppo-mcp",
        description="hyppo Model Context Protocol server.",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "http"),
        default="stdio",
        help="MCP transport (default: stdio).",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="HTTP bind host (HTTP only).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8082,
        help="HTTP bind port (default 8082, chosen to avoid wfonto-mcp / neqsim-mcp).",
    )
    args = parser.parse_args(argv)

    from hyppo.mcp import server as server_module

    if args.transport == "stdio":
        asyncio.run(server_module.run_stdio())
    else:
        server_module.run_http(args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
