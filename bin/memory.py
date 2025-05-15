#!/usr/bin/env python3
"""CLI for Memory Backbone."""

import argparse
from pathlib import Path

from ds_agent.memory import Memory


def main():
    """Run the Memory CLI."""
    parser = argparse.ArgumentParser(description="Memory Backbone CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest files or directories")
    ingest_parser.add_argument("path", type=Path, help="Path to file or directory")
    ingest_parser.add_argument(
        "--episode-id", type=str, help="Episode ID to associate with documents"
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Query memory")
    query_parser.add_argument("text", type=str, help="Query text")
    query_parser.add_argument("-k", type=int, default=5, help="Number of results to return")

    # Common arguments
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=Path("memory"),
        help="Path to memory directory",
    )

    args = parser.parse_args()

    # Initialize Memory
    memory = Memory(args.memory_dir)

    if args.command == "ingest":
        doc_id = memory.ingest(args.path, episode_id=args.episode_id)
        print(f"Ingested document(s) with ID(s): {doc_id}")

    elif args.command == "query":
        chunks = memory.query(args.text, k=args.k)
        for i, chunk in enumerate(chunks, 1):
            print(f"\nResult {i}:")
            print(f"Score: {chunk.score:.4f}")
            print(f"Source: {chunk.source_uri}")
            if chunk.episode_id:
                print(f"Episode: {chunk.episode_id}")
            print(f"Content: {chunk.content}")


if __name__ == "__main__":
    main()
