#!/usr/bin/env python3
"""Inspect nearest neighbors in a trained Volt graph-embedding artifact."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find cosine-nearest portfolios in embeddings.jsonl."
    )
    parser.add_argument("--input", required=True, type=Path)
    query = parser.add_mutually_exclusive_group()
    query.add_argument("--sample-id")
    query.add_argument("--portfolio-hash")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    return args


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            embedding = row.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise ValueError(f"line {line_number}: embedding is missing")
            if not all(math.isfinite(float(value)) for value in embedding):
                raise ValueError(f"line {line_number}: embedding is not finite")
            rows.append(row)
    if len(rows) < 2:
        raise ValueError("at least two embedding rows are required")
    dimensions = {len(row["embedding"]) for row in rows}
    if len(dimensions) != 1:
        raise ValueError("embedding dimensions are inconsistent")
    return rows


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def find_query(rows: Sequence[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    if args.sample_id is not None:
        matches = [row for row in rows if row["sample_id"] == args.sample_id]
    elif args.portfolio_hash is not None:
        matches = [
            row for row in rows if row["portfolio_hash"] == args.portfolio_hash
        ]
    else:
        # The first held-out graph makes the default inspection meaningful.
        matches = [row for row in rows if row.get("split") == "test"][:1]
    if len(matches) != 1:
        raise ValueError(f"expected one query row, found {len(matches)}")
    return matches[0]


def summarize(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "portfolio_hash": row["portfolio_hash"],
        "terminal_payoff_hash": row.get("terminal_payoff_hash"),
        "compiled_cog_hash": row.get("compiled_cog_hash"),
        "source_hash": row.get("source_hash"),
        "split": row.get("split"),
        "target": row.get("target"),
        "prediction": row.get("prediction"),
    }


def query_neighbors(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_rows(args.input)
    query = find_query(rows, args)
    candidates = [
        (
            cosine_similarity(query["embedding"], row["embedding"]),
            row,
        )
        for row in rows
        if row["sample_id"] != query["sample_id"]
    ]
    candidates.sort(key=lambda item: (-item[0], item[1]["sample_id"]))
    neighbors = []
    for similarity, row in candidates[: args.top_k]:
        summary = summarize(row)
        summary["cosine_similarity"] = similarity
        neighbors.append(summary)
    return {
        "query": summarize(query),
        "neighbors": neighbors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = query_neighbors(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
