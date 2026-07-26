#!/usr/bin/env python3
"""Render the verified Volt GNN experiment as a repository-friendly SVG."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


WIDTH = 1200
HEIGHT = 760
CONTRACTS = ("SPY-C-90", "SPY-C-100", "SPY-C-110", "SPY-P-90", "SPY-P-100", "SPY-P-110")
DISPLAY_CONTRACTS = ("C90", "C100", "C110", "P90", "P100", "P110")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("artifacts/supervised_gnn/embeddings.jsonl"),
    )
    parser.add_argument(
        "--graphs",
        type=Path,
        default=Path("artifacts/supervised_gnn/graphs.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/markup/gnn-learning.svg"),
    )
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def line(x1: float, y1: float, x2: float, y2: float, css: str) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{css}"/>'


def text(x: float, y: float, value: str, css: str, anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" class="{css}" '
        f'text-anchor="{anchor}">{html.escape(value)}</text>'
    )


def circle(x: float, y: float, radius: float, css: str) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" class="{css}"/>'


def rect(x: float, y: float, width: float, height: float, css: str, radius: float = 10) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'rx="{radius:.1f}" class="{css}"/>'
    )


def scale(domain: tuple[float, float], target: tuple[float, float], value: float) -> float:
    return target[0] + (value - domain[0]) * (target[1] - target[0]) / (domain[1] - domain[0])


def padded_extent(values: Sequence[float], padding: float = 0.05) -> tuple[float, float]:
    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum or 1.0
    return minimum - span * padding, maximum + span * padding


def objective_class(value: float) -> str:
    if value < -800:
        return "q1"
    if value < -500:
        return "q2"
    if value < -200:
        return "q3"
    if value < 0:
        return "q4"
    return "q5"


def render_pipeline(parts: list[str]) -> None:
    labels = (
        "Typed actions",
        "Candidate compiler",
        "Exact portfolio graph",
        "GNN embedding",
        "Supervised score",
    )
    centers = (112, 340, 586, 824, 1064)
    widths = (144, 174, 188, 160, 164)
    for index, (label, center, width) in enumerate(zip(labels, centers, widths, strict=True)):
        parts.append(rect(center - width / 2, 48, width, 42, "pipeline"))
        parts.append(text(center, 74, label, "pipeline-label", "middle"))
        if index < len(labels) - 1:
            start = center + width / 2 + 10
            end = centers[index + 1] - widths[index + 1] / 2 - 10
            parts.append(line(start, 69, end, 69, "arrow"))
            parts.append(f'<path d="M {end - 7:.1f} 63 L {end:.1f} 69 L {end - 7:.1f} 75" class="arrow"/>')


def selected_quantities(graph: dict[str, Any]) -> list[int]:
    quantities = {
        node["contract_id"]: node["quantity"]
        for node in graph["nodes"]
        if node["node_type"] == "option_position"
    }
    return [quantities.get(contract, 0) for contract in CONTRACTS]


def render_graph(parts: list[str], quantities: Sequence[int]) -> None:
    left, top = 32, 150
    parts.append(text(left, top - 18, "1 · Compiled portfolio graph", "section"))
    parts.append(text(left, top + 2, "Exact typed nodes and relationships", "muted"))
    underlying = (100, 208)
    expiry = (315, 208)
    portfolio = (208, 292)
    option_x = (55, 116, 177, 238, 299, 360)
    for index, quantity in enumerate(quantities):
        if quantity == 0:
            continue
        x, y = option_x[index], 397
        parts.append(line(x, y - 22, portfolio[0], portfolio[1] + 34, "position-edge"))
        parts.append(line(x, y - 22, underlying[0], underlying[1] + 20, "underlying-edge"))
        parts.append(line(x, y - 22, expiry[0], expiry[1] + 20, "expiry-edge"))
    parts.append(rect(underlying[0] - 64, underlying[1] - 23, 128, 46, "context"))
    parts.append(text(underlying[0], underlying[1] - 2, "Underlying", "node", "middle"))
    parts.append(text(underlying[0], underlying[1] + 15, "SPY · $100", "muted", "middle"))
    parts.append(rect(expiry[0] - 64, expiry[1] - 23, 128, 46, "context"))
    parts.append(text(expiry[0], expiry[1] - 2, "Expiry", "node", "middle"))
    parts.append(text(expiry[0], expiry[1] + 15, "30 days", "muted", "middle"))
    parts.append(circle(portfolio[0], portfolio[1], 43, "portfolio"))
    parts.append(text(portfolio[0], portfolio[1] - 3, "Portfolio", "portfolio-label", "middle"))
    parts.append(
        text(
            portfolio[0],
            portfolio[1] + 16,
            f"{sum(quantity != 0 for quantity in quantities)} legs",
            "portfolio-label",
            "middle",
        )
    )
    for index, quantity in enumerate(quantities):
        if quantity == 0:
            continue
        x, y = option_x[index], 397
        css = "option long" if quantity > 0 else "option short"
        parts.append(circle(x, y, 25, css))
        parts.append(text(x, y - 2, DISPLAY_CONTRACTS[index], "node", "middle"))
        parts.append(text(x, y + 16, f"{quantity:+d}", "muted", "middle"))


def render_embedding(parts: list[str], embeddings: list[dict[str, Any]], selected_id: str) -> None:
    left, top, right, bottom = 446, 178, 768, 424
    parts.append(text(left, 132, "2 · Learned embedding", "section"))
    parts.append(text(left, 152, "PCA view · nearby points are represented similarly", "muted"))
    matrix = np.asarray([row["embedding"] for row in embeddings], dtype=float)
    centered = matrix - matrix.mean(axis=0)
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ right_vectors[:2].T
    x_domain = padded_extent(projected[:, 0].tolist(), 0.04)
    y_domain = padded_extent(projected[:, 1].tolist(), 0.05)
    for index in range(5):
        x = left + (right - left) * index / 4
        y = top + (bottom - top) * index / 4
        parts.append(line(x, top, x, bottom, "grid"))
        parts.append(line(left, y, right, y, "grid"))
    parts.append(line(left, bottom, right, bottom, "axis"))
    parts.append(line(left, top, left, bottom, "axis"))
    for row, point in zip(embeddings, projected, strict=True):
        x = scale(x_domain, (left, right), float(point[0]))
        y = scale(y_domain, (bottom, top), float(point[1]))
        css = f"embedding {objective_class(float(row['target']))}"
        radius = 6.5 if row["sample_id"] == selected_id else 2.6
        if row["sample_id"] == selected_id:
            css += " selected"
        parts.append(circle(x, y, radius, css))
    parts.append(text((left + right) / 2, 450, "embedding component 1", "axis-label", "middle"))
    parts.append(text(left, top - 7, "component 2", "axis-label"))
    parts.append(text(left, 477, "lower objective", "muted"))
    for index, css in enumerate(("q1", "q2", "q3", "q4", "q5")):
        parts.append(circle(left + 113 + index * 20, 472, 5, css))
    parts.append(text(left + 225, 477, "higher objective", "muted"))


def render_performance(parts: list[str], embeddings: list[dict[str, Any]], selected_id: str) -> None:
    left, top, right, bottom = 840, 178, 1162, 424
    parts.append(text(left, 132, "3 · Held-out prediction", "section"))
    parts.append(text(left, 152, "GNN R² 0.992 · RMSE $28.25", "muted"))
    test = [row for row in embeddings if row["split"] == "test"]
    values = [float(value) for row in test for value in (row["target"], row["prediction"])]
    domain = padded_extent(values, 0.05)
    x = lambda value: scale(domain, (left, right), value)
    y = lambda value: scale(domain, (bottom, top), value)
    for index in range(5):
        value = domain[0] + (domain[1] - domain[0]) * index / 4
        px, py = x(value), y(value)
        parts.append(line(px, top, px, bottom, "grid"))
        parts.append(line(left, py, right, py, "grid"))
        parts.append(text(px, bottom + 18, f"{value:.0f}", "tick", "middle"))
        parts.append(text(left - 8, py + 4, f"{value:.0f}", "tick", "end"))
    parts.append(line(left, bottom, right, bottom, "axis"))
    parts.append(line(left, top, left, bottom, "axis"))
    parts.append(line(x(domain[0]), y(domain[0]), x(domain[1]), y(domain[1]), "identity"))
    for row in test:
        radius = 6.5 if row["sample_id"] == selected_id else 3.1
        css = "prediction selected" if row["sample_id"] == selected_id else "prediction"
        parts.append(circle(x(float(row["target"])), y(float(row["prediction"])), radius, css))
    parts.append(text((left + right) / 2, 466, "exact evaluated objective ($)", "axis-label", "middle"))
    parts.append(text(left, top - 7, "prediction ($)", "axis-label"))


def render_baselines(parts: list[str]) -> None:
    parts.append(text(32, 540, "Held-out error comparison", "section"))
    parts.append(text(32, 561, "Lower RMSE is better · same synthetic objective and grouped test split", "muted"))
    rows = (
        ("Training mean", 316.60, "baseline"),
        ("Pooled linear", 217.60, "linear"),
        ("Relation GNN", 28.25, "gnn"),
    )
    maximum_width = 840
    maximum = rows[0][1]
    for index, (label, value, css) in enumerate(rows):
        y = 596 + index * 47
        parts.append(text(32, y + 20, label, "bar-label"))
        parts.append(rect(170, y, maximum_width, 28, "bar-track", 4))
        parts.append(rect(170, y, maximum_width * value / maximum, 28, f"bar {css}", 4))
        parts.append(text(1028, y + 20, f"${value:.2f}", "bar-value"))
    parts.append(text(1138, 617, "mean", "muted", "middle"))
    parts.append(text(1138, 664, "linear", "muted", "middle"))
    parts.append(text(1138, 711, "GNN", "muted", "middle"))


def render_svg(embeddings: list[dict[str, Any]], graphs: list[dict[str, Any]]) -> str:
    selected = next(row for row in embeddings if row["split"] == "test")
    graph_by_sample = {row["sample_id"]: row for row in graphs}
    quantities = selected_quantities(graph_by_sample[selected["sample_id"]])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title description">',
        "<title id=\"title\">Volt supervised graph learning</title>",
        "<desc id=\"description\">Typed candidate compilation, one exact portfolio graph, the learned embedding space, held-out predictions, and baseline errors.</desc>",
        """<style>
          :root { color-scheme: light dark; }
          svg { background: transparent; font-family: ui-sans-serif, system-ui, sans-serif; }
          .pipeline { fill: #eef5ff; stroke: #9bb8d9; stroke-width: 1; }
          .pipeline-label, .section, .node, .bar-label, .bar-value { fill: #18212b; font-size: 15px; font-weight: 500; }
          .section { font-size: 18px; }
          .muted, .axis-label, .tick { fill: #586675; font-size: 13px; font-weight: 400; }
          .tick { font-size: 11px; }
          .arrow, .axis, .grid, .position-edge, .underlying-edge, .expiry-edge, .identity { fill: none; stroke: #aeb9c4; stroke-width: 1; }
          .grid { opacity: .45; }
          .position-edge { stroke: #4d5b68; opacity: .55; }
          .underlying-edge { stroke: #3c83c6; stroke-dasharray: 3 3; opacity: .35; }
          .expiry-edge { stroke: #8a70b8; stroke-dasharray: 7 4; opacity: .35; }
          .context { fill: #edf0f3; stroke: #b5c0ca; }
          .portfolio { fill: #287cc1; stroke: #287cc1; }
          .portfolio-label { fill: #ffffff; font-size: 14px; font-weight: 500; }
          .option { stroke-width: 1.5; }
          .long { fill: #d7f0e6; stroke: #278e68; }
          .short { fill: #f8dfdf; stroke: #c85b5b; }
          .embedding { stroke: #ffffff; stroke-width: .5; opacity: .68; }
          .q1 { fill: #b55565; }
          .q2 { fill: #d08b57; }
          .q3 { fill: #a79555; }
          .q4 { fill: #579c92; }
          .q5 { fill: #397fc0; }
          .selected { stroke: #18212b; stroke-width: 3; opacity: 1; }
          .prediction { fill: #397fc0; stroke: #ffffff; stroke-width: .7; opacity: .72; }
          .identity { stroke: #278e68; stroke-width: 2; }
          .bar-track { fill: #edf0f3; }
          .bar.baseline { fill: #b9c1c9; }
          .bar.linear { fill: #7899b8; }
          .bar.gnn { fill: #287cc1; }
          @media (prefers-color-scheme: dark) {
            .pipeline { fill: #26394b; stroke: #52708a; }
            .pipeline-label, .section, .node, .bar-label, .bar-value { fill: #f2f5f7; }
            .muted, .axis-label, .tick { fill: #acb8c3; }
            .arrow, .axis, .grid { stroke: #52606c; }
            .position-edge { stroke: #c1cad2; }
            .context, .bar-track { fill: #2d353d; stroke: #596570; }
            .portfolio-label { fill: #101418; }
            .long { fill: #24463c; }
            .short { fill: #522f35; }
            .embedding, .prediction { stroke: #161a1e; }
            .selected { stroke: #f2f5f7; }
          }
        </style>""",
    ]
    render_pipeline(parts)
    render_graph(parts, quantities)
    render_embedding(parts, embeddings, selected["sample_id"])
    render_performance(parts, embeddings, selected["sample_id"])
    render_baselines(parts)
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    embeddings = load_jsonl(args.embeddings)
    graphs = load_jsonl(args.graphs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_svg(embeddings, graphs), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
