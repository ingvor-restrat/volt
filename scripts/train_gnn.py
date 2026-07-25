#!/usr/bin/env python3
"""Train a small relation-aware GNN on Volt portfolio graph samples.

The exact contract, portfolio, payoff, and target semantics remain Rust-owned.
This program consumes the derived ``volt.portfolio_graph.v1`` representation,
learns a graph embedding under a supervised objective, and writes predictions
that retain the exact portfolio and terminal-payoff hashes.

PyTorch is the only third-party dependency.  PyTorch Geometric is deliberately
not required so the complete message-passing implementation remains visible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import torch
    from torch import Tensor, nn
except ImportError as error:  # pragma: no cover - exercised by the CLI
    raise SystemExit(
        "PyTorch is required. Install it with: "
        "python -m pip install -r requirements-ml.txt"
    ) from error


SCHEMA_VERSION = "volt.portfolio_graph.v1"
FEATURE_SCHEMA_VERSION = "volt.supervised_gnn_features.v1"
EXPERIMENT_SCHEMA_VERSION = "volt.supervised_gnn_experiment.v1"
SECONDS_PER_YEAR = 365.25 * 24.0 * 60.0 * 60.0
NANOSECONDS_PER_SECOND = 1_000_000_000.0

FEATURE_NAMES = (
    "is_portfolio",
    "is_underlying",
    "is_expiry",
    "is_option_position",
    "portfolio_leg_count",
    "portfolio_gross_quantity",
    "reference_spot_100d",
    "time_to_expiry_years",
    "right_call_put",
    "settlement_physical_cash",
    "exercise_european",
    "exercise_bermudan",
    "exercise_american",
    "strike_moneyness_offset",
    "mark_to_spot",
    "signed_quantity",
    "contract_multiplier_100",
    "exercise_opportunity_count",
)

FORWARD_RELATIONS = ("position_in", "written_on", "expires_at")
RELATION_NAMES = FORWARD_RELATIONS + tuple(
    f"reverse_{relation}" for relation in FORWARD_RELATIONS
)
RELATION_INDEX = {name: index for index, name in enumerate(RELATION_NAMES)}


@dataclass(frozen=True)
class EncodedGraph:
    """One validated JSON graph converted to numeric tensors."""

    sample_id: str
    portfolio_hash: str
    terminal_payoff_hash: str | None
    objective_name: str
    target: float
    node_features: list[list[float]]
    edges: list[tuple[int, int, int]]
    portfolio_node: int
    option_nodes: list[int]

    @property
    def group_key(self) -> str:
        # Equivalent terminal payoffs must never straddle train and test.
        return self.terminal_payoff_hash or f"exact:{self.portfolio_hash}"


@dataclass(frozen=True)
class SplitCounts:
    train: int
    validation: int
    test: int
    groups_train: int
    groups_validation: int
    groups_test: int


@dataclass(frozen=True)
class RegressionMetrics:
    rmse: float
    mae: float
    r_squared: float


@dataclass
class GraphBatch:
    """A disjoint union of graphs suitable for full-batch message passing."""

    x: Tensor
    edge_index: Tensor
    edge_type: Tensor
    graph_index: Tensor
    portfolio_nodes: Tensor
    option_mask: Tensor
    y: Tensor
    sample_ids: list[str]

    @property
    def graph_count(self) -> int:
        return len(self.sample_ids)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a supervised GNN on Volt graph JSONL."
    )
    parser.add_argument("--input", required=True, type=Path, help="graph JSONL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/supervised_gnn"),
        help="checkpoint, metrics, and embedding output directory",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=28695)
    parser.add_argument(
        "--quiet", action="store_true", help="only print the final JSON summary"
    )
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.patience <= 0:
        parser.error("--epochs and --patience must be positive")
    if args.hidden_dim <= 0 or args.layers <= 0:
        parser.error("--hidden-dim and --layers must be positive")
    return args


def _enum_value(value: Any) -> str:
    """Accept current strings and a future one-key externally tagged enum."""

    if isinstance(value, str):
        return value
    if isinstance(value, dict) and len(value) == 1:
        return next(iter(value))
    raise ValueError(f"expected enum string, received {value!r}")


def encode_sample(document: dict[str, Any], line_number: int) -> EncodedGraph:
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"line {line_number}: expected schema {SCHEMA_VERSION}, "
            f"received {document.get('schema_version')!r}"
        )
    target = document.get("target")
    if not isinstance(target, dict):
        raise ValueError(f"line {line_number}: supervised sample has no target")
    objective = float(target["objective"])
    if not math.isfinite(objective):
        raise ValueError(f"line {line_number}: objective is not finite")

    nodes = document.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError(f"line {line_number}: graph has no nodes")
    node_by_id: dict[str, int] = {}
    reference_spot = None
    time_to_expiry_years = None
    for index, node in enumerate(nodes):
        node_id = node.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError(f"line {line_number}: node {index} has no node_id")
        if node_id in node_by_id:
            raise ValueError(f"line {line_number}: duplicate node_id {node_id}")
        node_by_id[node_id] = index
        if node.get("node_type") == "underlying":
            reference_spot = float(node["reference_spot_micros"])
        elif node.get("node_type") == "expiry":
            time_to_expiry_years = (
                float(node["time_to_expiry_ns"])
                / NANOSECONDS_PER_SECOND
                / SECONDS_PER_YEAR
            )
    if reference_spot is None or reference_spot <= 0:
        raise ValueError(f"line {line_number}: positive underlying spot missing")
    if time_to_expiry_years is None or time_to_expiry_years < 0:
        raise ValueError(f"line {line_number}: valid expiry node missing")

    features: list[list[float]] = []
    portfolio_nodes: list[int] = []
    option_nodes: list[int] = []
    for index, node in enumerate(nodes):
        node_type = node.get("node_type")
        row = [0.0] * len(FEATURE_NAMES)
        if node_type == "portfolio":
            row[0] = 1.0
            row[4] = float(node["leg_count"]) / 6.0
            row[5] = float(node["gross_quantity"]) / 6.0
            portfolio_nodes.append(index)
        elif node_type == "underlying":
            row[1] = 1.0
            row[6] = reference_spot / 100_000_000.0
        elif node_type == "expiry":
            row[2] = 1.0
            row[7] = time_to_expiry_years
        elif node_type == "option_position":
            row[3] = 1.0
            right = _enum_value(node["right"])
            settlement = _enum_value(node["settlement"])
            exercise_style = _enum_value(node["exercise_style"])
            if right not in {"call", "put"}:
                raise ValueError(f"line {line_number}: unknown option right {right}")
            if settlement not in {"physical", "cash"}:
                raise ValueError(
                    f"line {line_number}: unknown settlement {settlement}"
                )
            if exercise_style not in {"european", "bermudan", "american"}:
                raise ValueError(
                    f"line {line_number}: unknown exercise style {exercise_style}"
                )
            row[8] = 1.0 if right == "call" else -1.0
            row[9] = 1.0 if settlement == "physical" else -1.0
            row[10] = 1.0 if exercise_style == "european" else 0.0
            row[11] = 1.0 if exercise_style == "bermudan" else 0.0
            row[12] = 1.0 if exercise_style == "american" else 0.0
            row[13] = float(node["strike_micros"]) / reference_spot - 1.0
            row[14] = float(node["mark_micros"]) / reference_spot
            row[15] = float(node["quantity"])
            row[16] = float(node["multiplier"]) / 100.0
            row[17] = float(node["exercise_opportunity_count"]) / 10.0
            option_nodes.append(index)
        else:
            raise ValueError(f"line {line_number}: unknown node_type {node_type!r}")
        features.append(row)
    if len(portfolio_nodes) != 1:
        raise ValueError(
            f"line {line_number}: expected exactly one portfolio root node"
        )

    encoded_edges: list[tuple[int, int, int]] = []
    for edge in document.get("edges", []):
        source_id = edge.get("source")
        target_id = edge.get("target")
        relation = _enum_value(edge.get("edge_type"))
        if source_id not in node_by_id or target_id not in node_by_id:
            raise ValueError(f"line {line_number}: edge references an unknown node")
        if relation not in FORWARD_RELATIONS:
            raise ValueError(f"line {line_number}: unknown edge type {relation!r}")
        source = node_by_id[source_id]
        destination = node_by_id[target_id]
        encoded_edges.append((source, destination, RELATION_INDEX[relation]))
        encoded_edges.append(
            (
                destination,
                source,
                RELATION_INDEX[f"reverse_{relation}"],
            )
        )

    return EncodedGraph(
        sample_id=str(document["sample_id"]),
        portfolio_hash=str(document["portfolio_hash"]),
        terminal_payoff_hash=document.get("terminal_payoff_hash"),
        objective_name=str(target["objective_name"]),
        target=objective,
        node_features=features,
        edges=encoded_edges,
        portfolio_node=portfolio_nodes[0],
        option_nodes=option_nodes,
    )


def load_graphs(path: Path) -> list[EncodedGraph]:
    samples: list[EncodedGraph] = []
    sample_ids: set[str] = set()
    objective_names: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                document = json.loads(line)
                sample = encode_sample(document, line_number)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"could not load {path}: {error}") from error
            if sample.sample_id in sample_ids:
                raise ValueError(f"duplicate sample_id {sample.sample_id}")
            sample_ids.add(sample.sample_id)
            objective_names.add(sample.objective_name)
            samples.append(sample)
    if len(samples) < 12:
        raise ValueError("at least 12 graph samples are required")
    if len(objective_names) != 1:
        raise ValueError(
            "one experiment may contain exactly one objective; received "
            + ", ".join(sorted(objective_names))
        )
    return samples


def grouped_split(
    samples: Sequence[EncodedGraph], seed: int
) -> dict[str, list[EncodedGraph]]:
    groups: dict[str, list[EncodedGraph]] = {}
    for sample in samples:
        groups.setdefault(sample.group_key, []).append(sample)
    if len(groups) < 3:
        raise ValueError("at least three terminal-payoff groups are required")

    def order_key(group_key: str) -> str:
        payload = f"{seed}:{group_key}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    ordered_groups = sorted(groups, key=order_key)
    train_end = max(1, round(len(ordered_groups) * 0.70))
    validation_end = max(train_end + 1, round(len(ordered_groups) * 0.85))
    validation_end = min(validation_end, len(ordered_groups) - 1)
    group_splits = {
        "train": set(ordered_groups[:train_end]),
        "validation": set(ordered_groups[train_end:validation_end]),
        "test": set(ordered_groups[validation_end:]),
    }
    return {
        split: [
            sample for sample in samples if sample.group_key in selected_groups
        ]
        for split, selected_groups in group_splits.items()
    }


def make_batch(samples: Sequence[EncodedGraph]) -> GraphBatch:
    x_rows: list[list[float]] = []
    edge_sources: list[int] = []
    edge_destinations: list[int] = []
    edge_types: list[int] = []
    graph_indices: list[int] = []
    portfolio_nodes: list[int] = []
    option_mask: list[bool] = []
    targets: list[float] = []
    node_offset = 0
    for graph_index, sample in enumerate(samples):
        x_rows.extend(sample.node_features)
        graph_indices.extend([graph_index] * len(sample.node_features))
        option_set = set(sample.option_nodes)
        option_mask.extend(
            local_index in option_set
            for local_index in range(len(sample.node_features))
        )
        portfolio_nodes.append(node_offset + sample.portfolio_node)
        for source, destination, relation in sample.edges:
            edge_sources.append(node_offset + source)
            edge_destinations.append(node_offset + destination)
            edge_types.append(relation)
        targets.append(sample.target)
        node_offset += len(sample.node_features)
    if edge_sources:
        edge_index = torch.tensor(
            [edge_sources, edge_destinations], dtype=torch.long
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return GraphBatch(
        x=torch.tensor(x_rows, dtype=torch.float32),
        edge_index=edge_index,
        edge_type=torch.tensor(edge_types, dtype=torch.long),
        graph_index=torch.tensor(graph_indices, dtype=torch.long),
        portfolio_nodes=torch.tensor(portfolio_nodes, dtype=torch.long),
        option_mask=torch.tensor(option_mask, dtype=torch.bool),
        y=torch.tensor(targets, dtype=torch.float32),
        sample_ids=[sample.sample_id for sample in samples],
    )


def segment_mean(
    values: Tensor, graph_index: Tensor, graph_count: int, mask: Tensor | None = None
) -> Tensor:
    if mask is not None:
        values = values[mask]
        graph_index = graph_index[mask]
    output = values.new_zeros((graph_count, values.shape[1]))
    counts = values.new_zeros((graph_count, 1))
    if values.shape[0]:
        output.index_add_(0, graph_index, values)
        counts.index_add_(
            0,
            graph_index,
            values.new_ones((values.shape[0], 1)),
        )
    return output / counts.clamp_min(1.0)


def graph_readout(node_states: Tensor, batch: GraphBatch) -> Tensor:
    root = node_states[batch.portfolio_nodes]
    option_mean = segment_mean(
        node_states,
        batch.graph_index,
        batch.graph_count,
        batch.option_mask,
    )
    all_mean = segment_mean(
        node_states, batch.graph_index, batch.graph_count
    )
    return torch.cat((root, option_mean, all_mean), dim=1)


class RelationMessageLayer(nn.Module):
    """One normalized message-passing step with relation-specific weights."""

    def __init__(self, hidden_dim: int, relation_count: int, dropout: float = 0.05):
        super().__init__()
        self.self_projection = nn.Linear(hidden_dim, hidden_dim)
        self.relation_projections = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(relation_count)
        )
        self.normalization = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        aggregate = torch.zeros_like(x)
        degree = x.new_zeros((x.shape[0], 1))
        if edge_index.shape[1]:
            source, destination = edge_index
            for relation, projection in enumerate(self.relation_projections):
                selected = edge_type == relation
                if not bool(selected.any()):
                    continue
                destinations = destination[selected]
                messages = projection(x[source[selected]])
                aggregate.index_add_(0, destinations, messages)
                degree.index_add_(
                    0,
                    destinations,
                    x.new_ones((destinations.shape[0], 1)),
                )
        update = self.self_projection(x) + aggregate / degree.clamp_min(1.0)
        return self.normalization(x + self.dropout(torch.relu(update)))


class PortfolioGNN(nn.Module):
    """Relation-aware encoder plus a scalar supervised regression head."""

    def __init__(
        self, input_dim: int, hidden_dim: int, layer_count: int, relation_count: int
    ):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.message_layers = nn.ModuleList(
            RelationMessageLayer(hidden_dim, relation_count)
            for _ in range(layer_count)
        )
        self.embedding_projection = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, batch: GraphBatch) -> tuple[Tensor, Tensor]:
        node_states = self.input_projection(batch.x)
        for layer in self.message_layers:
            node_states = layer(node_states, batch.edge_index, batch.edge_type)
        embedding = self.embedding_projection(graph_readout(node_states, batch))
        prediction = self.regression_head(embedding).squeeze(1)
        return prediction, embedding


class PooledLinearModel(nn.Module):
    """Linear baseline over the same node features without message passing."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.regression = nn.Linear(input_dim * 3, 1)

    def forward(self, batch: GraphBatch) -> tuple[Tensor, Tensor]:
        pooled = graph_readout(batch.x, batch)
        return self.regression(pooled).squeeze(1), pooled


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def target_statistics(targets: Tensor) -> tuple[float, float]:
    mean = float(targets.mean())
    standard_deviation = float(targets.std(unbiased=False))
    if standard_deviation <= 0:
        raise ValueError("training targets have zero variance")
    return mean, standard_deviation


def train_model(
    model: nn.Module,
    train: GraphBatch,
    validation: GraphBatch,
    target_mean: float,
    target_std: float,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    quiet: bool,
) -> tuple[nn.Module, int, float]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    normalized_train = (train.y - target_mean) / target_std
    normalized_validation = (validation.y - target_mean) / target_std
    best_state: dict[str, Tensor] | None = None
    best_validation = math.inf
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        prediction, _ = model(train)
        loss = nn.functional.mse_loss(prediction, normalized_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_prediction, _ = model(validation)
            validation_loss = float(
                nn.functional.mse_loss(
                    validation_prediction, normalized_validation
                )
            )
        if validation_loss < best_validation - 1e-8:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if not quiet and (epoch == 1 or epoch % 50 == 0):
            print(
                f"epoch={epoch:4d} train_mse={float(loss.detach()):.6f} "
                f"validation_mse={validation_loss:.6f}"
            )
        if stale_epochs >= patience:
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, best_epoch, best_validation


def predict(
    model: nn.Module,
    batch: GraphBatch,
    target_mean: float,
    target_std: float,
) -> tuple[Tensor, Tensor]:
    model.eval()
    with torch.no_grad():
        normalized, embeddings = model(batch)
    return normalized * target_std + target_mean, embeddings


def regression_metrics(target: Tensor, prediction: Tensor) -> RegressionMetrics:
    residual = prediction - target
    mse = float(torch.mean(residual.square()))
    mae = float(torch.mean(residual.abs()))
    total = float(torch.sum((target - target.mean()).square()))
    unexplained = float(torch.sum(residual.square()))
    r_squared = 1.0 - unexplained / total if total > 0 else math.nan
    return RegressionMetrics(rmse=math.sqrt(mse), mae=mae, r_squared=r_squared)


def dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, document: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")


def split_counts(splits: dict[str, list[EncodedGraph]]) -> SplitCounts:
    return SplitCounts(
        train=len(splits["train"]),
        validation=len(splits["validation"]),
        test=len(splits["test"]),
        groups_train=len({sample.group_key for sample in splits["train"]}),
        groups_validation=len(
            {sample.group_key for sample in splits["validation"]}
        ),
        groups_test=len({sample.group_key for sample in splits["test"]}),
    )


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    samples = load_graphs(args.input)
    splits = grouped_split(samples, args.seed)
    batches = {name: make_batch(split) for name, split in splits.items()}
    counts = split_counts(splits)
    target_mean, target_std = target_statistics(batches["train"].y)

    seed_everything(args.seed)
    gnn = PortfolioGNN(
        input_dim=len(FEATURE_NAMES),
        hidden_dim=args.hidden_dim,
        layer_count=args.layers,
        relation_count=len(RELATION_NAMES),
    )
    gnn, gnn_epoch, gnn_validation = train_model(
        gnn,
        batches["train"],
        batches["validation"],
        target_mean,
        target_std,
        args.epochs,
        args.patience,
        args.learning_rate,
        args.weight_decay,
        args.quiet,
    )

    seed_everything(args.seed + 1)
    linear = PooledLinearModel(len(FEATURE_NAMES))
    linear, linear_epoch, linear_validation = train_model(
        linear,
        batches["train"],
        batches["validation"],
        target_mean,
        target_std,
        args.epochs,
        args.patience,
        args.learning_rate,
        args.weight_decay,
        True,
    )

    mean_prediction = torch.full_like(batches["test"].y, target_mean)
    gnn_prediction, test_embeddings = predict(
        gnn, batches["test"], target_mean, target_std
    )
    linear_prediction, _ = predict(
        linear, batches["test"], target_mean, target_std
    )
    metrics = {
        "mean_baseline": asdict(
            regression_metrics(batches["test"].y, mean_prediction)
        ),
        "pooled_linear_baseline": asdict(
            regression_metrics(batches["test"].y, linear_prediction)
        ),
        "relation_gnn": asdict(
            regression_metrics(batches["test"].y, gnn_prediction)
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "input": str(args.input.resolve()),
        "dataset_sha256": dataset_sha256(args.input),
        "objective_name": samples[0].objective_name,
        "split_policy": (
            "70/15/15 deterministic split by terminal_payoff_hash; "
            "exact portfolio hash for the empty-payoff fallback"
        ),
        "counts": asdict(counts),
        "model": {
            "kind": "relation_aware_message_passing_gnn",
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "relation_names": RELATION_NAMES,
            "feature_names": FEATURE_NAMES,
            "best_epoch": gnn_epoch,
            "best_validation_normalized_mse": gnn_validation,
        },
        "pooled_linear_baseline": {
            "best_epoch": linear_epoch,
            "best_validation_normalized_mse": linear_validation,
        },
        "target_normalization": {
            "training_mean": target_mean,
            "training_standard_deviation": target_std,
        },
        "metrics_test": metrics,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "semantic_boundary": (
            "The model scores derived graphs only. Volt's Rust types, "
            "canonicalization, lifecycle runtime, and payoff evaluator remain "
            "authoritative."
        ),
    }
    write_json(args.output_dir / "metrics.json", experiment)

    torch.save(
        {
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "model_state_dict": gnn.state_dict(),
            "model": experiment["model"],
            "target_normalization": experiment["target_normalization"],
            "dataset_sha256": experiment["dataset_sha256"],
        },
        args.output_dir / "model.pt",
    )

    with (args.output_dir / "embeddings.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for split_name in ("train", "validation", "test"):
            prediction, embeddings = predict(
                gnn, batches[split_name], target_mean, target_std
            )
            for sample, predicted, embedding in zip(
                splits[split_name],
                prediction.tolist(),
                embeddings.tolist(),
                strict=True,
            ):
                row = {
                    "schema_version": FEATURE_SCHEMA_VERSION,
                    "sample_id": sample.sample_id,
                    "portfolio_hash": sample.portfolio_hash,
                    "terminal_payoff_hash": sample.terminal_payoff_hash,
                    "split": split_name,
                    "objective_name": sample.objective_name,
                    "target": sample.target,
                    "prediction": predicted,
                    "embedding": embedding,
                }
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")

    summary = {
        "samples": len(samples),
        "terminal_payoff_groups": len({sample.group_key for sample in samples}),
        "test": metrics,
        "artifacts": str(args.output_dir.resolve()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return experiment


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_experiment(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
