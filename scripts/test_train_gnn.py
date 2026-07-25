#!/usr/bin/env python3
"""Unit tests for the framework-free Volt graph encoder and split logic."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

import train_gnn


def fixture(sample_index: int, group: int) -> dict:
    portfolio_hash = f"portfolio-{sample_index}"
    return {
        "schema_version": "volt.portfolio_graph.v1",
        "sample_id": f"sample-{sample_index}",
        "portfolio_hash": portfolio_hash,
        "terminal_payoff_hash": f"payoff-{group}",
        "nodes": [
            {
                "node_id": "portfolio",
                "node_type": "portfolio",
                "leg_count": 1,
                "gross_quantity": 1,
            },
            {
                "node_id": "underlying",
                "node_type": "underlying",
                "underlying": "XYZ",
                "currency": "USD",
                "reference_spot_micros": 100_000_000,
            },
            {
                "node_id": "expiry",
                "node_type": "expiry",
                "expiry_ns": 1_800_000_000_000_000_000,
                "time_to_expiry_ns": 30 * 86_400 * 1_000_000_000,
            },
            {
                "node_id": "position:0",
                "node_type": "option_position",
                "contract_id": "XYZ-C-100",
                "right": "call",
                "settlement": "physical",
                "exercise_style": "european",
                "strike_micros": 100_000_000,
                "mark_micros": 5_000_000,
                "quantity": 1,
                "multiplier": 100,
                "first_exercise_ns": 1_800_000_000_000_000_000,
                "exercise_opportunity_count": 1,
            },
        ],
        "edges": [
            {
                "source": "position:0",
                "target": "portfolio",
                "edge_type": "position_in",
            },
            {
                "source": "position:0",
                "target": "underlying",
                "edge_type": "written_on",
            },
            {
                "source": "position:0",
                "target": "expiry",
                "edge_type": "expires_at",
            },
        ],
        "target": {
            "objective_name": "fixture",
            "objective": float(sample_index),
            "scenario_spots_micros": [90_000_000, 110_000_000],
            "scenario_pnl_micros": [-1, 1],
            "note": "test",
        },
    }


class GraphEncodingTest(unittest.TestCase):
    def test_typed_nodes_and_reverse_edges_are_encoded(self) -> None:
        encoded = train_gnn.encode_sample(fixture(0, 0), 1)
        self.assertEqual(len(encoded.node_features), 4)
        self.assertEqual(len(encoded.node_features[0]), len(train_gnn.FEATURE_NAMES))
        self.assertEqual(len(encoded.edges), 6)
        self.assertEqual(encoded.portfolio_node, 0)
        self.assertEqual(encoded.option_nodes, [3])

    def test_equivalent_payoffs_never_cross_splits(self) -> None:
        samples = [
            train_gnn.encode_sample(fixture(index, index // 2), index + 1)
            for index in range(20)
        ]
        splits = train_gnn.grouped_split(samples, 17)
        group_sets = {
            name: {sample.group_key for sample in split}
            for name, split in splits.items()
        }
        self.assertFalse(group_sets["train"] & group_sets["validation"])
        self.assertFalse(group_sets["train"] & group_sets["test"])
        self.assertFalse(group_sets["validation"] & group_sets["test"])

    def test_message_passing_model_has_graph_level_output(self) -> None:
        samples = [
            train_gnn.encode_sample(fixture(index, index), index + 1)
            for index in range(3)
        ]
        batch = train_gnn.make_batch(samples)
        model = train_gnn.PortfolioGNN(
            input_dim=len(train_gnn.FEATURE_NAMES),
            hidden_dim=16,
            layer_count=2,
            relation_count=len(train_gnn.RELATION_NAMES),
        )
        prediction, embedding = model(batch)
        self.assertEqual(tuple(prediction.shape), (3,))
        self.assertEqual(tuple(embedding.shape), (3, 16))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_jsonl_loader_rejects_duplicate_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.jsonl"
            document = fixture(0, 0)
            path.write_text(
                json.dumps(document) + "\n" + json.dumps(document) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
                train_gnn.load_graphs(path)


if __name__ == "__main__":
    unittest.main()
