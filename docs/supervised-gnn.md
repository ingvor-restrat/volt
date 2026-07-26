# Supervised Graph Learning

Volt now contains a runnable reference experiment that learns a portfolio graph
representation while keeping financial semantics in the Rust library.

This is the first learning vertical slice over the implemented structured COG
compiler. It is not a claim that the broader arbitrary-state/emission language
or a reinforcement-learning policy is complete.

## What runs end to end

```text
typed listed contracts
        |
        v
CogCompiler                             Rust: source → immutable artifact
        |
        v
CandidateGrammar::compile_action         Rust: exact action resolution
        |
        v
CanonicalPortfolio                    Rust: exact meaning
        |
        +--> terminal payoff + scenario evaluator --> scalar target
        |
        v
PortfolioGraphSample                  Rust: versioned graph projection
        |
        v
relation-aware message-passing GNN    Python/PyTorch: learned representation
        |
        +--> objective prediction
        |
        +--> learned embedding tied back to exact/payoff hashes
```

The graph has portfolio, underlying, expiry, and option-position nodes. Typed
edges describe position membership, underlying identity, and expiry. The
trainer adds explicit reverse relations for bidirectional message flow, uses a
different learned transformation for each relationship, pools the resulting
node states, and learns a graph-level embedding under the synthetic
risk-adjusted terminal-P&L target.

The fixture target proves the machinery, not an investment result. It uses
synthetic marks and scenario probabilities. A production experiment should
replace only the target producer and market context, not the graph/runtime
boundary.

## Run it

Use Python 3.11 or newer:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-ml.txt
PYTHON=.venv/bin/python scripts/run_supervised_demo.sh
```

Or run each boundary explicitly:

```bash
cargo run --example graph -- \
  --output artifacts/supervised_gnn/graphs.jsonl

.venv/bin/python scripts/train_gnn.py \
  --input artifacts/supervised_gnn/graphs.jsonl \
  --output-dir artifacts/supervised_gnn
```

The generated directory contains:

- `graphs.jsonl`: deterministic Rust-authored graph samples and labels;
- `model.pt`: learned model parameters plus feature and target metadata;
- `metrics.json`: data hash, split policy, model settings, and held-out metrics;
- `embeddings.jsonl`: prediction and embedding for every sample, retaining
  `sample_id`, `portfolio_hash`, and `terminal_payoff_hash`.

The [graphical experiment overview](markup/README.md#supervised-graph-learning)
shows the compiler path, one exact graph, the learned embedding space, held-out
predictions, and baseline errors. Its interactive version is
[`docs/markup/gnn-learning.html`](markup/gnn-learning.html).

Inspect the nearest portfolios in the learned space (by default, around the
first held-out sample):

```bash
.venv/bin/python scripts/query_embeddings.py \
  --input artifacts/supervised_gnn/embeddings.jsonl
```

Run the Python tests with:

```bash
PYTHONPATH=scripts .venv/bin/python -m unittest scripts/test_train_gnn.py
```

The regular Rust suite remains dependency-free:

```bash
cargo test --all-targets
```

## Evaluation discipline

The split is grouped by `terminal_payoff_hash`, with the exact portfolio hash
used only for the empty-payoff fallback. Therefore, portfolios known to have
the same terminal payoff cannot appear on opposite sides of the train/test
boundary.

The report compares:

1. the training-target mean;
2. a linear model over pooled raw graph features; and
3. the relation-aware message-passing GNN.

The comparison distinguishes learning a nonlinear graph representation from
merely recovering the target level or fitting a linear bag of legs. It does
not prove market efficacy; that requires point-in-time real data, a declared
label horizon, transaction costs, and strictly time-ordered validation.

## Verified fixture run

The reference command was run on 2026-07-25 with seed `28695`, PyTorch 2.13.0,
and dataset SHA-256
`8d3ecb063ea2a6d3f8f9b64967fffd18c8f552feadd4029cbc5bd8db2e175043`.
The 729 graphs formed 713 terminal-payoff groups. Grouped sample counts were
509 train, 109 validation, and 111 test. Every row was produced as a
`volt.compiled_portfolio_graph.v1` sample with source and artifact provenance.

| Held-out model | RMSE (dollars) | MAE (dollars) | R-squared |
|---|---:|---:|---:|
| Training mean | 316.60 | 252.79 | -0.005 |
| Pooled linear features | 217.60 | 168.66 | 0.525 |
| Relation-aware GNN | 28.25 | 18.96 | 0.992 |

These are reproducibility measurements for the synthetic evaluator. They
confirm that the implementation trains, exports an embedding, and generalizes
within this fixture universe; they are not live-market performance.

## Compiler boundary

There are now three explicit compilation boundaries in this experiment:

- `CogCompiler` parses and validates versioned structured source, emits
  diagnostics, canonicalizes rules, and produces a content-addressed
  `CompiledCog`;
- `CandidateGrammar::compile_action` resolves the implemented typed search
  vocabulary into exact listed-contract branches used to construct each
  candidate portfolio; and
- `PortfolioGraphSample::from_portfolio` validates and compiles an exact
  canonical portfolio into the versioned graph schema consumed by the model.

Compiler-bound rows use `volt.compiled_portfolio_graph.v1` and retain
`source_hash` plus `compiled_cog_hash` around the exact portfolio graph. The
initial compiler is options-design focused; broader typed emissions and account
or protocol state remain later language extensions.

## Path from supervised learning to RL

An RL environment should reuse this encoder but operate on a state-and-action
graph rather than the static terminal portfolio alone:

1. export `PortfolioState`, point-in-time market context, and every candidate
   action admitted by `CandidateGrammar`;
2. attach a legal-action mask produced by Rust;
3. let the policy rank only those admitted actions;
4. execute the selected compiled action through `CogRuntime` or
   `ScenarioSimulator`;
5. derive reward from the resulting journaled transition, including explicit
   risk, turnover, fill, and transaction-cost terms; and
6. retain state, action, observation, reward, and next-state hashes for replay.

The model must never learn or override hard legality. RL changes the search
policy; it does not replace the compiler, runtime, or financial semantics.
