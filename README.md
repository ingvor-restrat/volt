# Volt

Typed, canonical options portfolio logic with deterministic lifecycle,
simulation, and payoff analysis.

Volt is a typed, composable representation and runtime for options
portfolios. A portfolio is simultaneously a set of listed contracts, a payoff
shape, a collection of path risks, and a state that evolves through exercise,
assignment, expiry, and trading. Volt keeps those meanings together in one
canonical model.

Volt is part of the open-source analytical architecture behind strategynet.ai.
Read the
[Volt architecture overview](https://strategynet.ai/insights/volt-open-source-options-portfolio-logic).

The publishable Cargo package is named `volt-options`; its Rust library name
remains `volt`.

The initial Rust release provides:

- typed call and put contracts with European, Bermudan, and American exercise;
- cash and physical settlement;
- exact portfolio canonicalization and stable semantic hashes;
- terminal payoff projections without conflating them with dynamic equivalence;
- deterministic exercise, assignment, expiry, and replay journals;
- path simulation with explicit fill and transaction-cost models;
- a point-in-time candidate grammar for strategy discovery;
- a versioned structured COG compiler with diagnostics and immutable artifacts;
- versioned graph samples for optimization and learning systems; and
- generated and historical market observations behind one trait.

Volt is a financial-logic library. It does not contain brokerage connectivity,
live data vendors, account policy, or order-submission authority.

For the fuller story, read:

- [How Volt fits together](https://github.com/ingvor-restrat/volt/blob/main/docs/architecture.md)
- [Programs and naming](https://github.com/ingvor-restrat/volt/blob/main/docs/programs.md)
- [COG compiler](https://github.com/ingvor-restrat/volt/blob/main/docs/cog-compiler.md)
- [Supervised graph learning](https://github.com/ingvor-restrat/volt/blob/main/docs/supervised-gnn.md)
- [`risk` live walkthrough](https://github.com/ingvor-restrat/volt/blob/main/docs/markup/README.md)
- [Reproducible results](https://github.com/ingvor-restrat/volt/blob/main/docs/results.md)
- [Long-form LaTeX papers](https://github.com/ingvor-restrat/volt/blob/main/docs/latex/README.md)

## Quick start

```bash
cargo test
cargo run --example risk
cargo run --example risk_server
cargo run --example zap
cargo run --example cog -- --output /tmp/volt-cog.json
cargo run --example graph -- --output /tmp/volt-graphs.jsonl
cargo run --bin graph_export -- --input request.json --output graphs.json
cargo run --bin scenario_batch -- --input scenarios.json --output results.json
```

`risk` is the compact, `top`-like terminal monitor. `risk_server` serves the
same Rust simulation as an aligned browser dashboard at
[http://127.0.0.1:8081](http://127.0.0.1:8081). Both run a canonical four-leg
*iron condor* through a correlated spot/volatility path and display live marks,
Greeks, expiration P&L, and a spot/volatility shock grid. The pricer is an
educational adapter; Volt remains the authority for exact contracts,
canonicalization, and payoff algebra.

[![Volt simulation-driven browser dashboard](docs/markup/risk-web.gif)](docs/markup/README.md)

The native European quote-to-Greeks application adds implied-volatility inversion,
a volatility surface, portfolio Greeks and fully repriced scenarios. Its browser
and `top`-style terminal viewer consume the same native snapshot format.
[View both recordings and run commands](demo/live-greeks.html), or follow the
[demo guide](docs/markup/README.md#european-quote-to-greeks-viewer). The examples use
synthetic quotes; the valuation scope is European exercise.

[![Native European Greeks in the browser](docs/markup/live-greeks-web.gif)](demo/live-greeks.html)

[![Native European Greeks in the terminal](docs/markup/live-greeks-top.gif)](demo/live-greeks.html)

`zap` shows how exact portfolio canonicalization and terminal-payoff projection
reduce a small option search space. `cog` compiles structured options-design
business logic into an immutable artifact, produces a point-in-time legal
action mask and state/action graph, and journals a runtime transition. `graph`
emits deterministic JSON Lines fixtures built from synthetic marks and scenario
weights.

`graph_export` is the data-independent bridge used by research applications.
It validates declared contracts and marks, canonicalizes each quantity vector,
and returns portfolio hashes, terminal-payoff hashes, and graph samples.
`scenario_batch` evaluates complete paths and scheduled actions through the
scenario simulator, returning fills, pathwise P&L, runtime journals, final
states, and result hashes.

An optional reference training harness consumes the exact graph export and
learns a relation-aware portfolio embedding:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-ml.txt
PYTHON=.venv/bin/python scripts/run_supervised_demo.sh
```

It reports held-out performance against mean and linear pooled-feature
baselines and retains the exact portfolio/payoff hashes with every embedding.
See [Supervised graph learning](docs/supervised-gnn.md) for the semantic
boundary, evaluation policy, and route to reinforcement learning.

## Optional Rust compute

The [Volt Compute companion package](crates/volt-compute/README.md) provides native
CPU, Metal, and CUDA numerical execution, memory admission, matrix operations,
European option valuation, and a validated portfolio/market adapter. It builds
with the repository's pinned Rust 1.99.1 toolchain in a shared Cargo workspace
and leaves the core package dependencies unchanged. It also provides
Black–Scholes AAD Greeks and spot gamma, GBM Monte Carlo risk, Heston Fourier risk, time-stepped markets, native Deep Hedging
training/checkpoint/resume, and exact integer hedge reconciliation. Matched
benchmarks compare learned, genuine model and no-hedge strategies on identical
held-out paths. Heston path integration is a declared approximation; paper
replication has not been established. The companion README documents commands,
assumptions and the adaptive CPU-f64 Heston baseline.

See [training on a compute host](docs/compute-training.md) for checkout, CPU/CUDA
builds, background training, resume and held-out evaluation.

Run `cargo test --release --locked --workspace` to test both crates. Plain
`cargo test` selects the core; GPU support is enabled explicitly in `volt-compute`.

## Core model

```rust
use volt::{
    CanonicalPortfolio, ExerciseSchedule, OptionContract, OptionRight, Position,
    PriceMicros, Settlement, TimestampNs,
};

let expiry = TimestampNs(1_800_000_000_000_000_000);
let call = OptionContract::new(
    "XYZ-C-100",
    "XYZ",
    "USD",
    OptionRight::Call,
    PriceMicros(100_000_000),
    100,
    ExerciseSchedule::European { expiry },
    Settlement::Physical,
)?;
let portfolio = CanonicalPortfolio::from_positions(vec![Position::new(call, 1)?])?;
let payoff = portfolio.terminal_payoff_projection()?;

# Ok::<(), volt::VoltError>(())
```

All money values use integer micro-units. Timestamps use integer nanoseconds.
Serialized structures carry explicit schema versions, and semantic hashes are
computed from canonical serialized forms.

## Extraction boundary

This repository contains the reusable options-logic engine, its derivative
examples, and a small synthetic reference training harness. Application-
specific data archives, production cohort builders, trained policies, services,
and execution adapters remain outside Volt.

Licensed under Apache-2.0.

## Numerical application papers

The [computation and hedging papers](docs/latex/compute/README.md) explain pricing,
Greeks, hedge learning, live option analytics and straddle experiments, with
[canonical LaTeX sources and PDFs](docs/latex/compute/).
