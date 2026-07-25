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
- versioned graph samples for optimization and learning systems; and
- generated and historical market observations behind one trait.

Volt is a financial-logic library. It does not contain brokerage connectivity,
live data vendors, account policy, or order-submission authority.

For the fuller story, read:

- [How Volt fits together](https://github.com/ingvor-restrat/volt/blob/main/docs/architecture.md)
- [Programs and naming](https://github.com/ingvor-restrat/volt/blob/main/docs/programs.md)
- [`risk` live walkthrough](https://github.com/ingvor-restrat/volt/blob/main/docs/markup/README.md)
- [Reproducible results](https://github.com/ingvor-restrat/volt/blob/main/docs/results.md)
- [Long-form LaTeX papers](https://github.com/ingvor-restrat/volt/blob/main/docs/latex/README.md)

## Quick start

```bash
cargo test
cargo run --example risk
cargo run --example risk_server
cargo run --example zap
cargo run --example graph -- --output /tmp/volt-graphs.jsonl
```

`risk` is the compact, `top`-like terminal monitor. `risk_server` serves the
same Rust simulation as an aligned browser dashboard at
[http://127.0.0.1:8081](http://127.0.0.1:8081). Both run a canonical four-leg
*iron condor* through a correlated spot/volatility path and display live marks,
Greeks, expiration P&L, and a spot/volatility shock grid. The pricer is an
educational adapter; Volt remains the authority for exact contracts,
canonicalization, and payoff algebra.

[![Volt simulation-driven browser dashboard](docs/markup/risk-web.gif)](docs/markup/README.md)

`zap` shows how exact portfolio canonicalization and terminal-payoff projection
reduce a small option search space. `graph` emits deterministic JSON Lines
fixtures built from synthetic marks and scenario weights. Short names are
intentional: command names are part of the project vocabulary. The name `cog`
is reserved for a real compiler rather than attached to a formatter or demo.

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

This repository contains the reusable cog/options engine and its derivative
examples. Application-specific data archives, cohort builders, training loops,
services, and execution adapters remain outside Volt.

Licensed under Apache-2.0.
