# Programs

Volt uses short names for programs and examples. They are easier to remember,
say aloud, and keep stable than implementation-shaped names.

## `risk`

```bash
cargo run --example risk
```

`risk` is a live terminal monitor for a synthetic SIMX *iron condor*. Volt
constructs and canonicalizes the exact contracts and produces the terminal
payoff projection. A teaching adapter adds Black–Scholes marks to a correlated
spot/stochastic-volatility path, then reports leg P&L, portfolio Greeks,
expiration outcomes, and a spot/volatility shock grid.

The simulation is deterministic for a given seed:

```bash
cargo run -q --example risk -- --ticks 20 --seed 28695
```

It is educational synthetic output, not a market-data source or production
pricing model. See the [recorded walkthrough](markup/README.md).

## `zap`

```bash
cargo run --example zap
```

`zap` constructs a bounded synthetic option universe and removes two kinds of
duplication:

1. ordering duplication, by canonicalizing action sequences into exact
   portfolios; and
2. payoff duplication, by projecting exact portfolios into terminal-payoff
   classes.

It writes one JSON result document to standard output. The program does not
use market data or estimate returns.

## `graph`

```bash
cargo run --example graph -- --output /tmp/volt-graphs.jsonl
```

`graph` enumerates bounded synthetic portfolios and emits one
`PortfolioGraphSample` per line. The fixtures use declared synthetic marks and
scenario probabilities so graph consumers have deterministic infrastructure
data.

Without `--output`, JSON Lines are written to standard output. Progress is
written to standard error.

## Reserved: `cog`

`cog` is reserved for a proper financial-logic compiler. The present
`CandidateGrammar::compile_action` performs one specific compilation step
from typed candidate actions to exact listed-contract branches. It should not
be marketed as the larger compiler before a language, diagnostics model,
artifact format, and execution contract are implemented.

That distinction lets the short name retain meaning.
