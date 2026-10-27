# Numerical holder policies and stopped GBM episodes

A Bermudan holder compares intrinsic value with continuation value on each
contractual exercise date. The writer supplies a hedge until that decision,
liquidates it on settlement and finances the remaining cash balance. Volt's
numerical holder map and episode producer provide a reproducible reference for
that coupled problem.

The underlying pricing engine uses the finite-difference methods in
[exercise pricing](exercise-pricing.md). `build_holder_grid` saves both values
and continuation arrays in a single rollback. Exact integer dates include all
exercise opportunities, hedge dates and cash-dividend dates. Continuous American
pricing remains available from `exercise_pricing::price`; a holder-policy
experiment specifies its Bermudan exercise-date discretization explicitly.

## Rust API

```rust,ignore
use volt_compute::exercise_pricing::HolderGridCache;
use volt_compute::bermudan_market::{generate, generate_from_paths};

let mut cache = HolderGridCache::new(config.max_holder_scalar_values)?;
let result = generate(&config, range, &mut cache)?;
let holder = cache.get_or_build(&config.option, &config.grid, &config.dates)?;
let decision = holder.evaluate(config.dates[1], 90.0)?;
```

`HolderDecision` returns value, continuation value, immediate payoff, per-share
delta, permission to exercise and the resulting exercise decision. Positive
intrinsic strictly above interpolated continuation exercises; a tie continues.
Expiry always settles, including a zero payoff. The continuing hedge uses a
numerical continuation delta; an exercised contract has its intrinsic delta.
At endpoint states the delta stencil is one-sided. Spots outside the solved
spatial domain and dates absent from the saved map return explicit errors.

`HolderGridCache` retains one immutable, source-and-input-keyed grid. Repeated
batches reuse it; changed contracts, calendars or numerical parameters build a
new one. Callers can retain returned `Arc<HolderGrid>` objects for concurrent
read-only queries. Those retained references belong to the caller's memory
budget. Saved scalar admission counts the mesh and both arrays at every saved
date, with an eight-million-scalar maximum. Solver scratch and scenario-array
admission are separately included in the producer's working-memory estimate.
These are logical admission estimates rather than allocator quotas.

`generate_from_paths` accepts an explicit flat `[path,date]` pre-dividend stock
cohort. It enables paired numerical and learned-holder comparisons on the same
states. Distribution provenance for supplied cohorts stays with the caller.
`generate` supplies a deterministic GBM cohort itself: indexed Box–Muller normals
make path identity independent of chunk boundaries. Pricing rate and volatility
specify the PDE under Q; physical arithmetic drift is a separate input. Constant
volatility is shared between P and Q. This initial producer uses zero continuous
dividend yield and supports explicit dated cash dividends.

## A supplied learned holder

`generate_from_paths_with_stopping(config, range, spots, &stopping, &mut cache)`
assembles the same financed episode for another chosen holder. `SuppliedStopping`
contains `settlement_steps`, a `provenance` object with `policy_id` and
`policy_sha256`, and an optional `premium_override`. Every stop must be a
contractual exercise opportunity or expiry. The producer derives each cash
settlement from the contract, pre-dividend state and signed position units.

For a learned checkpoint, evaluate its causal stopping rule on each path, pass
those steps with the checkpoint's SHA256, and retain the checkpoint alongside the
result. The caller verifies the checkpoint digest and that each decision uses
only current and past observations.
The assembler queries numerical controls through the supplied stop. Changed
future states preserve an already stopped account. Complete market arrays still
receive shape and finite-value validation.

Each chosen stop reconstructs the corresponding pre-dividend settlement mark or
post-dividend continuing mark and dividend credit. This matters when the learned
and numerical holders choose different dates. `holder_policy` records the chosen
identity; dataset/content hashes bind it, and the content hash includes all
actual settlement steps and amounts. The market-only `physical_sha256` remains
shared across holders on the same cohort.

By default `economics.premium` is the signed optimal-holder PDE reference value,
labelled `numerical_pde_position_value`. An explicit `premium_override` is the
whole position's signed initial cash and receives the label
`caller_supplied_signed_cash`. `numerical_reference_premium` preserves the PDE
reference in both cases. The returned numerical delta is the optimal-holder PDE
delta evaluated until the chosen holder settles; it serves as a model-based
writer control for this supplied rule.

## Episode and cash-flow conventions

At an exercise date, the holder observes pre-dividend stock. A path that settles
uses that stock mark, pays signed `quantity × multiplier × intrinsic` and
receives zero cash-dividend credit. A path that continues uses post-dividend
stock `max(S−D,0)` and receives `min(D,S)` per carried share. Dividends therefore
enter `EpisodeBatch.cash_income` at the interval arrival. Every dividend date
must appear among the supplied decision/hedge dates.

`EpisodeBatch.exercise` records settlement steps from zero through expiry and
signed cash amounts owed by the writer. `payoffs` contains zeros, so settlement
enters the account once. Step-zero exercise opens no hedge. After settlement
only bank financing remains. The producer's `numerical_holdings` and
`numerical_pnl` use the same timing, actual proportional rebalancing fees and
one liquidation charge. `economics` provides the per-share marks convention,
premium, cash-growth arrays and cost rates for the native learner.

Decision features are log post-dividend stock over strike and remaining years.
The logarithm uses a declared floor of `1e-12` for the absorbing zero-stock state.
Units and multipliers affect liabilities and hedge holdings once. They leave
holder decisions and the market-only physical cohort hash unchanged.

## CLI

```sh
cargo build --release -p volt-compute --bin bermudan_market
cargo run --release -p volt-compute --bin bermudan_market -- \
  --input request.json --output episodes.json
```

The request is `{ "config": BermudanMarketConfig, "range": ScenarioRange }`.
Configuration contains an `ExerciseOption`, `ExerciseGrid`, `dates`, `seed`,
`physical_drift`, `quantity`, `multiplier`, proportional `trading_cost` and
`liquidation_cost`, `max_working_bytes` and `max_holder_scalar_values`.
The range specifies `split`, `first_path` and `paths`. Optional
`pre_dividend_spots` supplies a shared explicit cohort; adding `stopping` selects
its supplied holder. A supplied stopping rule requires that explicit cohort.
Output includes the stopped
batch, numerical control, economics, pre-dividend states and content/provenance
hashes. An output file is published atomically and an existing file is preserved.
Requests are capped at two MiB; callers can stream bounded path ranges using a
shared in-process holder cache. Financial and numerical admission errors stop a
batch rather than deleting paths.

The complete writer example uses the native learner:

```sh
cargo build --release -p volt-compute --bin bermudan_market --bin native_learner
python3 examples/bermudan_learning.py --output /tmp/volt-bermudan-example
```

It generates separate training and validation ranges, trains the stock hedge,
selects a checkpoint on validation, then generates fresh assessment paths and
compares the restored policy with the financed numerical delta control. The
one-share put has monthly hedge decisions and quarterly exercise. Output retains
the requests, episodes, selected optimizer checkpoint, validation history and
fresh-path report. The default 64-path, 100-update run demonstrates the complete
application sequence. Larger research studies declare their path pools, training
budget and comparisons before evaluation.

A bounded six-fit study used 16,384 training paths, 4,096 validation paths,
3,000 updates and three seeds at each of zero and 10-basis-point stock costs.
On 16,384 fresh common paths, average writer MSE was 4.013 and 4.079, respectively,
against numerical delta's 3.935 and 3.984 and the unhedged liability's 61.443.
Every policy selection preceded fresh assessment. The companion
[application paper](latex/compute/volt_deep_hedging.pdf) describes the financial
comparison. The [learned-stopping guide](../crates/volt-compute/docs-deep-stopping.md)
develops the holder's objective, valuation bounds and 2020 interval hedging.

## Validation and relation to learned stopping

Tests compare finite-difference refinements with an independent 3,650-step CRR
put tree and QuantLib1.40 spot-dividend FD controls. The dated benchmark uses
2026-01-01 valuation, Actual/365 Fixed, exercise at days90/180/270/365, strike100,
rate5%, volatility20% and zero continuous yield. QuantLib's 1,600-by-1,600 CN
scheme with two damping steps gives 5.955818029587064 for the spot100 put,
13.020881352130301 for the spot110 call with a day90 dividend of8, and
13.519042475279601 for the spot90 put with dividends3 at day90 and2 at day270.
Comparisons use explicit discretization tolerances; the saved grid is a numerical
reference whose accuracy is checked through refinement.

Additional tests cover exact saved dates, the zero-stock exercise decision,
cache identity/admission, path chunk identity, signed units, immediate settlement,
pre-dividend exercise, continuing dividend credits, actual costs and bank
financing after settlement. Supplied-holder tests cover matching numerical stops,
changed dividend-date stops, malformed opportunities/provenance, explicit initial
capital and post-stop state independence.

Becker, Cheridito and Jentzen's
[2019 deep optimal stopping paper](https://jmlr.org/papers/v20/18-232.html)
learns stopping decisions. Their
[2020 pricing and hedging paper](https://arxiv.org/abs/1912.11060)
combines learned stopping, price bounds and dynamic hedging. The numerical map
here supplies a separate holder/reference control. A learned-holder comparison
uses the same physical cohort and evaluates the writer's account under that
holder's actual stopping times.
