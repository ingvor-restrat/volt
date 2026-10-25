# Volt Compute

Native Rust numerical execution for Volt portfolios, using Burn and CubeCL on
CPU, Apple Metal, and NVIDIA CUDA. The companion package contains resident matrix
operations, logical memory budgeting, European option valuation, persistent
scenario execution, and an adapter from exact Volt portfolios and market inputs.

Both crates use the repository's pinned Rust 1.99.1 toolchain and shared Cargo
workspace and lockfile. The core `volt-options` package keeps its existing
dependencies; compute dependencies belong to this companion crate. It is currently
built from source; crate publication is disabled while the API develops.
First-order reverse-mode Black–Scholes Greeks and second-order forward spot gamma
are supported. GBM Monte Carlo risk, Heston Fourier risk, time-stepped markets,
native Deep Hedging training, and integer hedge reconciliation are also supported.
Paper replication has not been established; the supplied profiles declare their
model, simulation, numerical and training assumptions.

## Build and validate

From this directory, CPU execution needs no GPU SDK:

```sh
cargo test --release --locked
cargo clippy --release --locked --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --release --locked --no-deps
```

On an Apple silicon Mac with Metal support:

```sh
VOLT_TEST_DEVICE=metal cargo test --release --locked --features metal
```

On an NVIDIA host, install a compatible driver and CUDA toolkit with NVRTC and
make the toolkit libraries discoverable:

```sh
VOLT_TEST_DEVICE=cuda:0 cargo test --release --locked --features cuda
```

Device selection is explicit. A requested accelerator never silently falls back
to CPU. Metal rejects f64. CUDA uses CubeCL's CUDA C++ compiler through NVRTC for
accurate native special functions. The authored application and kernel code are
Rust; generated CUDA source is an internal backend detail.

The optional `fusion` feature is supported on Metal. CUDA execution with both
`cuda` and `fusion` enabled is rejected: shape-dependent fused arithmetic failed
cancellation checks. Use an unfused CUDA build. Each supported configuration must
pass the same numerical checks.

## Interfaces and financial meaning

`Session` selects a backend and precision. `ResidentMatrix` provides checked
row-major upload, device multiplication, and readback. `MatrixWorkspace` adds
shared logical leases for tensor, scratch, staging, and downloaded storage;
cloned workspaces coordinate admission and reject unowned matrices. CUDA matrix
multiplication explicitly selects CubeK's full-precision unit strategy rather
than automatic TF32 operand conversion. Direct Burn operations outside these
interfaces retain their upstream precision policies.

`ResidentOptions` uploads numeric option inputs once. `PreparedValuation` retains
the portfolio and device session across evaluations, applying positive uniform
spot shocks on device without re-uploading input arrays. Persistent/scaled spots
must remain positive normal values at device precision. `chunked::value_rows`
handles jobs that cannot fit as one resident portfolio, preserving ordered marks.

`VoltValuationRequest` projects an exact canonical portfolio and a versioned
point-in-time market snapshot into numeric rows. It checks schemas, identities,
common currency, observation age, matching market data, and supported exercise.
Source contracts, settlement, quantities, multipliers, timestamps, portfolio hash,
and market hash remain available alongside the projection. The initial pricing
scope is unexpired European calls and puts with positive volatility and explicit
Actual/365 Fixed maturity. Rates are continuously compounded annual fractions;
spot and strike are currency units.

Returned portfolio marks use ordered compensated f64 aggregation of prices and
original quantities/multipliers. Device reductions are diagnostics. Floating-point
valuation does not replace Volt's exact cash ledger, lifecycle, fills, or authority.

## Working memory

`MemoryBudget` admits known valuation workloads with compute and host allowances,
reserve, headroom, and row caps. CPU and Metal numeric storage also counts against
host working space. `MatrixBudget` models live tensors plus configured multiplication
scratch; RAII leases remain charged until their owning handles are dropped.

These are workload admission models, not OS or driver quotas. Caller-owned inputs,
retained prior results, other workspaces/processes, report history, and compiler or
driver memory are outside a single evaluation allowance. Allocator telemetry is
process/device shared and sampled; it does not establish peak physical memory.
An observed reserved-byte guard cannot prevent an earlier allocation failure.

## Run complete valuation jobs

The checked fixture is synthetic and includes 198 positions across three
underlyings and expiries, calls and puts, signed quantities, and both settlement
conventions. It validates infrastructure rather than market performance.

```sh
cargo run --locked --release --features metal -- \
  --chunked --device metal --max-batch-rows 17 \
  --volt-input tests/fixtures/portfolio-request.json \
  --output target/measurements/portfolio.json \
  --marks-output target/measurements/marks.jsonl
```

Use `--plan-only` to validate and print admission without initializing a GPU.
`--compute-budget-mib`, `--host-working-mib`, `--compute-reserve-mib`, and
`--headroom-percent` configure the working allowance. Raw `--input` numeric rows
are also supported, with financial consistency then owned by the caller.

## Measure resident and persistent performance

```sh
cargo run --locked --release --features metal --bin persistent_bench -- \
  --device metal --rows 1048576 --repeats 10 \
  --spot-scales 0.99,1,1.01 --output target/measurements/persistent.json

cargo run --locked --release --features metal --bin matrix_bench -- \
  --device metal --m 257 --k 193 --n 129 \
  --output target/measurements/matrix.json
```

Substitute `--features cuda --device cuda:0` on NVIDIA or use `--device cpu`.
Persistent-job timing includes evaluation, synchronized readback, and stable host
aggregation; preparation and independent scalar validation are reported separately.
Matrix reports separate input preparation, upload, cold and warm multiplication,
readback, and reference checking. Small/moderate matrices use a full independent
f64 dot-product reference with non-binary-fraction inputs; large workloads use a
labeled structured fixture with a closed-form reference for every output.

Measure release builds and distinguish resident operation timings from complete
jobs, cold initialization, transfers, and reference validation. The included scalar
reference is a correctness control rather than an optimized CPU benchmark.

## Calculate sensitivities

```sh
cargo run --locked --release --features metal --bin aad_bench -- \
  --device metal --rows 65536 --gamma --repeats 3 \
  --output target/measurements/greeks.json \
  --greeks-output target/measurements/greeks.jsonl
```

The same command accepts `--volt-input` for the canonical portfolio request, or
`--input` for numeric rows. `--plan-only` validates the complete input and its memory
plan without opening a GPU. The AAD model reserves retained graph/adjoint storage;
`--gamma` uses a larger model and returns a seventh output array. Admission budgets
remain estimates, with the same exclusions described above.

One reverse pass returns per-row delta, vega, rho, calendar theta, and dividend-yield
sensitivity. Each dispatch creates a fresh graph. Optional gamma is computed by a
second-order forward-mode tensor jet through the smooth pricing function. This is
not a Monte Carlo estimator for a discontinuous or kinked terminal payoff.

Vega, rho, and dividend sensitivities are per 1.0 annual fraction: multiply by 0.01
for one percentage point. Theta is currency per year of calendar passage with other
inputs fixed; divide by 365 for a daily convention. All row outputs are per underlying
unit. The report separately applies original signed quantities and multipliers and
uses compensated f64 sums for parallel portfolio sensitivities. Those parallel sums
do not replace instrument/underlying risk buckets or cross-gamma.

Each report checks every output against independent scalar analytic formulas;
tests also use central repricing and differences of separately computed delta.
The f64 regression catches an upstream Burn 0.22 scalar-division backward precision
issue; the shared CDF graph avoids it by multiplying by the reciprocal constant.
The analytic helpers validate the result; the production outputs come from AD.
Timing includes planning, device setup, upload, forward/backward, optional gamma,
all readback, and host aggregation, with scalar validation outside the interval.
Repeated dispatches reuse runtime caches, but do not imply persistent input buffers.
Report and JSON Lines output preserve numeric input hashes and row identities.

## Monte Carlo and Heston risk

```sh
cargo run --locked --release --features metal --bin monte_carlo_bench -- \
  --device metal --precision f32 --rows 4 --pairs 131072 \
  --output target/measurements/monte-carlo.json

cargo run --locked --release --features metal --bin heston_bench -- \
  --device metal --precision f32 --profile paper \
  --output target/measurements/heston.json
```

GBM simulation uses its exact terminal law, one reverse pass for five pathwise
first derivatives, and a likelihood-ratio gamma estimator. Indexed normal draws
are independent of batch boundaries; an antithetic pair is one observation when
calculating standard errors. Portfolio uncertainty uses signed position sums
within each shared pair, retaining numerical estimator covariance. This is not a
calibrated joint market P&L model across underlyings. Monte Carlo reports validate
against analytic expectations using a declared six-standard-error allowance plus
an explicit numerical allowance. This diagnostic is not guaranteed confidence
coverage, particularly for rare payoffs or small samples. The normal-input hash
records possible last-bit host math differences across machines.

`monte_carlo_bench` accepts `--input` numeric rows or `--volt-input` using the
existing exact projection. `--estimates-output` exports identity-bound means and
pair standard errors. Its batch cap counts expanded row × pair × 2 outcomes.
Its working model accounts for the terminal graph and backward state, not a
retained multi-step path. Source inputs, parsing and serialization remain outside
working allowances. Scoped cleanup also runs on failed jobs.

Heston has separate option and stochastic-variance types: Black–Scholes annual
volatility is not a Heston parameter. The shared characteristic-function graph
returns price, nine reverse first derivatives and gamma from differentiated
Fourier inversion. Variance sensitivities are per unit variance; theta is calendar
decay per year. The `paper` fixture uses the original short-horizon/high-vol-of-vol
parameters; `lewis` uses published external pricing controls. Both are numerical
risk fixtures, not Deep Hedging training or replication results.

`heston_bench --input` accepts `volt.compute.heston_request.v1` with `options`,
matching `params`, position metadata, and currency. This numeric request preserves
IDs and signed weights but does not revalidate canonical legal contracts. All
fields are explicit, and unknown model fields are rejected. The checked runner
preflights (N,U), (2N,U) and (2N,2U), then tests all eleven outputs separately for
node and cutoff convergence. Identical numeric model inputs share a cached
result across batches, preserving cancellation despite shape-dependent f32
rounding; row IDs and weights remain separate. Reports distinguish logical chunks,
unique evaluated rows and actual device batches. Passing finite refinements is not a global accuracy
certificate; tests also use independent published QuantLib prices, adaptive
integration Greek controls, repricing, variance limits and branch stress cases.
Current variance may be zero; its variance sensitivity is a right derivative.
An exact Black–Scholes control variate reduces cancellation of small prices.
Mapped quadrature supports the longer tail there, but callers must refine cutoff
as well as node count; fixed-node integration can remain unresolved for oscillatory
late-date states away from the strike. `heston_adaptive` provides an independent
CPU-f64 adaptive Gauss–Kronrod price/delta/initial-variance reference with explicit
work limits and separate tail checks. Its numerical error estimates are not
universal analytic bounds. Zero vol-of-variance is rejected rather than routed through an
incompatible sensitivity branch. `--plan-only` opens no device; the Heston cap
counts option × node cells. `--greeks-output` writes identity-bound row results.

## Integer hedge accounting

`hedge_ledger::HedgeLedger` provides repeated stock and funded variance-claim
trades, fractional quantity steps, proportional notional costs, lending/borrowing
interval accrual, terminal liquidation and explicit liability cashflows. A funded
variance claim differs from a conventional zero-value variance swap. The ledger
uses Volt currency/price/time types; it does not mutate Volt lifecycle contracts.

Price micros × quantity micros are retained in checked i128 currency picounits.
Cost and financing rational rounding is journaled; cash conversion to Volt micros
returns its residual explicitly. Amounts and quantities serialize as decimal
strings so exact values survive JSON consumers. Every journal entry reconciles
wealth against initial cash, mark gains, interest, costs and liability settlement.
Failed operations leave state and snapshot availability intact.
`hedge_reconciliation::reconcile_path` bridges simulated marks and policy holdings
to the ledger with explicit quote/quantity quantization and rounding bounds.
Canonical lifecycle handling, corporate-action cashflows and external execution
remain application responsibilities.


## Train, resume and benchmark Deep Hedging

European liabilities include calls, puts, same-strike/expiry straddles and
normalized call spreads. Training uses the aggregate terminal payoff and
after-cost hedge profit, without an analytic hedge label. Each fitted policy
is for the configured liability and responds to changing market state and
optional hedge inventory. See the workspace
[training guide](../../docs/compute-training.md#european-liabilities-and-the-trained-policy)
for JSON examples, premium/sign conventions and payoff-equivalent portfolios.

The `volatility_strategy` executable adds delta-hedged straddle controls with
separate physical-return and option-pricing volatility, shared nested market
paths, explicit stock costs, Greek/P&L attribution and exact ledger audits.
The library supplies IV inversion and bounded total-variance interpolation.
The optional `live-risk` feature adds a Hypercube-backed HTTP/SSE risk panel. Its default feed is
synthetic; complete normalized European quote/book frames can be posted by an
external producer. See the same guide for runnable examples and conventions.

`hedge_policy` implements a separate two-hidden-layer ReLU network at each decision
time, with BatchNorm, signed holdings and optional recurrence through previous
holdings. The Heston profile uses 30 decisions, `(log S, V)` features, two hedge
instruments and hidden width 17. Parameters and Adam moments stay on the selected
policy device. CPU affine products use native-dtype GEMM; accelerator products
use strict differentiable multiply-and-reduce operations to avoid CUDA's automatic
reduced-precision matrix path.

The runner generates exogenous markets in host f64 and uploads them to the policy
device. GBM uses its exact transition law. Heston samples exact CIR variance
endpoints, then approximates integrated variance using the declared left or
trapezoid rule in a simplified Broadie–Kaya construction. The resulting joint
Heston paths generally have martingale bias; exact endpoints do not make the full
path scheme exact. `market_paths_bench` reports martingale, variance and option-price
controls. Simulation bias and hedge discretization affect comparisons with a
continuous-time model hedge.

The following bounded CPU example exercises the workflow; its update count is
not a training-quality or paper-replication claim:

```sh
cargo run --release --locked --bin deep_hedging -- train \
  --profile bs --device cpu --updates 100 --validation-interval 25 \
  --validation-paths 256 --test-paths 512 --eval-batch-paths 256 \
  --checkpoint target/measurements/bs.checkpoint.json \
  --output target/measurements/bs.training.json

cargo run --release --locked --bin deep_hedging -- train \
  --resume target/measurements/bs.checkpoint.json --updates 200 \
  --checkpoint target/measurements/bs-resumed.checkpoint.json \
  --output target/measurements/bs-resumed.training.json

cargo run --release --locked --bin deep_hedging -- evaluate \
  --checkpoint target/measurements/bs-resumed.checkpoint.json \
  --paths 1024 --seed 900007 \
  --output target/measurements/bs.evaluation.json

cargo run --release --locked --bin deep_hedging -- benchmark \
  --checkpoint target/measurements/bs-resumed.checkpoint.json \
  --paths 1024 --seed 900001 --batch-paths 256 --baseline-threads 4 \
  --audit-paths 8 --samples-output target/measurements/bs.samples.jsonl \
  --output target/measurements/bs.benchmark.json
```

Use `--profile heston` for the two-instrument Heston application. Add
`--features metal` or `--features cuda` before `--bin`, and select the matching
`--device`, to train or infer on a GPU. `train --plan-only` and
`benchmark --plan-only` validate admission without initializing the policy device.
Complete settings are accepted through `train --config`; the report records the
resolved configuration. Feature, action and previous-holding scaling, a policy
without recurrence, proportional costs and the CVaR/entropy/squared objectives
are explicit settings. Such deviations must remain visible in comparisons.

Checkpoints retain parameters, Adam state, training position and BatchNorm
statistics. Resume preserves the experiment configuration while allowing a new
target, memory allowance and total update limit. Validation selects among the
trained policies and an initial zero-trading candidate. Frozen-weight BatchNorm
statistics are refreshed on a separate training-only calibration stream; validation
and test paths do not calibrate or train the model. Inference and matched
benchmarking use the selected snapshot unchanged.

`benchmark` compares the frozen candidate, a genuine model hedge and no hedge on
identical held-out paths. The BS baseline is analytic f64. The default Heston
benchmark baseline is **adaptive scalar CPU-f64**, independently of the policy's
GPU device; its thread count and execution policy are reported. It computes
Heston price/delta/initial-variance sensitivity, with no BS substitution when
integration fails. Work-limit or convergence failures stop the comparison.
Reports include paired risk comparisons, declared bootstrap subsets, scalar/device
accounting differences and optional exact integer-ledger audits. Path samples
are streamed only when explicitly requested.

Training and benchmark planners account for recurrent tape, expanded strict
products, optimizer state, calibration/evaluation buffers, adaptive reference
workspaces and bounded result storage. These remain admission estimates with
sampled allocator guards, rather than process or device quotas. Assess held-out
results, simulation refinements, numerical controls and multiple independent
training seeds before making a replication or performance claim.


For fresh Linux checkout, immutable CPU/CUDA binaries, background execution and
resume, see [training on a compute host](../../docs/compute-training.md). The
application also includes validation-tuned model-target bands, checked baseline
caches, compact paired P&L exports, pure checkpoint inspection and simulation
refinement with unchanged hedge dates. Paper replication remains in progress.
