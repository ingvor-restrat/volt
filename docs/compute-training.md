# Training with Volt Compute

The workspace contains the financial core and volt-compute, sharing Rust 1.99.1.
CPU execution needs no GPU toolkit. The application supports training, resume,
frozen evaluation, model comparisons and exact hedge-accounting audits.
Recovery of the original Deep Hedging numerical experiments remains in progress.

## Checkout and CPU build

~~~sh
git clone https://github.com/ingvor-restrat/volt.git
cd volt
rustup toolchain install 1.99.1 --profile minimal --component rustfmt --component clippy
cargo test --release --locked --workspace
cargo build --release --locked -p volt-compute --bin deep_hedging
~~~

An existing clean checkout can use git pull --ff-only. The repository owner is
ingvor-restrat. HTTPS checkout of the public repository needs no write credentials.
SSH checkout uses git@github.com:ingvor-restrat/volt.git when configured.
Its transport user is git; select the intended GitHub identity through SSH keys.

## Bounded example

From the repository root:

~~~sh
mkdir -p target/hedging/example
cp target/release/deep_hedging target/hedging/example/deep_hedging-cpu
RAYON_NUM_THREADS=1 target/hedging/example/deep_hedging-cpu train \
  --profile heston --device cpu --updates 2000 \
  --validation-interval 250 --validation-paths 8192 \
  --test-paths 4096 --eval-batch-paths 1024 \
  --checkpoint target/hedging/example/checkpoint.json \
  --output target/hedging/example/training.json \
  > target/hedging/example/stdout.log \
  2> target/hedging/example/stderr.log
~~~

This demonstrates a workflow, not a selected training-quality result or paper
replication. The Heston constructor uses left integrated-variance quadrature;
trapezoidal integration is available through JSON configuration. Exact CIR
endpoints do not make the joint stock/variance path exact.

Reports include resolved settings, validation selection, timing, diagnostic
evaluation and executable/lockfile hashes. Keep the executable copy immutable
while running; reports fingerprint the running executable. Separate copies also
prevent a later CUDA build replacing a CPU executable.

Small date-specific networks at batch 256 can train faster on CPU than GPU:
dispatch and synchronization can dominate. Matrices, pricing and larger batches
are different workloads; measure the actual workload.

## European liabilities and the trained policy

The JSON `liability` field describes the terminal amount owed by the option
seller. In addition to a call, the application supports a put or a same-strike,
same-expiry straddle:

~~~json
{"kind": "put", "strike": 100.0}
~~~

~~~json
{"kind": "straddle", "strike": 100.0}
~~~

The straddle owes `abs(terminal_spot - strike)`, equal to the sum of the call
and put payoffs. Values and holdings are per underlying unit in this application.
The liability is subtracted once from terminal trading wealth. A positive
liability describes a short option position; a long straddle requires the
opposite option exposure and an appropriate hedge, rather than negating fees.
The maturity is the configured market horizon. The current path-accounting
adapter requires zero rates and dividend yield and supports European exercise.

Set `premium_for_comparison` for the selected payoff and units. Changing the
liability while retaining a call's premium gives an incorrect comparison;
squared-loss training also uses that premium in its objective. The model
benchmark supplies the corresponding put or straddle hedge. At zero carry,
put-call parity gives put value `C - S + K` and straddle value `2*C - S + K`.
Stock hedge holdings are respectively `delta_call - 1` and `2*delta_call - 1`;
variance-claim holdings are respectively one and two times the call's. Fees are
calculated from actual resulting trades, so straddle costs are not generally
twice call-hedge costs. Checked baseline caches reuse call components, with the
requested liability bound to the composed result. Source changes intentionally
change the numerical-engine cache identity; use a new build's cache namespace.

A fitted policy responds to market states and, when recurrence is enabled,
current hedge inventory. It does not require training after every price move or
trade. The liability, market law, cost assumptions and objective are fixed for
that training run; this is not yet a policy conditioned on arbitrary portfolios.
No analytic hedge is supplied as a training label: simulated terminal liability
and after-cost trading gains determine the loss that is differentiated.

Two portfolio constructions with identical pathwise cashflows have the same
training objective when premium, hedging instruments, costs and constraints
also agree. Matching today's price or Greeks alone is insufficient. Contract
identity still determines settlement, funding and execution requirements.
Checkpoint resume requires the original configuration: mathematical equivalence
does not authorize editing a checkpoint's recorded liability or provenance.

## Frozen studies, background execution and resume

A complete HedgeTrainingConfig JSON fixes dynamics, integration, objectives,
costs, conditioning, dimensions, independent streams and working budgets.
Keep nonpublic settings and results in a private directory:

~~~sh
CONFIG=/path/to/frozen-profile.json
RUN=/path/to/private/run-directory
mkdir -p "$RUN"
cp target/release/deep_hedging "$RUN/deep_hedging-cpu"
"$RUN/deep_hedging-cpu" train --config "$CONFIG" --plan-only \
  --output "$RUN/memory-plan.json"
RAYON_NUM_THREADS=1 nohup "$RUN/deep_hedging-cpu" train \
  --config "$CONFIG" --checkpoint "$RUN/checkpoint.json" \
  --output "$RUN/training.json" \
  > "$RUN/stdout.log" 2> "$RUN/stderr.log" < /dev/null &
echo "$!" > "$RUN/controller.pid"
~~~

Checkpoints update at validation boundaries. Optional early_stopping contains
minimum_updates and patience_updates (for example 30000 each); omit it or use null
for a fixed update budget. Patience survives resume. Test outcomes never select
checkpoints or trigger stopping.

~~~sh
"$RUN/deep_hedging-cpu" train \
  --resume "$RUN/checkpoint.json" --updates 200000 \
  --checkpoint "$RUN/resumed-checkpoint.json" \
  --output "$RUN/resumed-training.json"
target/release/deep_hedging inspect \
  --checkpoint "$RUN/checkpoint.json" --output "$RUN/inspection.json"
~~~

Resume preserves mathematical settings while permitting a changed target, memory
allowance and total update limit. It retains optimizer state and the separately
selected inference policy. Inspection validates without opening a device and
distinguishes file, object and selected-policy serialization hashes.

## NVIDIA CUDA

Install a compatible driver and CUDA toolkit with NVRTC. Make the tools and
libraries discoverable in noninteractive sessions. Keep CUDA fusion disabled:

~~~sh
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
CARGO_TARGET_DIR=target/cuda cargo build --release --locked \
  -p volt-compute --features cuda --bin deep_hedging
mkdir -p target/hedging/cuda-example
cp target/cuda/release/deep_hedging target/hedging/cuda-example/deep_hedging-cuda
target/hedging/cuda-example/deep_hedging-cuda train \
  --profile heston --device cuda:0 --updates 100 \
  --validation-interval 100 --validation-paths 1024 \
  --test-paths 1024 --eval-batch-paths 1024 \
  --checkpoint target/hedging/cuda-example/checkpoint.json \
  --output target/hedging/cuda-example/training.json
~~~

Requested devices never silently fall back. Metal requires macOS/f32; CUDA
supports checked f32/f64. CUDA plus fusion is rejected by the accuracy guard.

## Held-out comparison

Choose a seed independent of training, calibration and validation:

~~~sh
target/release/deep_hedging benchmark \
  --checkpoint "$RUN/checkpoint.json" --device cpu \
  --paths 16384 --seed 810001 --batch-paths 256 --baseline-threads 8 \
  --audit-paths 8 --baseline-cache-dir "$RUN/model-cache" \
  --pnl-output "$RUN/paired-pnl.bin" --output "$RUN/benchmark.json"
~~~

The genuine adaptive Heston baseline is CPU f64 independently of the policy
device. Its first calculation can be expensive. Checked caches reuse compatible
model entries across policies, costs and objectives; corrupt/mismatched entries
fail. Compact paired P&L files use a versioned header, eight f64 fields per path,
market/financial cohort fingerprints and integrity checks. Optional full JSONL
holdings are larger. Market-only identity permits pairing liability and
no-liability controls despite different payoffs.

Simulation substeps 2 or 4 refine numerical evolution without adding hedge dates.
The model_band program separately tunes a declared no-trade-band grid on
validation paths and evaluates frozen widths on a different cohort. Widths are
physical inventory units. Its raw f64 P&L vector is described and hashed in its
report, distinct from the eight-column checkpoint-benchmark format.

## Straddle P&L with separate return and pricing volatility

An option bought at 20% implied volatility need not subsequently experience
20% realized volatility. The `volatility_strategy` application separates the
constant volatility generating GBM returns from the volatility used to price
and delta hedge a European straddle. Its generic example is a mechanism control,
not an observed-market performance result:

~~~sh
cargo build --release --locked -p volt-compute --bin volatility_strategy
mkdir -p target/straddle-example
target/release/volatility_strategy --example-config \
  --output target/straddle-example/request.json
target/release/volatility_strategy --config target/straddle-example/request.json \
  --plan-only --output target/straddle-example/plan.json
RAYON_NUM_THREADS=1 target/release/volatility_strategy \
  --config target/straddle-example/request.json \
  --output target/straddle-example/report.json
~~~

The request contains signed contract quantity, multiplier, physical volatility,
pricing volatility, fees and several hedge frequencies. Quantity times multiplier
is applied once. Positive quantity buys the straddle and shorts its delta;
negative quantity sells it and buys its delta. Opening trades, rebalances and
terminal liquidation incur the declared stock fees. Zero rates/dividends and
fractional stock holdings define this initial application; it has no option
execution spread, margin or market-impact model.

Coarser hedge grids read the same generated fine path and terminal stock.
Reports retain the physical-market fingerprint separately from pricing and
position settings. Each case includes mean profit and standard error, quantiles,
loss CVaR, actual turnover/costs, an independent analytic mean control, and exact
ledger audits. Gamma/theta attribution retains its higher-order residual.
An optional fixed inventory band can be predeclared in the request. It is not
tuned on the evaluation paths. Optional `--samples-output` exports streamed JSONL;
its size can greatly exceed the summary report.

## Quote chains, surfaces and a live risk panel

The optional `live-risk` feature connects European quote analytics to a pinned
public Hypercube engine. The financial core and default compute build do not
require Hypercube or a web service.

~~~sh
cargo build --release --locked -p volt-compute --features live-risk \
  --bin live_option_risk
target/release/live_option_risk --address 127.0.0.1:8088
~~~

Open `http://127.0.0.1:8088`. The default source is a changing synthetic chain,
clearly labelled in the panel. It shows bid/mid/ask IVs, a surface and expiry
smile, portfolio Greeks, full-repricing scenarios and source freshness. The
surface can be rotated or viewed as a heatmap. Freezing the browser view holds
its selected generation while the server continues to calculate newer ones.

A producer can supply complete normalized quote/book frames instead:

~~~sh
target/release/live_option_risk --example-frame > target/quote-frame.json
target/release/live_option_risk --input target/quote-frame.json \
  --once --output target/risk-snapshot.json
target/release/live_option_risk --input target/quote-frame.json --serve \
  --address 127.0.0.1:8088
curl -H 'Content-Type: application/json' --data-binary @target/newer-frame.json \
  http://127.0.0.1:8088/api/quotes
~~~

`GET /api/snapshot` returns one complete calculation and its Hypercube snapshot;
`GET /api/stream` emits those generations as SSE. A successful external frame
stops the synthetic generator. Frames bind a single underlying/common spot,
currency, quote timestamps, valuation cutoff and portfolio revision. Remaining
maturity is measured at that cutoff; the producer must update it as time passes.
Generations and cutoffs cannot go backwards. A changed book needs a newer
position revision. Rejected requests and failed atomic snapshot-file publication
preserve the last accepted state. `--config` supplies bounded `LiveRiskConfig`
settings, including quote age and request size. A vendor-specific feed adapter
is separate from this normalized JSON boundary.

The IV solver uses a safeguarded bracket and Newton/bisection steps, returning
explicit boundary, conditioning and failure states. The fitted surface linearly
interpolates total variance in log-forward moneyness and expiry; it refuses
extrapolation. Residual, put-call parity, strike-shape and calendar diagnostics
accompany the result. This interpolator reports violations and does not certify
or repair an arbitrage-free surface.

Default spot Greeks hold volatility fixed. `--sticky-log-moneyness` instead
holds the fitted `(k,w)` surface fixed while bumping spot/forward and repricing;
its delta/gamma are explicitly finite-bump results. Vega, theta and rho remain
model partials, in the units displayed by the panel. Missing risk on a held
contract makes portfolio totals unavailable rather than silently treating it
as zero. Derived Hypercube timestamps conservatively include contributing
surface observations; the original quote timestamps remain in the result.

The backend is CPU f64. Measure complete quote-to-risk/publication work before
choosing a GPU path; the current application does not claim a live vendor feed
or a measured exchange-to-screen latency. CI also exercises the browser script's
data/event wiring using a mocked DOM:

~~~sh
node crates/volt-compute/tests/live_risk_render.cjs target/risk-snapshot.json
~~~

That check complements the actual HTTP/SSE tests; browser layout review is a
separate check.
