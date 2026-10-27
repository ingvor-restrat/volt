# Correlated max-call markets and policy-tail price evaluation

`multi_asset_market` generates risk-neutral correlated geometric Brownian motion
and prepares stopped writer episodes for a max-call on one through one hundred
stocks. The holder receives `(max(S_i)−K)+` at an allowed exercise date. Rates,
annual volatility fractions, dividend yields, exercise dates and correlation are
explicit inputs. Exact GBM transitions eliminate Euler time-discretization error
between the supplied dates; stopping opportunities remain discrete.

`MaxCallMarket::paper_five()` specifies five independent stocks initially 100,
strike 100, continuously compounded rate 5%, dividend yield 10%, volatility 20%,
horizon 3 years and nine four-month exercise intervals. The [executed comparison](benchmarks/maxcall-five-asset-20261027/README.md)
trains three holders for this Becker, Cheridito and Jentzen case and values
their selected exercise decisions independently.

## Paths and conditional branches

`generate_paths(&market, paths, seed, first_path, namespace, max_working_bytes)`
returns `MaxCallPaths`. Its `values` layout is `[path,date,asset]`; `state(path,date)`
provides a checked slice. A supplied starting path ID preserves the generated states across different
streaming batch sizes. `market_sha256` binds model inputs;
`content_sha256` additionally binds the realized states, seed, namespace and path
range. Episode output preserves that identity as `physical_cohort_sha256` across
holder, position-size, premium and fee changes.

Correlation accepts positive semidefinite matrices, including perfect positive
or negative correlation. Semidefinite Cholesky adds no diagonal jitter. Unit
diagonals and finite entries are required. Symmetry tolerance is 1e-12; a negative
residual pivot within roundoff tolerance is clamped to zero and counted. The
factor reports rank and reconstruction error; a materially negative pivot or
nonzero residual at a zero pivot rejects the input. These diagnostics expose
roundoff treatment explicitly.

`simulate_tail(&market, date, &current_state, RandomKey, max_working_bytes)`
starts a conditional branch at the supplied state and returns
`[future_date_including_start,asset]` values. Outer and conditional streams have
separate purpose domains. SHA256 addresses a 128-bit stream key from seed,
namespace, global path ID and branch; two keyed SplitMix64 uniform words feed
Box–Muller normals indexed by global date and asset. Repeated logical addresses
reproduce paths. Independence statements use the usual idealized independent-RNG
assumption. Separate namespaces must be declared for training, selection and
independent evaluation.

## Financial episodes

`assemble_episodes` accepts a completed cohort, one admissible stopping index per
path and `MaxCallEpisodeConfig`. Settlement is derived from the observed state:

\[
 G_\tau=Q\,e^{-r t_\tau}(\max_i S_i(t_\tau)-K)^+,
 \qquad Q=\text{quantity}\times\text{multiplier}.
\]

The account uses the source 2020 discounted reinvested-security marks
`P_i(t)=exp((q_i−r)t)S_i(t)`. Cash growth and income growth are both one.
A holding `h_i` represents units of the reinvested security, corresponding to
`h_i exp(q_i t)` physical shares. Trading and liquidation fees are fractions of
discounted traded notionals. Reinvestment is part of the security convention.
The producer's `premium` is total signed initial cash for the complete position;
quantity and multiplier are applied once to settlement, with no additional
scaling of premium or holdings supplied by the caller.

`EpisodeBatch.exercise` carries stopping indices and discounted signed payments;
terminal `payoffs` are zero. A stop occurs on arrival before that date's hedge
action. The account liquidates once and settles; step-zero exercise opens no
hedge. Features contain the raw current stock vector followed by the discounted
unit max-call payoff. Holder identity/digest is recorded and bound into episode
content. The caller verifies the selected holder's checkpoint and causal use of
observations. `evaluate_episode_holdings` provides a scalar reference for the
same stopped discounted account and actual costs.

## Selected-policy-tail bounds

`MaxCallStoppingPolicy` offers scalar and batched decision callbacks. A selected
checkpoint supplies immutable inference weights and running normalization; a
resident predictor can override the batch callback. The policy's decision at a
date uses only that date's state. `evaluate_policy_tail` implements the nested
policy-tail construction of
[Becker, Cheridito and Jentzen (2019), Proposition 7](https://jmlr.org/papers/v20/18-232.html)
and their [2020 paper, Section 3.2](https://arxiv.org/abs/1912.11060).

For outer state `X_n`, each conditional simulation forces continuation at `n`,
then tests the selected policy first at `n+1`. Its discounted eventual reward
estimates `C_n=E[G_(tau_(n+1)) | X_n]`. Independent branches produce `Cbar_n`.
With the selected binary decision `f_n` on the actual outer state,

\[
 \Delta\widetilde M_n
 = f_n G_n+(1-f_n)\overline C_n-\overline C_{n-1},
 \qquad \overline C_N=0.
\]

Each conditional estimate is reused consistently in adjacent increments. Every
upper outer path runs through all exercise dates, including dates after the
holder would have stopped on that path. The lower estimate uses the selected
holder's first exercise. Inner branches use independent per-outer, per-date,
per-branch addresses; in particular each outer observation has its own estimate
of `C_0`. Batches are used for current active inner states to support efficient
resident inference. `conditional_outer_batch_rows` pools one through 32 parent
states, each with its own nested branches, into a shared inference batch. Its
default is one. This is independent of `outer_batch_rows`, which controls outer
cohort generation and lower-estimate streaming. Stream addresses and per-parent
reward summation remain identical across these grouping choices; callback
predictions must be independent of other rows in their batch.

The noisy process equals the policy-value Doob martingale plus errors with zero
conditional mean relative to the original outer observations. Conditional Jensen
therefore gives an upper **expectation** for `max_n(G_n−Mtilde_n)`. The estimator
records `becker_policy_tail_nested_v1`, distinct from the separate one-step
value-proxy method. Learned continuation values supply stopping decisions;
conditional means in this estimator come from actual policy-tail simulations.

Output includes lower/upper samples, means, outer-sample standard errors,
midpoint, noisy-increment diagnostics and an asymptotic joint 95% CLT enclosure.
A finite realized sample or estimated interval can miss the true price. The
lognormal max-call is unbounded; the output makes no bounded-payoff Hoeffding
claim. Transition-count and memory admission bound computational work, while
independent statistical evaluation addresses sampling error.

## Admission and executable interface

Market admission allows at most 100 assets, 1,024 time intervals, two million paths
per cohort and a declared working-memory allowance up to 8 GiB. The factor,
transition scratch, path/episode arrays and nested-estimator work are modeled;
caller-retained arrays, callback model/device workspace and the allocator's own
reserves remain caller-owned. The policy-tail evaluator also checks an explicit
worst-case asset-transition count before drawing samples. The evaluator streams
outer batches and permits up to eight million lower scalar observations, covering
the source 2020 evaluation count of 4,096,000 lower paths and 2,048 upper paths
with 2,048 nested branches. These counts require a correspondingly declared
transition/memory allowance. Numerical overflow
returns an error; paths are retained without selective deletion.

```sh
cargo build --release -p volt-compute --bin multi_asset_market
cargo run --release -p volt-compute --bin multi_asset_market -- \
  --input request.json > scenarios.json
cargo test --release -p volt-compute --test multi_asset_market
```

The JSON request contains `market`, `paths`, `seed`, `first_path`, `namespace`
and `max_working_bytes`. Optional `stopping_steps` and `episode` are supplied
together. Requests are capped at two MiB, with output serialized as a stream.
The Rust library supports chunked generation directly.

The [complete application example](maxcall-application.md) connects a selected
holder checkpoint to writer training, validation selection, checkpoint restoration
and fresh-path assessment. `maxcall_episodes` verifies the actual holder-file
digest and applies its first admissible exercise decision before assembling the
discounted stock account.

Tests cover correlated log moments and discounted-stock martingales, exact
one-asset European Black–Scholes sampling controls, pathwise max-call bounds,
perfect-correlation reduction, 100-asset production, chunk/conditional-stream
identity, actual fees and signed settlements, malformed PSD matrices, and
selected-policy-tail deterministic and sampling controls. A terminal-only
deterministic regression specifically checks the initial conditional mean;
omitting it would give a false zero price.
