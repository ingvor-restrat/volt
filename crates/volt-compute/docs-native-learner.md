# Hedge-training application API

Applications supply market scenarios and a liability, then ask the learner to improve a policy that chooses hedge holdings. Volt computes the financed portfolio path, trading costs and final risk. The API also provides continuous-holdings inference, evaluation and optimizer checkpoints. Jormungandr supplies an optional HTTP control service for external actors.

`native_learner` accepts bounded JSON-lines commands over stdin and writes one JSON response per command to stdout. Each response echoes its request `id` and includes `schema: "volt.native_learner.v2"`, `ok` and either `result` or `error`. Model tensors and Adam state stay in the process between requests.

```sh
cargo build -p volt-compute --release --bin native_learner
printf '%s\n' \
  '{"id":"a","op":"configure","config":{}}' \
  '{"id":"b","op":"status"}' \
  '{"id":"c","op":"shutdown"}' | target/release/native_learner
```

The default configuration selects CPU/f64, two decision dates, two input features, two hedging instruments, width eight, previous-holdings feedback, squared loss and zero funding/costs. `status` returns the complete resolved configuration. The policy engine is `volt_time_mlp`; native v2 requires BatchNorm disabled. Configure explicit growth and cost arrays when changing the number of dates or instruments.

For a fused Metal worker, build with `--features metal,fusion` and configure `target: "metal"`, `precision: "f32"`. Status distinguishes `fusion_compiled` from `fusion_active`; CPU/Flex execution is eager. The CUDA-fusion accuracy guard in `Session` also applies to this worker. Release builds use Cargo's release optimization settings. A separately measured CPU build can add `--config 'profile.release.lto="thin"' --config profile.release.codegen-units=1 --config 'build.rustflags=["-C","target-cpu=native"]'`; CPU-specific binaries should be built for their deployment host.

## Scenario batches

Register immutable dataset partitions once before uploading batches:

```json
{"id":"register-1","op":"register_dataset","dataset_id":"simulation-seed-7","partitions":[{"split":"train","first_path":"0","paths":"400000"},{"split":"validation","first_path":"400000","paths":"100000"},{"split":"test","first_path":"500000","paths":"100000"}]}
```

An `upload` command supplies `batch_id`, `split` and `batch`:

```json
{
  "id": "upload-1", "op": "upload", "batch_id": "train-1", "split": "train",
  "batch": {
    "paths": 1, "provenance": {"dataset_id": "simulation-seed-7", "first_path": "0"},
    "marks": [100, 3, 101, 3.4, 102, 3.8],
    "features": [0, 0, 1, 0.4], "payoffs": [1]
  }
}
```

`marks` has layout `[path, decision date including terminal, instrument]`; `features` has `[path, decision date excluding terminal, feature]`. `payoffs` contains the terminal liability owed on each path. Prices and holdings use one declared quoted-unit basis, with contract multipliers applied by the producer exactly once. Dataset identity and path ranges remain stable across actor retries. Each batch must lie entirely within its registered split. Partitions are disjoint, immutable and retained through checkpoint/restart. Range indices and counts are emitted as decimal strings to preserve exact u64 identities in browser clients; integer JSON inputs are also accepted. Features must contain only information available at their decision date.

`update` takes a training `batch_id` and `expected_version`. Evaluation and inference leave policy weights unchanged. `checkpoint` returns the configuration, policy parameters, complete Adam state and registered dataset partitions. `restore` accepts that object in a fresh worker; scenario tensors are uploaded again after restore. `drop_batch` releases device tensors while retaining dataset partitions. `shutdown` exits cleanly, and closing stdin also terminates the worker.

## Financial recurrence

At each decision the policy outputs target holdings. The cash account pays for the change in inventory and proportional transaction costs. Cash accrues by the interval's gross `cash_growth`; instruments are marked using the next date's price and `income_growth`. A stock with a continuous dividend yield can use `exp(q*dt)` as its gross income multiplier, matching the reinvestment convention. Options use a multiplier of one. At the horizon, charge declared liquidation fees and subtract the liability payoff. Negative cash accrues at the same configured rate as positive cash.

`trading_costs` and `liquidation_costs` contain one fraction of notional per instrument. `cash_growth` has one positive factor per interval; `income_growth` has `[interval, instrument]` positive factors. This profile assumes exogenous market paths and proportional trading costs. More detailed execution and funding models require an explicit recurrence extension.

### Exercise and intermediate cash payments

A Bermudan option can settle at one of several contractual dates. The producer
evaluates the holder's exercise rule along each path and supplies the resulting
settlement to the writer's learner. The policy then learns stock or other hedge
holdings until that settlement. The [Bermudan scenario application](../../docs/bermudan-market.md)
provides numerical exercise decisions, dated dividends and a financed delta control.

Three optional batch fields extend the same worker protocol:

```json
{
  "exercise": {"settlement_step": [1, 2], "settlement_amounts": [8.0, 3.0]},
  "cash_income": [0.5, 0.0, 0.5, 0.0],
  "initial_wealth": [6.0, 6.0]
}
```

This fragment describes two paths, two intervals and one hedge instrument;
the complete upload also contains marks, features, provenance and zero terminal
`payoffs`. Settlement indices range from zero through the final date. A positive
amount is paid by the writer; a negative amount is received. Contract quantity
and multiplier belong in the producer's settlement amounts exactly once.
`cash_income` has layout `[path, interval, instrument]` and gives the cash paid
per held instrument unit on arrival at the next date. `initial_wealth` supplies
one starting portfolio value per path; omission uses the configured `premium`.
This supports both whole-contract hedging and separately financed interval
regressions whose starting continuation value depends on the current state.

At decision date `t`, a contract with settlement index greater than `t` can
trade. Holdings earn the declared price return and cash income over the next
interval, while cash earns `cash_growth[t]`. Upon settlement the account closes
all hedge holdings, charges liquidation costs once and pays the settlement.
Its remaining cash accrues to the common final date. Settlement at date zero
pays from the initial cash before any hedge trade. Evaluation labels this P&L
convention `common_terminal_bank_account_after_settlement`. The loss therefore
compares every path in currency units at the same date.

After settlement, future features and market prices have zero financial effect
and contribute zero policy gradient. Applications stop requesting hedge actions
when the contract settles. The producer defines event order: for example, exercise
immediately before a cash dividend uses the pre-dividend stock mark; continuing
holders use the ex-dividend mark and receive the declared cash payment. The
scenario application records this convention with its output.

With `exercise` present, legacy terminal `payoffs` must all be zero. Omitting
all three fields retains the terminal-liability account and checkpoint format.
Memory admission includes settlement masks, payments and their differentiation
workspace. CPU f64/f32 and Metal/fusion controls reconcile signed settlements,
cash income, early liquidation and parameter gradients against separate scalar
calculations.

Objectives are `squared_loss`, `semi_squared_loss`, `cvar {alpha}` and `entropic {risk_aversion}`. P&L is positive for gains; CVaR and entropic risk operate on its negative. CVaR integrates exactly the declared fraction of the worst-loss observations, including a fractional last observation. Its gradient is an empirical subgradient; tied losses and zero trades contain nonsmooth points.

Requests/responses are limited to 16 MiB. At most four batches are resident, with conservative admission of the full job plus a second-job reserve for transactional updates. A candidate update replaces live policy/optimizer state only after successful completion. `expected_version` prevents duplicate retries from training twice. Keep timeout recovery and checkpoint persistence in the supervising service.

Dataset bookkeeping stores split ranges rather than individual path identifiers. A 600,000-path study with three contiguous splits needs only three range records. `provenance_budget_bytes` bounds serialized partition metadata (default 1 MiB); resident tensor batches have their own memory admission checks. Checkpoint size must fit the protocol message limit. The data producer supplies an immutable archive/generator identity and guarantees that different identities describe distinct source datasets.

Jormungandr's optional native backend manages this process through HTTP:
https://github.com/ingvor-restrat/jormungandr/blob/main/docs/native-learners.md

The integration tests cover scalar ledger reconciliation, all four objectives, finite-difference parameter gradients and exact optimizer continuation after restore. This initial policy profile establishes the native training interface; the JIVR surface simulator and the follow-on paper's architecture require additional research implementation.

## Shared reset LSTM policy

Set `policy_engine` to `"volt_shared_lstm_reset"` and set
`policy.shared_lstm_reset` to `{"dropout_probability":0.5,"dropout_seed":713}`.
The policy has two shared single-step LSTM layers, two shared ReLU dense layers
and a signed linear output. `hidden_width` applies to all four hidden layers.
Set `include_wealth: true` and `include_previous_holdings: true` for the source
profile: each decision receives features, current financed portfolio value and
previous holdings, in that order. Feature and holding transforms retain their
existing meanings. Wealth uses the portfolio's quoted currency units.
BatchNorm must remain disabled.

This follows the `RNNFNN` architecture in the [author implementation at
64aef8c](https://github.com/cpmendoza/deep-hedging_with_options/blob/64aef8c35b3cc4d31de8c499e2cc9e9f23fb5b3d/src/models/deep_agent.py).
Its configured profile uses width56 and dropout0.5 after each of the two dense
layers. The pinned TensorFlow-v1 code calls Dropout without an explicit
training flag; under an unfed default learning phase those layers are identity.
External session/notebook phase settings can change that behavior. Native
training uses its explicit `train` mode, so active dropout is a declared native
setting, not a claim of historical random-mask or learning-phase parity.
Every LSTM call has sequence length one and starts with zero cell and
hidden state. Weights are shared across decision dates; financial wealth and
holdings carry history. This engine does not carry a persistent LSTM memory
between dates. The source's four-LSTM variant, autoencoder and
rolling-option market construction are separate features.

Each LSTM stores a kernel, recurrent kernel and **one** bias in Keras gate order
`[input, forget, cell, output]`. Recurrent kernels and forget gates have zero
contribution with the declared reset state. They remain in checkpoints for an
explicit parameter mapping. Native initialization uses seeded Glorot uniform
kernels, zero biases and unit forget biases; it does not claim the same random
sample or orthogonal recurrent initializer as Keras. Imported explicit weights
are used for cross-framework tests. The six dense leaves follow the six LSTM
leaves, for twelve parameter tensors shared across all dates.

Training applies inverted dropout after each dense ReLU. Masks are determined
by `(dropout_seed, policy update, decision, dense layer, row, column)`. The seed
and update counter are checkpointed; restoring a checkpoint and repeating the
same batches reproduces optimizer continuation exactly. Evaluation/inference
use no dropout and do not advance any random stream. This deterministic stream
is a native convention, not a claim of Keras random-mask parity. Mask generation
currently runs on the host; accelerator throughput requires a measured workload.

`infer` accepts a `wealth` array with one finite value per path when
`include_wealth` is true. Missing, extra or incorrectly sized wealth is rejected.
For example, after configuring a two-feature/two-instrument policy:

```json
{"op":"infer","step":0,"paths":1,"features":[0.2,0.3],"previous_holdings":[0,0],"wealth":[2.5]}
```

The training rollout supplies attached wealth tensors internally, so gradients
include the effect of earlier holdings on later portfolio value and actions.
The low-level policy API adds `forward_at_with_wealth`; the existing `forward_at`
remains unchanged for policies without wealth inputs. Optional architecture and
wealth fields are omitted from legacy MLP serialization, preserving its existing
snapshot layout and restore behavior.

Admission reserves two candidate jobs and the common architecture-aware planner. Shared parameters are counted once;
CPU GEMM planning counts retained gate and activation vectors with conservative
forward/backward workspace. Accelerator planning uses a sixteenfold MLP tape
allowance for explicit products. The fixed runtime reserve is added once per
candidate. This estimate governs admission; runtime profiling records actual allocation. Larger
widths, horizons and batches need explicit memory budgets. Tests compare actual
Keras forward holdings, financed P&L, all shared parameter gradients and the first
canonical Adam update with dropout disabled; separate tests cover active-dropout
resume, wealth feedback and inference purity.


### Source small-trade rule and tracking report

The reset policy optionally accepts `small_trade_gate` with `sharpness: 1000`,
`force_threshold: 0.1`, `initial_trade_threshold: 0.001`, and
`inference_sharpen: true`. A thirteenth shared parameter is the nonnegative
trainable trade threshold, projected to zero after an update if necessary.
The activity fraction is `sigmoid(sharpness * (sum(abs(target - previous)) - threshold))`.
The sum adds changes in **physical holdings quantities across instruments**.
Contract denomination, multipliers and action units determine its numerical
scale. Producers specify any action normalization explicitly.

An optional `force_trade_signal` array has layout `[path, decision, instrument]`.
Its nonnegative observations produce a second sigmoid around `force_threshold`;
the greater activity/force fraction interpolates target and previous holdings.
Omitting the array supplies zero observations, appropriate for the static
84-day hedge call over the source 63-day experiment. A producer that rolls hedge instruments
supplies the settlement and roll cashflows alongside the force observations.

Training uses the smooth fraction. Source validation uses
`evaluate` with `sharpen_gate: false`. Inference defaults to the configured
sharpening `sigmoid(sharpness * (fraction - 0.5))`; `infer` and `evaluate` accept
an explicit override. Low-level callers use `forward_at_with_context` to supply
wealth, force observations and this phase choice. Existing APIs remain supported.

Optional learner `tracking_penalty` contains `liability_feature`,
`threshold_multiple` and `coefficient`. Before maturity it records whether
`liability_next - wealth_next > liability_initial * threshold_multiple` occurred
on any decision interval. It reports the path fraction and coefficient times
that fraction. The source uses a Boolean comparison with zero pathwise derivative; the native
calculation preserves this choice. `loss` / `loss_before_update` remain
the primary risk objective used for selection. `source_loss` /
`source_loss_before_update` additionally report this penalty. Coefficient units
therefore depend on the chosen risk objective.

Adam optionally accepts `epsilon_convention: "tensorflow_uncorrected"`. This
uses `lr * sqrt(1-beta2^step)/(1-beta1^step) * m/(sqrt(v)+epsilon)`, matching the
source epsilon-hat placement. The omitted default remains the existing
bias-corrected-denominator convention and retains legacy serialized configs.
Actual TensorFlow controls cover smooth/sharpened holdings, financed P&L, all
thirteen gradients and both first-step conventions. TensorFlow uses its own scalar types for exponentiation. The controls record
numerical agreement under those types, with dropout disabled.

Run the bounded synthetic example from the repository root after building the
`native_learner` and `joint_option_scenarios` release binaries:

```sh
python3 examples/joint_option_learning.py --output /tmp/volt-joint-example
```

The example exercises this rule, explicit soft validation, sharpening, source
Adam placement, tracking reports and checkpoint export. The example uses
illustrative generator coefficients and declared synthetic datasets. Historical
studies supply their calibration and dated observations.

The optional reset configuration `recurrent_activation: "hard_sigmoid"` matches
the pinned source's `tf.compat.v1.keras.layers.LSTM` default:
`clip(0.2 * x + 0.5, 0, 1)` for the input/forget/output gates. The omitted
default remains `sigmoid`, as in modern `tf.keras.layers.LSTM` and existing
native checkpoints. Both use tanh for the cell proposal/output. Source default
matching requires explicitly selecting hard sigmoid; the runnable example does
so. Direct tests execute the pinned author's complete `DeepAgent` class with
common weights and a static financial fixture, in addition to modern Keras
component controls.
