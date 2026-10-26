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

Objectives are `squared_loss`, `semi_squared_loss`, `cvar {alpha}` and `entropic {risk_aversion}`. P&L is positive for gains; CVaR and entropic risk operate on its negative. CVaR integrates exactly the declared fraction of the worst-loss observations, including a fractional last observation. Its gradient is an empirical subgradient; tied losses and zero trades contain nonsmooth points.

Requests/responses are limited to 16 MiB. At most four batches are resident, with conservative admission of the full job plus a second-job reserve for transactional updates. A candidate update replaces live policy/optimizer state only after successful completion. `expected_version` prevents duplicate retries from training twice. Keep timeout recovery and checkpoint persistence in the supervising service.

Dataset bookkeeping stores split ranges rather than individual path identifiers. A 600,000-path study with three contiguous splits needs only three range records. `provenance_budget_bytes` bounds serialized partition metadata (default 1 MiB); resident tensor batches have their own memory admission checks. Checkpoint size must fit the protocol message limit. The data producer supplies an immutable archive/generator identity and guarantees that different identities describe distinct source datasets.

Jormungandr's optional native backend manages this process through HTTP:
https://github.com/ingvor-restrat/jormungandr/blob/main/docs/native-learners.md

The integration tests cover scalar ledger reconciliation, all four objectives, finite-difference parameter gradients and exact optimizer continuation after restore. This initial policy profile establishes the native training interface; the JIVR surface simulator and the follow-on paper's architecture require additional research implementation.
