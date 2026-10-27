# A multi-asset exercise and hedging application

A max-call pays the excess of the largest underlying stock price over its strike.
The holder chooses when to exercise, and the writer trades a vector of stock
positions until settlement. Volt connects a learned holder to a complete writer
account through the same native learner interface used by the one-stock examples.

The supplied market has five independent stocks initially priced at 100, a strike
of 100, 20% annual volatility, a 5% interest rate and a 10% continuous dividend
yield. Its three-year contract permits exercise at nine four-month intervals.
`MaxCallMarket` also accepts different asset parameters and a supplied correlation
matrix, including positive semidefinite cases such as perfect correlation.
The [market and valuation guide](multi-asset-maxcall.md) defines these inputs.

## Run the complete example

From the repository root:

```sh
cargo build --release -p volt-compute \
  --bin maxcall_learning --bin maxcall_episodes --bin native_learner
python3 examples/maxcall_application.py --output /tmp/volt-maxcall-example
```

The example first fits a continuation-value holder on synthetic paths. It then
generates separate writer training and validation cohorts, applies that selected
holder's exercise decisions, and trains vector stock holdings against the resulting
liability. Validation selects the writer checkpoint. A further path range is
opened after both models are selected, and the restored writer is compared with
leaving the same liability in cash.

The output directory retains the holder inputs and training report, its checkpoint,
writer episode requests, generated episodes, writer configuration, validation
history, selected optimizer checkpoint and assessment summary. Existing output
directories are preserved. `--fee 0.001` adds a fee of ten basis points on stock
rebalancing and liquidation notionals. `--premium` supplies explicit initial
capital; the default uses the holder's training-derived continuation estimate.

Default example budgets are deliberately bounded: 4,096 holder training paths,
1,024 holder validation paths, 100 updates per fitted date, and 64 paths per writer
split with 200 writer updates. The research comparison with the published paper
has a separate fixed architecture, schedule, much larger datasets and independent
valuation estimates. The example demonstrates the complete application sequence.

## Financial units and the application interface

The account uses discounted dividend-reinvested stock values
`P_i(t)=exp((q_i-r)t)*S_i(t)`. Cash and income growth factors are both one.
The liability payment is discounted by `exp(-r*tau)` at the holder's exercise
step. A target holding `h_i` represents units of the reinvested stock security;
its physical share equivalent is `h_i*exp(q_i*t)`. Fees apply to discounted traded
notional. This puts the security gains, cash and settlement on a common basis.

The writer observes the current raw stock vector and the current discounted
unit payoff, together with previous holdings through the configured policy.
Its native engine uses date-specific MLPs and a complete-account squared-loss
objective. The source paper's interval hedge networks instead use the reinvested
stock vector and continuation-value changes as their regression targets. Each
application states its feature and loss convention explicitly.

`maxcall_episodes` connects a saved holder to the writer:

```sh
target/release/maxcall_episodes \
  --checkpoint holder.json --request episode-request.json --output episodes.json
```

The request contains path count, global first path ID, seed, namespace, inference
device/precision/batch limits and `MaxCallEpisodeConfig` economics. The program
checks that the declared holder SHA-256 matches the actual checkpoint file,
applies its decisions using current observations, and constructs the discounted
settlement internally. Quantity and multiplier enter the payment once. `premium`
is the complete position's signed starting capital.

Episode output includes the resolved costs, feature names, account convention,
actual stock-cohort digest and holder identity. The native `upload`, `update`,
`evaluate`, `infer` and `checkpoint` operations retain their existing schemas.
[Jormungandr's HTTP service](https://github.com/ingvor-restrat/jormungandr/blob/main/docs/native-learners.md)
can forward these same batches to a registered Volt worker.

Tests exercise the actual checkpoint-to-episode executable, signed settlement,
checkpoint-digest refusal and atomic preservation of an existing output file.
Separate market controls check correlated stock moments, independent conditional
streams, financed accounts and the selected-policy-tail valuation estimator.

## Published-case comparison and reading

The [executed study](benchmarks/maxcall-five-asset-20261027/README.md) records
three full-schedule holder fits and independent lower/upper valuation. Their
midpoints are 26.1576–26.1609 against the paper's 26.154, with wider sampling
intervals. It also records matched CPU/Metal/CUDA update measurements.
The [application paper](latex/compute/volt_multi_asset_hedging.pdf) develops the
mathematics, account convention and practical execution choices.
