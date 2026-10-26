# Factor scenarios and predictive hedging controls

The `volt-compute` factor generator turns **supplied** return-risk parameters into joint stock scenarios, European liabilities and native-learner episodes. It keeps a predictive physical drift model separate from the risk-neutral option-marking reference. The example is synthetic: its forecasting relation is chosen explicitly.

```sh
cargo run -p volt-compute --bin factor_scenarios -- --example > /tmp/factors.json
cargo run -p volt-compute --bin factor_scenarios -- \
  --input /tmp/factors.json --output /tmp/episodes.jsonl --batch-paths 128
cargo test -p volt-compute --test factor_scenarios
```

The JSONL output contains bounded scenario batches. Each has `batch` (the actual `native_learner::EpisodeBatch`), funding/cost arrays, named features, risk decomposition, physical-path hash and full input provenance. The producer streams chunks to an atomic output file; it checks range overlap, shape arithmetic and a working-memory allowance before generating them. Inputs are capped at 16 MiB; one request supports at most one million paths, 16 assets, 8 factors and 252 decisions. Public illustration seeds and ranges are separate from the research confirmation cohort.

## Specify the economic model

For stock `i`, annual instantaneous return covariance is

```text
C = B * factor_covariance * Bᵀ + diag(idiosyncratic_volatility²).
```

`B` is decimal stock return per decimal factor return. Factor covariance is annual return-squared; idiosyncratic volatility is annual decimal standard deviation. The implementation accepts positive-semidefinite factor covariance, including redundant factors, and checks symmetry and finiteness. Stock reference volatilities are `sqrt(C_ii)` and must be positive. The decomposition records both covariance and factor impacts.

A separately specified observable signal evolves once per interval:

```text
signal_next = persistence * signal + signal_innovation_std * independent_normal.
```

Persistence and signal innovation standard deviation are **per interval**. Each path has an uncertain annual response coefficient drawn once from the supplied mean and standard deviation. Its realized value remains latent. Physical excess drift is

```text
base_excess_return_i + latent_response_i * observed_signal.
```

The stock transition uses this drift plus `r-q_i`, the usual diffusion correction `-C_ii/2`, correlated factor shocks and independent idiosyncratic shocks. A counter RNG keys draws by seed, global path ID, step and shock domain. Chunking preserves the paths. Signal innovations, return innovations and latent coefficients use distinct domains.

Only the signal, mean forecast and its declared coefficient-prior uncertainty enter decision features. The realized latent coefficient is an audit output. This uncertainty describes the supplied prior across scenarios; the generator performs no Bayesian filtering or forecast fitting. Physical horizon covariance also includes drift/signal variation, so it can exceed the instantaneous diffusion covariance above.

European call/put marks use Black–Scholes at `sqrt(C_ii)` with the stated continuously compounded `r` and `q_i`. These are pricing-reference assumptions. Expected physical returns come from the separate signal specification. Liabilities carry signed contracts and multipliers, settle at expiry or at their closing mark if the experiment ends sooner, and are expressed as positive amounts owed by a short dealer.

An existing PCA snapshot supplies useful risk information, but creating this generator also requires factor innovation covariance, idiosyncratic risk, time normalization, initial prices and a physical transition law. A rank score additionally requires calibration into return units before it can serve as a drift forecast. `FactorProvenance` identifies supplied sources and checks that their knowledge time precedes the decision origin. Synthetic inputs are labelled; real calibration availability must be supplied.

## Episode and learner API

`factor_scenarios::generate(model, config, range)` returns:

| Field | Meaning |
|---|---|
|`batch.marks`|Stock prices in `[path,time including terminal,asset]` order|
|`batch.features`|Decision-time observations, excluding terminal observations|
|`batch.payoffs`|Signed, multiplied terminal liabilities|
|`economics.premium`|Initial quote of the liability portfolio|
|`economics.cash_growth`|Gross per-interval cash growth `exp(r*dt)`|
|`economics.income_growth`|Per-interval stock total-return multiplier `exp(q_i*dt)`|
|`economics.trading_costs`|Fractions of traded stock notional, including opening trades|
|`economics.liquidation_costs`|Explicit terminal charges|
|`physical_sha256`|Identity of stock/signal/latent-response paths, independent of observation masking and liability task|

Feature order is: relative log stock prices; horizon years remaining; liability mark; per-asset liability delta and gamma; observed signal; annual excess-return forecasts; annual drift-response uncertainty scales. `MarketOnly` masks all explicit signal/forecast/uncertainty channels while preserving the physical paths. Prices can still contain indirect information about previous signals.

For the two-stock example there are 13 features. Previous holdings are endogenous policy inputs. A native application can configure, register and upload directly:

```rust,ignore
let generated = factor_scenarios::generate(&model, &config, range)?;
let mut policy = HedgePolicyConfig::paper_heston(128);
policy.time_steps = config.steps;
policy.feature_count = generated.feature_names.len();
policy.hedge_count = model.asset_ids.len();
policy.hidden_width = 8;
policy.batch_norm.enabled = false;
let terms = &generated.economics;
let settings = LearnerConfig {
    target: ExecutionTarget::Cpu,
    precision: Precision::F64,
    policy,
    objective: Objective::Entropic { risk_aversion: 0.1 },
    premium: terms.premium,
    trading_costs: terms.trading_costs.clone(),
    liquidation_costs: terms.liquidation_costs.clone(),
    cash_growth: terms.cash_growth.clone(),
    income_growth: terms.income_growth.clone(),
    budget: MemoryBudget {
        compute_bytes: 512 * 1024 * 1024,
        host_working_bytes: 512 * 1024 * 1024,
        ..Default::default()
    },
    ..Default::default()
};
let mut learner = NativeLearner::new(settings)?;
learner.handle(Command::RegisterDataset {
    dataset_id: generated.batch.provenance.dataset_id.clone(),
    partitions: factor_scenarios::partitions(&declared_ranges)?,
})?;
learner.handle(Command::Upload {
    batch_id: "train-0".into(), split: Split::Train,
    batch: generated.batch,
})?;
let evaluation = learner.handle(Command::Evaluate {
    batch_id: "train-0".into(), retain_holdings: true,
})?;
learner.handle(Command::Update {
    batch_id: "train-0".into(), expected_version: 0,
})?;
```

The snippet assumes imports from `volt_compute::{factor_scenarios, native_learner::*, hedge_policy::HedgePolicyConfig, budget::MemoryBudget, ExecutionTarget, Precision}` and already supplied `model`, `config`, `range` and `declared_ranges`. Full wire-protocol and checkpoint examples are in the [native learner guide](../crates/volt-compute/docs-native-learner.md). The conservative planner reserves transactional updates; the 20-step two-stock double-precision example requires a larger allowance than the worker's default 256 MiB.

All dataset partitions must be declared together and have disjoint global path IDs. Training, validation and later confirmation share a model but use separate ranges. A policy version accompanies updates; validation/evaluation alone leaves it unchanged. Checkpoint weights, optimizer state, configuration and registered dataset identities travel together.

## Separate the three economic tasks

A meaningful comparison separates both information and liability:

1. **Hedge-only control:** train with the actual liability and mask explicit signal channels; select on liability validation risk.
2. **Signal-only trading:** train with zero liabilities and zero premium, observing signals; select on pure-trading validation risk.
3. **Combined:** train with the actual liability and observed signals; select on liability validation risk.

For confirmation, evaluate every selected strategy on identical physical paths and the same liability and premium. Also report each strategy's pure-trading P&L. With zero rates, the conversion is trading P&L = native P&L minus its task premium plus its task liability payoff. Different model inputs under the same liability objective would instead be an observation ablation.

The current native account permits signed stock holdings and financed negative cash at the configured funding rate. The synthetic control uses zero funding/dividends and an explicit common fee schedule. Broker margin, borrow availability and hard holding limits require additional application constraints. The quoted learner strategies are research accounts; the separate [lot allocator](financial-lot-qubo.md) demonstrates explicit finite cash/inventory constraints.

## Completed synthetic control

A prospectively declared mechanism study used two stocks, one return factor, a persistent signal, 20 daily decisions and one at-the-money call liability. The three tasks above each had three fits, 1,000 updates and the same fixed 2,048 training/4,096 validation paths. The objective was entropic risk with risk aversion 0.1 per currency unit. Stock fees were 0.05%; terminal stock fees were zero. After all nine selections were fixed, 32,768 fresh common paths supplied confirmation. Monetary values below are per one-unit call (multiplier 1), with initial stock prices 100 and 80. The policy used two hidden ReLU layers of width 8 per date, previous holdings, explicit feature scaling and no batch normalization. Adam used learning rate 0.001, beta values 0.9/0.999 and epsilon `1e-8` after bias correction. Mini-batches held 128 complete paths; validation ran every 100 updates.

| Strategy evaluated against the call | Entropic risk | Mean P&L | P&L SD | 95% loss CVaR | Mean fees |
|---|---:|---:|---:|---:|---:|
|Unhedged|0.744|−0.021|3.428|9.982|0.000|
|Hedge-only control|0.332|−0.224|1.434|3.857|0.213|
|Signal-only trading policy|0.599|+0.236|3.839|9.711|0.316|
|Combined|0.201|+0.189|2.808|5.850|0.426|

Values are averages of three fit-specific measures, except the common unhedged reference. Combined-minus-hedge entropic risk was −0.1302, with a family-adjusted 95% interval [−0.1650,−0.0959]; combined-minus-signal was −0.3980 [−0.4562,−0.3368]. Intervals used 1,999 paired hierarchical bootstrap draws and a two-comparison Bonferroni adjustment.

The combined policy improved the selected entropic objective while carrying more dispersion and worse 95% tail loss than the hedge-only control. The signal-only strategy's separate pure-trading mean P&L was +0.257, but its entropic risk was +0.00486 compared with zero for holding cash. Holding cash retained the lower pure-trading entropic risk in this comparison.

For 36 selected path and policy cases, a second calculation paid for holdings,
charged trading costs and accrued interest one date at a time using ordinary
floating-point numbers. Its final profit agreed with the native calculation
within 1e-9 currency units. A memory-admission refusal before the first numerical
evaluation prompted a recorded resource amendment from 256 to 512 MiB; financial
settings, samples, optimizer and selection rules remained as declared.

The predictive relation and its uncertainty were stipulated. The result demonstrates a functioning separation of hedging and statistical-trading tasks under that law. Empirical efficacy requires calibrated forecasts, point-in-time data and separate historical/out-of-sample evaluation.
