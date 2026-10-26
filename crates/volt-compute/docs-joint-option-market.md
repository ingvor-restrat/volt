# Joint return and option-surface scenarios

`joint_option_market` implements the supplied-parameter simulator used by [François, Gauthier, Godin and Pérez-Mendoza, *Deep Hedging with Options Using the Implied Volatility Surface*](https://arxiv.org/abs/2504.06208v3). The reference is the [author repository](https://github.com/cpmendoza/deep-hedging_with_options), commit `64aef8c35b3cc4d31de8c499e2cc9e9f23fb5b3d`, particularly `src/features/{jivr_simulation,nig_simulation,market_simulator}.py` and `src/data/full_data_loader.py`. Its MIT-licensed implementation is credited to Carlos P. Mendoza (2024).

This module produces market scenarios and episode inputs. Training architecture, policy state, constraints and empirical replication are separate work. All simulation and marking here use native CPU `f64`.

## Run a bounded example

From the repository root:

```sh
cargo run -p volt-compute --bin joint_option_scenarios -- --example > /tmp/joint-request.json
cargo run -p volt-compute --bin joint_option_scenarios -- \
  --input /tmp/joint-request.json --output /tmp/joint-scenarios.json
cargo test -p volt-compute --test joint_option_market
```

The example has invented coefficients, an identity copula and **linear synthetic quantiles**. It demonstrates the wire contract and explicitly sets `synthetic=true`. It supplies no calibrated NIG distribution or historical performance evidence. Replace the parameter matrix, initial state and quantile table with independently documented inputs for research. Actual parameter/state/table bytes enter the dataset hash, independently of the producer's source-digest declarations.

The CLI reads at most 16 MiB, validates disjoint declared partitions, and refuses an output alias of its input. Generation checks shapes, path-ID overflow and a conservative working-memory allowance before allocating scenario arrays. It writes through an exclusive temporary file followed by rename. The default 64 MiB modeled allowance supports small batches; stream successive path ranges into separate files for larger studies. The allowance includes conservative serialization/hash workspace, rather than an operating-system memory reservation. Caller-owned deserialized inputs and other processes remain outside it.

## Statistical inputs and equations

`JointParameters.rows` is the numeric 14×6 matrix from the reference CSV, excluding row/column labels. Columns are beta1 through beta5, then return. Rows are:

| Row | Meaning |
|---:|---|
|0|Five beta intercepts; return risk-price parameter lambda|
|1–5|VAR coefficients: row is input beta, column is output beta|
|6|Additional beta2 second-lag coefficient, column1|
|7|Beta unconditional scale; beta2–5 divide this by sqrt(252)|
|8|One-month ATM-IV scale for beta1 and return variance levels|
|9|Conditional variance persistence|
|10|ARCH coefficient|
|11|Innovation leverage coefficient|
|12–13|NIG zeta and phi|

`JointState` contains current stock, current/lagged beta vectors, six conditional variances and six previous standardized innovations. `state_from_history` reconstructs the latter from three beta observations, the current conditional variances and the source's last excess-return observation. The producer owns corporate-action normalization, source time and the historical return-column convention.

For beta1 and the index return, variance levels use the current one-month ATM surface volatility. Other beta levels use the row7 normalization above. Each channel follows the source recursion

```text
h_next = level + persistence*(h-level)
         + ARCH*h*(previous_e² - 1 - 2*previous_e*leverage).
```

The beta vector then follows the VAR, with the additional beta2 lag, and innovation `sqrt(h_next*dt)*e_next`. Index log return is

```text
(r-q)*dt + psi(-lambda*sqrt(h_return_next*dt))
         - psi((1-lambda)*sqrt(h_return_next*dt))
         + sqrt(h_return_next*dt)*e_return_next.
```

The cumulant `psi` uses the source return zeta/phi parametrization. Its finite-moment domain is checked at every step. Each conditional `h` retains the source channel's scale; an innovation always enters as `sqrt(h*dt)`. The beta-channel variances consequently have different economic units from index-return variance.

`advance` accepts supplied innovations for equation-level controls. Seeded `generate` obtains six correlated Gaussian draws, applies the source permutation `[1,2,3,4,5,0]`, maps their CDFs to `[0.0001,0.9995]`, then linearly interpolates the supplied marginal quantiles. `InnovationModel` carries the original return-first correlation matrix, increasing probability grid and six quantile columns in beta-first order. The reference uses1000 grid points generated with SciPy's NIG inverse CDF. Rust consumes the explicit table; it has no SciPy/fitter dependency. Shape, monotonicity, symmetry and positive-definite correlation checks reject malformed inputs.

The counter/Box–Muller RNG is a declared Rust contract. Its sequence differs from NumPy's RNG. A seed, global path ID and step determine each innovation independently of batch boundaries. Fixed supplied innovations isolate equation agreement from RNG differences.

## Surface, contracts and units

The surface uses the source coordinate `M=ln(F/K)/sqrt(tau)`, with `F=S*exp((r-q)*tau)`. It combines five coefficient functions: long-run level, exponentially decaying maturity slope, an asymmetric moneyness slope, smile attenuation and a downside smirk. `surface_volatility` implements those functions and the optional explicit IV floor. This coordinate differs from unscaled log-forward moneyness used by other surface modules.

`paper_case` sets 63 business-day decisions, `dt=1/252`, `r=0.026623194`, `q=0.01772245`, IV floor1%, stock cost0.0005 and option cost0.02. Costs are fractions of traded mark, including opening trades. The liability is one call plus one put, struck at initial stock and maturing after 63 days. The hedges are stock shares and a single initial-ATM 84-day call whose remaining maturity decreases each day. The reference configuration's empty hedging-option list must be populated to study this particular two-instrument case.

`EpisodeBatch.marks` is `[path,time including terminal,stock/call]`. `payoffs` is the positive amount owed for the short straddle, with multiplier1. At expiry it is `abs(S_T-K)`; if the configured horizon ends sooner, it is the closing straddle mark. Longer liability horizons are explicit configuration choices. Premium is the initial call-plus-put quote. Continuous carry is supplied separately as:

- `cash_growth[t]=exp(r*dt)`;
- `income_growth[t,stock]=exp(q*dt)`, and1 for the call;
- trading and terminal-liquidation cost arrays.

These are the arrays consumed by the funded recurrence in `native_learner::LearnerConfig`. Terminal costs default to zero, matching the source's terminal mark convention; alternative liquidation charges are explicit changes. Index return dynamics contain an estimated risk-price/cumulant adjustment. Option marks use Black–Scholes prices at the current IV surface. Those physical-return and pricing conventions serve different roles; their combination supplies no global arbitrage guarantee.

### Preserve the source gamma feature explicitly

The source's BS `gammas_f` discounts by `exp(-q*sqrt(tau))`. The mathematical fixed-volatility price derivative instead discounts by `exp(-q*tau)`. Their difference appears with nonzero dividend yield and was detected by executing the actual author functions against a fixed-innovation fixture.

`GammaFeatureConvention::AuthorSqrtTimeDividend` is the declared default for the **input gamma feature**, preserving source parity. `FixedVolatilityDerivative` selects the genuine derivative as a separate configuration. The selected convention is serialized and changes dataset identity. The public `option_value` price/delta/gamma API always returns genuine fixed-volatility derivatives. Changing the feature convention leaves option marks, premiums and liabilities unchanged. This distinction belongs in any baseline comparison or later ablation.

## Native learner boundary and available information

The 17 market features match the reference BS loader order:

```text
ln(S), horizon years remaining,
beta1..5, h_beta1..5, h_return,
straddle mark, straddle fixed-vol delta, straddle gamma feature,
hedge call mark
```

Only date-t state enters date-t features. Terminal states are retained separately for audit; the episode's feature array excludes the terminal date. Tests change a future innovation and verify that every earlier decision feature is unchanged. The transition's lagged betas and previous innovations remain in `JointState`; they are not additional hidden future features. The 17-element observation follows the source loader and is a partial observation of that transition state. A recurrent policy may use its past observations, subject to an explicitly tested state contract.

The author's policy additionally appends current funded portfolio value and previous holdings inside its cash recurrence. Those are endogenous policy state, supplied by the policy/application layer. This generator's 17 exogenous features do not by themselves reproduce that full policy observation. Existing native learner policies can consume the generated episodes using compatible dimensions and the supplied funding/cost arrays; recurrent/shared-policy experiments require their own architecture and state controls.

Declare all train/validation/test `ScenarioRange`s together and call `validate_ranges` or `dataset_partitions`. Range IDs must be disjoint across splits; a label or a different mini-batch ID alone provides no separation. Seeded generation keeps dataset identity stable across chunks of the same configuration. The fixed-innovation replay route binds its explicit range and bytes as its own dataset, suitable for parity fixtures. Separate processes still need a shared immutable partition registry.

Real source inputs require calibration/initial-state knowledge timestamps no later than `decision_origin_ns`. Synthetic fixtures may omit all three timestamps. These declarations enable validation of availability; they remain the producer's responsibility. The public author files named `*_random.csv` are explicitly reported as random-history fixtures. Their names and successful simulation establish no historical-data provenance or published out-of-sample result.

## Validation and reference boundaries

Focused tests cover independent transition and surface equations, put–call parity, zero conditional-variance channels, terminal settlement, fixed-innovation causality, seeded chunk invariance, memory/range failures, the gamma-feature distinction, source-availability failures and CLI input protection. A generated episode is registered/uploaded to the native learner, evaluated against an independent scalar funded recurrence, and used for one finite optimizer update.

The private reference comparison executes the pinned author's actual `data_preparation`, `betas_returns_simulations`, surface/pricing/Greek functions and NIG-table routine on a tiny fixture. Price derivatives retain the explicit source-gamma distinction above. Its parameter CSV is public, and its initial histories are labelled random. These controls establish implementation agreement within that scope.

The source also deletes paths with excessive stock levels or NaNs before selection. This generator reports the offending global path and step and rejects invalid variances/moment domains; it performs no automatic path deletion or resampling. Any conditioning/filtering rule requires a separately declared sampling protocol. The source's risk-penalty differentiation, one-step LSTM state handling, training/inference no-trade transforms and portfolio-value state are baseline conventions to audit separately. This scenario producer makes no changes to those learning rules and claims no completed follow-on replication.

## Reference implementation notice

The equation translation above is based on the author implementation under the following notice:

```text
MIT License

Copyright (c) [2024] [Carlos P. Mendoza]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```
