# Learned stopping and interval hedging

`deep_stopping` implements two distinct backward algorithms on a single-asset,
exact-transition, risk-neutral GBM Bermudan put. Exercise dates are explicit
floating year fractions; the first date is zero and the final date is mandatory.
Refining the exercise grid approximates continuous American exercise.

* `direct_stopping2019` optimizes a sigmoid decision probability against the
  discounted immediate reward versus the frozen future policy reward. Two hidden
  ReLU layers precede the logit. A probability of at least one half means stop.
* `continuation_regression2020` fits discounted future-policy rewards by squared
  error, with two hidden tanh layers and a linear continuation output. Exercise
  occurs when discounted immediate reward is at least estimated continuation.

These follow the respective algorithmic objectives of Becker, Cheridito and
Jentzen, [2019](https://jmlr.org/papers/v20/18-232.html) and
[2020](https://arxiv.org/abs/1912.11060).
This bounded implementation uses fixed training/validation pools, independently
initialized date networks, no batch normalization, and conditioned features
`[log(S/K), discounted_payoff/K]`; rewards are normalized by strike during fitting.
Validation selects the earliest maximum realized stopping reward. The papers'
large max-call experiments, training sizes, batch normalization, warm starts and
learning-rate schedules differ. Every date also has a continuation regression
for the alternate dual construction. Initial exercise is decided using only a
training estimate of future-policy value, as in the 2019 deterministic-state case.

## Bounds and sampling

After fitting, all networks are fixed. A fresh path stopped by the holder gives a
nonnegative discounted reward whose **expectation** is below the optimal value.
The sample average estimates that expectation and carries Monte Carlo uncertainty.

The implemented dual uses a **conditional-value martingale** built from the
selected continuation functions. The papers describe a nested policy-tail
estimator; the following construction supplies an alternative. Write
`v_n(x)=max(discounted_payoff_n(x), continuation_n(x))`, with terminal `v_N=g_N`.
At each outer transition, simulate independent one-step conditional children
from the current state. The increment is

```
v_(n+1)(outer_next) - mean_k v_(n+1)(independent_child_k)
```

Both terms have the same conditional mean. Reveal each step's independent
auxiliary draws only at the next date. The accumulated process is therefore a
martingale on the enlarged filtration; the original holder stopping times remain
adapted. The expectation of `max_allowed_dates(g_n-M_n)` is an upper bound for
any finite positive inner count. Inner noise can widen it. Replacing conditional
centering by a fitted continuation value alone would not supply this guarantee.

Outer future normals, conditional inner normals, training and validation use
separate Philox4x32-10 counter namespaces. Normal draws use open52 uniforms and
Box–Muller. Evaluation rejects seeds equal to training/validation or one another.
Paths have deterministic IDs and exact GBM transitions. Reports bind the actual
lower and upper outer spot streams with SHA-256. Every generated path remains in the cohort; invalid transitions fail explicitly.

Reports include separate means/standard errors, an asymptotic joint 95% price
enclosure and a conservative finite-sample Hoeffding enclosure. The latter uses
bounded put rewards and the explicit tanh-network output bound. Confidence
statements assume independent idealized RNG draws and a fixed trained model.
A finite-sample upper estimate may fall below the lower estimate by sampling
noise; results are never reordered or clipped. Numerical domains and modeled
workspace are checked before device creation. Supported spot/strike range is
`[1e-12,1e12]`, discounted payoff cap `1e12`, and finite checkpoint parameters
are bounded; unsupported scales fail rather than emitting nonfinite statistics.

## Interval hedging

Training uses the configured Burn device and precision. Selected stopping
decisions, value proxies and interval-hedge evaluation use CPU f64 arithmetic
over stored weights, including the exported f32 weights of a GPU-trained model.
The reported six-holder and three-hedge study uses CPU f64 throughout.

`train_interval_hedge` implements the distinct 2020 Section 4.2 regression, with
one rebalance per exercise interval:

```
min_h E[(C_n(X_n) + h_n(X_n)*(P_(n+1)-P_n) - v_(n+1)(X_(n+1)))^2]
C_n=max(0,continuation_n), P_t=S_t*exp((q-r)*t)
```

The tanh hedge networks are trained on fresh paths and validated independently.
`evaluate_interval_hedge` trades them only before the fixed holder's exercise and
reports discounted error, MSE and shortfall on another independent cohort. The
initial premium is explicit. Holdings are units of the dividend-reinvested
security; physical shares equal `h*exp(q*t)`. There are no transaction fees in
this source interval method. It is separate from the full-path native hedge
learner's stopped cash-account application. The initial version requires the
continuation-regression holder and every interior hedge-grid date must also be
a contractual exercise date. Quarterly exercise with monthly hedging is
rejected by this interval method; it needs additional within-interval models.
The holder-pricing API also supports calendars with fewer exercise dates.
The current experiments establish the complete one-stock application; the
papers' multidimensional max-call experiments provide a subsequent benchmark.

## Native application

Build `cargo build -p volt-compute --release --bin deep_stopping`. Then:

```sh
python3 examples/deep_stopping.py --output /tmp/volt-stopping-example
```

This standard-library client fits a continuation holder and interval hedge,
selects both models before independent assessment, and retains their inputs,
checkpoints, valuation estimates and hedge-error report. The default bounded
example uses 4,096 training paths and 100 updates per date. Individual commands
provide the same workflow with explicit configurations:

```
deep_stopping example --output example.json
# Split the example's training/evaluation objects into the respective files.
deep_stopping train --config training.json --checkpoint holder.json
deep_stopping evaluate --checkpoint holder.json --config evaluation.json --output bounds.json
deep_stopping hedge-train --holder holder.json --config hedge-training.json --initial-price 6.0 --checkpoint hedge.json
deep_stopping hedge-evaluate --checkpoint hedge.json --paths 32768 --seed 730031 --output hedge-result.json
```

The hedge-training configuration has the same market but fresh seeds.
`deep_stopping serve` also provides ordered JSON-line `train`, `evaluate`, `stop`
and `ping` operations. `stop_path` returns the exercise step and undiscounted owed
payoff for integration with an independently checked cash-account assembler.
It reads the observed prefix through settlement. The scenario producer validates
the complete supplied cohort and constructs the corresponding financial account.

## Completed one-stock comparison

A fixed campaign fitted three seeds per holder algorithm and three corresponding
2020 interval hedges. Each holder used 32,768 training paths, 8,192 validation
paths and 500 updates per fitted date. All nine selections preceded confirmation.
The one-year put has S0=K=100, volatility 20%, rate 5%, zero yield and exercise
at each of twelve monthly dates.

On 65,536 fresh payment paths, six holder estimates range from 5.952 to 5.983.
The alternate martingale upper estimates use 4,096 outer paths with 64 independent
conditional children and range from 6.262 to 6.292. Every asymptotic 95% price
enclosure contains the matching numerical value 6.0428. Finite-difference grids
600 × 600 and 1200 × 1200 differ by 0.00013. The finite-sample Hoeffding intervals
are substantially wider; the report retains both forms of sampling uncertainty.
The roughly 0.3 estimator gap remains a numerical improvement target.

On 32,768 additional paths, interval hedge MSE ranges 3.986–4.116 against
54.57–58.60 for cash alone at the declared training-derived starting premiums.
A separate integration check passes the selected holder's actual stopping times
through the cash-dividend-aware assembler and native writer, including a real
autodiff update. The complete-account writer example and its separate quarterly
numerical-holder comparison are described in the
[Bermudan application guide](../../docs/bermudan-market.md).
