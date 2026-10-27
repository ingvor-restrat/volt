# Multi-asset continuation learning

`maxcall_learning` fits a Bermudan max-call holder under the exact correlated
GBM market in [the market guide](multi-asset-maxcall.md). The checkpoint stores
all date networks, normalization statistics, market specification and initial
training estimate. Existing single-asset stopping and hedge-policy schemas are
independent of this module.

## The backward calculation

The default market is the five-asset row in Becker, Cheridito and Jentzen's
[2020 paper](https://arxiv.org/abs/1912.11060): initial stocks and strike 100,
volatility 20%, dividend yield 10%, funding 5%, independent Brownian components,
three years and nine exercise intervals. Each interval is four months.
`multi_asset_market::MaxCallMarket` also accepts heterogeneous stocks,
volatilities/yields and a positive-semidefinite correlation matrix.

At date `n`, the regression input is the raw vector of stock prices followed by
the **discounted unit payoff** `exp(-r*t_n)*max(max(S)-K,0)`. Its target is the
already selected future stopping policy's realized discounted payoff. Targets
retain their currency-per-option-unit scale throughout the regression. The network has
two tanh hidden layers of default width `d+50` and one linear output. Exercise
compares immediate discounted payoff with continuation. Initial exercise uses
an initial training estimate because the initial stock vector is deterministic.

The scheduled **final** update at each date is primary. All final parameters and
BatchNorm running statistics initialize the next earlier date. Adam's moments
and update counter reset at that date. A separate diagnostic checkpoint retains
the earliest minimum validation-MSE snapshot at each date. Their combined policy is an auxiliary candidate trained against the primary
future-policy labels. Earlier primary targets always use the final scheduled
future policy.

Training and validation market pools are generated once with disjoint seeds and
logical namespaces. Their SHA-256 identities are stored. Raw date features and
selected-future targets are uploaded once per date. Every update draws a
deterministic uniform batch **with replacement** from the resident training
split; updates transfer indices to the device and return a small completed-step check.
Scheduled validation and checkpoint operations download predictions or parameters. The fixed finite pool is an explicit reconstruction choice; the paper leaves
its path-reuse convention unspecified.

The explicit default schedules contain:

| Date | Learning rate and update count |
|---|---|
| Last fitted date | .1 × 1000, .01 × 2000, .001 × 2000, .0001 × 1000 |
| Earlier dates | .1 × 500, .01 × 1000, .001 × 1500, .0001 × 500 |

The paper gives the rates and total counts. These exact switch indices are
part of the declared reconstruction. The schedule, finite pools, normalization details and optimizer
convention must therefore accompany a numerical comparison. The public default
pools are 131,072 training and 16,384 diagnostic validation paths; a prospective
study can specify larger pools before fitting. The study configuration, rather
than an implicit software default, determines the experiment.

## Normalization and arithmetic

The reconstruction places BatchNorm on the input and on each hidden affine
output, before tanh. The output remains linear and unnormalized. Current batch variance divides
by batch size; the running variance also uses the population value, without a
Bessel correction. Epsilon defaults to .001. The running update is
`old*.99 + current*.01`; `new_weight` specifies the weight of the **new** moments.
Gamma starts at one, beta at zero. Xavier-uniform dense kernels and zero biases
initialize the last fitted date. Checkpoints record this complete contract.
Inference uses stored running moments, making each row independent of its
prediction peers.

Adam's beta values, epsilon and placement are explicit. The default
`bias_corrected` convention uses `mhat/(sqrt(vhat)+epsilon)`;
`tensorflow_uncorrected` places epsilon inside the uncorrected denominator and
uses the corresponding bias-corrected step size. These are different numerical
conventions. Coefficients and moments use the configured native dtype.

`strict_native_dtype` products use the existing dtype-preserving CPU GEMM and
explicit accelerator multiplication/reduction. The alternative `backend_matmul`
uses the backend's selected kernels, which can have different precision choices.
The request, timings and accuracy controls identify the actual arithmetic.

Every step computes loss, gradient norm and proposed-state finiteness, then reads
those three scalars together. Parameters, moments and running statistics commit
only after all checks pass. Full gradient and optimizer snapshots are available for independent controls.

### Optional fixed redundant translations

With hidden BatchNorm, a hidden affine bias shifts both an activation and its
batch mean, so its exact training effect cancels. Input BatchNorm beta likewise
contributes a constant offset removed by the next hidden BatchNorm. Floating
reductions can leave tiny residual gradients in these directions; Adam can
amplify them, and running-mean lag can affect inference.

`normalization.canonical_zero_translations: true` treats hidden affine biases
and, when input normalization is enabled, input beta as fixed zero constants.
Output bias and both hidden BatchNorm betas remain trainable. The default is
false and is omitted from serialized raw configurations, preserving that
contract. Reference reports preserve the original raw comparisons alongside the separate
fixed-translation checks, using the same numerical tolerances.

`MaxCallNetworkSnapshot::canonicalized_translations()` converts a nonzero warm
snapshot explicitly. It subtracts `old_hidden0_bias + old_input_beta*W0` from the
first hidden running mean and subtracts the second hidden bias from its running
mean, then fixes those parameter slots to zero. Variances are unchanged. This
preserves the represented inference function in real arithmetic, with numerical
agreement checked at the chosen precision. Requesting the option in a new date
learner applies this conversion to a raw warm snapshot. It requires hidden
normalization. The fixed slots remain in the wire layout and have zero gradients
and moments, corresponding to reference layers without those trainable variables.

Independent controls execute actual Keras and NumPy calculations for forward
values, every gradient, moments, running statistics and post-update inference.
Raw and fixed-translation contracts have separate reference fixtures. Actual
Keras f64 moving coefficients can originate as rounded f32 scalars; the exact
native f64 running-state contract is distinguished from that library detail.

## Native API and command line

The published-case comparison uses `target: "cpu", precision: "f64"` with the
original normalization representation. For single-precision CPU or accelerator
training, use `normalization.canonical_zero_translations: true`; independent
controls cover this representation on all three execution targets. Metal uses
`target: "metal", precision: "f32"` with `--features metal,fusion`; CUDA uses
`target: {"cuda":{"index":0}}, precision: "f32"` with `--features cuda`. CUDA uses an unfused build under the current accuracy
policy.
The CLI evaluation override uses `--target cuda:0` or `--target metal`. Device
support and arithmetic/accuracy controls are checked on the actual pinned build.
Reports identify precision, normalization and matrix arithmetic alongside the
independent accuracy checks used for the workload.

`MaxCallDateLearner` owns one date's resident splits and optimizer.
`MaxCallPredictor` uploads all selected date weights/statistics once and provides
batched continuation values and stopping decisions. It implements the market
module's `MaxCallStoppingPolicy` trait, including its batched override. The generic
policy-tail evaluator uses those decisions on independent outer paths and
conditional future paths. Its source estimator is separate from the one-asset
module's alternate one-step proxy dual.

```
cargo build --release -p volt-compute --bin maxcall_learning
maxcall_learning example --output defaults.json
maxcall_learning train --config training.json --checkpoint final.json --report training-report.json --progress-dir completed-dates
maxcall_learning evaluate --checkpoint final.json --config evaluation.json --target cpu --precision f64 --batch-rows 8192 --output bounds.json
```

Split the example's `training` and `evaluation` objects into the respective input
files. Progress is printed to stderr. Each completed date can be atomically
archived. `train --resume completed-dates/date-008.json --checkpoint ... --report ...`
regenerates the fixed pools, verifies their hashes, reconstructs future targets
and resumes the next earlier date with fresh Adam. An interrupted date starts again from its previous completed-date archive,
with fresh Adam state. Final output files and archives refuse overwrite.
Optional `--diagnostic-checkpoint` saves the separate validation candidate.

`bench --config ... --warmup 2 --updates 10 --output ...` measures one-date
resident regression on exact market samples. Its timings describe that update
workload; a complete pricing experiment additionally includes scenario
generation, all fitted dates and independent valuation.
`--synchronized-stages` completes each phase before recording it. Phase clocks
include their synchronization; `synchronization_seconds` is a nested diagnostic already included in those phase clocks. Unsynchronized phase clocks describe host enqueue
work; `total_seconds` includes the single scalar completion/readback. Inference
and readback are identified together where the API returns host predictions.

Device/dtype, logical path streams, matrix arithmetic, normalization, warm starts,
optimizer and separate training, validation and assessment samples belong in every benchmark record. The
working-byte plan estimates conservative admission requirements. Runtime
allocator telemetry measures actual physical allocation separately. Pricing reports include independent sampling uncertainty for their lower and
upper expectation estimates.
