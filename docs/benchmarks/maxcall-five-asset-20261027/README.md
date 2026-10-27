# Five-asset Bermudan max-call comparison

Three full-schedule CPU float64 fits recover midpoint prices 26.1576, 26.1609 and
26.1589 against the published 26.154. All three pass the prospective targets:
absolute midpoint difference at most 0.05 and finite, ordered joint enclosure
width at most 0.10. The published enclosure is narrower.

[The result record](comparison.json) retains each lower/upper mean, standard error,
path count, joint enclosure, training duration and selected checkpoint digest.
The training and evaluation requests here are the actual executed settings.
The three fits have separate initialization/training/validation seeds and use
common fresh assessment streams. Checkpoint selection preceded assessment.

## Reproduce a fit

Build the `maxcall_learning` release executable from the repository root, then:

```sh
target/release/maxcall_learning train \
  --config docs/benchmarks/maxcall-five-asset-20261027/training-41003.json \
  --checkpoint /tmp/maxcall-final.json --report /tmp/maxcall-training.json
```

The assessment request records the original checkpoint SHA-256. After a new fit,
set `policy_sha256` to the SHA-256 of the newly generated checkpoint; preserve
all other sampling and estimator settings. Then:

```sh
target/release/maxcall_learning evaluate --checkpoint /tmp/maxcall-final.json \
  --config /tmp/evaluation.json --target cpu --precision f64 \
  --batch-rows 8192 --output /tmp/maxcall-prices.json
```

Exact bitwise regeneration depends on the compiler, pinned dependencies and
native arithmetic. Budget about 0.8 GiB modeled memory per fit, with requests
allowing 4 GiB compute and 8 GiB host admission. Each fit used one CPU thread;
the three study fits ran concurrently. The training pools contain 1,048,576
paths, and each assessment uses 4,096,000 lower paths and 2,048 outer paths
with 2,048 conditional branches per state.

The [learning guide](../../maxcall-learning.md) describes the declared choices
for details left unspecified in the paper. The [application guide](../../maxcall-application.md)
connects a selected holder to a complete vector writer account. This comparison
covers one pricing row; the writer example has its own observations and loss.

## Device measurements

[Matched float32 timings](performance.json) cover 5/10/50 stocks, batches of
256/1,024/8,192, and hidden widths `d+50`. Both host sweeps completed 27 cases
(CPU f64, CPU f32 and GPU f32), with all 18 matched f32 CPU/GPU workload checks
passing. The optional fixed-zero normalization representation is explicit.
The scientific CPU-f64 price fits above retain the original representation.

On the M3 Max, CPU f32 beat Metal throughout this sweep. On the Threadripper
PRO 7985WX/RTX 4090 system, CUDA beat one-thread CPU f32 at batches 1,024 and
8,192: the 5-stock/8,192-row update took 11.91 ms versus 38.48 ms, while the
50-stock/8,192-row update took 16.13 ms versus 77.69 ms. Each measured update
includes gradient calculation, optimizer, synchronization and scalar checks.
The report retains first-use setup, inference/readback, dispersion and measured
host load. These observations describe the strict-arithmetic implementation
and recorded workloads. First-use timings include existing compiler/cache state.

## Full CUDA training and valuation

One prospectively configured CUDA float32 fit with fixed normalization
translations completed all 30,500 updates in 263 seconds, including process/generation
overhead. Full fresh-path pricing took about 66 seconds.
Its lower/upper estimates were 26.1407/26.1777, midpoint 26.1592 and joint 95%
enclosure [26.1220, 26.1969]. The [result record](cuda-full-fit.json) retains
the timings, hashes and comparison. [Training](training-cuda.json) and
[assessment](evaluation-cuda.json) requests are included.

The original CPU float64 policy's unchanged weights were replayed on the Linux
CPU to obtain exactly the same generated market cohort as the CUDA policy.
Across 4,096,000 paired paths, CUDA-minus-CPU discounted payment averaged 0.00136
with standard error 0.00106 and 95% interval [-0.00071, 0.00344]. This comparison
is compatible with zero mean difference for those selected policies.
The full-fit time comparison includes hardware, precision and normalization
choices; the matched float32 update measurements above retain a common workload.
