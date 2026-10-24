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
