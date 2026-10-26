# Incremental exercise-aware risk in Hypercube

The CPU-f64 exercise pricer supports European, Bermudan and American schedules.
The publication adapter takes its completed risk rows and publishes actual
Hypercube cells. A stock observation updates the affected underlying's options;
other underlyings retain their earlier values and explicit risk cutoffs.

Build both native processes:

```sh
cargo build --release -p volt-compute --features live-risk \
  --bin exercise_pricer --bin exercise_cube
python3 demo/exercise_incremental.py \
  --pricer target/release/exercise_pricer \
  --publisher target/release/exercise_cube < events.jsonl > snapshots.jsonl
```

The input is one JSON object per line. `replace_underlying` carries a complete
`volt.exercise_quote_frame.v1` for a single underlying. This replaces that
underlying's roster, including signed quantities. `spot` carries a stock price
and observation time and reuses the quote-paired implied-volatility calibration.
`remove_underlying` removes its roster; removing the last roster requires a new
empty-book session. All timestamps are exact nanoseconds, with decimal strings
used in browser-facing data. Generations increase globally; publication cutoffs
are nondecreasing. The following generator creates a small synthetic example:

```sh
python3 - <<'PY' > events.jsonl
import copy, json
now = 1767225600000000000
for generation, name in enumerate(['SIMX', 'SIMY'], 1):
    option = dict(row_id=name, contract_id=name+'-C', side='call',
                  spot=100., strike=100., valuation_time=now,
                  exercise=dict(style='american', first_exercise=now,
                                expiry=now+31536000000000000),
                  volatility=.2, rate=.05, dividend_yield=0., cash_dividends=[])
    quote = dict(underlying=name, bid=10.44, ask=10.46,
                 quote_ns=str(now), spot_ns=str(now),
                 quantity=-2, multiplier=100, option=option)
    print(json.dumps(dict(schema='volt.exercise_ingest.v1',
          action='replace_underlying', generation=generation,
          underlying=name, as_of_ns=str(now),
          frame=dict(schema='volt.exercise_quote_frame.v1',
                     source='synthetic', currency='USD', quotes=[quote]))))
print(json.dumps(dict(schema='volt.exercise_ingest.v1', action='spot',
      generation=3, underlying='SIMX', as_of_ns=str(now+60000000000),
      spot=101., spot_ns=str(now+60000000000))))
PY
```

The final response contains `snapshot.hypercube.values` and per-node missing
counts, the combined portfolio totals, and `snapshot.underlyings` with each risk
cutoff and its age relative to publication. In the example, SIMX has current risk
while SIMY's risk is sixty seconds old. The actual Hypercube process stays alive
between generations and reuses its compiled graph.

`IncrementalExerciseRisk.submit(event)` is also importable from
`demo/exercise_incremental.py` for an application adapter. For an HTTP application, run:

```sh
python3 demo/exercise_incremental_service.py \
  --pricer target/release/exercise_pricer \
  --publisher target/release/exercise_cube --port 8820
```

`POST /api/options-risk/event` accepts each ingest event. The browser reads
`GET /api/options-risk/snapshot`; `/health` also checks native-publisher liveness.
The default listener is local. A nonlocal binding requires the bearer token from
`VOLT_RISK_TOKEN`, sent as `Authorization: Bearer …`. Configured `--allow-origin`
values supplement local browser origins. There is no vendor connector. The
existing `demo/exercise_risk_service.py` remains the complete-frame HTTP interface.

## Units and missing values

Risk starts per underlying unit. The native publisher applies signed
`quantity × multiplier` once. It emits position value, delta dollars for a unit
fractional stock move, the quadratic gamma contribution for a 1% stock move,
vega per one percentage point of volatility, calendar theta per day and rho per
basis point. The gamma cell includes the factor one-half. These are local
sensitivities; an exercise event can make a symmetric local approximation
unhelpful. See [pricing conventions](exercise-pricing.md) for the solver and
calendar-event definitions.

An unpriced held contract makes portfolio totals unavailable. An unavailable
calendar theta makes only the held portfolio theta unavailable. The associated
Hypercube cells are missing, with explicit missing counts. An unheld contract
with unavailable risk contributes zero to portfolio totals.

Each cell's source observation time is the oldest quote, current stock or
calibration-stock observation. The original nanosecond times and risk cutoffs
remain in the snapshot. Older input snapshots that omit `calibration_spot_ns`
receive conservative epoch-zero lineage and
`calibration_stock_lineage_complete=false`. The current worker and incremental
coordinator supply the field.

## Admission and recovery

A session uses one source label and currency, with at most 2,048 combined
contracts and 128 contracts in each dirty-underlying worker frame. JSONL messages
are capped at 8 MiB, ingest events at 2 MiB, retained input frames at 8 MiB,
and combined calibration caches at 8,192 entries. Cache eviction changes
recomputation cost while preserving accepted results. The worker's solver/grid admission remains in force.
Contract identifiers are globally unique; changing a live identifier's right,
strike or exercise schedule requires removing that underlying first. A cash
dividend crossing requires a replacement frame with the new calibration and
remaining unpaid dividends. An expiry crossing requires a replacement roster.

Validation, pricing and native publication complete before ingest state is
committed. A rejected event can be corrected and retried with the same generation.
Process restart requires replay of accepted events; this example has no durable
journal. Concurrent callers to the importable coordinator are serialized. Private
market observations and production credentials belong outside the source tree.

## Verification

```sh
cargo test --release -p volt-compute --features live-risk --test exercise_publication
VOLT_EXERCISE_PRICER="$PWD/target/release/exercise_pricer" \
VOLT_EXERCISE_CUBE="$PWD/target/release/exercise_cube" \
  python3 -m unittest discover -s demo -p 'test_exercise_incremental*.py' -v
```

These checks run the Rust exercise solver and persistent native Hypercube process,
including signed-unit aggregation, per-underlying dirty updates, calibration
retention, clock lineage, missing risk, removals and rejected-update recovery.
