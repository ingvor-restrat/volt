# Exercise-aware option risk in an application

The worker connects observed option quotes and dated stock prices to Volt's
American/Bermudan pricing reference. It accepts a cross-section, calculates
implied volatility and Greeks, and publishes the completed snapshot for a
browser, terminal or application widget. The numerical calculation is performed
by the Rust `exercise_pricer`; the HTTP adapter uses Python's standard library.

```sh
cargo build --release --locked -p volt-compute --bin exercise_pricer
python3 demo/exercise_risk_service.py --pricer target/release/exercise_pricer
python3 demo/options_risk_top.py --url http://127.0.0.1:8820
```

Submit JSON to `POST /api/options-risk/frame`. Read the latest completed result
from `GET /api/options-risk/snapshot`. The worker permits one frame calculation
at a time and retains the previous completed snapshot until the next is ready.
The default 400×400 grid is a presentation setting. Refine the mesh against the
required numerical accuracy before choosing a production setting.

```json
{
  "schema": "volt.exercise_quote_frame.v1",
  "generation": 1,
  "source": "historical",
  "as_of_ns": "1767225600000000000",
  "currency": "USD",
  "assumptions": {"carry": "flat r=5%, q=0; empty cash-dividend schedule"},
  "quotes": [{
    "underlying": "XYZ", "bid": 10.44, "ask": 10.46,
    "quote_ns": "1767225600000000000", "spot_ns": "1767225600000000000",
    "quantity": -2, "multiplier": 100,
    "option": {
      "row_id": "xyz-call", "contract_id": "XYZ-20270101-C100",
      "side": "call", "spot": 100.0, "strike": 100.0,
      "valuation_time": 1767225600000000000,
      "exercise": {"style": "american", "first_exercise": 1767225600000000000,
                   "expiry": 1798761600000000000},
      "volatility": 0.2, "rate": 0.05, "dividend_yield": 0.0,
      "cash_dividends": []
    }
  }]
}
```

Save this as `frame.json`, then submit it:

```sh
curl -fsS -H 'Content-Type: application/json' \
  --data-binary @frame.json http://127.0.0.1:8820/api/options-risk/frame
```

The `volatility` input initializes IV inversion. Each quote's midpoint and paired
stock observation determine its calibrated volatility. Supply actual carry and
cash-dividend assumptions in the option. These are explicit model inputs.
`source` identifies `historical`, `live` or `synthetic` observations. Nanosecond
strings preserve exact source timestamps in browser-facing results.

When a stock observation advances while an option quote remains unchanged,
retain that quote's original `calibration_spot`, `calibration_spot_ns` and
`calibration_ns`. Update `option.spot`, `spot_ns`, `option.valuation_time` and
`as_of_ns` to the new common valuation time. The worker reuses the paired-quote
IV and calculates current Greeks. When the option quote changes, supply its new
paired calibration inputs. All observations must be available by their stated
time. A generation must increase for each accepted frame.

Contract Greeks are per underlying unit. Portfolio calculations apply signed
quantity × multiplier once. `delta_dollars` is delta × stock price × units;
`gamma_1pct` is half gamma × (1% of stock price)² × units. Vega is per one
percentage point of volatility, theta per calendar day, rho per basis point.
A position with unavailable risk makes aggregate portfolio risk unavailable.
Theta can be unavailable at an exact Bermudan exercise opportunity.

The HTTP adapter caps frames at 128 contracts and 2 MiB, caches up to 4,096 quote
calibrations, and splits pricing requests within the Rust work allowance.
`--workers 1..8` controls CPU subprocess concurrency (default four).
It binds to loopback by default. A nonlocal binding requires `VOLT_RISK_TOKEN`;
clients send it as `Authorization: Bearer ...`. Browser origins are local by
default; repeat `--allow-origin` for explicitly configured clients.

For a saved replay:

```sh
python3 demo/options_risk_top.py --input snapshots.jsonl --interval 0.4
VOLT_EXERCISE_PRICER=target/release/exercise_pricer \
  python3 -m unittest discover -s demo -p 'test_exercise*.py'
```

Historical quote files and producer adapters may live outside the public source
checkout. See [pricing conventions and independent controls](exercise-pricing.md)
and the [application papers](latex/compute/README.md).
