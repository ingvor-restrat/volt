# American and Bermudan pricing

`volt-compute::exercise_pricing` prices one-underlying vanilla calls and puts
with European, Bermudan or American exercise, including a delayed American
start and dated cash dividends. It is a native CPU-f64 reference under flat
Black–Scholes volatility, interest and continuous dividend yield. It returns
prices, spatial delta/gamma, volatility and rate sensitivities, calendar theta,
and a structured implied-volatility result.

The module uses Volt's existing `ExerciseSchedule` and `TimestampNs` types.
American intervals include their endpoints. Bermudan opportunities retain
their exact integer dates; the solver inserts them into its rollback grid.
The American calculation approximates continuous exercise by its rollback LCP.
Business calendars and broker cutoffs enter through the caller's contract
and exercise schedule. Prices are per underlying unit. The caller applies signed contract quantity,
multiplier, currency conversion and settlement accounting.

## Run a request

Save this synthetic one-year American put request as `request.json`:

```json
{
  "schema": "volt.compute.exercise_pricing.v1",
  "grid": {"space_steps": 600, "time_steps": 600},
  "rows": [{
    "option": {
      "row_id": "one-put",
      "contract_id": "SIMX-20270101-P100",
      "side": "put",
      "spot": 100.0,
      "strike": 100.0,
      "valuation_time": 1767225600000000000,
      "exercise": {
        "style": "american",
        "first_exercise": 1767225600000000000,
        "expiry": 1798761600000000000
      },
      "volatility": 0.20,
      "rate": 0.05,
      "dividend_yield": 0.0,
      "cash_dividends": [{"ex_date": 1775001600000000000, "amount": 3.0}]
    }
  }]
}
```

From the repository root:

```sh
cargo run --release --locked -p volt-compute --bin exercise_pricer -- \
  --input request.json --output prices.json
```

Input can also arrive on stdin. The CLI caps input at 4 MiB and writes an
output file atomically. The batch preserves input order and identities;
unsupported rows receive their own error. Duplicate row IDs and conflicting
exercise/strike/right definitions under one contract ID are explicit errors.
Grid configuration, schema and work-admission failures reject the whole request.

The library entry points are `price`, `price_and_greeks`,
`implied_volatility` and `price_batch`. Request and result structs support
Serde. `risk.valuation.price` is the option mark, with the volatility actually
used alongside it. The existing European APIs retain their own conventions.

## Numerical calculation

Between cash payments the value solves

\[
V_t+\tfrac12\sigma^2 S^2V_{SS}+(r-q)SV_S-rV=0.
\]

A nonuniform sinh mesh places extra points near strike. Crank–Nicolson
rollback uses implicit half-steps after payoff and exercise/dividend kinks.
American exercise solves the linear complementarity problem
`min(A V - b, V - payoff) = 0` by policy iteration. Its residual uses positive
diagonal row scaling and value normalization; reaching the iteration cap
returns an error. Centered spatial coefficients are used where monotone;
upwind coefficients handle drift-dominated nodes. Diagnostics report the
actual mesh, event dates, upwind nodes, policy iterations and residual.

The outer spot limit starts at `max(spot, strike) * s_max_multiple`. A
four-standard-deviation conditional lognormal-tail heuristic and cash-dividend
reserve enlarge it when needed, up to a hard cap of 1,000 times that price
scale. Inputs requiring more space receive an explicit unsupported result.
Diagnostics report the actual limit. This heuristic guides admission;
mesh/domain refinement measures the observed numerical error. Its call value
uses a discounted linear deep-in-the-money approximation over the permitted
future exercise dates; its put value is zero. At zero spot, the put uses the
best permitted discounted strike. These finite-domain approximations need
space, time and outer-domain refinement for the intended inputs.

Default space and base time counts are 800 each. Grid limits, event limits,
solver iteration limits and a conservative solve-cell admission check bound
each request. They describe this reference calculation's planned work,
with the caller responsible for service-level time and memory allocation.

## Dividends and event order

Each cash dividend has a strictly future ex-date before expiry. Dates are
ordered and amounts positive. At the ex-date, rollback applies
`V(t-, S) = V(t+, max(S - D, 0))` with linear mesh interpolation. This is a
spot-jump model with a zero stock floor. The continuous yield `q` describes
additional proportional carry under the selected model.

When a Bermudan opportunity coincides with an ex-date, exercise takes place
immediately before the stock jump. Continuous American rights also cover
that instant. Valuation is after any already-paid dividend, so the caller
supplies the post-event spot and removes paid cash dividends. A dividend
exactly at valuation or expiry is rejected to keep this event convention
explicit. Settlement delay, stock delivery and funding cashflows belong to
the surrounding financial ledger.

## Greek units and conditioning

| Result | Meaning |
|---|---|
| `delta` | Price change per one spot currency unit |
| `gamma` | Price curvature per squared spot unit |
| `vega_per_unit_volatility` | Sensitivity to a 1.0 annual volatility fraction |
| `vega_per_vol_point` | Sensitivity to a 0.01 volatility increase |
| `rho_per_unit_rate` | Sensitivity to a 1.0 annual continuously compounded rate |
| `rho_per_bp` | Sensitivity to a 0.0001 rate increase |
| `theta_per_year`, `theta_per_day` | Calendar-forward price change, Actual/365 Fixed |

Delta and gamma interpolate local nonuniform quadratic derivative stencils.
Vega and rho use central complete repricing. Calendar theta advances valuation
while retaining all future exercise/dividend dates and the original future
time-grid nodes. Retaining those nodes avoids changing the numerical damping
pattern when an event happens to coincide with a base time step.

The theta bump shortens before the next event. Theta is unavailable exactly
on a Bermudan opportunity, where losing the current right can cause a price
jump, or when a safe bump of at least one second cannot fit. The other results
remain available. The actual bump and reason accompany the result. Central
bumps must remain in the supported parameter range.

Spatial resolution can also become inadequate close to expiry or at very low
volatility. Greek evaluation requires at least two local cells per approximate
spot diffusion width, except for a positive immediately exercisable intrinsic
value. Price-only results retain mesh diagnostics. Exercise frontiers and
cash-event kinks can make sensitivities depend on grid and bump size; compare
refinements before using the estimates for a new regime.

## Implied volatility

Adding `"market_price": 6.1` beside a row's `option` requests IV inversion.
The solver bisects a configured positive-volatility bracket, checks the
repricing error and bracket width, and estimates local vega. Greeks are
returned for a converged, identified IV. Results distinguish convergence,
weak identification, an outside-bracket quote and iteration exhaustion.
An exercise-value quote can identify a range of volatilities, so its result
retains a weak-identification state without assigning an arbitrary IV.
`volatility_bracket` records the search interval; it is a numerical interval,
with quote uncertainty assessed separately from the spread and market inputs.

## Independent controls and next applications

The test suite checks the European limit against analytic prices and Greeks,
exercise ordering and delayed starts against a separate 3,650-step CRR tree,
and cash-dividend call/put prices and Greeks against independently generated
QuantLib 1.40 controls. Both implementations are checked at multiple meshes.
Exact date insertion, before-dividend exercise, grid/domain refinements,
calendar-bump stability, weak IV, identity failures and the actual JSON CLI
are exercised. These are numerical controls for the declared model.

An exercise-aware hedge study additionally needs the holder's stopping choice
and the writer's dated payment/delivery and hedge decisions. Becker,
Cheridito and Jentzen's [Deep Optimal Stopping](https://jmlr.org/papers/v20/18-232.html)
learns finite-date stopping rules, including Bermudan max-call examples.
Their [Pricing and hedging American-style options with deep learning](https://arxiv.org/abs/1912.11060)
combines a candidate exercise policy, price bounds and dynamic hedging.
These are useful references for a later joint stopping/hedging study; this
module supplies a one-asset finite-difference reference for that work.

Method references: [Rannacher damping](https://doi.org/10.1007/BF01390130),
[Reisinger and Witte on policy iteration](https://arxiv.org/abs/1012.4976), and
[QuantLib's spot cash-dividend convention](https://github.com/lballabio/QuantLib/blob/v1.40/ql/methods/finitedifferences/utilities/fdmdividendhandler.cpp).
