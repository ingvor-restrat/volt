# Volt live risk displays

The browser and terminal presentations consume the same deterministic Rust
simulation snapshots. JavaScript formats and charts the browser snapshot; it
does not reproduce the pricing or portfolio logic.

## Browser dashboard

![The live Volt browser risk dashboard](risk-web.gif)

Run the local server and open
[http://127.0.0.1:8081](http://127.0.0.1:8081):

```bash
cargo run --example risk_server
```

The browser uses semantic tables, tabular numerals, and bounded chart panels
so changing values remain aligned. Snapshots stream from Rust over
Server-Sent Events.

## Terminal monitor

![A live Volt options risk dashboard](risk.gif)

`risk` keeps the compact, keyboard-friendly feel of `top`:

```bash
cargo run --example risk
```

It constructs a cash-settled European *iron condor* on a fake index:

```text
+1 SIMX 90 put
-1 SIMX 95 put
-1 SIMX 105 call
+1 SIMX 110 call
```

Volt validates the four exact contracts, canonicalizes their signed
positions, hashes the portfolio, and derives its terminal hinge payoff. The
example then advances a correlated spot/volatility simulation and refreshes:

- each leg's theoretical mark, implied volatility, delta, and P&L;
- portfolio delta, gamma, vega, theta, and mark-to-model P&L;
- expiration P&L across spot shocks; and
- a live spot-by-volatility shock grid.

The spot process is geometric Brownian motion with log volatility following a
mean-reverting process and a negative spot/volatility correlation. The
Black–Scholes marks are an educational adapter in the example—not a claim that
Volt supplies a production pricing model or market quote.

Press `q` or Escape to quit. A bounded, ANSI-free run is useful in logs:

```bash
cargo run -q --example risk -- --ticks 1 --no-color
```

This division is visible in the code:

```text
Volt:      exact contracts -> canonical portfolio -> terminal payoff
example:   synthetic path  -> teaching marks      -> live risk display
```

## Rebuild the recordings

The terminal GIF comes from deterministic frames emitted by `risk`:

```bash
python3 docs/markup/record.py
```

The browser GIF comes from the live local server and headless Chrome:

```bash
python3 docs/markup/record_web.py
```

The recorders need Python and Pillow; the browser recorder also needs Chrome
and `websocket-client`. Both use a fixed seed and invoke the real examples, so
the animations are reproducible rather than hand-built mockups.
