# Long-form papers

These papers carry the derivatives research into the public Volt vocabulary
and API boundary:

- [Volt: Financial Semantics for Option Portfolio Graphs](volt_options_graph_logic.pdf)
  ([source](volt_options_graph_logic.tex)) — contract, payoff, lifecycle,
  compiler, and graph identities for learning and optimization;
- [Volt: Searching the Option Strategy Space](volt_strategy_discovery.pdf)
  ([source](volt_strategy_discovery.tex)) — a formal protocol for
  direct search over listed option positions, with pathwise evaluation,
  fixed-horizon hedge-policy analytics, parameter-matched Deep Sets and typed
  graph models, and a QUBO extension; and
- [Volt: Sequential Control of Option Portfolios](volt_sequential_control.pdf)
  ([source](volt_sequential_control.tex)) — state, actions, rewards, the
  reinforcement-learning loop, the distinction between set aggregation and
  typed message passing, a precommitted downside-aware checkpoint objective,
  multi-expiry runtime graph, generated control and compute benchmarks, and
  rollout allocation over the deterministic runtime; and
- [Option Portfolio Selection Around Two Event Shocks](volt_empirical_event_studies.pdf)
  ([source](volt_empirical_event_studies.tex)) — SPY tariff and USO conflict
  studies with dated market/option/strategy rollouts, directional probability
  menus, direction-neutral target-volatility scans, American lifecycle
  valuation, supervised model tests, and compiler-aware timing.

The corresponding PDFs are rebuilt from these plain LaTeX article sources.

Build from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  volt_options_graph_logic.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  volt_strategy_discovery.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  volt_sequential_control.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  volt_empirical_event_studies.tex
```

`latexmk -c` removes intermediate files without deleting the PDFs or sources.

## Computation and hedging applications

The [compute collection](compute/README.md) contains canonical sources and PDFs
for pricing/AAD, Deep Hedging, live option analytics and volatility strategies.
Build them with `python docs/latex/compute/build.py` from the repository root.
