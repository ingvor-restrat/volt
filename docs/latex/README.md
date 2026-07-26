# Long-form papers

These papers carry the derivatives research into the public Volt vocabulary
and API boundary:

- [Volt: Financial Semantics for Option Portfolio Graphs](volt_options_graph_logic.pdf)
  ([source](volt_options_graph_logic.tex)) — contract, payoff, lifecycle,
  compiler, and graph identities for learning and optimization;
- [Volt: Searching the Option Strategy Space](volt_strategy_discovery.pdf)
  ([source](volt_strategy_discovery.tex)) — a formal protocol for
  direct search over listed option positions, with pathwise evaluation,
  Deep Sets as the first invariant model, and GNN/QUBO extensions; and
- [Volt: Sequential Control of Option Portfolios](volt_sequential_control.pdf)
  ([source](volt_sequential_control.tex)) — state, actions, rewards, the
  reinforcement-learning loop, and rollout allocation over the deterministic
  runtime.

The corresponding PDFs are rebuilt from these plain LaTeX article sources.

Build from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  volt_options_graph_logic.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  volt_strategy_discovery.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  volt_sequential_control.tex
```

`latexmk -c` removes intermediate files without deleting the PDFs or sources.
