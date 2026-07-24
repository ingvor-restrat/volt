# Long-form papers

These papers carry the derivatives research into the public Volt vocabulary
and API boundary:

- [Volt: Options Graph Logic](volt_options_graph_logic.pdf)
  ([source](volt_options_graph_logic.tex)) — the options graph as a stable
  representation for learning and optimization, grounded in implemented
  option semantics, canonical identity, runtime, scenario, and graph
  contracts;
- [Volt: Strategy Search](volt_strategy_discovery.pdf)
  ([source](volt_strategy_discovery.tex)) — a proposed supervised discovery protocol
  grounded in Volt’s public synthetic fixtures; and
- [Volt: Portfolio Control](volt_sequential_control.pdf)
  ([source](volt_sequential_control.tex)) — a proposed sequential-control layer over the
  deterministic runtime.

The corresponding PDFs are rebuilt from these sources. The local class uses
the Palatino/Helvetica typefaces from `BRCentralRiskBook.tex` while retaining
the StrategyNet cover, hierarchy, spacing, and publication identity.

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
