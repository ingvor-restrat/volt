# Volt computation and hedging papers

These are the canonical LaTeX sources and compiled papers for Volt's numerical
applications. Each paper develops the financial problem, explains the supplied
interfaces and reports the corresponding completed experiments.

| Paper | Source | PDF |
|---|---|---|
| Options, Exercise and Learned Hedging: A Practical Primer | [LaTeX](volt_option_learning_primer.tex) | [PDF](volt_option_learning_primer.pdf) |
| Pricing and Hedging a Multi-Asset Option with Volt | [LaTeX](volt_multi_asset_hedging.tex) | [PDF](volt_multi_asset_hedging.pdf) |
| Pricing and Risk Analytics with Volt | [LaTeX](volt_gpu_pricing_aad.tex) | [PDF](volt_gpu_pricing_aad.pdf) |
| Learning to Hedge in Volt | [LaTeX](volt_deep_hedging.tex) | [PDF](volt_deep_hedging.pdf) |
| Live Option Analytics with Volt and Hypercube | [LaTeX](volt_hypercube_live_risk.tex) | [PDF](volt_hypercube_live_risk.pdf) |
| European Straddles | [LaTeX](volt_volatility_strategies.tex) | [PDF](volt_volatility_strategies.pdf) |

Start with the primer for contract terms, probability, financing, exercise,
Greeks, learning objectives, AD and the application architecture. It includes
worked examples, completed results, review questions and a reading route.

Build all six from the repository root with TeX Live and latexmk installed:

```sh
python docs/latex/compute/build.py
```

Use `--paper volt_deep_hedging` to select one. Intermediates stay in the ignored
`build/` directory; finished PDFs are placed beside their sources. This directory
uses a scoped 8pt journal template, with the main text after the contents and
references on a separate final page.

The companion [Jormungandr tutorial](https://github.com/ingvor-restrat/jormungandr/tree/main/docs/latex/hedging)
explains the continuous-action control problem and the complete native learner API.
Application documentation is in [the compute guide](../../../crates/volt-compute/README.md)
and [the native learner guide](../../../crates/volt-compute/docs-native-learner.md).
