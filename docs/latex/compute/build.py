#!/usr/bin/env python3
"""Build Volt application papers from the source checkout."""
from pathlib import Path
import argparse
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
PAPERS = ("volt_option_learning_primer", "volt_gpu_pricing_aad", "volt_deep_hedging",
          "volt_hypercube_live_risk", "volt_volatility_strategies")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", choices=PAPERS)
    args = parser.parse_args()
    build = HERE / "build"
    build.mkdir(exist_ok=True)
    for stem in (args.paper,) if args.paper else PAPERS:
        with (build / (stem + "-build.txt")).open("w") as output:
            subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                            "-outdir=" + str(build), stem + ".tex"], cwd=HERE, check=True,
                           stdout=output, stderr=subprocess.STDOUT)
        log = (build / (stem + ".log")).read_text(errors="replace")
        if any(w in log for w in ["Overfull", "undefined references", "undefined citations"]):
            raise RuntimeError("Resolve layout/reference warnings in " + stem)
        pdf = HERE / (stem + ".pdf")
        shutil.copy2(build / pdf.name, pdf)
        print(pdf.relative_to(HERE.parents[2]))

if __name__ == "__main__":
    main()
