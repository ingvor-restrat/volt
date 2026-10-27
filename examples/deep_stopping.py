#!/usr/bin/env python3
"""Native Bermudan holder, independent price estimates, and 2020 interval hedge.

Build: cargo build --release -p volt-compute --bin deep_stopping
Run: python3 examples/deep_stopping.py --output /tmp/volt-stopping-example
Python standard library only. Synthetic monthly one-share GBM put.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run(args):
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    binary = args.worker.resolve()
    environment = dict(os.environ, RAYON_NUM_THREADS='1')

    def native(*command):
        subprocess.run([str(binary), *map(str, command)], check=True, env=environment)

    defaults = directory / 'defaults.json'
    native('example', '--output', defaults)
    settings = json.loads(defaults.read_text())
    training = settings['training']
    training.update(width=8, training_paths=4096, validation_paths=1024,
                    batch_rows=256, updates_per_date=args.updates, validation_every=50)
    train_path, holder_path = directory / 'training.json', directory / 'holder.json'
    save(train_path, training)
    native('train', '--config', train_path, '--checkpoint', holder_path)
    holder = json.loads(holder_path.read_text())
    hedge = dict(training, training_seed=730043, validation_seed=730047,
                 initialization_seed=730049)
    hedge_config, hedge_path = directory / 'hedge-training.json', directory / 'hedge.json'
    save(hedge_config, hedge)
    native('hedge-train', '--holder', holder_path, '--config', hedge_config,
           '--initial-price', holder['initial_training_continuation'], '--checkpoint', hedge_path)
    # Both models are selected before either independent confirmation is opened.
    evaluation = settings['evaluation']
    evaluation.update(lower_paths=8192, upper_paths=256, inner_paths=16)
    eval_path, bounds_path = directory / 'evaluation.json', directory / 'bounds.json'
    save(eval_path, evaluation)
    native('evaluate', '--checkpoint', holder_path, '--config', eval_path, '--output', bounds_path)
    hedge_result_path = directory / 'hedge-result.json'
    native('hedge-evaluate', '--checkpoint', hedge_path, '--paths', 8192,
           '--seed', 730053, '--output', hedge_result_path)
    bounds = json.loads(bounds_path.read_text())
    result = json.loads(hedge_result_path.read_text())
    summary = dict(synthetic=True, exercise_dates='12 monthly dates over one year',
                   lower_expectation_estimate=bounds['lower'],
                   upper_expectation_estimate=bounds['upper'],
                   asymptotic_price_enclosure=bounds['asymptotic_joint_95_enclosure'],
                   interval_hedge_mse=result['mse'], zero_hedge_mse=result['zero_hedge_mse'],
                   discounted_mean_error=result['discounted_error']['mean'],
                   initial_price=holder['initial_training_continuation'],
                   basis='Training-derived premium; independent confirmation; one rebalance per exercise interval',
                   estimator='Alternate conditionally centered proxy dual; Monte Carlo expectation estimates')
    save(directory / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, default=Path('target/release/deep_stopping'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=100)
    arguments = parser.parse_args()
    if not 1 <= arguments.updates <= 1000:
        parser.error('--updates must be between 1 and 1000')
    run(arguments)
