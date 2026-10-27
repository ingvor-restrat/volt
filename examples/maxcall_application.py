#!/usr/bin/env python3
"""Fit a multi-asset holder and train its complete writer account (stdlib only).

Build:
  cargo build --release -p volt-compute --bin maxcall_learning --bin maxcall_episodes --bin native_learner
Run:
  python3 examples/maxcall_application.py --output /tmp/volt-maxcall-example

Synthetic five-asset max-call, three years and nine four-month exercise dates.
Small independent training/validation/test cohorts demonstrate the application.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from bermudan_learning import Worker


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    learner, generator, worker_binary = args.learner.resolve(), args.generator.resolve(), args.worker.resolve()
    env = dict(os.environ, RAYON_NUM_THREADS='1')

    def native(binary, *arguments):
        subprocess.run([str(binary), *map(str, arguments)], check=True, env=env)

    defaults = output / 'defaults.json'
    native(learner, 'example', '--output', defaults)
    training = json.loads(defaults.read_text())['training']
    training.update(target='cpu', precision='f64', hidden_width=16, batch_rows=256,
                    training_paths=4096, validation_paths=1024, validation_every=50,
                    final_date_schedule=[dict(updates=args.holder_updates, learning_rate=.01)],
                    earlier_date_schedule=[dict(updates=args.holder_updates, learning_rate=.01)],
                    initialization_seed=730101, training_seed=730103, validation_seed=730107)
    save(output / 'holder-training.json', training)
    holder_path = output / 'holder.json'
    native(learner, 'train', '--config', output / 'holder-training.json',
           '--checkpoint', holder_path, '--report', output / 'holder-training-report.json')
    holder = json.loads(holder_path.read_text())
    holder_hash = hashlib.sha256(holder_path.read_bytes()).hexdigest()
    market = holder['config']['market']
    assets, steps = len(market['spots']), len(market['times']) - 1
    premium = holder['initial_training_continuation'] if args.premium is None else args.premium

    def episodes(split):
        first = dict(train=0, validation=args.paths, test=2*args.paths)[split]
        request = dict(paths=args.paths, first_path=first, seed=730109, namespace=901,
                       target='cpu', precision='f64', inference_batch_rows=256,
                       inference_working_bytes=512*1024**2,
                       economics=dict(quantity=1, multiplier=1, premium=premium,
                                      trading_costs=[args.fee]*assets, liquidation_costs=[args.fee]*assets,
                                      holder_id='example-selected-holder', holder_sha256=holder_hash,
                                      max_working_bytes=64*1024**2))
        request_path, target = output / (split+'-request.json'), output / (split+'-episodes.json')
        save(request_path, request)
        native(generator, '--checkpoint', holder_path, '--request', request_path, '--output', target)
        return json.loads(target.read_text())

    train, validation = episodes('train'), episodes('validation')
    config = dict(target='cpu', precision='f64', seed=730111, learning_rate=.001,
                  objective=dict(kind='squared_loss'), premium=premium,
                  trading_costs=[args.fee]*assets, liquidation_costs=[args.fee]*assets,
                  cash_growth=train['cash_growth'], income_growth=train['income_growth'],
                  policy=dict(time_steps=steps, feature_count=assets+1, hedge_count=assets,
                              hidden_width=16, max_batch_rows=args.paths, include_previous_holdings=True,
                              batch_norm=dict(enabled=False, epsilon=1e-5, momentum=.1),
                              adam=dict(beta1=.9, beta2=.999, epsilon=1e-8),
                              feature_affine=dict(offsets=[100.]*assets+[0.], scales=[.02]*(assets+1)),
                              action_scales=[1.]*assets),
                  budget=dict(compute_bytes=4*1024**3, host_working_bytes=4*1024**3))
    save(output / 'writer-config.json', config)

    def upload(w, data, split):
        w.call('register_dataset', dataset_id=data['batch']['provenance']['dataset_id'],
               partitions=[dict(split=name, first_path=i*args.paths, paths=args.paths)
                           for i, name in enumerate(('train', 'validation', 'test'))])
        w.call('upload', batch_id=split, split=split, batch=data['batch'])

    writer = Worker(worker_binary)
    try:
        writer.call('configure', config=config)
        upload(writer, train, 'train'); upload(writer, validation, 'validation')
        score = writer.call('evaluate', batch_id='validation')['loss']
        best, best_score = writer.call('checkpoint'), score
        trace = [dict(update=0, validation_mse=score)]
        for i in range(args.writer_updates):
            writer.call('update', batch_id='train', expected_version=i)
            if (i+1)%25==0 or i+1==args.writer_updates:
                score = writer.call('evaluate', batch_id='validation')['loss']
                trace.append(dict(update=i+1, validation_mse=score))
                if score < best_score:
                    best, best_score = writer.call('checkpoint'), score
    finally:
        writer.close()
    save(output / 'writer-selected.json', best)
    # Holder and writer are selected before opening this independent path range.
    test = episodes('test')
    writer = Worker(worker_binary)
    try:
        writer.call('restore', checkpoint=best)
        upload(writer, test, 'test')
        result = writer.call('evaluate', batch_id='test')
    finally:
        writer.close()
    payments = test['batch']['exercise']['settlement_amounts']
    unhedged = sum((premium-x)**2 for x in payments)/len(payments)
    summary = dict(synthetic=True, scope='Small complete multi-asset application example',
                   assets=assets, years=market['times'][-1], exercise_intervals=steps,
                   writer_paths_per_split=args.paths, premium=premium,
                   premium_basis='holder training estimate' if args.premium is None else 'explicit supplied capital',
                   stock_fee_fraction=args.fee, account_basis=test['account_basis'],
                   holder_sha256=holder_hash, selected_writer_version=result['version'],
                   validation_trace=trace, fresh_writer_mse=result['loss'], fresh_unhedged_mse=unhedged,
                   fresh_mean_discounted_pnl=sum(result['pnl'])/len(result['pnl']),
                   early_settlements=sum(t<steps for t in test['batch']['exercise']['settlement_step']))
    assert math.isfinite(summary['fresh_writer_mse']) and math.isfinite(unhedged)
    save(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--learner', type=Path, default=Path('target/release/maxcall_learning'))
    parser.add_argument('--generator', type=Path, default=Path('target/release/maxcall_episodes'))
    parser.add_argument('--worker', type=Path, default=Path('target/release/native_learner'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--paths', type=int, default=64)
    parser.add_argument('--holder-updates', type=int, default=100)
    parser.add_argument('--writer-updates', type=int, default=200)
    parser.add_argument('--premium', type=float)
    parser.add_argument('--fee', type=float, default=0.)
    args=parser.parse_args()
    if not 2<=args.paths<=256 or not 1<=args.holder_updates<=1000 or not 1<=args.writer_updates<=3000 or not 0<=args.fee<=.01:
        parser.error('Require paths2..256, holder-updates1..1000, writer-updates1..3000 and fee0...01')
    main(args)
