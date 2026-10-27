#!/usr/bin/env python3
"""Generate a Bermudan liability, train its writer, select and evaluate a hedge.

Build: cargo build --release -p volt-compute --bin bermudan_market --bin native_learner
Run: python3 examples/bermudan_learning.py --output /tmp/volt-bermudan-example
Python standard library only. Synthetic one-share put; monthly hedging,
quarterly exercise, 5% funding, 20% GBM volatility, declared stock trading fees.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess


class Worker:
    def __init__(self, binary):
        env = dict(os.environ, RAYON_NUM_THREADS='1')
        self.process = subprocess.Popen([str(binary)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, env=env)
        self.serial = 0

    def call(self, op, **fields):
        self.serial += 1
        message = json.dumps(dict(id=str(self.serial), op=op, **fields), allow_nan=False)
        if len(message.encode()) >= 16 * 1024 * 1024:
            raise ValueError('Request exceeds the worker message allowance')
        self.process.stdin.write(message + '\n')
        self.process.stdin.flush()
        reply = json.loads(self.process.stdout.readline(16 * 1024 * 1024 + 1))
        if reply.get('id') != str(self.serial) or not reply.get('ok'):
            raise RuntimeError(reply)
        return reply['result']

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def generate(binary, directory, split, count, fee):
    start, year = 1767225600000000000, 365 * 86400 * 10**9
    dates = [start + year * n // 12 for n in range(13)]
    config = dict(
        option=dict(row_id='put', contract_id='SYNTH-PUT-100', side='put', spot=100,
                    strike=100, valuation_time=start, volatility=.2, rate=.05,
                    dividend_yield=0, cash_dividends=[],
                    exercise=dict(style='bermudan', opportunities=[dates[n] for n in (3, 6, 9, 12)])),
        grid=dict(space_steps=600, time_steps=600), dates=dates, seed=731941,
        physical_drift=.05, quantity=1, multiplier=1,
        trading_cost=fee, liquidation_cost=fee,
        max_working_bytes=64 * 1024**2, max_holder_scalar_values=100000)
    first = dict(train=0, validation=count, test=2 * count)[split]
    request = dict(config=config, range=dict(split=split, first_path=first, paths=count))
    source, target = directory / (split + '-request.json'), directory / (split + '-episodes.json')
    source.write_text(json.dumps(request) + '\n')
    subprocess.run([str(binary), '--input', str(source), '--output', str(target)], check=True)
    return json.loads(target.read_text())


def upload(worker, sample, split):
    batch = sample['batch']
    worker.call('register_dataset', dataset_id=batch['provenance']['dataset_id'],
                partitions=[dict(split=name, first_path=i * batch['paths'], paths=batch['paths'])
                            for i, name in enumerate(('train', 'validation', 'test'))])
    worker.call('upload', batch_id=split, split=split, batch=batch)


def run(args):
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    generator, binary = args.generator.resolve(), args.worker.resolve()
    train = generate(generator, directory, 'train', args.paths, args.fee)
    validation = generate(generator, directory, 'validation', args.paths, args.fee)
    economics = train['economics']
    config = dict(target='cpu', precision='f64', seed=27003, learning_rate=.001,
                  objective=dict(kind='squared_loss'), premium=economics['premium'],
                  trading_costs=[args.fee], liquidation_costs=[args.fee],
                  cash_growth=economics['cash_growth'], income_growth=economics['income_growth'],
                  budget=dict(compute_bytes=2 * 1024**3, host_working_bytes=2 * 1024**3),
                  policy=dict(time_steps=12, feature_count=2, hedge_count=1, hidden_width=16,
                              max_batch_rows=args.paths, include_previous_holdings=True,
                              feature_affine=dict(offsets=[0, .5], scales=[4, 2]),
                              adam=dict(beta1=.9, beta2=.999, epsilon=1e-8),
                              batch_norm=dict(enabled=False, epsilon=1e-5, momentum=.1)))
    worker = Worker(binary)
    try:
        worker.call('configure', config=config)
        upload(worker, train, 'train')
        upload(worker, validation, 'validation')
        before = worker.call('evaluate', batch_id='validation')['loss']
        best_loss, best = before, worker.call('checkpoint')
        trace = [dict(update=0, validation_mse=before)]
        for step in range(1, args.updates + 1):
            worker.call('update', batch_id='train', expected_version=step - 1)
            if step % 25 == 0 or step == args.updates:
                score = worker.call('evaluate', batch_id='validation')['loss']
                trace.append(dict(update=step, validation_mse=score))
                if score < best_loss:
                    best_loss, best = score, worker.call('checkpoint')
    finally:
        worker.close()
    (directory / 'selected-checkpoint.json').write_text(json.dumps(best) + '\n')
    # Fresh confirmation is generated only after validation selects the checkpoint.
    test = generate(generator, directory, 'test', args.paths, args.fee)
    worker = Worker(binary)
    try:
        worker.call('restore', checkpoint=best)
        upload(worker, test, 'test')
        assessment = worker.call('evaluate', batch_id='test')
        action = worker.call('infer', step=0, paths=1, features=test['batch']['features'][:2],
                             previous_holdings=[0])
    finally:
        worker.close()
    report = dict(synthetic=True, scope='Bounded application example with validation selection and fresh paths',
                  paths_per_split=args.paths, updates=args.updates, stock_fee_fraction=args.fee,
                  premium=economics['premium'], units='one put on one underlying share',
                  exercise='quarterly', hedge_frequency='monthly', selected_version=assessment['version'],
                  validation_trace=trace, selected_validation_mse=best_loss,
                  fresh_learned_mse=assessment['loss'],
                  fresh_numerical_delta_mse=sum(x*x for x in test['numerical_pnl']) / args.paths,
                  pnl_reference=assessment['pnl_reference'], initial_action=action,
                  early_settlements=sum(t < 12 for t in test['batch']['exercise']['settlement_step']),
                  physical_sha256=test['physical_sha256'])
    if not all(math.isfinite(report[k]) for k in ('fresh_learned_mse', 'fresh_numerical_delta_mse')):
        raise ValueError('Nonfinite assessment')
    (directory / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generator', type=Path, default=Path('target/release/bermudan_market'))
    parser.add_argument('--worker', type=Path, default=Path('target/release/native_learner'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--paths', type=int, default=64)
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--fee', type=float, default=.001)
    options = parser.parse_args()
    if not 2 <= options.paths <= 256 or not 1 <= options.updates <= 3000 or not 0 <= options.fee <= .01:
        parser.error('Use paths2..256, updates1..3000 and fee0...01')
    run(options)
