#!/usr/bin/env python3
"""Run a bounded synthetic joint-option/reset-LSTM workflow (Python stdlib only).

Build: cargo build -p volt-compute --release --bin joint_option_scenarios --bin native_learner
Run: python3 examples/joint_option_learning.py --output /tmp/volt-joint-demo
The default toy innovation table is illustrative, not a historical calibration.
"""
import argparse
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile

LIMIT = 16 * 1024 * 1024


def float32_token(value):
    bits = struct.pack('<f', float(value))
    number = struct.unpack('<f', bits)[0]
    if not math.isfinite(number):
        raise ValueError('Scenario value is not finite in f32')
    token = format(number, '.9g')
    if struct.pack('<f', float(token)) != bits:
        raise ValueError('f32 transport failed round-trip validation')
    return token


def episode_json(batch):
    fields = ['"paths":' + str(batch['paths']),
              '"provenance":' + json.dumps(batch['provenance'], separators=(',', ':'))]
    for key in ['marks', 'features', 'payoffs', 'force_trade_signal']:
        if batch.get(key) is not None:
            fields.append(json.dumps(key) + ':[' + ','.join(map(float32_token, batch[key])) + ']')
    return '{' + ','.join(fields) + '}'


class Worker:
    def __init__(self, binary):
        self.serial = 0
        self.process = subprocess.Popen([str(binary)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=None)

    def request(self, body=None, raw=None):
        self.serial += 1
        message = (json.dumps({**body, 'id': str(self.serial)}, separators=(',', ':'))
                   if raw is None else '{"id":"' + str(self.serial) + '",' + raw[1:])
        data = message.encode()
        if len(data) > LIMIT:
            raise ValueError(f'Request has {len(data)} bytes, exceeding 16 MiB')
        self.process.stdin.write(data + b'\n')
        self.process.stdin.flush()
        reply = json.loads(self.process.stdout.readline())
        if reply.get('id') != str(self.serial) or not reply.get('ok'):
            raise RuntimeError(reply)
        return reply['result']

    def close(self):
        if self.process.poll() is None:
            try:
                self.request({'op': 'shutdown'})
            finally:
                self.process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generator', type=Path, default=Path('target/release/joint_option_scenarios'))
    parser.add_argument('--worker', type=Path, default=Path('target/release/native_learner'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--paths', type=int, default=8)
    parser.add_argument('--days', type=int, default=3)
    parser.add_argument('--width', type=int, default=8)
    parser.add_argument('--updates', type=int, default=5)
    parser.add_argument('--active-dropout', action='store_true', help='Explicit native training dropout0.5; default matches an unfed TF1 learning phase')
    args = parser.parse_args()
    if not (1 <= args.paths <= 1000 and 2 <= args.days <= 63 and 1 <= args.width <= 56 and 1 <= args.updates <= 200):
        parser.error('Require paths1..1000,days2..63,width1..56,updates1..200')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'result.json').exists():
        raise ValueError('Refusing to replace an existing example result')
    generator, binary = args.generator.resolve(), args.worker.resolve()
    recipe = json.loads(subprocess.check_output([str(generator), '--example']))
    recipe['config'].update(steps=args.days, straddle_days=args.days,
                            hedge_call_days=args.days + 21, terminal_costs=[0, .02],
                            max_working_bytes=512 * 1024 * 1024)
    recipe['partitions'] = [{'split': 'train', 'first_path': 0, 'paths': args.paths}]
    recipe['range'] = recipe['partitions'][0]
    with tempfile.TemporaryDirectory(prefix='volt-joint-example-') as directory:
        request, scenarios = Path(directory) / 'request.json', Path(directory) / 'scenarios.json'
        request.write_text(json.dumps(recipe))
        subprocess.run([str(generator), '--input', str(request), '--output', str(scenarios)], check=True)
        sample = json.loads(scenarios.read_text())
    batch, terms = sample['batch'], sample['economics']
    config = {
        'target': 'cpu', 'precision': 'f32', 'policy_engine': 'volt_shared_lstm_reset',
        'seed': 730019, 'learning_rate': .0005, 'objective': {'kind': 'squared_loss'},
        'tracking_penalty': {'liability_feature': 13, 'threshold_multiple': 1, 'coefficient': 1},
        'premium': terms['premium'], 'trading_costs': terms['trading_costs'],
        'liquidation_costs': terms['liquidation_costs'], 'cash_growth': terms['cash_growth'],
        'income_growth': terms['income_growth'],
        'policy': {
            'shared_lstm_reset': {'dropout_probability': .5 if args.active_dropout else 0,
                                  'recurrent_activation': 'hard_sigmoid', 'dropout_seed': 730021},
            'small_trade_gate': {'sharpness': 1000, 'force_threshold': .1,
                                 'initial_trade_threshold': .001, 'inference_sharpen': True},
            'include_wealth': True, 'time_steps': args.days, 'feature_count': 17, 'hedge_count': 2,
            'include_previous_holdings': True, 'hidden_width': args.width,
            'max_batch_rows': args.paths,
            'batch_norm': {'enabled': False, 'epsilon': 1e-5, 'momentum': .1},
            'adam': {'beta1': .9, 'beta2': .999, 'epsilon': 1e-8,
                     'epsilon_convention': 'tensorflow_uncorrected'},
            'feature_affine': None, 'action_scales': None, 'previous_holding_scales': None},
        'budget': {'compute_bytes': 32 * 1024**3, 'host_working_bytes': 32 * 1024**3,
                   'compute_reserve_bytes': 32 * 1024**2, 'headroom_percent': 20,
                   'max_batch_rows': args.paths}}
    worker = Worker(binary)
    try:
        worker.request({'op': 'configure', 'config': config})
        worker.request({'op': 'register_dataset', 'dataset_id': batch['provenance']['dataset_id'],
                        'partitions': [{'split': 'train', 'first_path': '0', 'paths': str(args.paths)}]})
        worker.request(raw='{"op":"upload","batch_id":"demo","split":"train","batch":' + episode_json(batch) + '}')
        before = worker.request({'op': 'evaluate', 'batch_id': 'demo', 'sharpen_gate': False})
        updates = [worker.request({'op': 'update', 'batch_id': 'demo', 'expected_version': i})
                   for i in range(args.updates)]
        after = worker.request({'op': 'evaluate', 'batch_id': 'demo', 'sharpen_gate': False})
        inference = worker.request({'op': 'infer', 'step': 0, 'paths': 1,
                                    'features': batch['features'][:17], 'previous_holdings': [0, 0],
                                    'wealth': [terms['premium']], 'sharpen_gate': True})
        checkpoint = worker.request({'op': 'checkpoint'})
    finally:
        worker.close()
    (output / 'checkpoint.json').write_text(json.dumps(checkpoint))
    result = {'synthetic': True, 'historical_replication': False,
              'scope': 'Small training-batch mechanics; no held-out quality claim',
              'paths': args.paths, 'days': args.days, 'updates': args.updates,
              'initial_mse': before['loss'], 'final_mse': after['loss'],
              'tracking_violation_fraction': after['tracking_violation_fraction'],
              'source_loss': after['source_loss'], 'inference': inference,
              'finite_updates': all(math.isfinite(x['gradient_l2_norm']) for x in updates),
              'small_trade_threshold': checkpoint['policy']['parameters'][12]['values'][0],
              'units': terms['units'], 'feature_names': sample['feature_names']}
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
