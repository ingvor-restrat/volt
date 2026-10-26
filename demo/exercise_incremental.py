"""Bounded per-underlying ingest and native Hypercube publication.

Build: cargo build --release -p volt-compute --features live-risk --bin exercise_cube --bin exercise_pricer
Run: python demo/exercise_incremental.py --pricer target/release/exercise_pricer --publisher target/release/exercise_cube < events.jsonl

Each JSONL event uses schema volt.exercise_ingest.v1, a globally increasing
`generation`, `as_of_ns` and an `underlying`. `replace_underlying` supplies a
complete worker `frame` for that underlying. `spot` supplies `spot` and `spot_ns`;
it retains each quote's original calibration. `remove_underlying` removes its
roster. Other underlyings keep their previous risk cutoffs, exposed by the cube.
The class is reusable by an HTTP adapter; this command is a local replay driver.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import select
import subprocess
import sys
import threading
import time
from exercise_risk_service import calculate, decode_ns, finite

MAX_LINE = 8 * 1024 * 1024
MAX_EVENT = 2 * 1024 * 1024
MAX_CALIBRATIONS = 8192


class NativePublisher:
    def __init__(self, executable, timeout=30):
        self.timeout = timeout
        self.process = subprocess.Popen([str(Path(executable).resolve(strict=True))],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def publish(self, snapshot):
        data = json.dumps(snapshot, allow_nan=False, separators=(',', ':')).encode() + b'\n'
        if len(data) > MAX_LINE:
            raise ValueError('combined publication exceeds 8MiB')
        try:
            self.process.stdin.write(data)
            self.process.stdin.flush()
            if not select.select([self.process.stdout], [], [], self.timeout)[0]:
                self.close()
                raise TimeoutError('native publisher timed out; restart and replay accepted events')
            response = self.process.stdout.readline(MAX_LINE + 1)
            if not response.endswith(b'\n') or len(response) > MAX_LINE:
                self.close()
                raise ValueError('native publisher returned an oversized or incomplete response')
            result = json.loads(response)
            if not result.get('ok'):
                raise ValueError(result.get('error', 'native publication failed'))
            return result['publication']
        except (BrokenPipeError, OSError) as exc:
            self.close()
            raise RuntimeError('native publisher unavailable; restart and replay accepted events') from exc

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait(timeout=5)
        for stream in (self.process.stdin, self.process.stdout):
            if stream:
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def contract_definition(q):
    o = q['option']
    return json.dumps({k: o[k] for k in ('contract_id', 'side', 'strike', 'exercise')}, sort_keys=True)


class IncrementalExerciseRisk:
    """Serialized atomic ingest; one complete dirty-underlying calculation per event.

    No upstream state is committed before native publication succeeds. A process
    crash requires replay of accepted events. A dividend crossing requires a
    replacement frame, recording the new quote calibration and unpaid dividends.
    """
    def __init__(self, pricer, publisher, grid=None, timeout=30, workers=1):
        self.pricer = Path(pricer).resolve(strict=True)
        self.publisher = publisher
        self.grid = grid or {'space_steps': 400, 'time_steps': 400}
        self.timeout = timeout
        self.workers = workers
        self.frames = {}
        self.results = {}
        self.iv_caches = {}
        self.snapshot = None
        self.lock = threading.Lock()

    def submit(self, event):
        with self.lock:
            return self._submit(event)

    def _submit(self, event):
        if not isinstance(event, dict) or len(json.dumps(event, allow_nan=False).encode()) > MAX_EVENT:
            raise ValueError('ingest event must be an object within 2MiB')
        if event.get('schema') != 'volt.exercise_ingest.v1':
            raise ValueError('unsupported ingest schema')
        generation = event['generation']
        now = decode_ns(event['as_of_ns'])
        underlying = event['underlying']
        if isinstance(generation, bool) or not isinstance(generation, int) or not 0 < generation < 2**53:
            raise ValueError('generation must be a positive exact browser integer')
        if not isinstance(underlying, str) or not underlying or len(underlying) > 256:
            raise ValueError('underlying must be a bounded nonempty string')
        if self.snapshot and (generation <= self.snapshot['generation'] or now < decode_ns(self.snapshot['as_of_ns'])):
            raise ValueError('stale generation or publication cutoff')
        frames = dict(self.frames)
        results = dict(self.results)
        caches = dict(self.iv_caches)
        action = event.get('action')
        if action == 'remove_underlying':
            if underlying not in frames:
                raise ValueError('unknown underlying')
            del frames[underlying], results[underlying], caches[underlying]
            if not frames:
                raise ValueError('last underlying removal requires a new empty-book session')
        else:
            if action == 'replace_underlying':
                frame = copy.deepcopy(event['frame'])
                if not frame.get('quotes') or any(q['underlying'] != underlying for q in frame['quotes']):
                    raise ValueError('replacement must contain exactly one underlying')
                old = {q['option']['contract_id']: contract_definition(q) for q in frames.get(underlying, {}).get('quotes', [])}
                if any(q['option']['contract_id'] in old and old[q['option']['contract_id']] != contract_definition(q) for q in frame['quotes']):
                    raise ValueError('conflicting contract definition; remove its underlying before replacing identity')
                previous_quotes = {q['option']['contract_id']: q for q in frames.get(underlying, {}).get('quotes', [])}
                for q in frame['quotes']:
                    previous = previous_quotes.get(q['option']['contract_id'])
                    if previous:
                        before = decode_ns(previous['option']['valuation_time'])
                        crossed = [decode_ns(d['ex_date']) for d in previous['option'].get('cash_dividends', []) if before < decode_ns(d['ex_date']) <= now]
                        if crossed and min(decode_ns(q['quote_ns']), decode_ns(q.get('calibration_ns', now)), decode_ns(q.get('calibration_spot_ns', q['spot_ns']))) < max(crossed):
                            raise ValueError('dividend replacement requires fresh quote and calibration-stock observations')
            elif action == 'spot':
                if underlying not in frames:
                    raise ValueError('unknown underlying')
                frame = copy.deepcopy(frames[underlying])
                spot = finite(event['spot'], 'spot')
                observed = decode_ns(event['spot_ns'])
                if spot <= 0 or observed > now:
                    raise ValueError('invalid stock observation')
                for q in frame['quotes']:
                    o = q['option']
                    before = decode_ns(o['valuation_time'])
                    if observed < decode_ns(q['spot_ns']):
                        raise ValueError('stock observation moved backwards')
                    if any(before < decode_ns(d['ex_date']) <= now for d in o.get('cash_dividends', [])):
                        raise ValueError('cash-dividend crossing requires replace_underlying with new calibration')
                    exercise = o['exercise']
                    expiry = exercise.get('expiry', (exercise.get('opportunities') or [None])[-1])
                    if now >= decode_ns(expiry):
                        raise ValueError('expiry crossing requires a replacement roster')
                    q.setdefault('calibration_ns', str(before))
                    q.setdefault('calibration_spot_ns', q['spot_ns'])
                    q.setdefault('calibration_spot', o['spot'])
                    q['spot_ns'] = str(observed)
                    o.update(spot=spot, valuation_time=now)
            else:
                raise ValueError('action must be replace_underlying, spot or remove_underlying')
            frame.update(generation=generation, as_of_ns=str(now))
            if any(decode_ns(q['option']['valuation_time']) != now for q in frame['quotes']):
                raise ValueError('option valuation time must equal event cutoff')
            if sum(len(f['quotes']) for k, f in frames.items() if k != underlying) + len(frame['quotes']) > 2048:
                raise ValueError('combined contract cap is 2048')
            all_ids = [q['option']['contract_id'] for k, f in frames.items() if k != underlying for q in f['quotes']]
            if set(all_ids).intersection(q['option']['contract_id'] for q in frame['quotes']):
                raise ValueError('contract identity belongs to another underlying')
            retained_frames = [f for k, f in frames.items() if k != underlying] + [frame]
            if len(json.dumps(retained_frames, allow_nan=False).encode()) > MAX_LINE:
                raise ValueError('retained underlying frames exceed 8MiB')
            cache = dict(caches.get(underlying, {}))
            result = calculate(frame, self.pricer, self.grid, self.timeout, cache, self.workers)
            for row, quote in zip(result['rows'], frame['quotes']):
                row['calibration_spot_ns'] = str(quote.get('calibration_spot_ns', quote['spot_ns']))
            frames[underlying], results[underlying], caches[underlying] = frame, result, cache
            # Eviction affects recomputation cost, never accepted prices or IVs.
            for key in sorted((k for k in caches if k != underlying), key=lambda k: int(results[k]['as_of_ns'])):
                if sum(map(len, caches.values())) <= MAX_CALIBRATIONS:
                    break
                caches[key] = {}
        ordered = [results[k] for k in sorted(results)]
        if len({(r['currency'], r['source']) for r in ordered}) != 1:
            raise ValueError('one publication session requires one currency and source')
        combined = dict(ordered[0])
        combined.update(generation=generation, as_of_ns=str(now), rows=[q for r in ordered for q in r['rows']],
                        underlying_input_sha256={k: results[k]['input_sha256'] for k in sorted(results)},
                        calculated_at_ns=str(time.time_ns()),
                        underlying_calculated_at_ns={k: results[k]['calculated_at_ns'] for k in sorted(results)},
                        underlying_assumptions={k: results[k]['assumptions'] for k in sorted(results)},
                        calculation_ms=0. if action == 'remove_underlying' else result['calculation_ms'],
                        incremental={'action': action, 'underlying': underlying, 'repriced_underlyings': [] if action == 'remove_underlying' else [underlying]})
        # The single-worker frame hash would misdescribe this combined publication.
        combined.pop('input_sha256', None)
        native = self.publisher.publish(combined)
        combined.update(portfolio={**native['portfolio'], 'unpriced_positions': native['unpriced_positions']},
                        hypercube=native['cube'], underlyings=native['underlyings'],
                        publication_schema=native['schema'])
        self.frames, self.results, self.iv_caches = frames, results, caches
        self.snapshot = combined
        return copy.deepcopy(combined)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pricer', type=Path, required=True)
    parser.add_argument('--publisher', type=Path, required=True)
    parser.add_argument('--space-steps', type=int, default=400)
    parser.add_argument('--time-steps', type=int, default=400)
    parser.add_argument('--timeout', type=float, default=30)
    args = parser.parse_args()
    with NativePublisher(args.publisher, args.timeout) as native:
        engine = IncrementalExerciseRisk(args.pricer, native, {'space_steps': args.space_steps, 'time_steps': args.time_steps}, args.timeout)
        while True:
            line = sys.stdin.buffer.readline(MAX_LINE + 1)
            if not line:
                break
            if len(line) > MAX_LINE:
                raise ValueError('ingest line exceeds 8MiB')
            try:
                result = {'ok': True, 'snapshot': engine.submit(json.loads(line))}
            except (ValueError, KeyError, TypeError, RuntimeError, TimeoutError, subprocess.SubprocessError) as exc:
                result = {'ok': False, 'error': str(exc)}
            print(json.dumps(result, allow_nan=False, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
