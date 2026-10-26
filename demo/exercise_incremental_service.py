"""Exercise-aware incremental risk HTTP adapter with native Hypercube publication.

POST /api/options-risk/event accepts volt.exercise_ingest.v1 events.
GET /api/options-risk/snapshot returns the latest complete accepted snapshot.
Run --help for the native binary paths and local/authenticated binding options.
"""
from __future__ import annotations
import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import os
from pathlib import Path
import subprocess
import threading
from exercise_incremental import IncrementalExerciseRisk, NativePublisher
from exercise_risk_service import client_origin, decode_ns, PREFIX

MAX_BYTES = 2 * 1024 * 1024


def make_server(args, engine):
    """Construct a bounded server; caller owns engine/native-process lifetime."""
    token = os.environ.get(args.token_env, '')
    if not ipaddress.ip_address(args.host).is_loopback and not token:
        raise ValueError('a nonlocal binding requires a configured worker token')
    snapshot_lock, calculating = threading.Lock(), threading.Lock()
    state = {'snapshot': engine.snapshot}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def allowed_origin(self):
            return client_origin(self.headers.get('Origin'), args.allow_origin)

        def reply(self, code, data):
            body = json.dumps(data, allow_nan=False, separators=(',', ':')).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            origin = self.allowed_origin()
            if origin:
                self.send_header('Access-Control-Allow-Origin', origin)
                self.send_header('Vary', 'Origin')
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            try:
                self.allowed_origin()
            except PermissionError:
                self.send_error(403)
                return False
            if token and not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + token):
                self.reply(401, {'error': 'worker authorization required'})
                return False
            return True

        def do_OPTIONS(self):
            try:
                origin = self.allowed_origin()
            except PermissionError:
                self.send_error(403)
                return
            self.send_response(204)
            if origin:
                self.send_header('Access-Control-Allow-Origin', origin)
                self.send_header('Vary', 'Origin')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            self.end_headers()

        def do_GET(self):
            if not self.authorized():
                return
            if self.path == '/health':
                healthy = engine.publisher.process.poll() is None
                self.reply(200 if healthy else 503, {'ok': healthy, 'execution': 'cpu_f64_fd+hypercube'})
                return
            if self.path != PREFIX + '/snapshot':
                self.reply(404, {'error': 'unknown endpoint'})
                return
            with snapshot_lock:
                snapshot = state['snapshot']
            self.reply(200, snapshot or {'schema': 'volt.options_risk_snapshot.v1', 'generation': 0, 'source': 'waiting', 'rows': []})

        def do_POST(self):
            if not self.authorized():
                return
            if self.path != PREFIX + '/event':
                self.reply(404, {'error': 'unknown endpoint'})
                return
            try:
                if self.headers.get_content_type() != 'application/json':
                    self.reply(415, {'error': 'application/json required'})
                    return
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= MAX_BYTES:
                    self.reply(413, {'error': 'event exceeds request allowance'})
                    return
                data = self.rfile.read(size)
                if len(data) != size:
                    raise ValueError('incomplete request body')
                event = json.loads(data)
                if not isinstance(event, dict):
                    raise ValueError('event must be an object')
                if not calculating.acquire(blocking=False):
                    self.reply(429, {'error': 'worker is calculating an event'})
                    return
                try:
                    with snapshot_lock:
                        previous = state['snapshot']
                    if previous and (event.get('generation', 0) <= previous['generation'] or decode_ns(event.get('as_of_ns')) < int(previous['as_of_ns'])):
                        self.reply(409, {'error': 'generation and observation time must advance'})
                        return
                    snapshot = engine.submit(event)
                    with snapshot_lock:
                        state['snapshot'] = snapshot
                finally:
                    calculating.release()
                self.reply(200, snapshot)
            except (ValueError, KeyError, TypeError, RuntimeError, TimeoutError, subprocess.SubprocessError) as exc:
                self.reply(400, {'error': str(exc)[:600]})

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 16
        slots = threading.BoundedSemaphore(16)

        def get_request(self):
            sock, address = super().get_request()
            sock.settimeout(10)
            return sock, address

        def process_request(self, request, address):
            if not self.slots.acquire(blocking=False):
                request.close()
                return
            try:
                super().process_request(request, address)
            except Exception:
                self.slots.release()
                raise

        def process_request_thread(self, request, address):
            try:
                super().process_request_thread(request, address)
            finally:
                self.slots.release()

    return Server((args.host, args.port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pricer', type=Path, required=True)
    parser.add_argument('--publisher', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8820)
    parser.add_argument('--space-steps', type=int, default=400)
    parser.add_argument('--time-steps', type=int, default=400)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--token-env', default='VOLT_RISK_TOKEN')
    parser.add_argument('--allow-origin', action='append', default=[])
    args = parser.parse_args()
    if not 32 <= args.space_steps <= 4096 or not 8 <= args.time_steps <= 8192 or not 0 < args.timeout <= 300 or not 1 <= args.workers <= 8:
        parser.error('grid/time allowance outside supported limits')
    with NativePublisher(args.publisher, args.timeout) as publisher:
        engine = IncrementalExerciseRisk(args.pricer, publisher,
            {'space_steps': args.space_steps, 'time_steps': args.time_steps}, args.timeout, args.workers)
        server = make_server(args, engine)
        print(json.dumps({'host': args.host, 'port': server.server_address[1], 'execution': 'cpu_f64_fd+hypercube'}), flush=True)
        try:
            server.serve_forever()
        finally:
            server.server_close()


if __name__ == '__main__':
    main()
