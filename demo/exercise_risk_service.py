"""Serve dated exercise-aware option risk computed by Volt's Rust reference.

Build: cargo build --release -p volt-compute --bin exercise_pricer
Run: python demo/exercise_risk_service.py --pricer target/release/exercise_pricer
POST /api/options-risk/frame supplies observations; GET /api/options-risk/snapshot
returns the latest completed calculation. Historical producers and datasets can
remain outside the source checkout. Default binding and browser origins are local.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
from urllib.parse import urlparse

MAX_BYTES = 2 * 1024 * 1024
PREFIX = "/api/options-risk"


def decode_ns(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("timestamps must be exact integers or decimal strings")
    result = int(value)
    if not 0 <= result <= 2**63 - 1:
        raise ValueError("timestamp outside supported range")
    return result


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + " must be finite")
    return float(value)


def calculate(frame, pricer, grid, timeout, iv_cache=None, workers=4):
    iv_cache = {} if iv_cache is None else iv_cache
    start = time.perf_counter()
    if not isinstance(frame, dict):
        raise ValueError("quote frame must be an object")
    if not isinstance(workers, int) or isinstance(workers, bool) or not 1 <= workers <= 8:
        raise ValueError("workers must be within 1..8")
    if frame.get("schema") != "volt.exercise_quote_frame.v1":
        raise ValueError("unsupported quote frame schema")
    generation = frame.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int) or not 1 <= generation < 2**53:
        raise ValueError("generation must be a positive exact browser integer")
    source = frame.get("source")
    if source not in ("historical", "live", "synthetic"):
        raise ValueError("source must identify historical, live or synthetic observations")
    as_of = decode_ns(frame.get("as_of_ns"))
    quotes = frame.get("quotes")
    if not isinstance(quotes, list) or not 1 <= len(quotes) <= 128:
        raise ValueError("frame must contain 1..128 quotes")
    if not isinstance(frame.get("currency", "USD"), str) or len(frame.get("currency", "USD")) != 3 or not frame.get("currency", "USD").isascii() or not frame.get("currency", "USD").isupper():
        raise ValueError("currency must be a three-letter code")
    if not isinstance(frame.get("assumptions", {}), dict):
        raise ValueError("assumptions must be an object")
    ids, row_ids = set(), set()
    for quote in quotes:
        if not isinstance(quote, dict) or not isinstance(quote.get("option"), dict):
            raise ValueError("each quote requires an option object")
        o = quote["option"]
        contract = o["contract_id"]
        if not isinstance(contract, str) or not contract or contract in ids:
            raise ValueError("contract identities must be unique")
        ids.add(contract)
        row_id = o.get("row_id")
        if not isinstance(row_id, str) or not row_id or row_id in row_ids:
            raise ValueError("row identities must be unique")
        row_ids.add(row_id)
        if not isinstance(quote.get("underlying"), str) or not quote["underlying"].strip():
            raise ValueError("underlying identity is required")
        bid, ask = finite(quote["bid"], "bid"), finite(quote["ask"], "ask")
        if bid < 0 or ask <= 0 or bid > ask:
            raise ValueError("quotes must be positive and uncrossed")
        qt, st = decode_ns(quote["quote_ns"]), decode_ns(quote["spot_ns"])
        if max(qt, st) > as_of or decode_ns(o["valuation_time"]) != as_of:
            raise ValueError("source observations must precede the common valuation time")
        quantity, multiplier = quote.get("quantity", 0), quote.get("multiplier", 100)
        if any(isinstance(v, bool) or not isinstance(v, int) for v in [quantity, multiplier]) or not 1 <= multiplier <= 1_000_000 or abs(quantity) > 1_000_000 or abs(quantity * multiplier) > 2**53 - 1:
            raise ValueError("quantity and multiplier exceed exact economic-unit bounds")
    # Keep each Rust request within its declared solve-cell allowance.
    cells = grid.get("space_steps", 800) * (grid.get("time_steps", 800) + 512)
    chunk = max(1, min(128, 450_000_000 // (cells * 90)))
    def run(request_rows):
        if not request_rows:
            return []
        task_rows = min(chunk, max(1, math.ceil(len(request_rows) / workers)))
        batches = [request_rows[first:first + task_rows] for first in range(0, len(request_rows), task_rows)]
        def execute(batch):
            request = {"schema": "volt.compute.exercise_pricing.v1", "grid": grid, "rows": batch}
            remaining = timeout - (time.perf_counter() - start)
            if remaining <= 0:
                raise TimeoutError("frame calculation exceeded its time allowance")
            process = subprocess.run([str(pricer)], input=json.dumps(request, allow_nan=False),
                                     capture_output=True, text=True, timeout=remaining)
            if process.returncode:
                raise ValueError("Rust pricing request failed: " + process.stderr.strip()[:500])
            output = json.loads(process.stdout)
            if len(output["rows"]) != len(batch):
                raise ValueError("worker returned a different row count")
            for inp, out in zip(batch, output["rows"]):
                if inp["option"]["row_id"] != out["row_id"] or inp["option"]["contract_id"] != out["contract_id"]:
                    raise ValueError("worker row identity differs from input")
            return output["rows"]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return [row for batch in pool.map(execute, batches) for row in batch]
    calibrations, keys, missing_calibrations = {}, {}, []
    for quote in quotes:
        option = dict(quote["option"])
        calibration_ns = decode_ns(quote.get("calibration_ns", as_of))
        calibration_spot_ns = decode_ns(quote.get("calibration_spot_ns", quote["spot_ns"]))
        calibration_spot = finite(quote.get("calibration_spot", option["spot"]), "calibration spot")
        if calibration_spot <= 0 or not max(decode_ns(quote["quote_ns"]), calibration_spot_ns) <= calibration_ns <= as_of:
            raise ValueError("calibration inputs must be available at their recorded observation time")
        option.update(spot=calibration_spot, valuation_time=calibration_ns)
        key = hashlib.sha256(json.dumps({"option": option, "mid": (quote["bid"] + quote["ask"]) / 2,
                                        "quote_ns": str(quote["quote_ns"]), "spot_ns": str(calibration_spot_ns), "grid": grid},
                                       sort_keys=True, allow_nan=False).encode()).hexdigest()
        keys[option["contract_id"]] = key
        calibrations[option["contract_id"]] = (calibration_ns, calibration_spot)
        if key not in iv_cache:
            missing_calibrations.append({"option": option, "market_price": (quote["bid"] + quote["ask"]) / 2})
    for result in run(missing_calibrations):
        iv_cache[keys[result["contract_id"]]] = {"implied_volatility": result.get("implied_volatility"), "error": result.get("error")}
    risk_requests = []
    for quote in quotes:
        calibration = iv_cache[keys[quote["option"]["contract_id"]]]
        iv = calibration["implied_volatility"]
        if iv and iv["state"] == "converged" and iv["volatility"] is not None:
            option = dict(quote["option"], volatility=iv["volatility"])
            risk_requests.append({"option": option})
    risks = {row["contract_id"]: row for row in run(risk_requests)}
    results = []
    for quote in quotes:
        contract_id = quote["option"]["contract_id"]
        calibration = iv_cache[keys[contract_id]]
        results.append({"contract_id": contract_id, "row_id": quote["option"]["row_id"],
                        "risk": risks.get(contract_id, {}).get("risk"),
                        "implied_volatility": calibration["implied_volatility"],
                        "error": risks.get(contract_id, {}).get("error") or calibration["error"]})
    # Active quote calibrations survive bounded-cache pruning.
    keep = set(keys.values())
    if len(iv_cache) > 4096:
        for key in list(iv_cache):
            if key not in keep:
                del iv_cache[key]
            if len(iv_cache) <= 4096:
                break
    rows = []
    total = {"value": 0., "delta_dollars": 0., "gamma_1pct": 0., "vega_per_vol_point": 0., "theta_per_day": 0., "rho_per_bp": 0.}
    missing = []
    for quote, result in zip(quotes, results):
        o = quote["option"]
        if result["contract_id"] != o["contract_id"] or result["row_id"] != o["row_id"]:
            raise ValueError("worker row identity differs from input")
        risk, iv = result.get("risk"), result.get("implied_volatility")
        units = quote.get("quantity", 0) * quote.get("multiplier", 100)
        expiry = o["exercise"].get("expiry")
        if expiry is None:
            expiry = o["exercise"]["opportunities"][-1]
        row = {"contract_id": o["contract_id"], "underlying": quote["underlying"],
               "side": o["side"], "strike": o["strike"], "expiry_ns": str(expiry),
               "exercise_style": o["exercise"]["style"], "spot": o["spot"],
               "quote_ns": str(quote["quote_ns"]), "spot_ns": str(quote["spot_ns"]),
               "bid": quote["bid"], "ask": quote["ask"], "mid": (quote["bid"] + quote["ask"]) / 2,
               "quantity": quote.get("quantity", 0), "multiplier": quote.get("multiplier", 100),
               "position_units": units, "iv": iv, "risk": risk, "error": result.get("error"),
               "calibration_ns": str(calibrations[o["contract_id"]][0]),
               "calibration_spot": calibrations[o["contract_id"]][1], "risk_asof_ns": str(as_of)}
        rows.append(row)
        if units:
            if not risk:
                missing.append(o["contract_id"])
                continue
            total["value"] += units * risk["valuation"]["price"]
            total["delta_dollars"] += units * risk["delta"] * o["spot"]
            total["gamma_1pct"] += .5 * units * risk["gamma"] * (o["spot"] * .01)**2
            for key in ["vega_per_vol_point", "theta_per_day", "rho_per_bp"]:
                v = risk.get(key)
                if v is None:
                    total[key] = None
                elif total[key] is not None:
                    total[key] += units * v
    if missing:
        total = {key: None for key in total}
    for v in total.values():
        if v is not None and not math.isfinite(v):
            raise ValueError("portfolio aggregation overflow")
    return {"schema": "volt.options_risk_snapshot.v1", "generation": generation, "source": source,
            "as_of_ns": str(as_of), "calculated_at_ns": str(time.time_ns()), "currency": frame.get("currency", "USD"),
            "assumptions": frame.get("assumptions", {}), "rows": rows,
            "volatility_motion": "fixed_quote_calibration_between_quote_updates", "execution": "cpu_f64_fd",
            "portfolio": {**total, "unpriced_positions": missing},
            "input_sha256": hashlib.sha256(json.dumps(frame, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            "calculation_ms": (time.perf_counter() - start) * 1000}


def serve(args):
    token = os.environ.get(args.token_env, "")
    if not ipaddress.ip_address(args.host).is_loopback and not token:
        raise ValueError("a nonlocal binding requires a configured worker token")
    state = {"snapshot": None, "iv_cache": {}}
    lock, calculating = threading.Lock(), threading.Lock()
    grid = {"space_steps": args.space_steps, "time_steps": args.time_steps}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def origin(self):
            origin = self.headers.get("Origin")
            if not origin:
                return None
            parsed = urlparse(origin)
            if origin in args.allow_origin or (parsed.scheme in ("http", "https") and parsed.hostname in ("localhost", "127.0.0.1", "::1")):
                return origin
            raise PermissionError("origin is outside the worker's configured clients")
        def send(self, status, data):
            body = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            origin = self.origin()
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.end_headers()
            self.wfile.write(body)
        def authorized(self):
            try:
                self.origin()
            except PermissionError:
                self.send_error(403)
                return False
            if token and not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.send(401, {"error": "worker authorization required"})
                return False
            return True
        def do_OPTIONS(self):
            try:
                origin = self.origin()
            except PermissionError:
                self.send_error(403); return
            self.send_response(204)
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.end_headers()
        def do_GET(self):
            if not self.authorized(): return
            if self.path == "/health": self.send(200, {"ok": True}); return
            if self.path != PREFIX + "/snapshot": self.send(404, {"error": "unknown endpoint"}); return
            with lock: snapshot = state["snapshot"]
            self.send(200, snapshot or {"schema": "volt.options_risk_snapshot.v1", "generation": 0, "source": "waiting", "rows": []})
        def do_POST(self):
            if not self.authorized(): return
            if self.path != PREFIX + "/frame": self.send(404, {"error": "unknown endpoint"}); return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BYTES: self.send(413, {"error": "frame exceeds request allowance"}); return
                frame = json.loads(self.rfile.read(size))
                if not isinstance(frame, dict):
                    raise ValueError("quote frame must be an object")
                if not calculating.acquire(blocking=False): self.send(429, {"error": "worker is calculating a frame"}); return
                try:
                    with lock: current = state["snapshot"]
                    if current and frame.get("generation", 0) <= current["generation"]:
                        self.send(409, {"error": "generation must advance"}); return
                    if current and decode_ns(frame.get("as_of_ns")) < int(current["as_of_ns"]):
                        self.send(409, {"error": "observation time must advance"}); return
                    cache = dict(state["iv_cache"])
                    snapshot = calculate(frame, args.pricer, grid, args.timeout, cache, args.workers)
                    with lock:
                        state["snapshot"], state["iv_cache"] = snapshot, cache
                    self.send(200, snapshot)
                finally: calculating.release()
            except (ValueError, KeyError, TypeError, TimeoutError, subprocess.TimeoutExpired) as exc:
                self.send(400, {"error": str(exc)[:600]})
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
                request.close(); return
            try:
                super().process_request(request, address)
            except Exception:
                self.slots.release(); raise
        def process_request_thread(self, request, address):
            try:
                super().process_request_thread(request, address)
            finally:
                self.slots.release()
    print(json.dumps({"url": f"http://{args.host}:{args.port}", "engine": "exercise_pricer", "grid": grid}), flush=True)
    Server((args.host, args.port), Handler).serve_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pricer", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8820)
    parser.add_argument("--space-steps", type=int, default=400)
    parser.add_argument("--time-steps", type=int, default=400)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--token-env", default="VOLT_RISK_TOKEN")
    parser.add_argument("--allow-origin", action="append", default=[])
    args = parser.parse_args()
    args.pricer = args.pricer.resolve(strict=True)
    if not 32 <= args.space_steps <= 4096 or not 8 <= args.time_steps <= 8192 or not 0 < args.timeout <= 300 or not 1 <= args.workers <= 8:
        parser.error("grid/time allowance outside supported limits")
    serve(args)
