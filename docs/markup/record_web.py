#!/usr/bin/env python3
"""Record the live Rust-backed browser dashboard as an animated GIF."""

from __future__ import annotations

import base64
import io
import json
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).with_name("risk-web.gif")
SEED = 28_695
FRAME_COUNT = 24
FRAME_DELAY_MS = 160
WIDTH = 1280
HEIGHT = 900


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def wait_for(url: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.5):
                return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError(f"timed out waiting for {url}")


class DevTools:
    def __init__(self, url: str) -> None:
        self.socket = websocket.create_connection(
            url,
            suppress_origin=True,
            timeout=30,
        )
        self.next_id = 0

    def call(self, method: str, **params):
        self.next_id += 1
        message_id = self.next_id
        self.socket.send(
            json.dumps(
                {
                    "id": message_id,
                    "method": method,
                    "params": params,
                }
            )
        )
        while True:
            reply = json.loads(self.socket.recv())
            if reply.get("id") != message_id:
                continue
            if "error" in reply:
                raise RuntimeError(reply["error"])
            return reply.get("result", {})

    def close(self) -> None:
        self.socket.close()


def terminate(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main() -> None:
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None:
        raise RuntimeError("recording requires google-chrome or chromium")

    server_port = available_port()
    debug_port = available_port()
    dashboard = f"http://127.0.0.1:{server_port}/"
    server = subprocess.Popen(
        [
            "cargo",
            "run",
            "-q",
            "--example",
            "risk_server",
            "--",
            "--address",
            f"127.0.0.1:{server_port}",
            "--interval-ms",
            str(FRAME_DELAY_MS),
            "--seed",
            str(SEED),
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    with tempfile.TemporaryDirectory(prefix="volt-risk-chrome-") as profile:
        browser = None
        tools = None
        try:
            wait_for(dashboard)
            browser = subprocess.Popen(
                [
                    chrome,
                    "--headless=new",
                    "--disable-gpu",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-background-networking",
                    "--disable-extensions",
                    "--hide-scrollbars",
                    "--no-first-run",
                    f"--user-data-dir={profile}",
                    f"--remote-debugging-port={debug_port}",
                    "about:blank",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            wait_for(f"http://127.0.0.1:{debug_port}/json/version")
            request = urllib.request.Request(
                f"http://127.0.0.1:{debug_port}/json/new",
                method="PUT",
            )
            with urllib.request.urlopen(request) as response:
                target = json.load(response)
            tools = DevTools(target["webSocketDebuggerUrl"])
            tools.call("Page.enable")
            tools.call("Runtime.enable")
            tools.call(
                "Emulation.setDeviceMetricsOverride",
                width=WIDTH,
                height=HEIGHT,
                deviceScaleFactor=1,
                mobile=False,
            )
            tools.call("Page.navigate", url=f"{dashboard}?capture=1")

            for _ in range(200):
                ready = tools.call(
                    "Runtime.evaluate",
                    expression=(
                        "document.readyState === 'complete' "
                        "&& typeof render === 'function'"
                    ),
                    returnByValue=True,
                )
                if ready["result"].get("value") is True:
                    break
                time.sleep(0.05)
            else:
                state = tools.call(
                    "Runtime.evaluate",
                    expression=(
                        "JSON.stringify({href: location.href, "
                        "title: document.title, state: document.readyState, "
                        "body: document.body?.innerText.slice(0, 80)})"
                    ),
                    returnByValue=True,
                )
                raise RuntimeError(
                    "dashboard JavaScript did not become ready: "
                    f"{state['result'].get('value')}"
                )

            frames = []
            for _ in range(FRAME_COUNT):
                tools.call(
                    "Runtime.evaluate",
                    expression="""
                      fetch('/api/snapshot')
                        .then(response => response.json())
                        .then(snapshot => {
                          latest = snapshot;
                          render(latest);
                          connection.textContent = 'SIMULATION · RUST SNAPSHOTS';
                          return snapshot.step;
                        })
                    """,
                    awaitPromise=True,
                    returnByValue=True,
                )
                shot = tools.call(
                    "Page.captureScreenshot",
                    format="png",
                    fromSurface=True,
                )
                frame = Image.open(
                    io.BytesIO(base64.b64decode(shot["data"]))
                ).convert("RGB")
                frames.append(
                    frame.quantize(
                        colors=160,
                        method=Image.Quantize.MEDIANCUT,
                        dither=Image.Dither.NONE,
                    )
                )
                time.sleep(FRAME_DELAY_MS / 1_000)

            durations = [FRAME_DELAY_MS] * len(frames)
            durations[-1] = 1_000
            frames[0].save(
                OUTPUT,
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=0,
                disposal=2,
                optimize=True,
            )
        finally:
            if tools is not None:
                tools.close()
            if browser is not None:
                terminate(browser)
            terminate(server)

    size_mb = OUTPUT.stat().st_size / (1024 * 1024)
    print(
        f"wrote {OUTPUT.relative_to(ROOT)} "
        f"({FRAME_COUNT} browser frames, {size_mb:.1f} MiB)"
    )


if __name__ == "__main__":
    main()
