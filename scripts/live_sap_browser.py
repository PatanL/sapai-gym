#!/usr/bin/env python3
"""Run the Unity client with a VS Code-friendly live control dashboard."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from queue import Empty, Queue
import threading
import time
from urllib.parse import urlparse

from sapai_gym1.deployment import CANONICAL_CANVAS


DEFAULT_URL = (
    "https://html-classic.itch.zone/html/18047452/"
    "Production%20WebGL/index.html"
)


DASHBOARD_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>SAP live bot</title>
<style>
body{margin:0;background:#111827;color:#e5e7eb;font:14px system-ui,sans-serif}
header{display:flex;gap:10px;align-items:center;padding:10px 14px;background:#0f172a;position:sticky;top:0}
button{padding:7px 12px;border:0;border-radius:6px;background:#334155;color:white;cursor:pointer}
button:hover{background:#475569}.danger{background:#991b1b}.pill{padding:4px 8px;border-radius:999px;background:#1e293b}
main{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:12px;padding:12px}
#stage{position:relative;background:#000;line-height:0;min-height:300px}
#frame{width:100%;height:auto;cursor:crosshair;user-select:none;-webkit-user-drag:none}
#cursor{display:none;position:absolute;width:18px;height:18px;border:3px solid #f43f5e;border-radius:50%;transform:translate(-50%,-50%);pointer-events:none;box-shadow:0 0 0 3px rgba(255,255,255,.7);transition:left .12s linear,top .12s linear}
aside{background:#0f172a;padding:12px;border-radius:8px;overflow-wrap:anywhere}pre{white-space:pre-wrap;color:#cbd5e1}
.ok{color:#4ade80}.warn{color:#fbbf24}@media(max-width:1000px){main{grid-template-columns:1fr}}
</style></head><body>
<header><strong>Super Auto Pets live bot</strong><span id="connection" class="pill">connecting</span>
<button onclick="command('/pause')">Pause</button><button onclick="command('/resume')">Resume</button>
<button onclick="command('/step')">Single frame</button><button class="danger" onclick="command('/stop')">Emergency stop</button>
<button onclick="command('/login')">Auto login</button>
</header><main><section id="stage"><img id="frame" src="/frame.png"><div id="cursor"></div></section>
<aside><h3>Session</h3><pre id="status"></pre><h3>Controls</h3>
<p>Canvas clicks are forwarded only when the server was started with <code>--allow-manual-input</code>.</p>
<p><input id="typed" type="password" autocomplete="off" placeholder="Type into focused game field" style="box-sizing:border-box;width:100%;padding:8px">
<button style="margin-top:6px" onclick="sendText()">Type securely</button>
<button style="margin-top:6px" onclick="sendKey('Tab')">Tab</button>
<button style="margin-top:6px" onclick="sendKey('Enter')">Enter</button>
<button style="margin-top:6px" onclick="sendKey('Backspace')">Backspace</button></p>
<p>Bot decisions and parsed state will appear here as the perception/controller layers are connected.</p></aside></main>
<script>
const frame=document.getElementById('frame'), statusEl=document.getElementById('status'), connection=document.getElementById('connection'), cursor=document.getElementById('cursor');
async function command(path,body={}){await fetch(path,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});}
async function sendText(){const input=document.getElementById('typed'),text=input.value;input.value='';if(text)await command('/type',{text});}
async function sendKey(key){await command('/key',{key});}
frame.addEventListener('click',async e=>{const r=frame.getBoundingClientRect(),x=(e.clientX-r.left)/r.width,y=(e.clientY-r.top)/r.height;cursor.style.display='block';cursor.style.left=(x*100)+'%';cursor.style.top=(y*100)+'%';await command('/click',{x,y});});
let lastFrameIndex=-1;
let lastInputTimestamp='';
function showAutomationInput(input){
  if(!input||!input.timestamp||input.timestamp===lastInputTimestamp)return;
  lastInputTimestamp=input.timestamp;
  const p=input.normalized||input.target||input.source;
  if(!p||typeof p.x!=='number'||typeof p.y!=='number')return;
  cursor.style.display='block';cursor.style.left=(p.x*100)+'%';cursor.style.top=(p.y*100)+'%';
  cursor.animate([{transform:'translate(-50%,-50%) scale(1.7)',opacity:.35},{transform:'translate(-50%,-50%) scale(1)',opacity:1}],{duration:350});
}
async function refresh(){try{const s=await (await fetch('/status.json',{cache:'no-store'})).json();statusEl.textContent=JSON.stringify(s,null,2);connection.textContent=s.stopped?'stopped':s.paused?'paused':'live';connection.className='pill '+(s.lastError?'warn':'ok');showAutomationInput(s.lastInput);if(s.frameIndex!==lastFrameIndex){lastFrameIndex=s.frameIndex;frame.src='/frame.png?i='+s.frameIndex;}}catch(e){connection.textContent='disconnected';connection.className='pill warn';}}
setInterval(refresh,1000);refresh();
</script></body></html>"""


@dataclass
class SharedState:
    frame: bytes = b""
    frame_index: int = 0
    paused: bool = False
    step_requested: bool = False
    stopped: bool = False
    last_error: str | None = None
    last_input: dict | None = None
    page: dict = field(default_factory=dict)
    agent: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def status(self, allow_manual_input: bool) -> dict:
        with self.lock:
            return {
                "frameIndex": self.frame_index,
                "paused": self.paused,
                "stopped": self.stopped,
                "allowManualInput": allow_manual_input,
                "lastError": self.last_error,
                "lastInput": self.last_input,
                "page": self.page,
                "agent": self.agent,
            }


def make_handler(
    state, commands, allow_manual_input, credential_username
):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def send_bytes(self, payload, content_type, status=HTTPStatus.OK):
            try:
                self.send_response(status)
                self.send_header("content-type", content_type)
                self.send_header("content-length", str(len(payload)))
                self.send_header("cache-control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                # Browsers legitimately cancel an older frame when a newer
                # frame becomes available; this is not a dashboard failure.
                return

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self.send_bytes(DASHBOARD_HTML.encode(), "text/html; charset=utf-8")
            elif path == "/frame.png":
                with state.lock:
                    frame = state.frame
                if not frame:
                    self.send_bytes(b"frame unavailable", "text/plain", HTTPStatus.SERVICE_UNAVAILABLE)
                else:
                    self.send_bytes(frame, "image/png")
            elif path == "/status.json":
                self.send_bytes(
                    json.dumps(state.status(allow_manual_input)).encode(),
                    "application/json",
                )
            else:
                self.send_bytes(b"not found", "text/plain", HTTPStatus.NOT_FOUND)

        def do_POST(self):
            path = urlparse(self.path).path
            length = int(self.headers.get("content-length", "0"))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self.send_bytes(b"invalid json", "text/plain", HTTPStatus.BAD_REQUEST)
                return
            if path == "/click":
                if not allow_manual_input:
                    self.send_bytes(b"manual input disabled", "text/plain", HTTPStatus.FORBIDDEN)
                    return
                try:
                    x, y = float(body["x"]), float(body["y"])
                except (KeyError, TypeError, ValueError):
                    self.send_bytes(b"x/y required", "text/plain", HTTPStatus.BAD_REQUEST)
                    return
                if not 0 <= x <= 1 or not 0 <= y <= 1:
                    self.send_bytes(b"x/y outside canvas", "text/plain", HTTPStatus.BAD_REQUEST)
                    return
                commands.put(("click", {"x": x, "y": y}))
            elif path == "/drag":
                if not allow_manual_input:
                    self.send_bytes(b"manual input disabled", "text/plain", HTTPStatus.FORBIDDEN)
                    return
                try:
                    source = body["source"]
                    target = body["target"]
                    x1, y1 = float(source["x"]), float(source["y"])
                    x2, y2 = float(target["x"]), float(target["y"])
                except (KeyError, TypeError, ValueError):
                    self.send_bytes(b"source/target required", "text/plain", HTTPStatus.BAD_REQUEST)
                    return
                if any(value < 0 or value > 1 for value in (x1, y1, x2, y2)):
                    self.send_bytes(b"drag outside canvas", "text/plain", HTTPStatus.BAD_REQUEST)
                    return
                commands.put(
                    (
                        "drag",
                        {
                            "source": {"x": x1, "y": y1},
                            "target": {"x": x2, "y": y2},
                        },
                    )
                )
            elif path == "/type":
                if not allow_manual_input:
                    self.send_bytes(b"manual input disabled", "text/plain", HTTPStatus.FORBIDDEN)
                    return
                text = body.get("text")
                if not isinstance(text, str) or not text or len(text) > 512:
                    self.send_bytes(b"invalid text", "text/plain", HTTPStatus.BAD_REQUEST)
                    return
                commands.put(("type", {"text": text}))
            elif path == "/key":
                if not allow_manual_input:
                    self.send_bytes(b"manual input disabled", "text/plain", HTTPStatus.FORBIDDEN)
                    return
                key = body.get("key")
                if key not in {"Tab", "Enter", "Backspace", "Escape"}:
                    self.send_bytes(b"unsupported key", "text/plain", HTTPStatus.BAD_REQUEST)
                    return
                commands.put(("key", {"key": key}))
            elif path == "/login":
                if not allow_manual_input or credential_username is None:
                    self.send_bytes(b"credential login unavailable", "text/plain", HTTPStatus.FORBIDDEN)
                    return
                commands.put(("login", {}))
            elif path == "/agent":
                # Local policy/perception processes publish only diagnostic
                # state here. This endpoint never executes a click.
                with state.lock:
                    state.agent = body
            elif path in {"/pause", "/resume", "/step", "/stop"}:
                commands.put((path[1:], body))
            else:
                self.send_bytes(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
                return
            self.send_bytes(b"ok", "text/plain")

    return Handler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument(
        "--additional-bind",
        action="append",
        default=[],
        help="Also serve the same dashboard on this specific interface.",
    )
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--width", type=int, default=CANONICAL_CANVAS.width)
    parser.add_argument("--height", type=int, default=CANONICAL_CANVAS.height)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument(
        "--record-video",
        type=Path,
        help="Save a smooth recording of the live Unity page as WebM.",
    )
    parser.add_argument("--allow-manual-input", action="store_true")
    parser.add_argument("--credential-service", default="sap-live-browser")
    parser.add_argument("--credential-username")
    return parser.parse_args()


def canvas_metadata(page):
    return page.evaluate(
        """() => ({title:document.title,url:location.href,viewport:{width:innerWidth,height:innerHeight},canvases:[...document.querySelectorAll('canvas')].map(c=>{const r=c.getBoundingClientRect();return{id:c.id,bufferWidth:c.width,bufferHeight:c.height,x:r.x,y:r.y,width:r.width,height:r.height}})})"""
    )


def ensure_recording_overlay(page):
    """Install a video-only HUD and visible automation pointer."""

    page.evaluate(
        """() => {
          if (document.getElementById('sap-recording-hud')) return;
          const hud=document.createElement('div');
          hud.id='sap-recording-hud';
          hud.style.cssText='position:fixed;left:16px;top:16px;z-index:2147483647;pointer-events:none;background:rgba(15,23,42,.86);color:#fff;padding:10px 14px;border-radius:9px;font:600 16px/1.35 system-ui,sans-serif;white-space:pre-wrap;box-shadow:0 2px 12px rgba(0,0,0,.4)';
          hud.textContent='SAP bot · initializing';
          document.body.appendChild(hud);
          const pointer=document.createElement('div');
          pointer.id='sap-recording-pointer';
          pointer.style.cssText='display:none;position:fixed;z-index:2147483647;pointer-events:none;width:24px;height:24px;border:4px solid #f43f5e;border-radius:50%;transform:translate(-50%,-50%);box-shadow:0 0 0 4px rgba(255,255,255,.85);transition:left .10s linear,top .10s linear';
          document.body.appendChild(pointer);
          const label=document.createElement('div');
          label.id='sap-recording-pointer-label';
          label.style.cssText='display:none;position:fixed;z-index:2147483647;pointer-events:none;background:#be123c;color:white;padding:3px 7px;border-radius:5px;font:700 13px system-ui,sans-serif;transform:translate(14px,14px)';
          document.body.appendChild(label);
        }"""
    )


def update_recording_hud(page, agent):
    if not isinstance(agent, dict):
        return
    state = agent.get("state") or {}
    recommendation = agent.get("recommendation") or {}
    action = recommendation.get("action") or recommendation.get("actionType")
    if isinstance(action, dict):
        action = action.get("type", action)
    lines = [
        f"SAP bot · checkpoint {agent.get('checkpointUpdate', '?')}",
        (
            f"Turn {state.get('turn', '?')} · {state.get('gold', '?')}g · "
            f"{state.get('lives', '?')} hearts · {state.get('wins', '?')} wins"
        ),
        f"Action {agent.get('loopAction', '?')} · {agent.get('status', 'thinking')}",
    ]
    if action is not None:
        lines.append(f"Policy: {action}")
    page.evaluate(
        "text => { const el=document.getElementById('sap-recording-hud'); "
        "if(el) el.textContent=text; }",
        "\n".join(lines),
    )


def show_recording_pointer(page, x, y, label):
    page.evaluate(
        """({x,y,label}) => {
          const p=document.getElementById('sap-recording-pointer');
          const l=document.getElementById('sap-recording-pointer-label');
          if(!p||!l) return;
          p.style.display='block';p.style.left=x+'px';p.style.top=y+'px';
          l.style.display='block';l.style.left=x+'px';l.style.top=y+'px';l.textContent=label;
          p.animate([{transform:'translate(-50%,-50%) scale(1.8)',opacity:.35},{transform:'translate(-50%,-50%) scale(1)',opacity:1}],{duration:350});
        }""",
        {"x": x, "y": y, "label": label},
    )


def main():
    args = parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    args.profile_dir.mkdir(parents=True, exist_ok=True)
    try:
        from playwright.sync_api import (
            TimeoutError as PlaywrightTimeoutError,
            sync_playwright,
        )
    except ImportError as error:
        raise SystemExit("Install Playwright and Chromium first") from error

    state = SharedState()
    commands: Queue = Queue()
    handler = make_handler(
        state,
        commands,
        args.allow_manual_input,
        args.credential_username,
    )
    binds = list(dict.fromkeys([args.bind, *args.additional_bind]))
    servers = [
        ThreadingHTTPServer((bind, args.port), handler) for bind in binds
    ]
    for bind, server in zip(binds, servers):
        threading.Thread(
            target=server.serve_forever, daemon=True
        ).start()
        print(f"dashboard=http://{bind}:{args.port}", flush=True)

    launch_args = [
        "--use-gl=swiftshader",
        "--enable-webgl",
        "--ignore-gpu-blocklist",
        "--disable-dev-shm-usage",
        "--autoplay-policy=no-user-gesture-required",
    ]
    video_temp_dir = None
    if args.record_video is not None:
        args.record_video.parent.mkdir(parents=True, exist_ok=True)
        video_temp_dir = args.record_video.parent / (
            f".{args.record_video.stem}-raw-{time.time_ns()}"
        )
        video_temp_dir.mkdir()
    try:
        with sync_playwright() as playwright:
            context_options = {
                "headless": not args.headed,
                "viewport": {"width": args.width, "height": args.height},
                "device_scale_factor": 1.0,
                "args": launch_args,
            }
            if args.record_video is not None:
                context_options.update(
                    record_video_dir=str(video_temp_dir),
                    record_video_size={
                        "width": args.width,
                        "height": args.height,
                    },
                )
            context = playwright.chromium.launch_persistent_context(
                str(args.profile_dir), **context_options
            )
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(args.url, wait_until="domcontentloaded", timeout=120_000)
            recorded_video = page.video if args.record_video is not None else None
            if recorded_video is not None:
                ensure_recording_overlay(page)
            interval = 1.0 / args.fps
            while True:
                loop_started = time.monotonic()
                try:
                    while True:
                        command, payload = commands.get_nowait()
                        with state.lock:
                            if command == "pause":
                                state.paused = True
                            elif command == "resume":
                                state.paused = False
                            elif command == "step":
                                state.step_requested = True
                            elif command == "stop":
                                state.stopped = True
                            elif command == "click":
                                x = payload["x"] * args.width
                                y = payload["y"] * args.height
                                if recorded_video is not None:
                                    ensure_recording_overlay(page)
                                    show_recording_pointer(page, x, y, "CLICK")
                                page.mouse.click(x, y)
                                state.last_input = {
                                    "kind": "manual-click",
                                    "normalized": payload,
                                    "pixels": {"x": round(x), "y": round(y)},
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                }
                            elif command == "drag":
                                source = payload["source"]
                                target = payload["target"]
                                x1, y1 = source["x"] * args.width, source["y"] * args.height
                                x2, y2 = target["x"] * args.width, target["y"] * args.height
                                if recorded_video is not None:
                                    ensure_recording_overlay(page)
                                    show_recording_pointer(page, x1, y1, "DRAG")
                                page.mouse.move(x1, y1)
                                page.mouse.down()
                                page.mouse.move(x2, y2, steps=15)
                                if recorded_video is not None:
                                    show_recording_pointer(page, x2, y2, "DROP")
                                page.mouse.up()
                                state.last_input = {
                                    "kind": "manual-drag",
                                    "source": source,
                                    "target": target,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                }
                            elif command == "type":
                                # Unity WebGL consumes keydown/keyup events on
                                # its canvas; DOM insertText does not reach the
                                # in-game text fields.
                                page.keyboard.type(payload["text"], delay=15)
                                state.last_input = {
                                    "kind": "manual-type",
                                    "characterCount": len(payload["text"]),
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                }
                            elif command == "key":
                                page.keyboard.press(payload["key"])
                                state.last_input = {
                                    "kind": "manual-key",
                                    "key": payload["key"],
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                }
                            elif command == "login":
                                from keyrings.alt.file import PlaintextKeyring

                                credential_store = PlaintextKeyring()
                                password = credential_store.get_password(
                                    args.credential_service,
                                    args.credential_username,
                                )
                                if password is None:
                                    raise RuntimeError(
                                        "Saved game credential was not found"
                                    )
                                # Canonical login-screen controls measured at
                                # 1920x1080. This is a user-invoked helper, not
                                # an automatic screen-agnostic click sequence.
                                page.mouse.click(args.width * 0.5, args.height * 0.318)
                                page.keyboard.press("Control+A")
                                page.keyboard.type(args.credential_username, delay=15)
                                page.mouse.click(args.width * 0.5, args.height * 0.43)
                                page.keyboard.press("Control+A")
                                page.keyboard.type(password, delay=15)
                                page.mouse.click(args.width * 0.5, args.height * 0.602)
                                state.last_input = {
                                    "kind": "saved-credential-login",
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                }
                except Empty:
                    pass
                with state.lock:
                    stopped = state.stopped
                    should_capture = not state.paused or state.step_requested
                    agent_overlay = state.agent
                    state.step_requested = False
                if stopped:
                    break
                if recorded_video is not None:
                    ensure_recording_overlay(page)
                    update_recording_hud(page, agent_overlay)
                if should_capture:
                    try:
                        screenshot_options = {
                            "type": "png",
                            "full_page": False,
                            "timeout": 15_000,
                        }
                        if recorded_video is not None:
                            # The HUD/pointer belong in the human-facing
                            # video, not in perception frames. In particular,
                            # a pointer over a just-bought pet can otherwise
                            # obscure enough of its sprite to look empty.
                            screenshot_options["style"] = (
                                "#sap-recording-hud,#sap-recording-pointer,"
                                "#sap-recording-pointer-label"
                                "{visibility:hidden!important}"
                            )
                        frame = page.screenshot(**screenshot_options)
                        metadata = canvas_metadata(page)
                        with state.lock:
                            state.frame = frame
                            state.frame_index += 1
                            state.page = metadata
                            state.last_error = None
                    except PlaywrightTimeoutError as error:
                        # CPU contention can delay a screenshot while the
                        # underlying page/video remains healthy. Keep serving
                        # the last good frame and retry instead of killing the
                        # browser and aborting a live match.
                        with state.lock:
                            state.last_error = repr(error)
                elapsed = time.monotonic() - loop_started
                time.sleep(max(0.0, interval - elapsed))
            context.close()
            if recorded_video is not None:
                raw_videos = list(video_temp_dir.glob("*.webm"))
                if len(raw_videos) != 1:
                    raise RuntimeError(
                        "Expected exactly one raw browser video, found "
                        f"{len(raw_videos)} in {video_temp_dir}"
                    )
                raw_videos[0].replace(args.record_video)
                video_temp_dir.rmdir()
                print(f"recording={args.record_video}", flush=True)
    except Exception as error:
        with state.lock:
            state.last_error = repr(error)
            state.stopped = True
        raise
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
