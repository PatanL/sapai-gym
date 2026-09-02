#!/usr/bin/env python3
"""Capture the official Unity WebGL client at a reproducible viewport.

This recorder deliberately performs no clicks.  Privacy/terms consent and
account authentication must be completed by the user in a persistent headed
profile before unattended screenshot collection is enabled.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from sapai_gym1.deployment import CANONICAL_CANVAS


DEFAULT_URL = (
    "https://html-classic.itch.zone/html/18047452/"
    "Production%20WebGL/index.html"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--width", type=int, default=CANONICAL_CANVAS.width)
    parser.add_argument("--height", type=int, default=CANONICAL_CANVAS.height)
    parser.add_argument("--device-scale-factor", type=float, default=1.0)
    parser.add_argument("--wait-seconds", type=float, default=35.0)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Open an interactive browser for user consent/login and recording.",
    )
    return parser.parse_args()


def canvas_metadata(page):
    return page.evaluate(
        """() => ({
          title: document.title,
          viewport: {width: innerWidth, height: innerHeight},
          canvases: [...document.querySelectorAll('canvas')].map(canvas => {
            const rect = canvas.getBoundingClientRect();
            return {
              id: canvas.id,
              bufferWidth: canvas.width,
              bufferHeight: canvas.height,
              x: rect.x,
              y: rect.y,
              width: rect.width,
              height: rect.height
            };
          })
        })"""
    )


def main():
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("Viewport dimensions must be positive")
    if args.count <= 0 or args.interval_seconds < 0 or args.wait_seconds < 0:
        raise SystemExit("Capture count/timing arguments are invalid")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as error:
        raise SystemExit(
            "Install Playwright and Chromium: pip install playwright && "
            "playwright install chromium"
        ) from error

    launch_args = [
        "--use-gl=swiftshader",
        "--enable-webgl",
        "--ignore-gpu-blocklist",
        "--disable-dev-shm-usage",
        "--autoplay-policy=no-user-gesture-required",
    ]
    with sync_playwright() as playwright:
        if args.profile_dir is not None:
            args.profile_dir.mkdir(parents=True, exist_ok=True)
            context = playwright.chromium.launch_persistent_context(
                str(args.profile_dir),
                headless=not args.headed,
                viewport={"width": args.width, "height": args.height},
                device_scale_factor=args.device_scale_factor,
                args=launch_args,
            )
            page = context.pages[0] if context.pages else context.new_page()
            browser = None
        else:
            browser = playwright.chromium.launch(
                headless=not args.headed,
                args=launch_args,
            )
            context = browser.new_context(
                viewport={"width": args.width, "height": args.height},
                device_scale_factor=args.device_scale_factor,
            )
            page = context.new_page()

        page.goto(args.url, wait_until="domcontentloaded", timeout=120_000)
        time.sleep(args.wait_seconds)
        records = []
        for index in range(args.count):
            timestamp = datetime.now(timezone.utc)
            stem = timestamp.strftime("sap-%Y%m%dT%H%M%S.%fZ")
            screenshot = args.output_dir / f"{stem}.png"
            page.screenshot(path=str(screenshot), full_page=False)
            digest = hashlib.sha256(screenshot.read_bytes()).hexdigest()
            records.append(
                {
                    "schemaVersion": 1,
                    "timestamp": timestamp.isoformat(),
                    "url": args.url,
                    "screenshot": screenshot.name,
                    "sha256": digest,
                    "requestedViewport": {
                        "width": args.width,
                        "height": args.height,
                        "deviceScaleFactor": args.device_scale_factor,
                    },
                    "page": canvas_metadata(page),
                }
            )
            if index + 1 < args.count:
                time.sleep(args.interval_seconds)

        manifest = args.output_dir / "capture-manifest.json"
        manifest.write_text(json.dumps(records, indent=2) + "\n")
        print(f"captures={len(records)} manifest={manifest}")
        context.close()
        if browser is not None:
            browser.close()


if __name__ == "__main__":
    main()
