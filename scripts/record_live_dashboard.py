#!/usr/bin/env python3
"""Record the live control dashboard while the real automation is running."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8770")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--completion-hold", type=float, default=4.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.duration <= 0:
        raise SystemExit("--duration must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    webm_output = args.output.with_suffix(".webm")

    from playwright.sync_api import sync_playwright

    completion_seen = None
    started = time.monotonic()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": args.width, "height": args.height},
            record_video_dir=str(args.output.parent),
            record_video_size={"width": args.width, "height": args.height},
        )
        page = context.new_page()
        page.goto(args.url, wait_until="domcontentloaded", timeout=30_000)
        video = page.video
        while time.monotonic() - started < args.duration:
            try:
                status = page.evaluate(
                    "async () => await (await fetch('/status.json', "
                    "{cache:'no-store'})).json()"
                )
                if status.get("agent", {}).get("status") == "arena-complete":
                    completion_seen = completion_seen or time.monotonic()
                    if time.monotonic() - completion_seen >= args.completion_hold:
                        break
            except Exception:
                # A brief dashboard refresh/reload should remain visible in
                # the recording rather than aborting the capture.
                pass
            time.sleep(1.0)
        context.close()
        video.save_as(str(webm_output))
        browser.close()

    if args.output.suffix.lower() == ".mp4":
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            ffmpeg = next(
                Path.home().glob(
                    ".cache/ms-playwright/ffmpeg-*/ffmpeg-linux"
                ),
                None,
            )
        if ffmpeg is None:
            raise RuntimeError("Could not locate ffmpeg for MP4 conversion")
        subprocess.run(
            [
                str(ffmpeg),
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(webm_output),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(args.output),
            ],
            check=True,
        )
    print(args.output)


if __name__ == "__main__":
    main()
