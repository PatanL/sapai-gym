"""HTTP client used by the autonomous agent to observe/control Playwright."""

from __future__ import annotations

import json
import time
from urllib.request import Request, urlopen

import numpy as np


class DashboardClient:
    def __init__(self, base_url: str, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    def _get(self, path):
        with urlopen(self.base_url + path, timeout=self.timeout) as response:
            return response.read()

    def _post(self, path, payload=None):
        request = Request(
            self.base_url + path,
            data=json.dumps(payload or {}).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            return response.read()

    def status(self):
        return json.loads(self._get("/status.json"))

    def frame(self):
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("Frame decoding requires opencv-python") from error
        encoded = np.frombuffer(self._get("/frame.png"), dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("Dashboard returned an invalid frame")
        return frame

    def click(self, point):
        return self._post("/click", {"x": point.x, "y": point.y})

    def drag(self, source, target):
        return self._post(
            "/drag",
            {"source": source.__dict__, "target": target.__dict__},
        )

    def publish_agent(self, payload):
        return self._post("/agent", payload)

    def execute_gesture(self, gesture, inter_click_delay=0.4):
        from .action_executor import (
            ClickGesture,
            ClickSequenceGesture,
            DragGesture,
        )

        if isinstance(gesture, ClickGesture):
            self.click(gesture.point)
        elif isinstance(gesture, DragGesture):
            self.drag(gesture.source, gesture.target)
        elif isinstance(gesture, ClickSequenceGesture):
            for point in gesture.points:
                self.click(point)
                time.sleep(inter_click_delay)
        else:
            raise TypeError(f"Unknown gesture {type(gesture)}")

    def wait_for_stable_frame(
        self,
        *,
        interval: float = 0.6,
        maximum_wait: float = 8.0,
        mean_difference: float = 1.2,
    ):
        """Wait for two semantically stable frames, ignoring the timer area."""

        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("Frame stability requires opencv-python") from error
        deadline = time.monotonic() + maximum_wait
        previous = None
        while time.monotonic() < deadline:
            current = self.frame()
            # Exclude bottom-right timer, whose digits change continuously.
            sample = current.copy()
            sample[int(sample.shape[0] * 0.82) :, int(sample.shape[1] * 0.88) :] = 0
            sample = cv2.resize(sample, (480, 270))
            if previous is not None:
                difference = float(
                    np.abs(sample.astype(np.int16) - previous.astype(np.int16)).mean()
                )
                if difference <= mean_difference:
                    return current, difference
            previous = sample
            time.sleep(interval)
        raise TimeoutError("Live UI did not reach a stable frame")

    def wait_for_changed_then_stable(
        self,
        reference,
        *,
        interval: float = 0.5,
        maximum_wait: float = 12.0,
        minimum_change: float = 0.35,
        mean_difference: float = 1.2,
    ):
        """Wait until queued input changes Unity, then wait for animation settle."""

        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("Frame synchronization requires opencv-python") from error

        def sample(frame):
            result = frame.copy()
            result[
                int(result.shape[0] * 0.82) :,
                int(result.shape[1] * 0.88) :,
            ] = 0
            return cv2.resize(result, (480, 270))

        baseline = sample(reference).astype(np.int16)
        deadline = time.monotonic() + maximum_wait
        changed = False
        previous = None
        last_change = 0.0
        while time.monotonic() < deadline:
            current = self.frame()
            current_sample = sample(current).astype(np.int16)
            reference_difference = float(
                np.abs(current_sample - baseline).mean()
            )
            if reference_difference >= minimum_change:
                changed = True
                last_change = reference_difference
            if changed and previous is not None:
                settle_difference = float(
                    np.abs(current_sample - previous).mean()
                )
                if settle_difference <= mean_difference:
                    return current, settle_difference, last_change
            previous = current_sample
            time.sleep(interval)
        raise TimeoutError("Queued UI input did not visibly change and settle")
