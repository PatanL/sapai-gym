"""Deterministic menu navigation for the authenticated Arena client."""

from __future__ import annotations

import time

from .geometry import NormalizedPoint


class ArenaNavigator:
    play = NormalizedPoint(0.5, 0.33)
    arena = NormalizedPoint(0.331, 0.333)
    turtle_pack = NormalizedPoint(0.5, 0.46)
    start = NormalizedPoint(0.91, 0.92)
    dismiss_detail = NormalizedPoint(0.70, 0.15)

    def __init__(self, dashboard):
        self.dashboard = dashboard

    def from_main_menu(self, *, existing_run: bool = True):
        """Enter Arena from a known authenticated main-menu frame."""

        # The Unity canvas reaches its final dimensions before the main-menu
        # controls finish accepting input. Require both a settling delay and
        # a stable frame so the first Play click cannot be silently dropped.
        time.sleep(20.0)
        self.dashboard.wait_for_stable_frame(maximum_wait=12.0)
        self.dashboard.click(self.play)
        # WebGL navigation can take several seconds even after the outgoing
        # menu looks static. Clicking Arena before the Play screen is live is
        # silently ignored, so retain a conservative transition allowance.
        time.sleep(5.0)
        self.dashboard.click(self.arena)
        time.sleep(5.0)
        if not existing_run:
            # The selected pack persists globally. On the current Arena setup
            # screen Turtle Pack already carries the green selection check;
            # clicking the pack card opens the separate Pack collection and
            # prevents the game from starting. Start the selected pack.
            self.dashboard.click(self.start)
            time.sleep(5.0)
        frame = self.dashboard.wait_for_stable_frame(maximum_wait=12.0)
        # The Arena-tile click can propagate into the newly loaded shop and
        # open a pet detail card at the same coordinates.
        self.dashboard.click(self.dismiss_detail)
        time.sleep(1.0)
        return frame
