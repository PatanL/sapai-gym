"""Conservative temporal state used between live visual observations."""

from __future__ import annotations

from collections import deque

from .state_schema import BoardView, LiveGameState


class LiveStateTracker:
    def __init__(self):
        self.opponent_history = deque(maxlen=2)
        self.latest_state: LiveGameState | None = None

    def reset(self):
        self.opponent_history.clear()
        self.latest_state = None

    def observe_opponent_board(self, board: BoardView):
        if not self.opponent_history or board != self.opponent_history[-1]:
            self.opponent_history.append(board)
        return tuple(self.opponent_history)

    def update(self, state: LiveGameState):
        for board in state.opponent_history:
            self.observe_opponent_board(board)
        normalized = LiveGameState(
            **{
                **state.to_dict(),
                "team": state.team,
                "shop": state.shop,
                "opponent_history": tuple(self.opponent_history),
            }
        )
        self.latest_state = normalized
        return normalized
