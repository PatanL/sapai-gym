import numpy as np
import pytest
import torch

from sapai_gym1.rl.opponent_distribution import OpponentDistributionConfig, AutoregressiveOpponentBoard, MarginalOpponentBoard
from sapai_gym1.rl.opponent_forecast_quality import calibration, paired_match_interval, evaluate_forecast, board_events
from test_opponent_forecast_data import complete_match


def test_calibration_and_match_clusters_are_not_independent_turns():
    result = calibration([0, 1, .5, .5], [0, 1, 0, 1])
    assert result['brier'] == pytest.approx(.125) and result['ece'] == 0
    rows = complete_match('a')+complete_match('b')
    result = paired_match_interval(rows, [2]*4+[4]*4, [1]*8, repeats=100)
    assert result['matches'] == 2 and result['meanMatchNllImprovement'] == 2
    with pytest.raises(ValueError):
        calibration([1.5], [1])


@pytest.mark.parametrize('variant', ['history_marginal', 'causal_marginal', 'causal_autoregressive'])
def test_quality_metrics_are_finite_seeded_and_do_not_touch_training_rng(variant):
    cls = AutoregressiveOpponentBoard if variant == 'causal_autoregressive' else MarginalOpponentBoard
    model = cls(OpponentDistributionConfig(('none', 'pet-fish'), ('none',), width=32))
    rows = complete_match('a')
    rng = torch.get_rng_state()
    report = evaluate_forecast(model, rows, variant, batch_size=2, samples=4)
    assert torch.equal(rng, torch.get_rng_state())
    assert report == evaluate_forecast(model, rows, variant, batch_size=2, samples=4)
    assert np.isfinite(report['negativeLogLikelihoodPerBoard'])
    assert report['syntacticallyInvalidBoardRate'] == 0
    assert report['exactHiddenShopReachabilityRate'] is None
    assert report['rows'] == 4 and len(report['nllPerRow']) == 4


def test_known_board_event_labels():
    config = OpponentDistributionConfig(('none', 'pet-shark', 'pet-fly', 'pet-deer', 'pet-monkey'), ('none', 'status-extra-life'), width=32)
    board = torch.zeros(1, 5, 6, dtype=torch.long)
    board[0, :, 0] = torch.tensor([1, 2, 3, 4, 0])
    board[0, 2, 2] = 1; board[0, 3, 1] = 2
    assert board_events(board, config)[0].tolist() == [True, False, False, True, False, False, True, False]
