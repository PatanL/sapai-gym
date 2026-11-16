from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.benchmark_puct_equal_compute import equal_compute_puct_config


def test_equal_compute_uses_requested_production_selection_parameters():
    config = equal_compute_puct_config(
        SimpleNamespace(
            horizon_battles=5,
            leaf_value_weight=0.5,
            validation_rollouts=32,
            puct_cpuct=2.0,
            puct_prior_floor=0.10,
            puct_min_root_visits=2,
            puct_transition_budget_scale=1.0,
            puct_selection_only=False,
            puct_external_confirmation_only=False,
            puct_root_screening_top_k=8,
            puct_root_screening_rollouts=8,
            puct_root_screening_rounds=3,
            puct_root_screening_min_survivors=4,
        ),
        selection_budget=7_500,
    )
    assert config.max_simulator_transitions == 7_500
    assert config.horizon_battles == 5
    assert config.cpuct == 2.0
    assert config.prior_floor == 0.10
    assert config.min_root_visits == 2


def test_equal_wall_ablation_can_scale_puct_selection_budget():
    config = equal_compute_puct_config(
        SimpleNamespace(
            horizon_battles=5,
            leaf_value_weight=0.5,
            validation_rollouts=32,
            puct_cpuct=2.0,
            puct_prior_floor=0.10,
            puct_min_root_visits=1,
            puct_transition_budget_scale=0.2,
            puct_selection_only=True,
            puct_external_confirmation_only=False,
            puct_root_screening_top_k=8,
            puct_root_screening_rollouts=8,
            puct_root_screening_rounds=3,
            puct_root_screening_min_survivors=4,
        ),
        selection_budget=7_500,
    )
    assert config.max_simulator_transitions == 1_500
    assert config.selection_only
