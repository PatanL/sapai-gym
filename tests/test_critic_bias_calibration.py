from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_critic_bias_calibration import analyze  # noqa: E402


def test_shard_held_out_bias_calibration_improves_synthetic_rows():
    rng = np.random.default_rng(31)
    rows = []
    for shard in range(8):
        for index in range(40):
            turn = (2, 6, 11)[index % 3]
            position = ("ahead", "even", "behind")[(index // 3) % 3]
            observed = float(rng.normal())
            systematic_bias = (
                0.5
                + (0.4 if turn >= 9 else -0.2 if turn >= 5 else 0.0)
                + (0.6 if position == "behind" else -0.1 if position == "even" else 0.0)
            )
            rows.append(
                {
                    "shard": shard,
                    "turn": turn,
                    "position": position,
                    "predicted": observed + systematic_bias,
                    "monteCarlo": observed,
                }
            )
    report = analyze({"rows": rows}, ridge=1.0)
    assert report["recommended"] == "use-additive-bias"
    assert report["maeImprovement"] > 0.3
    assert (
        report["calibratedMaxAbsSliceBias"]
        < report["rawMaxAbsSliceBias"]
    )
