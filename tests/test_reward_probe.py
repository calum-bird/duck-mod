"""The design's central invariant, as a test: stillness must not be a trap.

Run 1 trained a statue because the reward stack paid one. These tests run the
probe's behaviour scoring through the real reward terms and pin the three facts
the redesign rests on. If a future weight or std edit breaks any of them, this
fails on CPU in seconds instead of on an H100 in dollars.
"""

import importlib.util
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_spec = importlib.util.spec_from_file_location("reward_probe", _SCRIPTS / "reward_probe.py")
probe = importlib.util.module_from_spec(_spec)
sys.modules["reward_probe"] = probe
_spec.loader.exec_module(probe)

from microduck_dance.mdp import ENERGY_WEIGHT, POWER_WEIGHT  # noqa: E402

CLUMSY = probe.BEHAVIOURS["clumsy"]
GOOD = probe.BEHAVIOURS["good"]


def _total(behaviour, std, power, energy, weights=None):
    saved = dict(probe.BASE_WEIGHTS)
    if weights is not None:
        probe.BASE_WEIGHTS.clear()
        probe.BASE_WEIGHTS.update(weights)
    try:
        return sum(probe.score(*behaviour, std, power, energy).values())
    finally:
        probe.BASE_WEIGHTS.clear()
        probe.BASE_WEIGHTS.update(saved)


def test_stage_one_has_a_gradient_out_of_stillness():
    """A clumsy attempt — 40% amplitude, a quarter-beat late, fallen 45% of
    steps — must out-earn a statue under the shipped stage-1 weights."""
    still = _total((0.0, 0.0, 0.0), 0.006, POWER_WEIGHT, ENERGY_WEIGHT)
    clumsy = _total(CLUMSY, 0.006, POWER_WEIGHT, ENERGY_WEIGHT)
    assert clumsy - still > 0.3, (clumsy, still)


def test_late_stage_never_pays_reverting_to_a_statue():
    full = {"beat_bob": 4.0, "beat_sway": 2.0,
            "station_keeping_penalty": 20.0, "heading_drift_penalty": 4.0}
    still = _total((0.0, 0.0, 0.0), 0.006, POWER_WEIGHT, ENERGY_WEIGHT * 0.5, full)
    mediocre = _total((0.7, 0.05, 0.20), 0.006, POWER_WEIGHT, ENERGY_WEIGHT * 0.5, full)
    assert mediocre - still > 1.0, (mediocre, still)


def test_probe_still_retrodicts_run_one():
    """The probe is only trustworthy while it explains the failure that
    actually happened: run 1's stack (position Gaussian at full weight,
    std=0.010, no motion terms) must score the statue ABOVE the clumsy
    attempt. If this starts passing the old config, the probe has been
    tuned into a yes-machine and every verdict it gives is worthless."""
    run1 = {"beat_bob": 4.0, "beat_sway": 0.0,
            "station_keeping_penalty": 20.0, "heading_drift_penalty": 4.0}
    still = _total((0.0, 0.0, 0.0), 0.010, 0.0, 0.0, run1)
    clumsy = _total(CLUMSY, 0.010, 0.0, 0.0, run1)
    assert still - clumsy > 0.5, (still, clumsy)
