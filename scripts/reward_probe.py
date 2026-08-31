#!/usr/bin/env python3
"""Ask the reward stack what behaviour it actually pays for — before renting a GPU.

Run 1 did not fail at training. It failed at arithmetic: every positive term
was a position-tracking Gaussian, standing at the reference's mean collected a
risk-free ~45% of it, and clumsy first attempts at dancing — which fall — earned
LESS than doing nothing. A local optimum with no gradient out of it. PPO found
it in twenty minutes and $1.30; the arithmetic was knowable for free.

The question that decides a run is therefore NOT "does good dancing out-earn
standing still" (it always does — run 1's config passes that test, which is how
the first version of this probe green-lit the very failure it was written to
explain). It is:

    does CLUMSY tracking out-earn standing still?

A clumsy policy here moves smoothly but wrong: attenuated amplitude, lagged
phase, frequent falls. That is what early PPO actually produces — not
iid position noise, whose finite-difference velocity would be nonsense.

This drives the REAL reward terms in `microduck_dance.mdp` through the tests'
stub env, so it prices their interactions (the gate zeroing falls, station-
keeping charging the sway) and any term you add lands here automatically.

    python scripts/reward_probe.py             # verdict on the shipped stack
    python scripts/reward_probe.py --sweep     # energy x power weight table
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

import torch

from conftest_shim import StubEnv
from microduck_dance import beat_clock as bc
from microduck_dance import mdp as dance_mdp

# The SHIPPED stage-1 stack from beat_dance_env_cfg: precision Gaussians start
# small (bob) or off (sway) and ramp later; the motion terms carry the early
# incentive. test_beat_dance_cfg asserts these match the cfg's step-0 values.
BASE_WEIGHTS = {
    "beat_bob": 1.0,
    "beat_sway": 0.0,
    "station_keeping_penalty": 20.0,
    "heading_drift_penalty": 4.0,
}


def score(
    amp_factor: float,
    phase_lag: float,
    fall_fraction: float,
    std: float,
    power_weight: float,
    energy_weight: float,
    bpm: float = 120.0,
    seconds: float = 8.0,
    bob: float = 0.014,
    sway: float = 0.012,
    wander: float = 0.0,
) -> dict[str, float]:
    """Mean per-step weighted reward for one behaviour.

    The behaviour executes the commanded choreography at ``amp_factor`` of its
    amplitude, ``phase_lag`` bars late, fallen for ``fall_fraction`` of steps.
    (0, 0, 0) is standing still; (1, ~0, low) is the trained destination.
    """
    env = StubEnv(num_envs=1)
    dt = 0.02
    bar_period = 120.0 / bpm
    steps = int(seconds / dt)
    totals: dict[str, float] = defaultdict(float)

    weights = dict(BASE_WEIGHTS)
    weights["beat_bob_power"] = power_weight
    weights["beat_sway_power"] = power_weight * 0.5
    weights["beat_bob_energy"] = energy_weight
    weights.setdefault("fallen_tax", 0.0)

    for i in range(steps):
        phase = (i * dt / bar_period) % 1.0
        twist = torch.zeros(1, 3)
        twist[0, 0] = math.cos(2 * math.pi * phase)
        twist[0, 1] = math.sin(2 * math.pi * phase)
        twist[0, 2] = float(bc.tempo_to_norm(torch.tensor([bpm])))
        body = torch.zeros(1, 6)
        body[0, 1], body[0, 2] = sway, bob
        env.set_commands(twist=twist, body_pose=body)

        lagged = torch.tensor([(phase - phase_lag) % 1.0])
        a_bob = torch.tensor(amp_factor * bob)
        a_sway = torch.tensor(amp_factor * sway)
        bpm_t = torch.tensor([bpm])

        z = float(bc.bob_reference(lagged, a_bob, bc.NOMINAL_HEIGHT))
        lat = float(bc.sway_reference(lagged, a_sway)) + wander * (i / steps)
        vz = float(bc.bob_velocity_reference(lagged, a_bob, bpm_t))
        vlat = float(bc.sway_velocity_reference(lagged, a_sway, bpm_t))

        fallen = (i % 100) < int(100 * fall_fraction)
        env.set_height(0.050 if fallen else z)
        env.set_xy(torch.tensor([[0.0, lat]]))
        env.set_lin_vel(torch.tensor([[0.0, vlat, vz]]))

        for name, weight in weights.items():
            if weight == 0.0:
                continue
            fn = {
                "beat_bob": lambda e: dance_mdp.beat_bob_tracking(e, std=std),
                "beat_sway": dance_mdp.beat_sway_tracking,
                "beat_bob_power": dance_mdp.beat_bob_power,
                "beat_bob_energy": dance_mdp.beat_bob_energy,
                "fallen_tax": dance_mdp.fallen_tax,
                "beat_sway_power": dance_mdp.beat_sway_power,
                "station_keeping_penalty": dance_mdp.station_keeping_penalty,
                "heading_drift_penalty": dance_mdp.heading_drift_penalty,
            }[name]
            totals[name] += float(fn(env).mean()) * weight

    return {k: v / steps for k, v in totals.items()}


BEHAVIOURS = {
    # name: (amp_factor, phase_lag in bars, fall fraction)
    "still": (0.0, 0.0, 0.0),
    "clumsy": (0.4, 0.12, 0.45),   # falls per run 1's measured 53%-open gate
    "good": (1.0, 0.02, 0.10),
}


def gradient(std: float, power_weight: float, energy_weight: float) -> tuple[float, float, dict, dict]:
    still = score(*BEHAVIOURS["still"], std, power_weight, energy_weight)
    clumsy = score(*BEHAVIOURS["clumsy"], std, power_weight, energy_weight)
    good = score(*BEHAVIOURS["good"], std, power_weight, energy_weight)
    s, c, g = (sum(x.values()) for x in (still, clumsy, good))
    return c - s, g - s, still, clumsy


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--std", type=float, default=0.006)
    p.add_argument("--power-weight", type=float, default=None,
                   help="beat_bob_power weight (sway power rides at half); "
                        "default reads the value the cfg ships")
    p.add_argument("--sweep", action="store_true", help="table over std x power weight")
    args = p.parse_args()

    if args.sweep:
        print("Gradient out of stillness (clumsy - still), per step. Positive = escapable.")
        print("Rows: energy weight. Columns: power weight. std = 6 mm.\n")
        header = f"{'energy w':>9}" + "".join(f"{f'pw={w:.0f}':>9}" for w in (0.0, 2.0, 4.0, 6.0))
        print(header)
        print("-" * len(header))
        for ew in (0.0, 1.0, 2.0, 3.0, 4.0):
            row = f"{ew:9.1f}"
            for pw in (0.0, 2.0, 4.0, 6.0):
                out, _, _, _ = gradient(0.006, pw, ew)
                row += f"{out:9.2f}"
            print(row)
        print("\nRetrodiction row: energy 0 / power 0 must stay negative, or the probe")
        print("no longer explains run 1.")
        return 0

    from microduck_dance.mdp import ENERGY_WEIGHT, POWER_WEIGHT  # noqa: PLC0415

    power = args.power_weight if args.power_weight is not None else POWER_WEIGHT
    energy = ENERGY_WEIGHT if args.power_weight is None else (
        0.0 if args.power_weight == 0.0 else ENERGY_WEIGHT
    )
    out, dest, still, clumsy = gradient(args.std, power, energy)
    terms = sorted(set(still) | set(clumsy))
    print(f"std={args.std:.3f}  power={power:.1f}  energy={energy:.1f}  "
          f"(clumsy: 40% amplitude, 0.12 bar late, 45% fallen)\n")
    print(f"{'term':<28}{'stand still':>13}{'clumsy':>10}")
    print("-" * 51)
    for t in terms:
        print(f"{t:<28}{still.get(t, 0):13.3f}{clumsy.get(t, 0):10.3f}")
    print("-" * 51)
    print(f"{'TOTAL per step':<28}{sum(still.values()):13.3f}{sum(clumsy.values()):10.3f}")
    print(f"\ngradient out of stillness : {out:+.3f}/step")
    print(f"destination advantage     : {dest:+.3f}/step")
    if out <= 0:
        print("\nVERDICT: STILLNESS IS A TRAP — clumsy tracking earns less than doing\n"
              "nothing; the policy will stand still however long you train (run 1).")
        return 1
    print("\nVERDICT: a gradient out of stillness exists and the destination pays.\n"
          "Worth a GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
