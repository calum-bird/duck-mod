"""The beat clock command term.

Occupies the ``twist`` command slot (see ``beat_clock`` for why) and advances a
bar phase at whatever tempo the episode -- or, at deployment, the host -- is
running. Subclasses ``UniformVelocityCommand`` for the same reason
``GroundPickPhaseCommand`` upstream does: the velocity command already owns the
3-wide slot and its plumbing (buffer allocation, manager registration, obs
wiring), so overriding ``compute`` is the whole change.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)

from microduck_dance import beat_clock


class BeatClockCommand(UniformVelocityCommand):
    """Cyclic beat clock: ``[cos(2πφ), sin(2πφ), tempo_norm]``.

    Phase and tempo are per-environment. Tempo is drawn on episode reset, so
    a single batch spans the whole BPM envelope and the policy must actually
    read twist[2] rather than memorising one cadence. With
    ``tempo_change_prob > 0`` the tempo can also jump mid-episode at the
    resampling interval (phase stays continuous, as when a track changes),
    which is what teaches the policy to re-lock instead of free-running.
    """

    cfg: "BeatClockCommandCfg"

    def __init__(self, cfg: "BeatClockCommandCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._bar_phase = torch.zeros(self.num_envs, device=self.device)
        self._bpm = torch.full(
            (self.num_envs,), float(cfg.bpm_range[0]), device=self.device
        )

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    @property
    def bar_phase(self) -> torch.Tensor:
        """Live bar phase, for tests and debug visualisation."""
        return self._bar_phase

    @property
    def bpm(self) -> torch.Tensor:
        return self._bpm

    def _sample_bpm(self, n: int) -> torch.Tensor:
        lo, hi = self.cfg.bpm_range
        return torch.empty(n, device=self.device).uniform_(float(lo), float(hi))

    def compute(self, dt: float) -> None:
        # One bar spans two beats: period = 2 * 60/bpm = 120/bpm seconds.
        bar_period = 120.0 / self._bpm.clamp(min=1.0)
        self._bar_phase = torch.remainder(self._bar_phase + dt / bar_period, 1.0)
        self.vel_command_b[:, 0] = torch.cos(beat_clock.TWO_PI * self._bar_phase)
        self.vel_command_b[:, 1] = torch.sin(beat_clock.TWO_PI * self._bar_phase)
        # Encode against the FIXED global envelope, never cfg.bpm_range: the
        # curriculum widens bpm_range over training, and encoding against a
        # moving range would (a) shift what twist[2] means under the policy's
        # feet mid-run, (b) disagree with the reward terms decoding it via
        # norm_to_tempo's global constants, and (c) disagree with the host's
        # beat_source at deployment. The range narrows what is SAMPLED, not
        # how it is REPORTED.
        self.vel_command_b[:, 2] = beat_clock.tempo_to_norm(self._bpm)

    def reset(self, env_ids: torch.Tensor | None) -> dict:
        if env_ids is not None and len(env_ids) > 0:
            n = len(env_ids)
            self._bpm[env_ids] = self._sample_bpm(n)
            if self.cfg.randomize_phase:
                # Decorrelate envs: without this the whole batch shares one
                # downbeat and PPO sees a single phase trajectory per update.
                self._bar_phase[env_ids] = torch.rand(n, device=self.device)
            else:
                self._bar_phase[env_ids] = 0.0
        return {}

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # Tempo changes only; the phase is continuous and never resampled (a
        # jump would be a beat the policy could not have anticipated, and the
        # footfall reward would charge it for a strike it had no way to place).
        if self.cfg.tempo_change_prob <= 0.0 or len(env_ids) == 0:
            return
        n = len(env_ids)
        change = torch.rand(n, device=self.device) < self.cfg.tempo_change_prob
        if change.any():
            targets = env_ids[change]
            self._bpm[targets] = self._sample_bpm(int(change.sum()))

    def _update_command(self) -> None:
        pass  # advanced in compute()

    def _update_metrics(self) -> None:
        pass  # no velocity-tracking metrics: this slot is not a velocity


@dataclass(kw_only=True)
class BeatClockCommandCfg(UniformVelocityCommandCfg):
    class_type: type = BeatClockCommand
    # Tempo envelope sampled per episode. Narrowed early in training by the
    # beat_tempo_range_curriculum, which widens it once the bob is stable.
    bpm_range: tuple[float, float] = (beat_clock.BPM_MIN, beat_clock.BPM_MAX)
    # False -> every episode starts on a downbeat, matching a deployment where
    # the host starts the clock when the music starts.
    randomize_phase: bool = True
    # Probability that a resample tick changes the tempo mid-episode.
    tempo_change_prob: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> "BeatClockCommand":
        return BeatClockCommand(self, env)
