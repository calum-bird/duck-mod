"""Task registration.

Discovered automatically through the ``mjlab.tasks`` entry point declared in
``pyproject.toml``, so ``uv run train Mjlab-BeatDance-Flat-MicroDuck`` and
``uv run list-envs`` see these tasks without any CLI wrapper -- the same
mechanism upstream uses for its own thirteen task families.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_BACKLASH_ROBOT_CFG
from mjlab_microduck.tasks import MicroduckOnPolicyRunner
from mjlab_microduck.tasks.backlash import make_backlash_variant

from microduck_dance.beat_dance_env_cfg import (
    MicroduckBeatDanceRlCfg,
    make_microduck_beat_dance_env_cfg,
)

register_mjlab_task(
    task_id="Mjlab-BeatDance-Flat-MicroDuck",
    env_cfg=make_microduck_beat_dance_env_cfg(),
    play_env_cfg=make_microduck_beat_dance_env_cfg(play=True),
    rl_cfg=MicroduckBeatDanceRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Backlash twin: +-1 deg of serial gear play per servo, with the encoder reading
# back THROUGH the play. Every shipped task family has one, and for this task it
# is not optional garnish -- the whole skill is timing, and gear play is exactly
# what puts a real foot strike a few milliseconds off where the policy placed
# it. Train the backlash variant before deploying to hardware.
register_mjlab_task(
    task_id="Mjlab-BeatDance-Flat-Backlash-MicroDuck",
    env_cfg=make_backlash_variant(
        make_microduck_beat_dance_env_cfg(), MICRODUCK_WALK_BACKLASH_ROBOT_CFG
    ),
    play_env_cfg=make_backlash_variant(
        make_microduck_beat_dance_env_cfg(play=True), MICRODUCK_WALK_BACKLASH_ROBOT_CFG
    ),
    rl_cfg=MicroduckBeatDanceRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
