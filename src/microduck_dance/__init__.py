"""Beat-synced dance task for the Microduck, layered on ``mjlab_microduck``.

This package does not fork the upstream training stack -- it imports it and
registers additional mjlab tasks, so upstream fixes to the robot model, the BAM
actuator model and the domain randomisation flow straight through.
"""

__all__ = ["beat_clock", "mdp", "commands", "beat_dance_env_cfg", "tasks"]
