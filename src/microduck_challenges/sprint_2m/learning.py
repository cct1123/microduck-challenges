"""Explicit, bounded learning controls; no change to the robot or reward objective."""

import math
from dataclasses import dataclass

from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg


@dataclass
class SprintRefinePpoCfg(PpoWithSymmetryCfg):
    """Record intentional phase-boundary noise scaling in saved agent configs."""

    exploration_std_scale: float = 1.0


def phase_settings(training, phase):
    """Select a validated profile without silently falling back to another phase."""
    if phase not in ("explore", "exploit"):
        raise ValueError(f"Unknown Sprint learning phase: {phase!r}")
    settings = dict(training["phases"][phase])
    for key in ("max_iterations", "save_interval"):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f"{phase}.{key} must be a positive integer")
    if settings["save_interval"] > settings["max_iterations"]:
        raise ValueError("Checkpoint interval must fit within the phase budget")
    for key in ("learning_rate", "entropy_coef", "clip_param", "exploration_std_scale"):
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{phase}.{key} must be finite and positive")
    if settings["clip_param"] >= 1 or settings["exploration_std_scale"] > 1:
        raise ValueError("Clip must be below one and noise scale must not exceed one")
    return settings
