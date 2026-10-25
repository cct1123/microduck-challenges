"""Small opt-in reward ablations; no implicit combination or training launch."""

import math


REWARD_PROFILES = {
    "A0": {},
    "A1": {"track_linear_velocity": 0.0},
    "B1": {"air_time": 3.0},
    "B2": {"speed_forward": 18.0},
    "C1": {"track_linear_velocity": 0.0, "air_time": 3.0},
}


def reward_overrides(profile):
    """Return a copy so a caller cannot mutate the registered experiment."""
    if not isinstance(profile, str) or profile not in REWARD_PROFILES:
        raise ValueError(f"Unknown Sprint reward profile: {profile!r}")
    return dict(REWARD_PROFILES[profile])


def resolve_reward_profile(training, stage, override=None):
    """Use the shared recipe for refinement; archived stages stay unmodified."""
    profile = override if override is not None else (
        training.get("reward_profile", "A0") if stage == "refine" else "A0"
    )
    reward_overrides(profile)
    if stage != "refine" and profile != "A0":
        raise ValueError("Reward experiments require the strict refinement warm-start path")
    return profile


def apply_reward_profile(cfg, profile):
    """Apply ONLY after source validation; freeze matching curriculum endpoints."""
    overrides = reward_overrides(profile)
    for name, weight in overrides.items():
        if name not in cfg.rewards or not math.isfinite(weight):
            raise ValueError(f"Reward profile {profile} requires {name}")
        cfg.rewards[name].weight = weight
        for curriculum in cfg.curriculum.values():
            params = curriculum.params
            if name in params.get("targets", {}):
                params["targets"][name] = (weight, weight)
            if params.get("reward_name") == name:
                for entry in params.get("weight_stages", []):
                    entry["weight"] = weight
    return cfg
