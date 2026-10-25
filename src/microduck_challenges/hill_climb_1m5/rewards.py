"""H1 hypotheses on generated Hill terrain, not fixed-slope race progress.

Forward means horizontal world velocity projected onto the spawn heading.
Upright remains the optimum; only nose-down pitch tolerance is relaxed while
actually travelling forward. No robot/actuator limits are changed.
"""
import math

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

H1 = {"speed_weight": 5.0, "vel_ref": 1.0, "forward_variance": .30,
      "motion_speed": .1, "hold_steps": 50 * 24, "ramp_steps": 200 * 24}


def _fraction(env, origin_step, hold_steps, ramp_steps):
    if (type(origin_step) is not int or origin_step <= 0 or type(hold_steps) is not int
            or hold_steps < 0 or type(ramp_steps) is not int or ramp_steps <= 0):
        raise ValueError("H1 requires a positive source origin and bounded hold/ramp")
    return min(1., max(0., (env.common_step_counter - origin_step - hold_steps) / ramp_steps))


def _motion(env):
    data = env.scene["robot"].data
    quat = data.root_link_quat_w
    w, x, y, z = quat.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    # Match heading_hold on reset, without depending on reward execution order.
    reference = torch.where(env.episode_length_buf <= 1, yaw, getattr(env, "_heading_ref", yaw))
    velocity = data.root_link_lin_vel_w
    speed = reference.cos() * velocity[:, 0] + reference.sin() * velocity[:, 1]
    gravity = quat_apply_inverse(quat, torch.tensor([0., 0., -1.], device=quat.device).expand_as(velocity))
    valid = (torch.isfinite(quat).all(-1) & torch.isfinite(velocity).all(-1)
             & torch.isfinite(gravity).all(-1) & torch.isfinite(reference)
             & (gravity[:, 2] <= -.5))
    guard = getattr(env, "_hill_tilt_guard", None)
    if guard is not None:
        valid &= ~guard.failed
    return torch.where(valid, speed, 0.), gravity, valid


def horizontal_forward_speed(env, origin_step, hold_steps, ramp_steps, vel_ref=1.):
    """Bounded SIGNED horizontal spawn-course speed, not along-slope distance."""
    if not math.isfinite(vel_ref) or vel_ref <= 0:
        raise ValueError("vel_ref must be finite and positive")
    speed, _, valid = _motion(env)
    value = _fraction(env, origin_step, hold_steps, ramp_steps) * torch.tanh(speed / vel_ref)
    return torch.where(valid, value, 0.)


def drive_upright(env, std, forward_variance, motion_speed, origin_step, hold_steps, ramp_steps):
    """World-up optimum; relax positive gravity-x (nose-down) pitch only."""
    if (not all(math.isfinite(v) and v > 0 for v in (std, forward_variance, motion_speed))
            or forward_variance < std**2):
        raise ValueError("Invalid H1 upright tolerance")
    speed, gravity, valid = _motion(env)
    blend = _fraction(env, origin_step, hold_steps, ramp_steps) * (speed / motion_speed).clamp(0., 1.)
    variance = std**2 + blend * (forward_variance - std**2)
    pitch_variance = torch.where(gravity[:, 0] > 0, variance, std**2)
    value = torch.exp(-gravity[:, 0].square() / pitch_variance - gravity[:, 1].square() / std**2)
    return torch.where(valid, value, 0.)


class ArenaTiltGuard:
    """Independent >60 degree / 0.3 s fall guard; existing hard guard unchanged."""
    def __init__(self, cfg, env):
        del cfg
        self.hold_steps = max(1, round(.3 / env.step_dt))
        self.steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._hill_tilt_guard = self

    def __call__(self, env):
        quat = env.scene["robot"].data.root_link_quat_w
        gravity = quat_apply_inverse(quat, torch.tensor([0., 0., -1.], device=quat.device).expand(quat.shape[0], 3))
        tipped = gravity[:, 2] > -.5
        self.steps[:] = torch.where(tipped, (self.steps + 1).clamp_max(self.hold_steps), 0)
        self.failed[:] = (self.steps >= self.hold_steps) | ~torch.isfinite(gravity).all(-1)
        return self.failed

    def reset(self, env_ids=None):
        rows = slice(None) if env_ids is None else env_ids
        self.steps[rows], self.failed[rows] = 0, False


def configure_h1(cfg, origin_step):
    from mjlab.managers import RewardTermCfg, TerminationTermCfg
    schedule = {k: H1[k] for k in ("hold_steps", "ramp_steps")}
    schedule["origin_step"] = origin_step
    cfg.rewards["horizontal_forward_speed"] = RewardTermCfg(
        func=horizontal_forward_speed, weight=H1["speed_weight"],
        params={**schedule, "vel_ref": H1["vel_ref"]})
    upright = cfg.rewards["upright"]
    # Keep the source's weight and baseline std; do not weaken roll globally.
    upright.func = drive_upright
    upright.params = {**schedule, "std": upright.params["std"],
                      **{k: H1[k] for k in ("forward_variance", "motion_speed")}}
    cfg.terminations["arena_held_fall"] = TerminationTermCfg(func=ArenaTiltGuard, time_out=False)
    return cfg
