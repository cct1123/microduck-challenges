"""Sock Heist's own terms: freezing while the giant watches, walking on green, being caught, and what the critic alone is
told of the light and the lane. Each reads the LightsCommand (lights.py) named `command_name`."""

import torch

from . import lights


def freeze(env, command_name: str, std: float) -> torch.Tensor:
    """(n,): exp(-(drift / std)²) on the steps the giant watches, else 0. At std = the tolerance, a duck
    on its mark earns 1 and one at the edge of being caught e^-1."""
    term = env.command_manager.get_term(command_name)
    _, _, watch = term.stepped()
    return torch.where(watch, torch.exp(-((term.drift() / std) ** 2)), 0.0)


def progress(env, command_name: str) -> torch.Tensor:
    """Forward speed, up to the command: pays walking on green and nothing on red (the command is zero)."""
    vx = env.scene["robot"].data.root_link_lin_vel_b[:, 0].clamp(min=0.0)
    return torch.minimum(vx, env.command_manager.get_command(command_name)[:, 0])


def caught(env, command_name: str) -> torch.Tensor:
    """(n,) bool: watched, and more than TOLERANCE_M from its mark, as run_once catches a duck."""
    term = env.command_manager.get_term(command_name)
    _, _, watch = term.stepped()
    return watch & (term.drift() > lights.TOLERANCE_M)


def off_lane(env, command_name: str, half_width: float) -> torch.Tensor:
    """(n,) bool: the trunk more than `half_width` off its lane. The training floor has no walls, so this
    stands for them: heist3 learned to ignore its turn command and veered, which the Arena's wall ends
    within a second, caught or fallen, where nothing in training did."""
    term = env.command_manager.get_term(command_name)
    return (term.trunk[:, 1] - term.lane_y).abs() > half_width


def light_state(env, command_name: str) -> torch.Tensor:
    """(n, 4), for the step about to run: standing (the settle or a red), watched, the seconds until the
    twist next changes (the settle ends, a red starts or ends), and the drift over TOLERANCE_M once
    marked (0 before the mark: in the settle, on green and while the head turns)."""
    term = env.command_manager.get_term(command_name)
    standing, watch = term.ahead()
    k = env.episode_length_buf[:, None]
    settle = torch.full_like(term.onset[:, :1], lights.steps(lights.SETTLE_S))
    changes = torch.cat([settle, term.onset, term.onset + term.length], dim=1)
    until = (changes - k).masked_fill(changes <= k, torch.iinfo(changes.dtype).max).amin(dim=1)
    drift = torch.where(watch, term.drift() / lights.TOLERANCE_M, 0.0)
    return torch.stack([standing.float(), watch.float(), until * lights.CTRL_DT, drift], dim=-1)


def lane_state(env, command_name: str) -> torch.Tensor:
    """(n, 2): the trunk's y off its lane (+ left of it) and its yaw, what the lane's hold steers by."""
    term = env.command_manager.get_term(command_name)
    trunk = term.trunk
    return torch.stack([trunk[:, 1] - term.lane_y, lights.yaw_of(trunk[:, 3:7])], dim=-1)
