"""The Sock Heist's rules for training: the event's numbers and the Arena's arithmetic, copied, and
LightsCommand, the twist that plays them to the policy.

The training job installs no Arena, so nothing here is imported from it;
tests/test_sock_heist_6m.py pins each copy against it.
"""
import math

import torch
from mjlab_microduck.tasks import mdp as microduck_mdp

CTRL_DT = 0.02                 # arena.constants
SETTLE_S = 0.5                 # [start] settle_s
FIRST_GREEN_S = (0.6, 1.0)     # [lights] first_green_s
GREEN_S = (1.6, 3.0)           # [lights] green_s
RED_S = (1.4, 2.4)             # [lights] red_s
TURN_S = 0.4                   # [lights] turn_s
TOLERANCE_M = 0.03             # [lights] tolerance_m
LOOKAHEAD_M = 0.5              # [steering] lookahead_m
GAIN = 2.0                     # [steering] gain
MAX_YAW_RATE = 1.0             # [steering] max_yaw_rate
# The Arena's [course] width_m halved, less 0.1 m: the trunk's |y| at its first wall contact measured 0.28-0.37 m.
LANE_HALF_WIDTH_M = 0.3


def steps(seconds: float) -> int:
    return round(seconds / CTRL_DT)


def _draw(span: tuple[float, float], shape: tuple[int, ...], device) -> torch.Tensor:
    """Whole steps, both ends included: arena.lights.timetable's draw."""
    return torch.randint(steps(span[0]), steps(span[1]) + 1, shape, device=device)


def draw_reds(n: int, episode_steps: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Each env's reds as (onset, length), (n, R) int64, onsets counted from the episode's first step.
    R is enough for the last onset never to be reached."""
    settle = steps(SETTLE_S)
    r = math.ceil((episode_steps - settle - steps(FIRST_GREEN_S[0])) / (steps(GREEN_S[0]) + steps(RED_S[0]))) + 1
    length, green = _draw(RED_S, (n, r), device), _draw(GREEN_S, (n, r), device)
    gap = length + green
    onset = settle + _draw(FIRST_GREEN_S, (n, 1), device) + torch.cumsum(gap, dim=1) - gap
    return onset, length


def light_steps(onset: torch.Tensor, length: torch.Tensor, turn: int, k: torch.Tensor):
    """(red, mark, watch) at each env's step k, (n,) bool: arena.runner._LightSteps at one step."""
    k = k[:, None]
    red = (onset <= k) & (k < onset + length)
    mark = k == onset + turn - 1
    watch = (onset + turn <= k) & (k < onset + length)
    return red.any(1), mark.any(1), watch.any(1)


def yaw_of(quat: torch.Tensor) -> torch.Tensor:
    """(n,): the yaw of the trunk's wxyz quaternion (n, 4), tilted or not, as arena.runner._steered reads it."""
    w, qx, qy, qz = quat.unbind(-1)
    return torch.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def hold_yaw_rate(y: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """The yaw rate that turns the duck toward the lane's centre line `LOOKAHEAD_M` ahead (arena.runner._steered).
    `y` is the offset from the lane (n,), `quat` the trunk's wxyz (n, 4)."""
    error = torch.remainder(torch.atan2(-y, torch.full_like(y, LOOKAHEAD_M)) - yaw_of(quat) + math.pi, 2.0 * math.pi) - math.pi
    return torch.clamp(GAIN * error, -MAX_YAW_RATE, MAX_YAW_RATE)


class LightsCommandCfg(microduck_mdp.VelocityCommandCommandOnlyCfg):
    """The twist under the giant's eye: LightsCommand."""

    def build(self, env) -> "LightsCommand":
        return LightsCommand(self, env)


class LightsCommand(microduck_mdp.VelocityCommandCommandOnly):
    """The twist the policy reads, as arena.runner.run_once asks it: zeros through the settle and every
    red, from its first step to its last, and on green `[speed, 0, hold]`, the hold turning the duck
    toward its lane. Each episode draws one forward speed (the base draw, over the sweep) and its reds
    (draw_reds). The lane runs along +x through the spawn point, as the Arena's runs along its x axis
    from y = 0. The mark is where the trunk stands on the floor once the head has finished turning,
    and where it entered the episode until then: mjlab's train starts rsl_rl's learn at random episode
    lengths, moving every env to a random step after its first reset without another, so an env may
    start inside a watch, past its mark step.

    mjlab calls _update_command after a step's physics, terminations, rewards and resets: `stepped()`
    is the step just simulated, and the twist is for the step about to run.
    """

    turn: int = steps(TURN_S)

    def __init__(self, cfg: LightsCommandCfg, env):
        super().__init__(cfg, env)
        self.speed = torch.zeros(self.num_envs, device=self.device)
        self.onset, self.length = draw_reds(self.num_envs, env.max_episode_length, self.device)
        self.lane_y = torch.zeros(self.num_envs, device=self.device)
        self.mark = torch.zeros(self.num_envs, 2, device=self.device)

    @property
    def trunk(self) -> torch.Tensor:
        """(n, 7): the trunk's free-joint qpos, its position then its wxyz quaternion, as run_once reads it."""
        return self.robot.data.data.qpos[:, self.robot.indexing.free_joint_q_adr]

    def stepped(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(red, mark, watch) at the step just simulated."""
        return light_steps(self.onset, self.length, self.turn, self._env.episode_length_buf - 1)

    def ahead(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(standing, watch) at the step about to run: standing through the settle and every red."""
        k = self._env.episode_length_buf
        red, _, watch = light_steps(self.onset, self.length, self.turn, k)
        return (k < steps(SETTLE_S)) | red, watch

    def drift(self) -> torch.Tensor:
        """(n,): how far the trunk stands across the floor from its mark, in m."""
        return torch.linalg.vector_norm(self.trunk[:, 0:2] - self.mark, dim=-1)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        self.speed[env_ids] = self.vel_command_b[env_ids, 0]
        self.onset[env_ids], self.length[env_ids] = draw_reds(len(env_ids), self._env.max_episode_length, self.device)

    def _update_command(self) -> None:
        trunk, fresh = self.trunk, self._env.episode_length_buf == 0
        _, mark, _ = self.stepped()
        self.mark[:] = torch.where((mark | fresh)[:, None], trunk[:, 0:2], self.mark)
        self.lane_y[:] = torch.where(fresh, trunk[:, 1], self.lane_y)
        standing, _ = self.ahead()
        hold = hold_yaw_rate(trunk[:, 1] - self.lane_y, trunk[:, 3:7])
        walk = torch.stack([self.speed, torch.zeros_like(hold), hold], dim=-1)
        self.vel_command_b[:] = torch.where(standing[:, None], 0.0, walk)
