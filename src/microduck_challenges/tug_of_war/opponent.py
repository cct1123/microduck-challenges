"""The Tug of War's opponent: the Arena's reference duck, frozen (Plan F, Decisions 3–7).

`reference.onnx` is t20-500, trained walking upright (Plan I), the bytes the Arena vendors as its
reference duck. `load_actor` reads it into torch, so it runs on the training's device with the envs'
batch. `OpponentAction` drives the `opponent` entity with it: an action term that takes none of the
agent's action. mjlab calls it at the start of each step, on the state the learner's observation
saw, so both ducks act on one state, as in the Arena's `runner._step`. It builds the 61 numbers
`runner._observe` gives a duck and sets the opponent's joint targets as the learner's are set, into
its own BAM servos.
`ReferenceActor`, the learner's actor in RL_CFG, starts it as the same duck (Plan F, lever 5).
The HF job installs no Arena: nothing here imports it; tests/test_tug_of_war.py pins each copy.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import onnx
import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from onnx import helper, numpy_helper
from rsl_rl.models import MLPModel

REFERENCE = Path(__file__).with_name("reference.onnx")  # the Arena's reference duck, t20-500, walking upright (Plan I)
REFERENCE_SEEN = 1003 * 24 * 4096  # t20-500's normalizer count: rollouts (its walker start's 502, its own 501) × steps × envs
WARM_START_STD = 0.5  # the warm-started learner's action noise: t20-500's own at 500, 0.36 to 0.70, mean 0.50, it walked upright under

# The actor rsl_rl exports (Decision 4), node by node: its op and attributes, ONNX's defaults in.
_GEMM = ("Gemm", {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 1})
_ELU = ("Elu", {"alpha": 1.0})
_ACTOR = (("Sub", {}), ("Div", {}), _GEMM, _ELU, _GEMM, _ELU, _GEMM, _ELU, _GEMM)
_ONNX_DEFAULTS = {"Gemm": {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0}, "Elu": {"alpha": 1.0}}


class Normalize(torch.nn.Module):
    """(x - mean) / std: rsl_rl's obs_normalizer as it exports it, its epsilon in `std`."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def load_actor(path: Path) -> torch.nn.Sequential:
    """The actor at `path` as torch: Normalize, then a Linear per Gemm and an ELU per Elu.

    Reads `Sub(mean) → Div(std) → 3 × (Gemm transB=1 → Elu α=1) → Gemm` and nothing else: a
    ValueError names the first node outside it.
    """
    graph = onnx.load(path).graph
    nodes = list(graph.node)
    for node, (op, attributes) in zip(nodes, _ACTOR):
        given = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type != op or {**_ONNX_DEFAULTS.get(op, {}), **given} != attributes:
            raise ValueError(f"{path}: node {node.name!r} is {node.op_type} {given}, the actor's is {op} {attributes}")
    if len(nodes) != len(_ACTOR):
        raise ValueError(f"{path}: {len(nodes)} nodes, the actor has {len(_ACTOR)}")
    weights = {w.name: torch.tensor(numpy_helper.to_array(w)) for w in graph.initializer}
    sub, div, *layers = nodes
    modules = [Normalize(weights[sub.input[1]], weights[div.input[1]])]
    for node in layers:
        if node.op_type == "Elu":
            modules.append(torch.nn.ELU())
            continue
        weight, bias = weights[node.input[1]], weights[node.input[2]]
        linear = torch.nn.Linear(weight.shape[1], weight.shape[0])
        linear.load_state_dict({"weight": weight, "bias": bias})
        modules.append(linear)
    return torch.nn.Sequential(*modules)


class ReferenceActor(MLPModel):
    """The learner's actor (RL_CFG's), starting as the reference duck (`--agent.actor.class-name
    MLPModel` starts from scratch instead).

    Its layers and its observation normalizer are `reference.onnx`'s (a shape or a missing
    normalizer refuses to load), its action noise std is WARM_START_STD; the critic starts fresh.
    The ONNX's std carries the normalizer's epsilon. The normalizer counts the observations t20-500
    had read: from 0, PPO's first update would replace its statistics with one batch's.
    """

    def __init__(self, *args, distribution_cfg: dict, **kwargs):
        super().__init__(*args, distribution_cfg={**distribution_cfg, "init_std": WARM_START_STD}, **kwargs)
        normalize, *layers = load_actor(REFERENCE)
        self.mlp.load_state_dict(torch.nn.Sequential(*layers).state_dict())
        std = normalize.std - self.obs_normalizer.eps
        self.obs_normalizer.load_state_dict(
            {"_mean": normalize.mean, "_var": std**2, "_std": std, "count": torch.tensor(REFERENCE_SEEN)}
        )


@dataclass(kw_only=True)
class OpponentActionCfg(JointPositionActionCfg):
    """The opponent's servos, driven by `reference.onnx` on the event's `command` (env.COMMAND)."""

    entity_name: str = "opponent"
    actuator_names: tuple[str, ...] = (".*",)
    command: tuple[float, ...]

    def build(self, env) -> "OpponentAction":
        return OpponentAction(self, env)


class OpponentAction(JointPositionAction):
    """The opponent duck, as the Arena's `runner._Duck`: each control step it observes, its policy
    acts, and its targets are the default pose plus the action (scale 1.0, the Arena's
    ACTION_SCALE), into its own BAM servos (the reset and fifth drives come with the scene's).
    Nothing randomises it (Decision 7), as on the Arena: no observation noise, delay or encoder
    bias, and its servos run on the Arena's nominal battery (env.OPPONENT_CFG)."""

    policy: Callable[[torch.Tensor], torch.Tensor]

    def __init__(self, cfg: OpponentActionCfg, env):
        super().__init__(cfg=cfg, env=env)
        self.policy = load_actor(REFERENCE).to(self.device).eval()
        self._command = torch.tensor(cfg.command, device=self.device).expand(self.num_envs, -1)
        # The parent sized its raw action by action_dim, 0: the opponent's is its policy's, (n, 14).
        self._raw_actions = torch.zeros(self.num_envs, self._num_targets, device=self.device)
        self.prev_joint_vel = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        """None of the agent's action: the opponent's comes from its own policy."""
        return 0

    @property
    def last_action(self) -> torch.Tensor:
        """(n, 14): the action its policy took at its previous step, `_Duck.last_action`."""
        return self._raw_actions

    def observe(self) -> torch.Tensor:
        """(n, 61): `runner._observe` for this duck (Decision 6). Its trunk's angular velocity and
        gravity in the trunk's frame, its joints off the default pose, the joint velocity read at its
        previous step, its previous action, the event's command."""
        data = self._entity.data
        return torch.cat(
            [
                data.root_link_ang_vel_b,
                data.projected_gravity_b,
                data.joint_pos[:, self.target_ids] - data.default_joint_pos[:, self.target_ids],
                self.prev_joint_vel,
                self.last_action,
                self._command,
            ],
            dim=-1,
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        """`actions` is the agent's empty slice. The duck observes, acts, and keeps this step's joint
        velocity for its next observation, as `_Duck.act` does; the parent sets its targets."""
        obs = self.observe()
        with torch.no_grad():
            action = self.policy(obs)
        self.prev_joint_vel[:] = self._entity.data.joint_vel[:, self.target_ids]
        super().process_actions(action)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """A new bout: no joint velocity kept (a reset's are zeros), and the parent zeroes the last action."""
        self.prev_joint_vel[env_ids] = 0.0
        super().reset(env_ids)
