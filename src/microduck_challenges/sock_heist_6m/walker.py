"""The learner's start: the Arena's reference walker, `walker.onnx` (the bytes of the Arena's
`BEST_alpha_walking.onnx`), as the Tug of War's reference t20 started from it. It walks and stands, but
was never trained to stop and start again: that is what this training adds.
The HF job installs no Arena: nothing here imports it; tests/test_sock_heist_6m.py pins the copy.
"""

from pathlib import Path

import torch
from rsl_rl.models import MLPModel

from ..tug_of_war.opponent import load_actor

WALKER = Path(__file__).with_name("walker.onnx")
WALKER_SEEN = 502 * 24 * 4096  # the normalizer count t20's start from the walker used; the walker's own is unknown
WARM_START_STD = 0.3  # the warm-started learner's action noise: t20's, from the walker (commit 8391bf7)


class WalkerActor(MLPModel):
    """The learner's actor (RL_CFG's), starting as the walker (`--agent.actor.class-name MLPModel`
    starts from scratch instead).

    Its layers and its observation normalizer are `walker.onnx`'s (a shape or a missing normalizer
    refuses to load), its action noise std is WARM_START_STD; the critic starts fresh. The ONNX's std
    carries the normalizer's epsilon. The normalizer counts WALKER_SEEN observations: from 0, PPO's
    first update would replace its statistics with one batch's.
    """

    def __init__(self, *args, distribution_cfg: dict, **kwargs):
        super().__init__(*args, distribution_cfg={**distribution_cfg, "init_std": WARM_START_STD}, **kwargs)
        normalize, *layers = load_actor(WALKER)
        self.mlp.load_state_dict(torch.nn.Sequential(*layers).state_dict())
        std = normalize.std - self.obs_normalizer.eps
        self.obs_normalizer.load_state_dict(
            {"_mean": normalize.mean, "_var": std**2, "_std": std, "count": torch.tensor(WALKER_SEEN)}
        )
