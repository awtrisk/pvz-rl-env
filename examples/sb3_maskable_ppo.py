"""Train your own agent on PvZ with sb3-contrib's MaskablePPO.

The env is plain Gymnasium; the only PvZ-specific glue is exposing the engine's
per-step legality mask through the ``action_masks()`` hook MaskablePPO looks for.

Install: pip install -e ".[sb3]"
Run from the repo root:    python examples/sb3_maskable_ppo.py [total_timesteps]
"""

import sys
from pathlib import Path

import gymnasium as gym

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pvz_portable_gym  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv  # noqa: E402


class PvZMaskEnv(gym.Wrapper):
    """Expose info["action_mask"] as action_masks() for MaskablePPO."""

    def __init__(self, env):
        super().__init__(env)
        self._mask = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._mask = info["action_mask"]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._mask = info["action_mask"]
        return obs, reward, terminated, truncated, info

    def action_masks(self):
        if self._mask is None:
            raise RuntimeError("call reset() before action_masks()")
        return self._mask


def make_env(rank: int):
    def _init():
        # Monitor (innermost) gives you ep_rew/ep_len in the logs. The globals
        # vector is unnormalized (sun reaches 9990); consider VecNormalize
        # (norm_obs_only) if training from scratch.
        env = Monitor(pvz_portable_gym.PvZGymEnv(reward_mode="sparse", obs_version=2))
        return PvZMaskEnv(env)

    return _init


def main() -> None:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    # SubprocVecEnv is load-bearing: one engine per process (singleton gLawnApp).
    venv = SubprocVecEnv([make_env(rank) for rank in range(8)])
    model = MaskablePPO(
        "MultiInputPolicy",
        venv,
        seed=0,
        n_steps=256,
        batch_size=512,
        verbose=1,
    )
    # Random play dies in ~4 waves; ~100k steps gets a first signal, real runs
    # want millions (see the repo README for the reference PPO recipe).
    model.learn(total_timesteps=steps)
    model.save("pvz_maskable_ppo")


if __name__ == "__main__":
    main()
