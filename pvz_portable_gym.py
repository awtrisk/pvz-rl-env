"""
Gymnasium wrapper for the headless PvZ C++ bridge (pvz_env).

This is the Phase 3 deliverable: a gymnasium.Env that wraps the raw pybind11
PvZEnv into the standard Gymnasium API, ready for Phase 4 (neural network)
and Phase 5 (PPO training).

Observation (Dict):
    spatial: Box(5, 9, 36) float32
        - Channels 0-2: plant ID, plant HP (0-1), plant state
        - Channels 3-35: per-zombie-type aggregate HP (33 zombie types)
    global: Box(24,) float32
        - [0] sun bank / 9990
        - [1] wave / num_waves  (per-stage fraction: under chain_stages this
          RESETS at every stage boundary; the absolute wave is info["wave"])
        - [2-11] seed cooldowns (0=ready, 1=just used)
        - [12-16] lawnmower readiness by lane
        - [17-21] nearest zombie X by lane (clamped to [0,1])
        - [22] sky-sun countdown
        - [23] next-zombie countdown

Action: Discrete(496)
    [0]          Wait
    [1..45]      Shovel at (row, col): 1 + row*9 + col
    [46..495]    Plant seed s at (row, col): 46 + s*45 + row*9 + col

Action mask (via info["action_mask"]): bool(496,)

Each step advances 100 engine frames (FRAMES_PER_STEP=100).

Reward modes (constructor arg ``reward_mode``):
    "legacy" (default): the bridge's dense shaping reward. Kept for exact
        comparability with results trained under the original dense objective. Note the
        clipped threat-potential term breaks potential-shaping telescoping,
        so this reward is NOT policy-invariant to the sparse objective.
    "sparse": max(absolute-wave advance, 0) + 1 on stage_complete - 1 on lost
        per step — the principled objective (mirrors ppo.py
        frontier_sparse_reward exactly). The legacy value stays available in
        info["reward_legacy"] under this mode.

Registered as Gymnasium id "PvZ-v0" (see module bottom).

obs_version (constructor arg):
    1 (default): 5x9x36 spatial, 24 globals — every existing checkpoint.
    2: 5x9x38 spatial (channels 36-37 = normal-plant layer: the plant
       beneath a Pumpkin shell) and 26 globals (24 = absolute
       wave/(wavesPerStage+1), 25 = survival stage — unbounded under
       chain_stages, so the global Box high is inf in v2).

deck (constructor arg): list of 10 seed-type ints overriding the default
    10-seed meta-deck (Sunflower, Twin Sunflower, Melon-pult, Winter
    Melon, Gloom-shroom, Fume-shroom, Pumpkin, Garlic, Squash, Jalapeno).
    Applied at reset; costs, cooldowns and the action mask follow the seed
    bank generically, so any legal seed combination works.
"""

import importlib
import os
import sys
from contextlib import suppress
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ── Bootstrap: locate the compiled C++ extension ──────────────────
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# SDL2 runtime DLLs live in the MSYS2 ucrt64 bin (Windows only)
_UCRT64 = "D:/msys2/ucrt64/bin"
if os.path.isdir(_UCRT64) and hasattr(os, "add_dll_directory"):
    with suppress(OSError):
        os.add_dll_directory(_UCRT64)

_BUILD_DIR = os.path.join(_REPO_ROOT, "pvz-portable", "build")
# Repo root must OUTRANK the build dir even if it already sits lower in
# sys.path (PYTHONPATH/cwd), or a stale build-dir module shadows a fresh
# repo-root build. On Modal the image ignores *.pyd/*.so at repo root, so
# the cloud always resolves to build/.
sys.path.insert(0, _BUILD_DIR)
sys.path.insert(0, _REPO_ROOT)

_pvz = importlib.import_module("pvz_env")

NUM_ACTIONS = 496
SPATIAL_SHAPE = (5, 9, 36)
GLOBAL_SIZE = 24
MAX_STEPS = 2000


class PvZGymEnv(gym.Env):
    """
    Gymnasium environment wrapping the pvz_env C++ bridge.

    The C++ engine handles all game logic, reward computation, and temporal
    resolution (100 frames per step). This wrapper adapts the raw C++ tuple
    to Gymnasium's API contract and declares proper observation/action spaces.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        render_mode=None,
        chdir: bool = True,
        resdir: str = "pvz-portable/",
        savedir: str = "pvz-portable/savedata/",
        reward_mode: str = "legacy",
        max_steps: int = MAX_STEPS,
        obs_version: int = 1,
        deck: list | None = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        if reward_mode not in ("legacy", "sparse"):
            raise ValueError(f"reward_mode must be 'legacy' or 'sparse', got {reward_mode!r}")
        self.reward_mode = reward_mode
        self.max_steps = int(max_steps)  # pi-lens-ignore: unchecked-throwing-call-python
        if obs_version not in (1, 2):
            raise ValueError(f"obs_version must be 1 or 2, got {obs_version!r}")
        self.obs_version = obs_version
        if deck is not None and len(deck) != 10:
            raise ValueError(f"deck must contain exactly 10 seed types, got {len(deck)}")
        self.deck = deck

        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        if chdir:
            os.chdir(_REPO_ROOT)

        # Observation space: Dict of spatial tensor + global vector.
        # v2 adds the normal-plant layer (+2 spatial channels) and absolute
        # wave + stage (+2 globals, unbounded — hence high=inf there).
        spatial_shape = SPATIAL_SHAPE if obs_version == 1 else (5, 9, SPATIAL_SHAPE[2] + 2)
        global_size = GLOBAL_SIZE if obs_version == 1 else GLOBAL_SIZE + 2
        self.observation_space = spaces.Dict(
            {
                "spatial": spaces.Box(
                    low=0.0, high=np.inf, shape=spatial_shape, dtype=np.float32
                ),
                "global": spaces.Box(
                    low=0.0,
                    high=1.0 if obs_version == 1 else np.inf,
                    shape=(global_size,),
                    dtype=np.float32,
                ),
            }
        )

        # Action space: flattened Discrete(496)
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Create the underlying C++ engine instance
        self._env: Any = _pvz.PvZEnv(resdir, savedir)
        self._env.set_obs_version(obs_version)
        if deck is not None:
            self._env.set_deck(list(deck))

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment.

        Args:
            seed: Seeds Gymnasium and, unless options overrides it, the C++ game RNG.
            options: Optional dict. Supports {"wave": int} and optional
                     {"seed": int} for paired deterministic rollouts.

        Returns:
            (observation, info) — Gymnasium 5-tuple reset contract.
        """
        super().reset(seed=seed)

        wave = 1
        if options and "wave" in options:
            # pi-lens-ignore: unchecked-throwing-call-python
            wave = int(options["wave"])
        if options and "max_steps" in options:
            # pi-lens-ignore: unchecked-throwing-call-python
            self.max_steps = int(options["max_steps"])
        self._steps = 0
        self._max_wave_seen = wave
        episode_seed = (
            # pi-lens-ignore: unchecked-throwing-call-python
            int(options["seed"])
            if options and "seed" in options
            # pi-lens-ignore: unchecked-throwing-call-python
            else (-1 if seed is None else int(seed))
        )

        self._env.reset(wave, episode_seed)

        if options and "sun" in options:
            # pi-lens-ignore: unchecked-throwing-call-python
            self._env.set_sun_money(int(options["sun"]))

        obs = self._get_obs()
        info = self._get_info()
        info["wave"] = wave
        return obs, info

    def squash_targetable_lanes(self):
        """Return lanes where a Squash in the teacher's fixed column can acquire now."""
        return np.asarray(self._env.squash_targetable_lanes(), dtype=bool)

    def step(self, action):
        """
        Execute one action, advance 10 game frames, return next state.

        Args:
            action: int in [0, 496). See module docstring for encoding.

        Returns:
            (observation, reward, terminated, truncated, info) — Gymnasium
            5-tuple. info["action_mask"] carries the bool(496,) validity mask.
        """
        # pi-lens-ignore: unchecked-throwing-call-python
        action = int(action)

        # The C++ step returns a 6-tuple:
        #   (obs_dict, mask_bool496, reward, done, truncated, info_dict)
        # pi-lens-ignore: unchecked-throwing-call-python
        raw_obs, mask, reward, done, truncated, raw_info = self._env.step(action)

        obs = self._wrap_obs(raw_obs)
        info = dict(raw_info)
        info["action_mask"] = np.asarray(mask, dtype=bool)

        self._steps += 1
        # Sparse objective mirrors ppo.py frontier_sparse_reward: absolute
        # wave advance (floored at 0; the stage-boundary wave collision is
        # compensated by the stage_complete bonus), +1 per stage completion,
        # -1 on loss.
        wave_now = int(info.get("wave", self._max_wave_seen))  # pi-lens-ignore: unchecked-throwing-call-python
        sparse = (
            # pi-lens-ignore: unchecked-throwing-call-python
            float(max(wave_now - self._max_wave_seen, 0))
            + (1.0 if info.get("stage_complete") else 0.0)
            - (1.0 if info.get("lost") else 0.0)
        )
        self._max_wave_seen = max(self._max_wave_seen, wave_now)
        if self.reward_mode == "sparse":
            info["reward_legacy"] = float(reward)  # pi-lens-ignore: unchecked-throwing-call-python
            reward = sparse
        else:
            info["reward_sparse"] = sparse

        truncated = bool(truncated) or self._steps >= self.max_steps

        # pi-lens-ignore: unchecked-throwing-call-python
        return obs, float(reward), bool(done), truncated, info

    def _get_obs(self):
        return self._wrap_obs(self._env.get_obs())

    def _get_info(self):
        obs = self._env.get_obs()
        mask = np.asarray(self._env.get_action_mask(), dtype=bool)
        return {
            "action_mask": mask,
            "sun": obs["global"][0] * 9990,
            "wave": obs["global"][1],
        }

    @staticmethod
    def _wrap_obs(raw_obs):
        """Convert C++ numpy arrays into owned float32 arrays matching spaces."""
        return {
            "spatial": np.ascontiguousarray(raw_obs["spatial"], dtype=np.float32),
            "global": np.ascontiguousarray(raw_obs["global"], dtype=np.float32),
        }

    def set_plant_bonus_scale(self, scale):
        """Set the per-plant placement bonus scale in the C++ engine."""
        # pi-lens-ignore: unchecked-throwing-call-python
        self._env.set_plant_bonus_scale(float(scale))

    def set_sun_money(self, sun):
        """Override the current sun bank. Used by the curriculum for staged starting resources."""
        # pi-lens-ignore: unchecked-throwing-call-python
        self._env.set_sun_money(int(sun))

    def set_sun_penalty_scale(self, scale):
        """Set the per-sun per-step holding penalty scale."""
        # pi-lens-ignore: unchecked-throwing-call-python
        self._env.set_sun_penalty_scale(float(scale))

    def set_cooldown_scale(self, scale):
        """Set the multiplier applied to seed-packet cooldown durations."""
        # pi-lens-ignore: unchecked-throwing-call-python
        self._env.set_cooldown_scale(float(scale))

    def set_chain_stages(self, chain):
        """Enable endless chaining of survival stages on the live board."""
        self._env.set_chain_stages(bool(chain))

    def set_deck(self, deck):
        """Configure the 10-seed deck (applied on the next reset)."""
        self._env.set_deck([int(seed) for seed in deck])

    def get_render_state(self):
        """Return bridge metadata used to verify the playing-board render path."""
        return dict(self._env.get_render_state())

    def render(self):
        """Return an RGB array of the current board state."""
        if self.render_mode == "rgb_array":
            return np.ascontiguousarray(self._env.render(), dtype=np.uint8)
        return None

    def record_step(self, action, frame_interval=5):
        """Step while capturing frames; returns (obs, reward, done, truncated, info, frames)."""
        # pi-lens-ignore: unchecked-throwing-call-python
        action = int(action)
        raw_obs, mask, reward, done, truncated, raw_info, frames = (
            self._env.record_step(
                action,
                # pi-lens-ignore: unchecked-throwing-call-python
                int(frame_interval),
            )
        )
        obs = self._wrap_obs(raw_obs)
        info = dict(raw_info)
        info["action_mask"] = np.asarray(mask, dtype=bool)
        frames = [np.ascontiguousarray(f, dtype=np.uint8) for f in frames]
        # pi-lens-ignore: unchecked-throwing-call-python
        return obs, float(reward), bool(done), bool(truncated), info, frames

    def telemetry_step(self, action, frame_interval=5):
        """Step while collecting Squash telemetry without allocating video frames."""
        raw_obs, mask, reward, done, truncated, raw_info = self._env.telemetry_step(
            # pi-lens-ignore: unchecked-throwing-call-python
            int(action),
            # pi-lens-ignore: unchecked-throwing-call-python
            int(frame_interval),
        )
        obs = self._wrap_obs(raw_obs)
        info = dict(raw_info)
        info["action_mask"] = np.asarray(mask, dtype=bool)
        # pi-lens-ignore: unchecked-throwing-call-python
        return obs, float(reward), bool(done), bool(truncated), info

    def close(self):
        self._env = None


if "PvZ-v0" not in gym.registry:
    gym.register(
        id="PvZ-v0",
        entry_point="pvz_portable_gym:PvZGymEnv",
    )
