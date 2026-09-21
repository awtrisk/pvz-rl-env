"""Env-contract regression tests: registration, sparse reward mode, max_steps.

Boots the engine (imports torch first per the DLL bootstrap convention).
"""

import torch  # noqa: F401  (resolves MSYS2 DLLs before pvz_env loads)

import gymnasium as gym

import unittest

from pvz_portable_gym import MAX_STEPS, PvZGymEnv

WAIT = 0


def plant_action(slot: int, row: int, col: int) -> int:
    """Discrete(496) index for planting deck `slot` at (row, col)."""
    return 46 + slot * 45 + row * 9 + col


SEED_SUNFLOWER = 1
SEED_PUMPKINSHELL = 30
SEED_PEASHOOTER = 0


class RegistrationTest(unittest.TestCase):
    def test_registered_id_resolves(self):
        env = gym.make("PvZ-v0")
        assert isinstance(env.unwrapped, PvZGymEnv)
        env.close()


class SparseRewardTest(unittest.TestCase):
    def test_sparse_mode_wave_advance_pays_and_legacy_kept(self):
        env = PvZGymEnv(reward_mode="sparse", max_steps=MAX_STEPS)
        env.reset(seed=275990)
        collected = 0.0
        saw_legacy_key = False
        for _ in range(400):
            _, reward, done, truncated, info = env.step(WAIT)
            assert info["reward_legacy"] is not None
            saw_legacy_key = True
            collected += reward
            if reward > 0:
                break
            if done or truncated:
                break
        assert collected > 0.0, "no wave advance observed in 400 wait steps"
        assert saw_legacy_key
        env.close()

    def test_legacy_mode_default_reports_sparse_in_info(self):
        env = PvZGymEnv()
        env.reset(seed=275991)
        for _ in range(3):
            obs, reward, done, truncated, info = env.step(WAIT)
            assert isinstance(reward, float)
            assert "reward_sparse" in info
            assert obs["spatial"].shape == (5, 9, 36)
            assert obs["global"].shape == (24,)
        env.close()


class MaxStepsTest(unittest.TestCase):
    def test_wrapper_truncates_at_max_steps(self):
        env = PvZGymEnv(max_steps=3)
        env.reset(seed=275992)
        truncated = False
        steps = 0
        for _ in range(5):
            _, _, done, truncated, _ = env.step(WAIT)
            steps += 1
            if done or truncated:
                break
        assert truncated and steps == 3
        env.close()


class ObsV2Test(unittest.TestCase):
    def test_v2_shapes_and_absolute_wave_globals(self):
        env = PvZGymEnv(obs_version=2)
        obs, _ = env.reset(seed=275994)
        assert obs["spatial"].shape == (5, 9, 38)
        assert obs["global"].shape == (26,)
        assert abs(obs["global"][24] - 1.0 / 21.0) < 1e-4  # absolute wave 1
        assert obs["global"][25] == 0.0  # stage 0
        env.close()

    def test_v2_normal_layer_reveals_plant_under_pumpkin(self):
        env = PvZGymEnv(obs_version=2)
        env.reset(seed=275995)
        env.step(plant_action(0, 0, 0))  # Sunflower slot 0
        obs, _, _, _, _ = env.step(plant_action(6, 0, 0))  # Pumpkin slot 6 on top
        assert obs["spatial"][0, 0, 36] == float(SEED_SUNFLOWER + 1)
        assert obs["spatial"][0, 0, 37] > 0.0
        env.close()


class DeckConfigTest(unittest.TestCase):
    def test_custom_deck_plants_peashooter(self):
        env = PvZGymEnv(deck=[SEED_PEASHOOTER] * 10)
        obs, _ = env.reset(seed=275996)
        obs, _, _, _, info = env.step(plant_action(0, 0, 0))
        assert obs["spatial"][0, 0, 0] == float(SEED_PEASHOOTER + 1)
        assert info["sun"] < 1000  # peashooter cost charged
        env.close()

    def test_invalid_deck_rejected(self):
        with self.assertRaises(ValueError):
            PvZGymEnv(deck=[0] * 9)
        env = PvZGymEnv()
        with self.assertRaises(ValueError):
            env._env.set_deck([99] + [0] * 9)
        env.close()


class ResetOptionsTest(unittest.TestCase):
    def test_reset_options_max_steps_override(self):
        env = PvZGymEnv()
        env.reset(seed=275993, options={"max_steps": 2})
        assert env.max_steps == 2
        truncated = False
        for _ in range(4):
            _, _, done, truncated, _ = env.step(WAIT)
            if done or truncated:
                break
        assert truncated
        env.close()
