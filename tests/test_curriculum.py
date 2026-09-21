"""Unit tests for the staged action-space curriculum manager."""

import unittest

import numpy as np

from curriculum import CurriculumManager, PHASES

NUM_ACTIONS = 496


class TestCurriculumManager(unittest.TestCase):
    def test_default_phases(self):
        manager = CurriculumManager()
        self.assertEqual(manager.current()["start_wave"], 1)
        self.assertEqual(manager.current()["start_sun"], 3000)
        self.assertEqual(manager.current()["allowed_seeds"], [0, 1, 2, 6, 8])

    def test_initial_start_wave_maps_to_phase(self):
        manager = CurriculumManager(initial_start_wave=10)
        self.assertEqual(manager.current()["start_wave"], 10)
        self.assertEqual(manager.current()["start_sun"], 1500)
        self.assertEqual(manager.current()["allowed_seeds"], list(range(10)))

    def test_initial_start_wave_unknown_defaults_to_first_phase(self):
        manager = CurriculumManager(initial_start_wave=99)
        self.assertEqual(manager.current()["start_wave"], 1)

    def test_seed_mask(self):
        manager = CurriculumManager()
        mask = manager.seed_mask(NUM_ACTIONS)
        self.assertEqual(mask.shape, (NUM_ACTIONS,))
        # Wait and shovel are always allowed.
        self.assertEqual(mask[0], 1.0)
        self.assertTrue(np.all(mask[1:46] == 1.0))
        # Disallowed seeds are disabled.
        for seed in set(range(10)) - set(manager.current()["allowed_seeds"]):
            start = 46 + seed * 45
            end = start + 45
            self.assertTrue(np.all(mask[start:end] == 0.0))
        # Allowed seeds are enabled (the env mask will still gate placement).
        for seed in manager.current()["allowed_seeds"]:
            start = 46 + seed * 45
            end = start + 45
            self.assertTrue(np.all(mask[start:end] == 1.0))

    def test_is_win(self):
        manager = CurriculumManager(initial_start_wave=1)
        self.assertTrue(manager.is_win(6))
        self.assertTrue(manager.is_win(7))
        self.assertFalse(manager.is_win(5))

    def test_economy_gate_keeps_emergency_actions(self):
        phases = [
            {
                "start_wave": 1,
                "start_sun": 0,
                "allowed_seeds": list(range(10)),
                "economy_floor": 3,
            }
        ]
        manager = CurriculumManager(phases=phases)
        env_mask = np.ones((1, NUM_ACTIONS), dtype=bool)
        spatial = np.zeros((1, 5, 9, 36), dtype=np.float32)
        gated = manager.action_mask(env_mask, spatial)[0].astype(bool)
        self.assertTrue(gated[46])
        self.assertFalse(gated[47])
        self.assertFalse(gated[46 + 45])
        self.assertFalse(gated[46 + 45 + 1])
        self.assertFalse(gated[46 + 2 * 45])
        self.assertFalse(gated[46 + 8 * 45 + 6])

        spatial[0, 2, 2, 3] = 270.0
        no_economy = env_mask.copy()
        no_economy[0, [46, 55, 64, 73, 82, 91, 100, 109, 118, 127]] = False
        gated = manager.action_mask(no_economy, spatial)[0].astype(bool)
        self.assertTrue(gated[46 + 8 * 45 + 2 * 9 + 6])
        self.assertTrue(gated[46 + 9 * 45 + 2 * 9 + 6])

        spatial[0, :, :, :] = 0.0
        spatial[0, 0, 0, 0] = 2.0
        gated = manager.action_mask(env_mask, spatial)[0].astype(bool)
        self.assertTrue(gated[46 + 45])
        self.assertFalse(gated[46 + 2 * 45])
        spatial[0, 1, 0, 0] = 42.0
        gated = manager.action_mask(env_mask, spatial)[0].astype(bool)
        self.assertTrue(gated[46 + 2 * 45])

        spatial[0, :, :, :] = 0.0
        spatial[0, 0, 0, 0] = 2.0
        spatial[0, 1, 0, 0] = 2.0
        spatial[0, 2, 0, 0] = 2.0
        gated = manager.action_mask(env_mask, spatial)[0].astype(bool)
        self.assertTrue(gated[46 + 2 * 45])

    def test_phase_specific_target_wave(self):
        phases = [
            {
                "start_wave": 1,
                "start_sun": 0,
                "allowed_seeds": [0],
                "waves_to_survive": 2,
            },
            {
                "start_wave": 1,
                "start_sun": 0,
                "allowed_seeds": list(range(10)),
                "waves_to_survive": 19,
            },
        ]
        manager = CurriculumManager(
            phases=phases, promotion_window=2, promotion_rate=1.0
        )
        self.assertEqual(manager.target_wave, 3)
        self.assertTrue(manager.is_win(3))
        self.assertFalse(manager.is_win(2))
        manager.record(True)
        manager.record(True)
        self.assertEqual(manager.target_wave, 20)

    def test_promotion(self):
        manager = CurriculumManager(
            promotion_window=10,
            promotion_rate=0.9,
            demotion_window=5,
            demotion_rate=0.6,
        )
        # Record 9 wins out of 10 -> 90% win rate, should promote.
        for _ in range(9):
            self.assertFalse(manager.record(True))
        self.assertEqual(manager.current()["start_wave"], 1)
        changed = manager.record(True)
        self.assertTrue(changed)
        self.assertEqual(manager.phase_index, 1)
        # start_wave is still 1 in phase 1, but seed set expanded.
        self.assertEqual(manager.current()["start_wave"], 1)
        self.assertEqual(manager.current()["allowed_seeds"], [0, 1, 2, 3, 6, 8, 9])
        # Windows reset after promotion.
        self.assertEqual(manager.stats()["promotion_window_size"], 0)
        self.assertEqual(manager.stats()["target_wave"], 6)

    def test_demotion(self):
        manager = CurriculumManager(
            initial_start_wave=20,
            promotion_window=10,
            promotion_rate=0.9,
            demotion_window=5,
            demotion_rate=0.6,
        )
        # Record 5 losses out of 5 -> 0% win rate, should demote.
        for _ in range(4):
            self.assertFalse(manager.record(False))
        changed = manager.record(False)
        self.assertTrue(changed)
        self.assertEqual(manager.current()["start_wave"], 10)
        self.assertEqual(manager.current()["allowed_seeds"], list(range(10)))
        self.assertEqual(manager.stats()["demotion_window_size"], 0)

    def test_no_change_below_threshold(self):
        manager = CurriculumManager(
            promotion_window=10,
            promotion_rate=0.9,
        )
        for _ in range(10):
            manager.record(True)
        # Already promoted to phase 1.
        self.assertEqual(manager.phase_index, 1)
        # 8/10 wins is below 90%, no promotion.
        for _ in range(10):
            manager.record(True)
            manager.record(True)
            manager.record(True)
            manager.record(True)
            manager.record(True)
            manager.record(True)
            manager.record(True)
            manager.record(True)
            manager.record(False)
            manager.record(False)
        self.assertEqual(manager.phase_index, 1)

    def test_cannot_demote_from_first_phase(self):
        manager = CurriculumManager()
        for _ in range(20):
            manager.record(False)
        self.assertEqual(manager.phase_index, 0)
        self.assertEqual(manager.current()["start_wave"], 1)

    def test_cannot_promote_from_last_phase(self):
        manager = CurriculumManager(initial_start_wave=50)
        # Move to the last phase by forcing phase_index to the end.
        manager._advance(10)
        for _ in range(100):
            manager.record(True)
        self.assertEqual(manager.phase_index, len(PHASES) - 1)
        self.assertEqual(manager.current()["start_wave"], 1)
        self.assertEqual(manager.current()["start_sun"], 50)

    def test_stats(self):
        manager = CurriculumManager()
        for _ in range(5):
            manager.record(True)
            manager.record(False)
        stats = manager.stats()
        self.assertEqual(stats["phase_index"], 0)
        self.assertEqual(stats["start_wave"], 1)
        self.assertEqual(stats["win_rate"], 0.5)
        self.assertEqual(stats["allowed_seeds"], [0, 1, 2, 6, 8])


if __name__ == "__main__":
    unittest.main()
