"""Endless-deck experiment config tests: bcwm<sun> preset + full-deck suite."""

import unittest

import train_modal


class BcwmPresetTest(unittest.TestCase):
    def test_bcwm_phase_parses_with_eight_seed_mask(self):
        cfg = train_modal.build_training_config(
            num_envs=2,
            num_updates=3,
            start_wave=1,
            steps_per_update=4,
            promotion_window=50,
            promotion_rate=0.9,
            demotion_window=20,
            demotion_rate=0.6,
            waves_to_survive=5,
            checkpoint_path="resume.pt",
            checkpoint_interval=1,
            wandb_project="pvz-test",
            wandb_run_name="",
            cooldown_scale=1.0,
            phase="bcwm750",
        )
        phase = cfg["phases"][0]
        seeds = phase["allowed_seeds"]
        self.assertEqual(len(seeds), 8)
        self.assertIn(3, seeds)  # Winter Melon
        self.assertIn(4, seeds)  # Gloom-shroom
        self.assertIn(5, seeds)  # Fume-shroom
        self.assertNotIn(7, seeds)  # Garlic masked
        self.assertNotIn(9, seeds)  # Jalapeno masked
        self.assertEqual(phase["start_sun"], 750)
        self.assertEqual(phase["waves_to_survive"], 19)

    def test_bc_phase_still_uses_five_seed_mask(self):
        cfg = train_modal.build_training_config(
            num_envs=2,
            num_updates=3,
            start_wave=1,
            steps_per_update=4,
            promotion_window=50,
            promotion_rate=0.9,
            demotion_window=20,
            demotion_rate=0.6,
            waves_to_survive=5,
            checkpoint_path="resume.pt",
            checkpoint_interval=1,
            wandb_project="pvz-test",
            wandb_run_name="",
            cooldown_scale=1.0,
            phase="bc750",
        )
        self.assertEqual(len(cfg["phases"][0]["allowed_seeds"]), 5)

    def test_endless_eval_full_suite_registered(self):
        from evaluation import EXTRA_SUITES

        suite = EXTRA_SUITES["endless_eval_full"]
        self.assertTrue(suite["full_deck"])
        self.assertTrue(suite["chain_stages"])
        self.assertEqual(suite["start_sun"], 3000)


class BcfullPresetTest(unittest.TestCase):
    def test_bcfull_phase_parses_with_all_ten_slots(self):
        cfg = train_modal.build_training_config(
            num_envs=2,
            num_updates=3,
            start_wave=1,
            steps_per_update=4,
            promotion_window=50,
            promotion_rate=0.9,
            demotion_window=20,
            demotion_rate=0.6,
            waves_to_survive=5,
            checkpoint_path="resume.pt",
            checkpoint_interval=1,
            wandb_project="pvz-test",
            wandb_run_name="",
            cooldown_scale=1.0,
            phase="bcfull750",
            deck="1,41,39,44,42,10,30,35,17,20",
        )
        phase = cfg["phases"][0]
        self.assertEqual(phase["allowed_seeds"], list(range(10)))
        self.assertEqual(phase["start_sun"], 750)
        self.assertEqual(phase["waves_to_survive"], 19)
        self.assertEqual(len(cfg["deck"]), 10)
        self.assertEqual(cfg["deck"][7], 35)  # Coffee Bean in the Garlic slot

    def test_endless_eval_coffee_suite_registered(self):
        from evaluation import EXTRA_SUITES

        suite = EXTRA_SUITES["endless_eval_coffee"]
        self.assertTrue(suite["full_deck"])
        self.assertTrue(suite["chain_stages"])
        self.assertEqual(len(suite["deck"]), 10)
        self.assertEqual(suite["deck"][7], 35)

    def test_bcnarrow_parses_with_discovered_mask(self):
        from train_modal import build_training_config

        cfg = build_training_config(
            num_envs=2,
            num_updates=3,
            start_wave=1,
            steps_per_update=4,
            promotion_window=50,
            promotion_rate=0.9,
            demotion_window=20,
            demotion_rate=0.6,
            waves_to_survive=5,
            checkpoint_path="resume.pt",
            checkpoint_interval=1,
            wandb_project="pvz-test",
            wandb_run_name="",
            cooldown_scale=1.0,
            phase="bcnarrow750",
            deck="1,41,39,44,42,10,30,35,17,20",
        )
        phase = cfg["phases"][0]
        # Twin Sunflower (1) and Pumpkin (6) dropped, Coffee Bean (7) kept.
        self.assertEqual(phase["allowed_seeds"], [0, 2, 3, 4, 5, 7, 8, 9])
        self.assertEqual(phase["start_sun"], 750)
        self.assertEqual(phase["waves_to_survive"], 19)
        self.assertEqual(cfg["deck"][7], 35)

    def test_endless_eval_narrow_suite_registered(self):
        from evaluation import EXTRA_SUITES

        suite = EXTRA_SUITES["endless_eval_narrow"]
        self.assertFalse(suite["full_deck"])
        self.assertTrue(suite["chain_stages"])
        self.assertEqual(suite["allowed_seeds"], [0, 2, 3, 4, 5, 7, 8, 9])
        self.assertEqual(suite["deck"][7], 35)
