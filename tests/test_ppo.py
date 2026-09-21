"""Phase 5 PPO logic tests using a dummy vector environment.

These tests exercise the PPO trainer (collect, GAE, update) without the C++
engine, so they can run quickly on CPU. They cover the action-mask contract
used by the staged curriculum and the Modal configuration boundary.
"""

import copy
import inspect
import json
import os
import sys
import tempfile
import types
import unittest

import numpy as np
import torch

import ppo
import train_modal
from curriculum import CurriculumManager
from network import GLOBAL_SIZE, PvZActorCritic

os.environ.setdefault("WANDB_MODE", "disabled")


class DummyVecEnv:
    """Minimal vector env that matches the PvZVecEnv API."""

    def __init__(self, num_envs, horizon):
        self.num_envs = num_envs
        self._horizon = horizon
        self._step = 0
        self._start_wave = 1

    def reset(self):
        self._step = 0
        obs = {
            "spatial": np.zeros((self.num_envs, 5, 9, 36), dtype=np.float32),
            "global": np.zeros((self.num_envs, GLOBAL_SIZE), dtype=np.float32),
        }
        infos = [
            {"action_mask": np.ones(496, dtype=np.float32), "wave": self._start_wave}
            for _ in range(self.num_envs)
        ]
        return obs, infos

    def step(self, actions):
        self._step += 1
        obs = {
            "spatial": np.zeros((self.num_envs, 5, 9, 36), dtype=np.float32),
            "global": np.zeros((self.num_envs, GLOBAL_SIZE), dtype=np.float32),
        }
        rewards = np.ones(self.num_envs, dtype=np.float32) * 0.1
        dones = np.zeros(self.num_envs, dtype=np.float32)
        truncated = np.zeros(self.num_envs, dtype=np.float32)
        if self._step >= self._horizon:
            truncated[:] = 1.0
        infos = [
            {
                "action_mask": np.ones(496, dtype=np.float32),
                "wave": self._start_wave + self._step,
            }
            for _ in range(self.num_envs)
        ]
        return obs, rewards, dones, truncated, infos

    def set_start_wave(self, wave):
        self._start_wave = wave

    def set_start_sun(self, sun):
        pass

    def set_plant_bonus_scale(self, scale):
        pass

    def set_sun_penalty_scale(self, scale):
        pass

    def set_cooldown_scale(self, scale):
        pass

    def close(self):
        pass

    def reset_at(self, indices):
        self._step = 0
        infos = [
            {"action_mask": np.ones(496, dtype=np.float32), "wave": self._start_wave}
            for _ in indices
        ]
        return [
            (
                {
                    "spatial": np.zeros((5, 9, 36), dtype=np.float32),
                    "global": np.zeros((GLOBAL_SIZE,), dtype=np.float32),
                },
                infos[i],
            )
            for i in range(len(indices))
        ]


class TestPPOLogic(unittest.TestCase):
    def test_collect_rollout_shapes(self):
        env = DummyVecEnv(2, 16)
        agent = PvZActorCritic()
        cfg = {"steps_per_update": 16, "gamma": 0.99, "gae_lambda": 0.95}
        manager = CurriculumManager()
        rollout, next_value, outcomes, _ = ppo.collect_rollout(
            agent, env, cfg, torch.device("cpu"), manager
        )

        self.assertEqual(rollout["obs_spatial"].shape, (16, 2, 5, 9, 36))
        self.assertEqual(rollout["obs_global"].shape, (16, 2, GLOBAL_SIZE))
        self.assertEqual(rollout["actions"].shape, (16, 2))
        self.assertEqual(rollout["logprobs"].shape, (16, 2))
        self.assertEqual(rollout["values"].shape, (16, 2))
        self.assertEqual(rollout["rewards"].shape, (16, 2))
        self.assertEqual(rollout["dones"].shape, (16, 2))
        self.assertEqual(rollout["truncated"].shape, (16, 2))
        self.assertEqual(rollout["episode_starts"].shape, (16, 2))
        self.assertTrue(np.all(rollout["episode_starts"][0] == 1.0))
        self.assertEqual(rollout["masks"].shape, (16, 2, 496))
        self.assertEqual(rollout["optimization_mask"].shape, (16, 2))
        self.assertTrue(np.all(rollout["optimization_mask"] == 1.0))
        self.assertEqual(next_value.shape, (2,))
        self.assertIn("last_caches", rollout)
        self.assertEqual(len(rollout["last_caches"]), agent.mamba_config.n_layers)

    def test_frontier_mask_trains_only_at_or_beyond_requested_wave(self):
        env = DummyVecEnv(2, 16)
        agent = PvZActorCritic()
        cfg = {
            "steps_per_update": 6,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "frontier_wave": 4,
        }
        rollout, _, _, _ = ppo.collect_rollout(
            agent, env, cfg, torch.device("cpu"), CurriculumManager()
        )

        expected = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
        np.testing.assert_array_equal(rollout["optimization_mask"][:, 0], expected)
        np.testing.assert_array_equal(rollout["optimization_mask"][:, 1], expected)
        self.assertFalse(rollout["dones"].any())

    def test_frontier_sparse_reward_uses_only_wave_and_terminal_outcome(self):
        advancing_env = DummyVecEnv(1, 16)
        cfg = {
            "steps_per_update": 2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "frontier_wave": 1,
            "frontier_sparse_reward": True,
        }
        advancing, _, _, _ = ppo.collect_rollout(
            PvZActorCritic(),
            advancing_env,
            cfg,
            torch.device("cpu"),
            CurriculumManager(),
        )
        np.testing.assert_array_equal(advancing["rewards"][:, 0], [1.0, 1.0])

        env = DummyVecEnv(2, 16)
        original_step = env.step

        def terminal_step(actions):
            obs, reward, done, truncated, infos = original_step(actions)
            reward[:] = 99.0
            done[:] = 1.0
            truncated[:] = 1.0
            for info in infos:
                info["wave"] = 1
            infos[0]["lost"] = True
            infos[1]["stage_complete"] = True
            return obs, reward, done, truncated, infos

        env.step = terminal_step
        rollout, _, _, _ = ppo.collect_rollout(
            PvZActorCritic(),
            env,
            cfg,
            torch.device("cpu"),
            CurriculumManager(),
        )

        np.testing.assert_array_equal(rollout["rewards"][:, 0], [-1.0, -1.0])
        np.testing.assert_array_equal(rollout["rewards"][:, 1], [1.0, 1.0])
        self.assertTrue(rollout["dones"].all())
        self.assertFalse(rollout["truncated"].any())
        self.assertFalse(rollout["terminal_values"].any())

    def test_frontier_mode_uses_native_episode_termination(self):
        env = DummyVecEnv(2, 16)
        agent = PvZActorCritic()
        cfg = {
            "steps_per_update": 6,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "frontier_wave": 2,
        }
        manager = CurriculumManager(
            phases=[
                {
                    "start_wave": 1,
                    "start_sun": 0,
                    "allowed_seeds": list(range(10)),
                    "waves_to_survive": 2,
                }
            ]
        )
        rollout, _, outcomes, _ = ppo.collect_rollout(
            agent, env, cfg, torch.device("cpu"), manager
        )

        self.assertFalse(rollout["dones"].any())
        self.assertFalse(rollout["truncated"].any())
        self.assertEqual(outcomes, [])

    def test_rollout_boundary_preserves_recurrent_context(self):
        env = DummyVecEnv(2, 100)
        agent = PvZActorCritic()
        agent.memory_gate.data.fill_(0.1)
        anchor = copy.deepcopy(agent).eval()
        with torch.no_grad():
            next(agent.mamba.parameters()).add_(0.1)
        cfg = {"steps_per_update": 4, "gamma": 0.99, "gae_lambda": 0.95}
        manager = CurriculumManager()
        first, _, _, normalizer = ppo.collect_rollout(
            agent,
            env,
            cfg,
            torch.device("cpu"),
            manager,
            anchor_agent=anchor,
        )
        second, _, _, _ = ppo.collect_rollout(
            agent,
            env,
            cfg,
            torch.device("cpu"),
            manager,
            obs=first["last_obs"],
            infos=first["last_infos"],
            reward_normalizer=normalizer,
            caches=first["last_caches"],
            anchor_agent=anchor,
            anchor_caches=first["anchor_last_caches"],
        )

        self.assertTrue(np.all(second["episode_starts"][0] == 0.0))
        self.assertTrue(
            any(
                not torch.equal(current[0], frozen[0])
                for current, frozen in zip(
                    second["initial_caches"],
                    second["anchor_initial_caches"],
                    strict=True,
                )
            )
        )
        with torch.no_grad():
            _, logprob, _, _ = agent.get_action_and_value(
                torch.from_numpy(second["obs_spatial"].transpose(1, 0, 2, 3, 4)),
                torch.from_numpy(second["obs_global"].transpose(1, 0, 2)),
                torch.from_numpy(second["masks"].transpose(1, 0, 2)),
                action=torch.from_numpy(second["actions"].transpose(1, 0)),
                episode_starts=torch.from_numpy(
                    second["episode_starts"].transpose(1, 0)
                ),
                initial_caches=second["initial_caches"],
            )
        np.testing.assert_allclose(
            logprob.numpy(), second["logprobs"].transpose(1, 0), atol=1e-5, rtol=1e-5
        )

    def test_target_wave_is_real_terminal_and_auxiliary_cannot_promote(self):
        env = DummyVecEnv(5, 100)
        agent = PvZActorCritic()
        cfg = {
            "steps_per_update": 4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "auxiliary_start_sun": 3000,
        }
        manager = CurriculumManager(
            phases=[
                {
                    "start_wave": 1,
                    "start_sun": 0,
                    "allowed_seeds": list(range(10)),
                    "waves_to_survive": 2,
                }
            ]
        )
        rollout, _, outcomes, _ = ppo.collect_rollout(
            agent, env, cfg, torch.device("cpu"), manager
        )
        self.assertTrue(np.all(rollout["dones"][1] == 1.0))
        self.assertTrue(all(outcome["phase_index"] == 0 for outcome in outcomes))
        self.assertFalse(outcomes[0]["promotion_eligible"])
        self.assertTrue(all(outcome["promotion_eligible"] for outcome in outcomes[1:5]))

    def test_rollout_masks_curriculum_locked_seeds(self):
        env = DummyVecEnv(2, 16)
        agent = PvZActorCritic()
        cfg = {"steps_per_update": 16, "gamma": 0.99, "gae_lambda": 0.95}
        manager = CurriculumManager()

        rollout, _, _, _ = ppo.collect_rollout(
            agent, env, cfg, torch.device("cpu"), manager, use_bc=True
        )

        allowed = manager.seed_mask(agent.num_actions).astype(bool)
        self.assertTrue(np.all(rollout["masks"][:, :, ~allowed] == 0.0))
        self.assertTrue(np.all(allowed[rollout["actions"]]))

    def test_modal_config_has_defined_curriculum_and_logging_keys(self):
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
            cooldown_scale=0.13,
        )

        self.assertEqual(cfg["promotion_window"], 50)
        self.assertEqual(cfg["promotion_rate"], 0.9)
        self.assertEqual(cfg["wandb_project"], "pvz-test")
        self.assertEqual(cfg["checkpoint_path"], "resume.pt")
        self.assertEqual(cfg["checkpoint_interval"], 1)
        self.assertEqual(cfg["wandb_run_name"], "pvz-ppo")
        self.assertEqual(cfg["evaluation_episodes"], 20)
        self.assertEqual(cfg["lr"], 1e-4)
        self.assertEqual(cfg["gae_lambda"], 0.95)
        self.assertFalse(cfg["extend_schedule"])
        self.assertIsNone(cfg["frontier_wave"])

        zero_cfg = train_modal.build_training_config(
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
            wandb_project="",
            wandb_run_name="",
            cooldown_scale=1.0,
            phase="zero",
            frontier_wave=8,
            frontier_sparse_reward=True,
        )
        self.assertEqual(zero_cfg["phases"][0]["waves_to_survive"], 19)
        self.assertEqual(zero_cfg["gae_lambda"], 0.99)
        self.assertEqual(zero_cfg["frontier_wave"], 8)
        self.assertTrue(zero_cfg["frontier_sparse_reward"])
        self.assertFalse(zero_cfg["use_reward_norm"])

    def test_extended_resume_schedule_is_explicit(self):
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
            extend_schedule=True,
        )
        self.assertTrue(cfg["extend_schedule"])

        memory_cfg = train_modal.build_training_config(
            num_envs=8,
            num_updates=390,
            start_wave=1,
            steps_per_update=256,
            promotion_window=50,
            promotion_rate=0.7,
            demotion_window=20,
            demotion_rate=0.6,
            waves_to_survive=5,
            checkpoint_path="plan_zero_best.pt",
            checkpoint_interval=10,
            wandb_project="",
            wandb_run_name="",
            cooldown_scale=1.0,
            phase="zero_memory",
            auxiliary_start_sun=3000,
            lr=3e-4,
            anchor_kl_coef=0.01,
            extend_schedule=True,
            restart_schedule=True,
            memory_only=True,
            reset_curriculum_state=True,
        )
        self.assertEqual(
            [p["waves_to_survive"] for p in memory_cfg["phases"]], [6, 9, 12, 15, 19]
        )
        self.assertTrue(memory_cfg["memory_only"])
        self.assertTrue(memory_cfg["restart_schedule"])
        self.assertTrue(memory_cfg["reset_curriculum_state"])
        self.assertEqual(memory_cfg["gae_lambda"], 0.99)
        self.assertEqual(memory_cfg["curriculum_distill_coef"], 0.0)

        curriculum_cfg = train_modal.build_training_config(
            num_envs=2,
            num_updates=3,
            start_wave=1,
            steps_per_update=4,
            promotion_window=50,
            promotion_rate=0.9,
            demotion_window=20,
            demotion_rate=0.6,
            waves_to_survive=19,
            checkpoint_path="resume.pt",
            checkpoint_interval=1,
            wandb_project="pvz-test",
            wandb_run_name="",
            cooldown_scale=1.0,
            phase="zero_defense",
        )
        self.assertEqual(
            [phase["start_sun"] for phase in curriculum_cfg["phases"]],
            [0, 0, 0, 0, 0],
        )
        self.assertEqual(
            [phase["waves_to_survive"] for phase in curriculum_cfg["phases"]],
            [1, 3, 6, 11, 19],
        )
        self.assertEqual(
            [phase.get("economy_floor", 0) for phase in curriculum_cfg["phases"]],
            [3, 5, 5, 5, 0],
        )

        ramp_cfg = train_modal.build_training_config(
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
            phase="zero_ramp",
        )
        self.assertEqual(
            [phase["start_sun"] for phase in ramp_cfg["phases"]],
            [3000, 1500, 1000, 750, 500, 300, 150, 50, 0],
        )
        self.assertEqual(ramp_cfg["phases"][3]["start_sun"], 750)
        self.assertEqual(ramp_cfg["phases"][-1]["waves_to_survive"], 19)

        offense_cfg = train_modal.build_training_config(
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
            phase="zero_offense_ramp",
        )
        self.assertEqual(offense_cfg["phases"][3]["start_sun"], 750)
        self.assertEqual(offense_cfg["phases"][3]["economy_floor"], 5)
        self.assertEqual(offense_cfg["phases"][3]["offense_floor"], 6)
        self.assertEqual(offense_cfg["phases"][-1]["start_sun"], 0)

    def test_curriculum_offense_gate_preserves_emergency_actions(self):
        phases = [
            {
                "start_wave": 1,
                "start_sun": 0,
                "allowed_seeds": list(range(10)),
                "economy_floor": 2,
                "offense_floor": 1,
            }
        ]
        manager = CurriculumManager(phases=phases)
        env_mask = np.ones((1, 496), dtype=np.float32)
        spatial = np.zeros((1, 5, 9, 36), dtype=np.float32)

        economy_mask = manager.action_mask(env_mask, spatial)[0].astype(bool)
        sunflower_actions = [46 + row * 9 for row in range(5)]
        self.assertTrue(np.all(economy_mask[sunflower_actions]))
        self.assertFalse(economy_mask[46 + 2 * 45])

        spatial[0, 0, 0, 0] = 2.0
        spatial[0, 1, 0, 0] = 2.0
        offense_mask = manager.action_mask(env_mask, spatial)[0].astype(bool)
        melon_actions = np.arange(46 + 2 * 45, 46 + 3 * 45)
        winter_actions = np.arange(46 + 3 * 45, 46 + 4 * 45)
        self.assertTrue(np.all(offense_mask[melon_actions]))
        self.assertTrue(np.all(offense_mask[winter_actions]))
        self.assertFalse(offense_mask[46])

        spatial[0, 0, 0, 0] = 0.0
        spatial[0, 1, 0, 0] = 0.0
        spatial[0, 0, 3] = 1000.0
        emergency_mask = manager.action_mask(env_mask, spatial)[0].astype(bool)
        self.assertTrue(emergency_mask[46 + 8 * 45 + 6])
        self.assertTrue(emergency_mask[46 + 9 * 45 + 6])
        self.assertFalse(emergency_mask[46 + 2 * 45])

    def test_fixed_evaluation_suites_preserve_deployment_conditions(self):
        from evaluation import fixed_evaluation_suites

        suites = {
            suite["name"]: suite
            for suite in fixed_evaluation_suites(episodes=3, max_steps=40)
        }
        self.assertEqual(
            set(suites), {"bootstrap_eval", "transfer_eval", "zero_eval", "target_eval"}
        )
        self.assertEqual(
            suites["bootstrap_eval"],
            {
                "name": "bootstrap_eval",
                "start_wave": 1,
                "start_sun": 3000,
                "cooldown_scale": 0.13,
                "full_deck": False,
                "deterministic": True,
                "episodes": 3,
                "max_steps": 40,
                "seed_start": 272000,
            },
        )
        self.assertEqual(suites["target_eval"]["start_sun"], 50)
        self.assertTrue(suites["target_eval"]["full_deck"])

    def test_evaluation_scheduler_runs_each_suite_in_order(self):
        from evaluation import fixed_evaluation_suites, schedule_checkpoint_evaluations

        calls = []

        def evaluator(checkpoint_path, suite):
            calls.append((checkpoint_path, suite["name"]))
            return {"suite": suite["name"]}

        summaries = schedule_checkpoint_evaluations(
            "checkpoint_000001.pt", fixed_evaluation_suites(1, 2), evaluator
        )
        self.assertEqual(
            calls,
            [
                ("checkpoint_000001.pt", "bootstrap_eval"),
                ("checkpoint_000001.pt", "transfer_eval"),
                ("checkpoint_000001.pt", "zero_eval"),
                ("checkpoint_000001.pt", "target_eval"),
            ],
        )
        self.assertEqual(
            [summary["suite"] for summary in summaries], [call[1] for call in calls]
        )

    def test_wait_offense_override_requires_wait_offense_and_no_emergency(self):
        offense = 46 + 2 * 45
        emergency = 46 + 8 * 45
        mask = np.zeros(496, dtype=bool)
        mask[[0, offense]] = True
        logits = np.zeros(496, dtype=np.float32)

        self.assertEqual(
            train_modal._wait_offense_override_action(0, mask, logits), offense
        )
        self.assertEqual(
            train_modal._wait_offense_override_action(offense, mask, logits), offense
        )
        mask[emergency] = True
        self.assertEqual(train_modal._wait_offense_override_action(0, mask, logits), 0)

    def test_override_advances_recurrent_state_once_with_active_memory(self):
        class CountingAgent:
            def __init__(self):
                self.calls = 0

            def step_logits(self, spatial, global_vec, mask, caches):
                self.calls += 1
                logits = torch.zeros(1, 496)
                logits[0, 0] = 2
                logits[0, 46 + 2 * 45] = 1
                return logits, torch.zeros(1, 1), [cache + 1 for cache in caches]

        agent = CountingAgent()
        mask = torch.zeros(1, 496, dtype=torch.bool)
        mask[0, [0, 46 + 2 * 45]] = True
        action, caches, overridden, _ = train_modal._evaluation_policy_step(
            agent,
            torch.zeros(1, 5, 9, 36),
            torch.zeros(1, 24),
            mask,
            [torch.zeros(1)],
            deterministic=True,
            override=True,
        )
        self.assertEqual(agent.calls, 1)
        self.assertEqual(caches[0].item(), 1)
        self.assertTrue(overridden)
        self.assertEqual(action, 46 + 2 * 45)

    def test_manual_evaluation_accepts_seed_start_and_wait_offense_override(self):
        from train_modal import evaluate

        entrypoint = next(
            value
            for name, value in vars(evaluate).items()
            if name.startswith("_sync_original_")
        )
        parameters = inspect.signature(entrypoint._info.raw_f).parameters
        self.assertIn("seed_start", parameters)
        self.assertIn("wait_offense_override", parameters)

    def test_wait_offense_comparison_accepts_locked_seed_range(self):
        entrypoint = next(
            value
            for name, value in vars(train_modal.compare_wait_offense_override).items()
            if name.startswith("_sync_original_")
        )
        parameters = inspect.signature(entrypoint._info.raw_f).parameters
        self.assertEqual(parameters["episodes"].default, 100)
        self.assertEqual(parameters["seed_start"].default, 272000)
        self.assertEqual(parameters["start_sun"].default, 0)
        self.assertEqual(parameters["cooldown_scale"].default, 1.0)
        self.assertTrue(parameters["full_deck"].default)

    def test_paired_teacher_comparison_matches_explicit_seeds(self):
        from evaluation import paired_teacher_comparison

        baseline = [
            {"seed": 2, "terminal_reason": "lost", "max_wave": 12},
            {"seed": 1, "terminal_reason": "stage_complete", "max_wave": 21},
        ]
        candidate = [
            {"seed": 1, "terminal_reason": "lost", "max_wave": 18},
            {"seed": 2, "terminal_reason": "stage_complete", "max_wave": 21},
        ]
        comparison = paired_teacher_comparison(baseline, candidate)

        self.assertEqual([pair["seed"] for pair in comparison["pairs"]], [1, 2])
        self.assertEqual(comparison["completion_wins"], 1)
        self.assertEqual(comparison["completion_losses"], 1)
        self.assertEqual(comparison["mean_max_wave_delta"], 3)

    def test_paired_teacher_comparison_rejects_missing_seed(self):
        from evaluation import paired_teacher_comparison

        with self.assertRaises(ValueError):
            paired_teacher_comparison(
                [{"seed": 1, "terminal_reason": "lost", "max_wave": 1}],
                [{"seed": 2, "terminal_reason": "lost", "max_wave": 1}],
            )

        from evaluation import write_evaluation_results

        suite = {
            "name": "target_eval",
            "start_wave": 1,
            "start_sun": 50,
            "cooldown_scale": 1.0,
            "full_deck": True,
            "episodes": 2,
            "max_steps": 20,
        }
        rows = [
            {
                "checkpoint": "checkpoint_000001.pt",
                "suite": suite,
                "episode": 0,
                "max_wave": 6,
                "lost": False,
            },
            {
                "checkpoint": "checkpoint_000001.pt",
                "suite": suite,
                "episode": 1,
                "max_wave": 4,
                "lost": True,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            summary = write_evaluation_results(
                f"{directory}/checkpoint_000001.pt", suite, rows
            )
            with open(summary["results_path"], encoding="utf-8") as stream:
                saved_rows = [json.loads(line) for line in stream]

        self.assertEqual(saved_rows, rows)
        self.assertEqual(summary["median_max_wave"], 5)
        self.assertEqual(summary["loss_rate"], 0.5)

    def test_evaluation_action_geometry_decoding(self):
        action_counts = {
            "seeds": [0] * 10,
            "plant_rows": [0] * 5,
            "plant_columns": [0] * 9,
            "early_plant_rows": [0] * 5,
            "early_plant_columns": [0] * 9,
        }
        action_id = 46 + 7 * 45 + 3 * 9 + 8
        plant_cell = (action_id - 46) % 45
        action_counts["seeds"][(action_id - 46) // 45] += 1
        action_counts["plant_rows"][plant_cell // 9] += 1
        action_counts["plant_columns"][plant_cell % 9] += 1
        action_counts["early_plant_rows"][plant_cell // 9] += 1
        action_counts["early_plant_columns"][plant_cell % 9] += 1
        self.assertEqual(action_counts["seeds"][7], 1)
        self.assertEqual(action_counts["plant_rows"], [0, 0, 0, 1, 0])
        self.assertEqual(action_counts["plant_columns"], [0, 0, 0, 0, 0, 0, 0, 0, 1])
        self.assertEqual(action_counts["early_plant_rows"], [0, 0, 0, 1, 0])
        self.assertEqual(
            action_counts["early_plant_columns"], [0, 0, 0, 0, 0, 0, 0, 0, 1]
        )

    def test_danger_rows_uses_two_house_columns(self):
        spatial = np.zeros((5, 9, 36), dtype=np.float32)
        spatial[1, 1, 3] = 2
        spatial[3, 2, 3] = 3
        self.assertEqual(train_modal._danger_rows({"spatial": spatial}), [1])

    def test_terminal_board_snapshot_groups_rows(self):
        spatial = np.zeros((5, 9, 36), dtype=np.float32)
        spatial[2, 4, 0] = 1
        spatial[2, 6, 3] = 2.5
        spatial[4, 1, 4] = 1.5
        snapshot = train_modal._terminal_board_snapshot({"spatial": spatial})
        self.assertEqual(snapshot["plant_occupancy_by_row"], [0, 0, 1, 0, 0])
        self.assertEqual(snapshot["zombie_hp_by_row"], [0, 0, 2.5, 0, 1.5])

    def test_teacher_transfer_gate_requires_stage_completion_and_wave_twenty(self):
        from evaluation import teacher_transfer_gate

        summary = {"episodes": 5, "median_max_wave": 21}
        results = [
            {"stage_complete": True, "max_wave": 21},
            {"stage_complete": True, "max_wave": 21},
            {"stage_complete": True, "max_wave": 21},
            {"stage_complete": True, "max_wave": 21},
            {"stage_complete": False, "max_wave": 21},
        ]
        gate = teacher_transfer_gate(summary, results)

        self.assertEqual(gate["successful_stage_completions"], 4)
        self.assertEqual(gate["success_rate"], 0.8)
        self.assertTrue(gate["passed"])

    def test_ppo_update_runs(self):
        env = DummyVecEnv(2, 16)
        agent = PvZActorCritic()
        cfg = {"steps_per_update": 16, "gamma": 0.99, "gae_lambda": 0.95}
        manager = CurriculumManager()
        anchor = copy.deepcopy(agent).eval()
        rollout, next_value, _, _ = ppo.collect_rollout(
            agent,
            env,
            cfg,
            torch.device("cpu"),
            manager,
            anchor_agent=anchor,
        )
        rollout = ppo.compute_gae(rollout, next_value, cfg)
        rollout["optimization_mask"][:8] = 0.0

        optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
        train_cfg = {
            "num_updates": 2,
            "lr": 1e-3,
            "clip_coef": 0.2,
            "vf_coef": 0.5,
            "ent_coef_start": 0.01,
            "ent_coef_end": 0.001,
            "max_grad_norm": 0.5,
            "num_epochs": 2,
            "anchor_kl_coef": 0.1,
        }
        metrics = ppo.ppo_update(
            agent,
            optimizer,
            rollout,
            train_cfg,
            0,
            anchor_agent=anchor,
        )
        self.assertIn("policy_loss", metrics)
        self.assertIn("value_loss", metrics)
        self.assertIn("entropy", metrics)
        self.assertIn("anchor_kl", metrics)
        self.assertEqual(metrics["optimization_fraction"], 0.5)
        self.assertGreaterEqual(metrics["anchor_kl"], 0.0)
        self.assertTrue(np.isfinite(metrics["policy_loss"]))
        self.assertTrue(np.isfinite(metrics["value_loss"]))

    def test_ppo_losses_ignore_pre_frontier_targets(self):
        env = DummyVecEnv(1, 16)
        agent = PvZActorCritic()
        cfg = {"steps_per_update": 8, "gamma": 0.99, "gae_lambda": 0.95}
        anchor = copy.deepcopy(agent).eval()
        rollout, next_value, _, _ = ppo.collect_rollout(
            agent,
            env,
            cfg,
            torch.device("cpu"),
            CurriculumManager(),
            anchor_agent=anchor,
        )
        rollout = ppo.compute_gae(rollout, next_value, cfg)
        rollout["optimization_mask"][:4] = 0.0
        with torch.no_grad():
            agent.actor.residual.bias[0] += 0.1
        train_cfg = {
            "num_updates": 1,
            "lr": 0.0,
            "clip_coef": 0.2,
            "vf_coef": 0.5,
            "ent_coef_start": 0.01,
            "ent_coef_end": 0.01,
            "max_grad_norm": 0.5,
            "num_epochs": 1,
            "anchor_kl_coef": 0.1,
        }
        baseline = ppo.ppo_update(
            agent,
            torch.optim.Adam(agent.parameters(), lr=0.0),
            rollout,
            train_cfg,
            0,
            anchor_agent=anchor,
        )
        perturbed = copy.deepcopy(rollout)
        perturbed["returns"][:4] += 1_000_000
        perturbed["logprobs"][:4] += 100
        perturbed["actions"][:4] = 0
        changed = ppo.ppo_update(
            agent,
            torch.optim.Adam(agent.parameters(), lr=0.0),
            perturbed,
            train_cfg,
            0,
            anchor_agent=anchor,
        )

        for metric in ("policy_loss", "value_loss", "entropy"):
            self.assertAlmostEqual(baseline[metric], changed[metric], places=6)
        self.assertGreater(baseline["anchor_kl"], 0.0)
        self.assertAlmostEqual(baseline["anchor_kl"], changed["anchor_kl"], places=6)

    def test_train_smoke_on_dummy_env(self):
        """A tiny end-to-end run on the dummy env to catch integration issues."""
        env = DummyVecEnv(2, 8)
        cfg = {
            "num_envs": 2,
            "steps_per_update": 8,
            "num_updates": 2,
            "start_wave": 1,
            "lr": 1e-3,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_coef": 0.2,
            "vf_coef": 0.5,
            "ent_coef_start": 0.01,
            "ent_coef_end": 0.001,
            "max_grad_norm": 0.5,
            "num_epochs": 2,
            "device": "cpu",
            "resdir": "",
            "savedir": "",
            "checkpoint_dir": "",
            "checkpoint_interval": 10,
        }

        class _Factory:
            def __init__(self, env_):
                self.env = env_

            def __call__(self, num_envs, resdir, savedir, start_wave):
                return self.env

        fake_vec_env = types.ModuleType("vec_env")
        fake_vec_env.__dict__["PvZVecEnv"] = _Factory(env)
        original = sys.modules.get("vec_env")
        sys.modules["vec_env"] = fake_vec_env
        try:
            ppo.train(cfg)
        finally:
            if original is None:
                del sys.modules["vec_env"]
            else:
                sys.modules["vec_env"] = original

    def test_checkpoint_resume_restores_training_state(self):
        with tempfile.TemporaryDirectory() as directory:

            class _Factory:
                def __init__(self, env):
                    self.env = env

                def __call__(self, num_envs, resdir, savedir, start_wave):
                    return self.env

            fake_vec_env = types.ModuleType("vec_env")
            original = sys.modules.get("vec_env")
            sys.modules["vec_env"] = fake_vec_env
            try:
                base = {
                    "num_envs": 1,
                    "steps_per_update": 4,
                    "start_wave": 1,
                    "lr": 1e-3,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_coef": 0.2,
                    "vf_coef": 0.5,
                    "ent_coef_start": 0.01,
                    "ent_coef_end": 0.001,
                    "max_grad_norm": 0.5,
                    "num_epochs": 1,
                    "device": "cpu",
                    "resdir": "",
                    "savedir": "",
                    "checkpoint_interval": 1,
                    "use_reward_norm": True,
                }

                reference_dir = os.path.join(directory, "reference")
                torch.manual_seed(123)
                fake_vec_env.__dict__["PvZVecEnv"] = _Factory(DummyVecEnv(1, 4))
                ppo.train({**base, "num_updates": 2, "checkpoint_dir": reference_dir})
                reference = torch.load(
                    os.path.join(reference_dir, "checkpoint_000001.pt"),
                    weights_only=True,
                )

                torch.manual_seed(123)
                fake_vec_env.__dict__["PvZVecEnv"] = _Factory(DummyVecEnv(1, 4))
                ppo.train({**base, "num_updates": 1, "checkpoint_dir": directory})
                first = os.path.join(directory, "checkpoint_000000.pt")
                saved = torch.load(first, weights_only=True)
                self.assertEqual(saved["resume_state_version"], 1)
                self.assertEqual(saved["selection_protocol_version"], 1)
                self.assertIn("optimizer_state", saved)
                self.assertIn("reward_normalizer_state", saved)
                self.assertIn("torch_rng_state", saved)
                with self.assertRaisesRegex(ValueError, "leaves no updates"):
                    ppo.train(
                        {
                            **base,
                            "num_updates": 1,
                            "checkpoint_path": first,
                            "checkpoint_dir": directory,
                        }
                    )

                torch.manual_seed(999)
                fake_vec_env.__dict__["PvZVecEnv"] = _Factory(DummyVecEnv(1, 4))
                ppo.train(
                    {
                        **base,
                        "num_updates": 2,
                        "checkpoint_path": first,
                        "checkpoint_dir": directory,
                    }
                )
                resumed = torch.load(
                    os.path.join(directory, "checkpoint_000001.pt"), weights_only=True
                )
                self.assertEqual(resumed["update"], 1)
                self.assertEqual(resumed["schedule_num_updates"], 1)
                for name, tensor in reference["agent_state"].items():
                    self.assertTrue(
                        torch.equal(tensor, resumed["agent_state"][name]), name
                    )
            finally:
                if original is None:
                    del sys.modules["vec_env"]
                else:
                    sys.modules["vec_env"] = original

    def test_generalizing_selection_requires_broad_domain_gates(self):
        summaries = [
            {"suite": {"name": "bootstrap_eval"}, "mean_max_wave": 21.0},
            {"suite": {"name": "transfer_eval"}, "mean_max_wave": 19.9},
            {"suite": {"name": "zero_eval"}, "mean_max_wave": 9.5},
        ]
        self.assertIsNone(ppo._generalizing_selection(summaries, {}))
        summaries[1]["mean_max_wave"] = 20.0
        selected = ppo._generalizing_selection(
            summaries, {"selection_suite": "zero_eval"}
        )
        self.assertIsNotNone(selected)
        if selected is None:
            self.fail("expected a broad-domain selection")
        self.assertEqual(selected["mean_max_wave"], 9.5)
        incumbent = {
            "bootstrap_eval": {"mean_max_wave": 21.0},
            "transfer_eval": {"mean_max_wave": 20.1},
            "zero_eval": {"mean_max_wave": 9.4},
        }
        self.assertIsNone(ppo._generalizing_selection(summaries, {}, incumbent))

    def test_train_schedules_fixed_evaluations_after_checkpoint(self):
        env = DummyVecEnv(1, 4)
        calls = []

        def evaluator(checkpoint_path, suite):
            calls.append((checkpoint_path, suite["name"]))
            return {
                "suite": suite,
                "mean_max_wave": 21.0 if suite["name"] != "zero_eval" else 9.0,
            }

        cfg = {
            "num_envs": 1,
            "steps_per_update": 4,
            "num_updates": 1,
            "start_wave": 1,
            "lr": 1e-3,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_coef": 0.2,
            "vf_coef": 0.5,
            "ent_coef_start": 0.01,
            "ent_coef_end": 0.001,
            "max_grad_norm": 0.5,
            "num_epochs": 1,
            "device": "cpu",
            "resdir": "",
            "savedir": "",
            "checkpoint_interval": 1,
            "checkpoint_evaluator": evaluator,
            "evaluation_suites": __import__("evaluation").fixed_evaluation_suites(
                episodes=1, max_steps=4
            ),
        }

        class _Factory:
            def __call__(self, num_envs, resdir, savedir, start_wave):
                return env

        fake_vec_env = types.ModuleType("vec_env")
        fake_vec_env.__dict__["PvZVecEnv"] = _Factory()
        original = sys.modules.get("vec_env")
        sys.modules["vec_env"] = fake_vec_env
        try:
            with tempfile.TemporaryDirectory() as directory:
                cfg["checkpoint_dir"] = directory
                ppo.train(cfg)
                self.assertTrue(
                    os.path.isfile(os.path.join(directory, "checkpoint_000000.pt"))
                )
        finally:
            if original is None:
                del sys.modules["vec_env"]
            else:
                sys.modules["vec_env"] = original

        self.assertEqual(
            [name for _, name in calls],
            ["bootstrap_eval", "transfer_eval", "zero_eval", "target_eval"],
        )

    def test_train_with_wandb_disabled(self):
        """Smoke test the W&B code path without actually logging."""
        env = DummyVecEnv(2, 8)
        cfg = {
            "num_envs": 2,
            "steps_per_update": 8,
            "num_updates": 2,
            "start_wave": 1,
            "lr": 1e-3,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_coef": 0.2,
            "vf_coef": 0.5,
            "ent_coef_start": 0.01,
            "ent_coef_end": 0.001,
            "max_grad_norm": 0.5,
            "num_epochs": 2,
            "device": "cpu",
            "resdir": "",
            "savedir": "",
            "checkpoint_dir": "",
            "checkpoint_interval": 10,
            "wandb_project": "test_pvz_ppo",
            "wandb_run_name": "dummy_run",
        }

        class _Factory:
            def __init__(self, env_):
                self.env = env_

            def __call__(self, num_envs, resdir, savedir, start_wave):
                return self.env

        fake_vec_env = types.ModuleType("vec_env")
        fake_vec_env.__dict__["PvZVecEnv"] = _Factory(env)
        original = sys.modules.get("vec_env")
        sys.modules["vec_env"] = fake_vec_env
        try:
            ppo.train(cfg)
        finally:
            if original is None:
                del sys.modules["vec_env"]
            else:
                sys.modules["vec_env"] = original


class HardenedSelectionGatesTest(unittest.TestCase):
    """Phase 4 promotion-gate hardening: presence, sample size, completion."""

    @staticmethod
    def _summary(name, episodes=20, mean=21.0, median=21.0, completion=1.0):
        return {
            "suite": {"name": name},
            "episodes": episodes,
            "mean_max_wave": mean,
            "median_max_wave": median,
            "loss_rate": 0.0,
            "completion_rate": completion,
        }

    def test_missing_required_suite_never_promotes(self):
        cfg = {"selection_suite": "bc1500_eval", "required_suites": ["bootstrap_eval"]}
        self.assertIsNone(
            ppo._generalizing_selection([self._summary("bc1500_eval")], cfg)
        )

    def test_short_suite_never_promotes(self):
        cfg = {"selection_suite": "bc1500_eval"}
        self.assertIsNone(
            ppo._generalizing_selection(
                [self._summary("bc1500_eval", episodes=10)], cfg
            )
        )

    def test_completion_floor_gates_selection(self):
        cfg = {
            "selection_suite": "bc1500_eval",
            "min_selection_completion_rate": 0.8,
        }
        self.assertIsNone(
            ppo._generalizing_selection(
                [self._summary("bc1500_eval", completion=0.5)], cfg
            )
        )
        self.assertIsNotNone(
            ppo._generalizing_selection(
                [self._summary("bc1500_eval", completion=0.8)], cfg
            )
        )

    def test_incumbent_median_and_completion_regressions_block(self):
        cfg = {"selection_suite": "bc1500_eval"}
        incumbent = {
            "bc1500_eval": {
                "mean_max_wave": 21.0,
                "median_max_wave": 21.0,
                "completion_rate": 1.0,
            }
        }
        self.assertIsNone(
            ppo._generalizing_selection(
                [self._summary("bc1500_eval", median=19.0)], cfg, incumbent
            )
        )
        self.assertIsNone(
            ppo._generalizing_selection(
                [self._summary("bc1500_eval", completion=0.8)], cfg, incumbent
            )
        )
        # Exactly one median wave of regression stays within tolerance.
        self.assertIsNotNone(
            ppo._generalizing_selection(
                [self._summary("bc1500_eval", median=20.0)], cfg, incumbent
            )
        )

    def test_resolve_evaluation_suites_by_name(self):
        from evaluation import resolve_evaluation_suites

        suites = resolve_evaluation_suites(
            ["bc1500_eval", "transfer_eval"], episodes=7, seed_start=274540
        )
        self.assertEqual(
            [suite["name"] for suite in suites], ["bc1500_eval", "transfer_eval"]
        )
        self.assertEqual(suites[0]["start_sun"], 1500)
        self.assertFalse(suites[0]["full_deck"])
        self.assertEqual(suites[0]["episodes"], 7)
        self.assertEqual(suites[0]["seed_start"], 274540)
        with self.assertRaises(ValueError):
            resolve_evaluation_suites(["nope"], episodes=7)

    def test_write_evaluation_results_counts_truncation_as_noncompletion(self):
        from evaluation import write_evaluation_results

        rows = [
            {"seed": 1, "max_wave": 21, "lost": False, "stage_complete": True},
            {
                "seed": 2,
                "max_wave": 21,
                "lost": False,
                "stage_complete": False,
                "terminal_reason": "truncated",
            },
            {"seed": 3, "max_wave": 12, "lost": True, "terminal_reason": "lost"},
            {
                "seed": 4,
                "max_wave": 21,
                "lost": False,
                "terminal_reason": "stage_complete",
            },
        ]
        suite = {"name": "t", "episodes": 4, "max_steps": 10}
        with tempfile.TemporaryDirectory() as tmp:
            summary = write_evaluation_results(
                os.path.join(tmp, "ckpt.pt"), suite, rows
            )
        self.assertEqual(summary["completion_rate"], 0.5)
        self.assertEqual(summary["episodes"], 4)

    def test_bc1500_phase_preset_matches_bc_training_mask(self):
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
            checkpoint_path="",
            checkpoint_interval=1,
            wandb_project="",
            wandb_run_name="",
            cooldown_scale=1.0,
            phase="bc1500",
            sparse_reward=True,
            selection_suite="bc1500_eval",
            evaluation_suite_names="bc1500_eval,transfer_eval",
            evaluation_seed_start=274540,
            required_suites="bc1500_eval,transfer_eval",
            min_selection_completion_rate=0.8,
        )
        phase = cfg["phases"][0]
        self.assertEqual(phase["start_wave"], 1)
        self.assertEqual(phase["start_sun"], 1500)
        self.assertEqual(phase["waves_to_survive"], 19)
        self.assertEqual(phase["allowed_seeds"], [0, 1, 2, 6, 8])
        self.assertTrue(cfg["sparse_reward"])
        self.assertFalse(cfg["use_reward_norm"])
        self.assertEqual(cfg["selection_suite"], "bc1500_eval")
        self.assertEqual(cfg["required_suites"], ["bc1500_eval", "transfer_eval"])
        self.assertEqual(
            [suite["name"] for suite in cfg["evaluation_suites"]],
            ["bc1500_eval", "transfer_eval"],
        )
        self.assertEqual(cfg["evaluation_suites"][0]["seed_start"], 274540)
        self.assertEqual(cfg["gae_lambda"], 0.99)

    def test_bc1000_phase_preset_descends_from_bc1500(self):
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
            checkpoint_path="",
            checkpoint_interval=1,
            wandb_project="",
            wandb_run_name="",
            cooldown_scale=1.0,
            phase="bc1000",
            sparse_reward=True,
            selection_suite="bc1000_eval",
            evaluation_suite_names="bc1000_eval",
            evaluation_seed_start=274580,
            required_suites="bc1000_eval",
            min_selection_completion_rate=0.8,
        )
        phase = cfg["phases"][0]
        self.assertEqual(phase["start_sun"], 1000)
        self.assertEqual(phase["allowed_seeds"], [0, 1, 2, 6, 8])
        self.assertEqual(phase["waves_to_survive"], 19)
        self.assertEqual(cfg["gae_lambda"], 0.99)


if __name__ == "__main__":
    unittest.main()
