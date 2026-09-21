"""Unit tests for deterministic teacher-policy action selection."""

import tempfile
import unittest
from typing import Any, cast

import numpy as np

from teacher_data import load_episode_shard, save_episode_shard, split_episode_paths
from teacher_policy import (
    PREFERRED_ACTIONS,
    StatefulTeacher,
    choose_action,
    choose_actions,
    preferred_actions,
)


class TestTeacherPolicy(unittest.TestCase):
    def test_prefers_structured_legal_placement(self):
        mask = np.zeros(496, dtype=bool)
        mask[0] = True
        mask[PREFERRED_ACTIONS[2]] = True
        mask[PREFERRED_ACTIONS[1]] = True

        self.assertEqual(choose_action(mask), PREFERRED_ACTIONS[1])

    def test_waits_after_structured_layout_is_unavailable(self):
        mask = np.zeros(496, dtype=bool)
        mask[0] = True
        mask[46 + 8 * 45 + 44] = True
        mask[46 + 4 * 45 + 7] = True

        self.assertEqual(choose_action(mask), 0)

    def test_vector_policy_never_returns_masked_action(self):
        masks = np.zeros((2, 496), dtype=bool)
        masks[:, 0] = True
        masks[0, PREFERRED_ACTIONS[0]] = True
        masks[1, 46 + 9 * 45] = True

        actions = choose_actions(masks)
        self.assertTrue(np.all(masks[np.arange(2), actions]))
        self.assertEqual(actions.tolist(), [PREFERRED_ACTIONS[0], 0])

    def test_sun1000_profile_builds_two_economy_columns_before_emergency_line(self):
        actions = preferred_actions("structured-v3-sun1000")

        expected_economy = [46 + row * 9 + col for col in (0, 1) for row in range(5)]
        expected_squash = [46 + 8 * 45 + row * 9 + 6 for row in range(5)]
        self.assertEqual(list(actions[:10]), expected_economy)
        self.assertEqual(list(actions[10:15]), expected_squash)

    def test_sun1000_repair_waits_for_an_incomplete_stage(self):
        mask = np.zeros(496, dtype=bool)
        mask[0] = True
        mask[46 + 8 * 45 + 6] = True  # squash is available before economy finishes
        spatial = np.zeros((5, 9, 36), dtype=np.float32)

        self.assertEqual(
            choose_action(mask, "structured-v3-sun1000-r3", spatial),
            0,
        )

        first_economy_action = 46
        mask[first_economy_action] = True
        self.assertEqual(
            choose_action(mask, "structured-v3-sun1000-r3", spatial),
            first_economy_action,
        )

    def test_stateful_sun1000_teacher_places_squash_in_most_threatened_lane(self):
        mask = np.zeros(496, dtype=bool)
        mask[0] = True
        mask[46 + 8 * 45 + 1 * 9 + 6] = True
        mask[46 + 8 * 45 + 3 * 9 + 6] = True
        spatial = np.zeros((5, 9, 36), dtype=np.float32)
        spatial[1, 8, 3] = 270.0
        spatial[3, 2, 3] = 270.0

        teacher = StatefulTeacher("structured-v4-sun1000")

        self.assertEqual(
            teacher.choose_action(mask, spatial),
            46 + 8 * 45 + 3 * 9 + 6,
        )

    def test_stateful_teacher_skips_squash_lane_outside_engine_acquisition_range(self):
        mask = np.zeros(496, dtype=bool)
        mask[0] = True
        squash_lane_one = 46 + 8 * 45 + 1 * 9 + 6
        squash_lane_three = 46 + 8 * 45 + 3 * 9 + 6
        mask[[squash_lane_one, squash_lane_three]] = True
        spatial = np.zeros((5, 9, 36), dtype=np.float32)
        spatial[1, 2, 3] = 540.0
        spatial[3, 2, 3] = 270.0

        action, diagnostic = StatefulTeacher(
            "structured-v4-sun1000"
        ).choose_action_with_diagnostics(
            mask, spatial, squash_targetable_lanes=np.array([0, 0, 0, 1, 0])
        )

        self.assertEqual(action, squash_lane_three)
        self.assertEqual(
            diagnostic["squash_targetable_lanes"], [False, False, False, True, False]
        )

    def test_cooldown_repair_arms_ready_squash_before_emergency(self):
        mask = np.zeros(496, dtype=bool)
        mask[0] = True
        squash = 46 + 8 * 45 + 2 * 9 + 6
        mask[squash] = True
        spatial = np.zeros((5, 9, 36), dtype=np.float32)
        spatial[2, 5, 3] = 100.0  # 100 HP × (9 - 5) = 400 pre-emergency danger

        self.assertEqual(
            StatefulTeacher("structured-v4-sun1000").choose_action(mask, spatial), 0
        )
        self.assertEqual(
            StatefulTeacher("structured-v5-sun1000-cooldown").choose_action(
                mask, spatial
            ),
            squash,
        )

    def test_stateful_sun1000_teacher_records_emergency_diagnostics(self):
        mask = np.zeros(496, dtype=bool)
        mask[0] = True
        spatial = np.zeros((5, 9, 36), dtype=np.float32)
        spatial[3, 2, 3] = 270.0
        spatial[3, 6, 0] = 1.0
        global_observation = np.zeros(24, dtype=np.float32)
        global_observation[0] = 49.0 / 9990.0
        global_observation[10] = 0.5
        teacher = StatefulTeacher("structured-v4-sun1000")

        action, diagnostic = teacher.choose_action_with_diagnostics(
            mask, spatial, global_observation
        )

        self.assertEqual(action, 0)
        self.assertEqual(diagnostic["selected_action"], 0)
        self.assertEqual(diagnostic["emergency_lane"], 3)
        self.assertFalse(diagnostic["emergency_squash_legal"])
        self.assertEqual(diagnostic["squash_seed_cooldown"], 0.5)
        self.assertAlmostEqual(cast(float, diagnostic["available_sun"]), 49.0)
        self.assertFalse(diagnostic["squash_affordable"])
        self.assertTrue(diagnostic["squash_target_occupied"])
        self.assertEqual(len(cast(list[Any], diagnostic["lane_danger"])), 5)

    def test_stateful_sun1000_teacher_records_and_resets_squash_history(self):
        teacher = StatefulTeacher("structured-v4-sun1000")
        squash_lane_one = 46 + 8 * 45 + 1 * 9 + 6
        squash_mask = np.zeros(496, dtype=bool)
        squash_mask[0] = True
        squash_mask[squash_lane_one] = True
        lane_one_emergency = np.zeros((5, 9, 36), dtype=np.float32)
        lane_one_emergency[1, 2, 3] = 270.0

        action, first = teacher.choose_action_with_diagnostics(
            squash_mask, lane_one_emergency
        )
        self.assertEqual(action, squash_lane_one)
        self.assertIsNone(first["last_squash_lane"])
        self.assertIsNone(first["decisions_since_last_squash"])
        self.assertEqual(
            first["emergency_squash_history_classification"], "no_recent_squash"
        )

        wait_mask = np.zeros(496, dtype=bool)
        wait_mask[0] = True
        lane_three_emergency = np.zeros((5, 9, 36), dtype=np.float32)
        lane_three_emergency[3, 2, 3] = 270.0
        _, conflict = teacher.choose_action_with_diagnostics(
            wait_mask, lane_three_emergency
        )
        self.assertEqual(conflict["last_squash_lane"], 1)
        self.assertEqual(conflict["decisions_since_last_squash"], 1)
        self.assertEqual(
            conflict["emergency_squash_history_classification"],
            "different_lane_conflict",
        )
        self.assertEqual(conflict["last_squash_commitment_lane"], 1)
        self.assertEqual(conflict["last_squash_commitment_decision"], 0)
        self.assertEqual(
            conflict["last_squash_commitment_danger"], [0.0, 1890.0, 0.0, 0.0, 0.0]
        )
        self.assertEqual(conflict["emergency_lane_danger_at_last_squash"], 0.0)
        self.assertFalse(conflict["emergency_lane_had_danger_when_squash_committed"])

        _, repeat = teacher.choose_action_with_diagnostics(
            wait_mask, lane_one_emergency
        )
        self.assertEqual(repeat["decisions_since_last_squash"], 2)
        self.assertEqual(
            repeat["emergency_squash_history_classification"], "same_lane_repeat"
        )
        self.assertEqual(repeat["emergency_lane_danger_at_last_squash"], 1890.0)
        self.assertTrue(repeat["emergency_lane_had_danger_when_squash_committed"])

        teacher.reset()
        _, reset = teacher.choose_action_with_diagnostics(wait_mask, lane_one_emergency)
        self.assertIsNone(reset["last_squash_lane"])
        self.assertIsNone(reset["decisions_since_last_squash"])
        self.assertEqual(
            reset["emergency_squash_history_classification"], "no_recent_squash"
        )
        self.assertIsNone(reset["last_squash_commitment_lane"])
        self.assertIsNone(reset["last_squash_commitment_decision"])
        self.assertIsNone(reset["last_squash_commitment_danger"])
        self.assertIsNone(reset["emergency_lane_danger_at_last_squash"])
        self.assertIsNone(reset["emergency_lane_had_danger_when_squash_committed"])

    def test_squash_commitment_outcome_records_cleared_lane_before_different_emergency(
        self,
    ):
        teacher = StatefulTeacher("structured-v4-sun1000")
        squash_lane_one = 46 + 8 * 45 + 1 * 9 + 6
        squash_mask = np.zeros(496, dtype=bool)
        squash_mask[[0, squash_lane_one]] = True
        wait_mask = np.zeros(496, dtype=bool)
        wait_mask[0] = True
        lane_one_emergency = np.zeros((5, 9, 36), dtype=np.float32)
        lane_one_emergency[1, 2, 3] = 270.0
        lane_three_emergency = np.zeros((5, 9, 36), dtype=np.float32)
        lane_three_emergency[3, 2, 3] = 270.0

        teacher.choose_action_with_diagnostics(squash_mask, lane_one_emergency)
        _, clear_diagnostic = teacher.choose_action_with_diagnostics(
            wait_mask, np.zeros((5, 9, 36), dtype=np.float32)
        )
        _, emergency_diagnostic = teacher.choose_action_with_diagnostics(
            wait_mask, lane_three_emergency
        )
        outcomes = teacher.close_episode("lost")

        self.assertEqual(clear_diagnostic["active_squash_commitment_lane"], 1)
        self.assertEqual(clear_diagnostic["committed_lane_current_danger"], 0.0)
        self.assertTrue(
            clear_diagnostic["committed_lane_cleared_before_first_emergency"]
        )
        self.assertEqual(emergency_diagnostic["first_emergency_lane_since_squash"], 3)
        self.assertEqual(emergency_diagnostic["first_emergency_age_since_squash"], 2)
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["outcome_category"], "zero_danger_lane_emergence")
        self.assertTrue(outcomes[0]["committed_lane_cleared_before_first_emergency"])
        self.assertEqual(outcomes[0]["close_reason"], "lost")

    def test_squash_commitment_outcome_records_same_lane_recurrence(self):
        teacher = StatefulTeacher("structured-v4-sun1000")
        squash_lane_one = 46 + 8 * 45 + 1 * 9 + 6
        squash_mask = np.zeros(496, dtype=bool)
        squash_mask[[0, squash_lane_one]] = True
        wait_mask = np.zeros(496, dtype=bool)
        wait_mask[0] = True
        lane_one_emergency = np.zeros((5, 9, 36), dtype=np.float32)
        lane_one_emergency[1, 2, 3] = 270.0

        teacher.choose_action_with_diagnostics(squash_mask, lane_one_emergency)
        _, diagnostic = teacher.choose_action_with_diagnostics(
            wait_mask, lane_one_emergency
        )
        outcome = teacher.close_episode("truncated")[0]

        self.assertEqual(diagnostic["first_emergency_lane_since_squash"], 1)
        self.assertEqual(outcome["outcome_category"], "same_lane_recurrence")
        self.assertFalse(outcome["committed_lane_cleared_before_first_emergency"])
        self.assertEqual(outcome["first_emergency_age"], 1)

    def test_squash_commitment_outcome_attributes_known_competing_lane(self):
        teacher = StatefulTeacher("structured-v4-sun1000")
        squash_lane_one = 46 + 8 * 45 + 1 * 9 + 6
        squash_mask = np.zeros(496, dtype=bool)
        squash_mask[[0, squash_lane_one]] = True
        wait_mask = np.zeros(496, dtype=bool)
        wait_mask[0] = True
        commitment_board = np.zeros((5, 9, 36), dtype=np.float32)
        commitment_board[1, 2, 3] = 270.0
        commitment_board[3, 8, 3] = 50.0
        lane_three_emergency = np.zeros((5, 9, 36), dtype=np.float32)
        lane_three_emergency[3, 2, 3] = 270.0

        teacher.choose_action_with_diagnostics(squash_mask, commitment_board)
        teacher.choose_action_with_diagnostics(wait_mask, lane_three_emergency)
        outcome = teacher.close_episode("stage_complete")[0]

        self.assertEqual(outcome["outcome_category"], "known_competing_lane_emergence")
        self.assertEqual(outcome["first_emergency_lane"], 3)
        self.assertEqual(outcome["first_emergency_lane_danger_at_commitment"], 50.0)

    def test_squash_commitment_outcomes_reset_without_leakage(self):
        teacher = StatefulTeacher("structured-v4-sun1000")
        squash_lane_one = 46 + 8 * 45 + 1 * 9 + 6
        squash_mask = np.zeros(496, dtype=bool)
        squash_mask[[0, squash_lane_one]] = True
        lane_one_emergency = np.zeros((5, 9, 36), dtype=np.float32)
        lane_one_emergency[1, 2, 3] = 270.0

        teacher.choose_action_with_diagnostics(squash_mask, lane_one_emergency)
        teacher.reset()

        self.assertEqual(teacher.close_episode("lost"), [])

    def test_squash_commitment_records_target_lifecycle(self):
        teacher = StatefulTeacher("structured-v4-sun1000")
        squash_lane_one = 46 + 8 * 45 + 1 * 9 + 6
        squash_mask = np.zeros(496, dtype=bool)
        squash_mask[[0, squash_lane_one]] = True
        wait_mask = np.zeros(496, dtype=bool)
        wait_mask[0] = True
        lane_one_emergency = np.zeros((5, 9, 36), dtype=np.float32)
        lane_one_emergency[1, 2, 3] = 270.0
        visible_squash = np.zeros((5, 9, 36), dtype=np.float32)
        visible_squash[1, 6, 0] = 18.0
        visible_squash[1, 6, 1] = 0.75
        visible_squash[1, 6, 2] = 2.0

        _, commitment = teacher.choose_action_with_diagnostics(
            squash_mask, lane_one_emergency
        )
        _, visible = teacher.choose_action_with_diagnostics(wait_mask, visible_squash)
        _, absent = teacher.choose_action_with_diagnostics(
            wait_mask, np.zeros((5, 9, 36), dtype=np.float32)
        )
        outcome = teacher.close_episode("stage_complete")[0]

        self.assertEqual(
            commitment["nearest_zombie_columns"], [None, 2, None, None, None]
        )
        self.assertTrue(visible["committed_squash_visible_at_target"])
        self.assertEqual(visible["committed_squash_state_at_target"], 2.0)
        self.assertEqual(visible["committed_squash_health_at_target"], 0.75)
        self.assertFalse(absent["committed_squash_visible_at_target"])
        self.assertTrue(outcome["squash_seen_after_commitment"])
        self.assertEqual(outcome["first_squash_visible_age"], 1)
        self.assertEqual(outcome["nearest_zombie_column_at_commitment"], 2)
        self.assertIsNone(outcome["first_post_commit_nearest_zombie_column"])
        self.assertEqual(outcome["first_squash_absent_age"], 2)

    def test_squash_commitment_records_unseen_target(self):
        teacher = StatefulTeacher("structured-v4-sun1000")
        squash_lane_one = 46 + 8 * 45 + 1 * 9 + 6
        squash_mask = np.zeros(496, dtype=bool)
        squash_mask[[0, squash_lane_one]] = True
        wait_mask = np.zeros(496, dtype=bool)
        wait_mask[0] = True
        lane_one_emergency = np.zeros((5, 9, 36), dtype=np.float32)
        lane_one_emergency[1, 2, 3] = 270.0

        teacher.choose_action_with_diagnostics(squash_mask, lane_one_emergency)
        _, diagnostic = teacher.choose_action_with_diagnostics(
            wait_mask, np.zeros((5, 9, 36), dtype=np.float32)
        )
        outcome = teacher.close_episode("lost")[0]

        self.assertFalse(diagnostic["committed_squash_visible_at_target"])
        self.assertFalse(outcome["squash_seen_after_commitment"])
        self.assertIsNone(outcome["first_squash_visible_age"])
        self.assertEqual(outcome["first_squash_absent_age"], 1)

    def test_stateful_sun1000_teacher_latches_destroyed_economy_milestone(self):
        mask = np.zeros(496, dtype=bool)
        mask[0] = True
        economy = [46 + row * 9 for row in range(5)]
        melon = 46 + 2 * 45 + 3
        mask[economy] = True
        mask[melon] = True
        spatial = np.zeros((5, 9, 36), dtype=np.float32)
        teacher = StatefulTeacher("structured-v4-sun1000")

        for action in economy:
            self.assertEqual(teacher.choose_action(mask, spatial), action)
            mask[action] = False

        self.assertEqual(teacher.choose_action(mask, spatial), melon)

    def test_stateful_sun1000_teacher_defers_destroyed_sunflower_rebuild_until_clear(
        self,
    ):
        mask = np.zeros(496, dtype=bool)
        mask[0] = True
        economy = [46 + row * 9 for row in range(5)]
        melon = 46 + 2 * 45 + 3
        mask[economy] = True
        mask[melon] = True
        clear_board = np.zeros((5, 9, 36), dtype=np.float32)
        active_attack = clear_board.copy()
        active_attack[2, 8, 3] = 100.0
        teacher = StatefulTeacher("structured-v4-sun1000")

        for action in economy:
            self.assertEqual(teacher.choose_action(mask, clear_board), action)
            mask[action] = False

        self.assertEqual(teacher.choose_action(mask, clear_board), melon)
        mask[melon] = False
        mask[economy[2]] = True

        self.assertEqual(teacher.choose_action(mask, active_attack), 0)
        self.assertEqual(teacher.choose_action(mask, clear_board), economy[2])


class TestTeacherShardData(unittest.TestCase):
    def test_saved_shard_preserves_owned_arrays_and_effective_mask(self):
        trajectory = {
            "spatial": np.zeros((2, 5, 9, 36), dtype=np.float32),
            "global": np.zeros((2, 12), dtype=np.float32),
            "masks": np.eye(496, dtype=bool)[[0, 46]],
            "actions": np.array([0, 46]),
            "rewards": np.array([1.0, 2.0]),
            "dones": np.array([False, True]),
            "imitation_mask": np.array([False, True]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/episode_000.npz"
            save_episode_shard(path, trajectory, {"suite": "bootstrap_eval"})
            loaded = load_episode_shard(path)

        self.assertEqual(loaded["config"]["suite"], "bootstrap_eval")
        self.assertTrue(np.all(loaded["masks"][np.arange(2), loaded["actions"]]))
        np.testing.assert_array_equal(loaded["imitation_mask"], [False, True])
        self.assertTrue(loaded["spatial"].flags["C_CONTIGUOUS"])

    def test_saved_shard_rejects_mismatched_imitation_mask(self):
        trajectory = {
            "spatial": np.zeros((2, 5, 9, 36), dtype=np.float32),
            "global": np.zeros((2, 12), dtype=np.float32),
            "masks": np.ones((2, 496), dtype=bool),
            "actions": np.array([0, 1]),
            "rewards": np.zeros(2),
            "dones": np.array([False, True]),
            "imitation_mask": np.array([True]),
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "imitation_mask"),
        ):
            save_episode_shard(f"{directory}/episode_000.npz", trajectory, {})

    def test_load_quarantines_legacy_search_shard(self):
        trajectory = {
            "spatial": np.zeros((1, 5, 9, 36), dtype=np.float32),
            "global": np.zeros((1, 12), dtype=np.float32),
            "masks": np.eye(496, dtype=bool)[[0]],
            "actions": np.array([0]),
            "rewards": np.zeros(1),
            "dones": np.array([True]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/episode_legacy.npz"
            save_episode_shard(path, trajectory, {"searched": {"max_wave": 8}})
            with self.assertRaisesRegex(ValueError, "quarantined"):
                load_episode_shard(path)

    def test_behavior_cloning_rejects_overlapping_partitions(self):
        from behavior_cloning import train_behavior_cloning

        with self.assertRaisesRegex(ValueError, "episode-disjoint"):
            train_behavior_cloning(
                object(),
                train_paths=["episode_001.npz"],
                validation_paths=["episode_001.npz"],
            )

    def test_episode_split_does_not_mix_episode_paths(self):
        train, validation = split_episode_paths(
            ["episode_2.npz", "episode_1.npz", "episode_3.npz"]
        )
        self.assertEqual({path.name for path in validation}, {"episode_1.npz"})
        self.assertEqual(
            {path.name for path in train}, {"episode_2.npz", "episode_3.npz"}
        )

    def test_collector_records_only_effective_legal_actions(self):
        from teacher_data import collect_teacher_episode

        class Env:
            def reset(self, options=None):
                return {"spatial": np.zeros((5, 9, 36)), "global": np.zeros(24)}, {
                    "action_mask": np.eye(496, dtype=bool)[46]
                }

            def step(self, action):
                self.action = action
                return (
                    {"spatial": np.zeros((5, 9, 36)), "global": np.zeros(24)},
                    1.0,
                    True,
                    False,
                    {"action_mask": np.eye(496, dtype=bool)[0]},
                )

        env = Env()
        trajectory = collect_teacher_episode(env, np.ones(496, dtype=bool), max_steps=2)
        self.assertEqual(trajectory["actions"].tolist(), [46])
        self.assertTrue(trajectory["dones"][0])


if __name__ == "__main__":
    unittest.main()
