"""Minibatch behavior-cloning trainer regression tests on synthetic shards.

Run from repo root:
    python -m unittest tests.test_behavior_cloning -v
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from behavior_cloning import train_behavior_cloning
from network import GLOBAL_SIZE, NUM_ACTIONS, PvZActorCritic  # noqa: E402
from teacher_data import save_episode_shard

PLANT_ACTION = NUM_ACTIONS - 400  # any legal plant id; wait is 0


def _write_shard(directory: Path, name: str, actions, imitation_mask=None) -> Path:
    steps = len(actions)
    trajectory = {
        "spatial": np.random.default_rng(0).random((steps, 5, 9, 36), dtype=np.float32),
        "global": np.random.default_rng(1).random(
            (steps, GLOBAL_SIZE), dtype=np.float32
        ),
        "masks": np.ones((steps, NUM_ACTIONS), dtype=bool),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.zeros(steps, dtype=np.float32),
        "dones": np.zeros(steps, dtype=bool),
    }
    if imitation_mask is not None:
        trajectory["imitation_mask"] = np.asarray(imitation_mask, dtype=bool)
    path = directory / name
    save_episode_shard(path, trajectory, {"teacher": "test"})
    return path


class MinibatchBehaviorCloningTest(unittest.TestCase):
    def test_minibatch_updates_and_balanced_weights(self):
        actions = [0, 0, PLANT_ACTION] * 4  # 12 steps, mostly wait
        with tempfile.TemporaryDirectory() as tmp:
            paths = [
                _write_shard(Path(tmp), f"episode_{i:05d}.npz", actions)
                for i in range(4)
            ]
            agent = PvZActorCritic()
            metrics = train_behavior_cloning(
                agent,
                paths,
                epochs=2,
                background_weight=0.5,
                seq_len=5,  # 12 steps -> ragged 5+5+2 chunks per episode
                batch_size=4,
            )
        # Minibatching: more optimizer updates than epochs (old code did 1/epoch).
        self.assertGreater(metrics["updates"], 2)
        self.assertEqual(metrics["background_weight"], 0.5)
        self.assertTrue(np.isfinite(metrics["validation"]["nll"]))
        self.assertGreater(metrics["validation"]["nll"], 0.0)

    def test_zero_weight_labels_rejected(self):
        # Explicit all-background imitation_mask with weight 0 leaves no labels.
        with tempfile.TemporaryDirectory() as tmp:
            paths = [
                _write_shard(
                    Path(tmp),
                    f"episode_{i:05d}.npz",
                    [0] * 8,
                    imitation_mask=[False] * 8,
                )
                for i in range(4)
            ]
            agent = PvZActorCritic()
            with self.assertRaises(ValueError):
                train_behavior_cloning(agent, paths, epochs=1, background_weight=0.0)


if __name__ == "__main__":
    unittest.main()
