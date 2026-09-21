"""Episode-sharded teacher trajectory storage for behavior cloning."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from teacher_policy import StatefulTeacher, choose_action


def save_episode_shard(
    path: str | Path, trajectory: dict[str, Any], config: dict[str, Any]
) -> None:
    """Save one owned episode trajectory without allowing timestep leakage."""
    required = ("spatial", "global", "masks", "actions", "rewards", "dones")
    missing = [key for key in required if key not in trajectory]
    if missing:
        raise ValueError(f"missing trajectory fields: {missing}")
    steps = len(trajectory["actions"])
    if any(len(trajectory[key]) != steps for key in required):
        raise ValueError("trajectory fields must have the same timestep count")
    if "imitation_mask" in trajectory and len(trajectory["imitation_mask"]) != steps:
        raise ValueError("imitation_mask must match the trajectory length")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "spatial": np.ascontiguousarray(trajectory["spatial"], dtype=np.float32),
        "global_": np.ascontiguousarray(trajectory["global"], dtype=np.float32),
        "masks": np.ascontiguousarray(trajectory["masks"], dtype=np.bool_),
        "actions": np.ascontiguousarray(trajectory["actions"], dtype=np.int64),
        "rewards": np.ascontiguousarray(trajectory["rewards"], dtype=np.float32),
        "dones": np.ascontiguousarray(trajectory["dones"], dtype=np.bool_),
        "config": np.array(json.dumps(config, sort_keys=True)),
    }
    if "imitation_mask" in trajectory:
        arrays["imitation_mask"] = np.ascontiguousarray(
            trajectory["imitation_mask"], dtype=np.bool_
        )
    np.savez_compressed(target, **arrays)


def load_episode_shard(path: str | Path) -> dict[str, Any]:
    """Load a shard and validate that each label remains legal under its mask."""
    with np.load(path, allow_pickle=False) as data:
        try:
            config = json.loads(str(data["config"]))
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("invalid episode shard config") from error
        if (
            "searched" in config
            and config.get("branch_method") != "fresh_process_seeded_replay"
        ):
            raise ValueError("legacy snapshot-labelled search shards are quarantined")
        result = {
            "spatial": data["spatial"],
            "global": data["global_"],
            "masks": data["masks"],
            "actions": data["actions"],
            "rewards": data["rewards"],
            "dones": data["dones"],
            "config": config,
        }
        if "imitation_mask" in data:
            result["imitation_mask"] = data["imitation_mask"]
    if not np.all(
        result["masks"][np.arange(len(result["actions"])), result["actions"]]
    ):
        raise ValueError(
            "teacher shard contains an action disabled by its effective mask"
        )
    return result


def collect_teacher_episode(
    env: Any,
    curriculum_mask: np.ndarray,
    max_steps: int,
    reset_options: dict[str, Any] | None = None,
    profile: str = "structured-v2",
) -> dict[str, Any]:
    """Collect one profile-driven episode using only the effective action mask."""
    obs, info = env.reset(options=reset_options)
    teacher = StatefulTeacher(profile) if profile == "structured-v4-sun1000" else None
    rows = {
        key: [] for key in ("spatial", "global", "masks", "actions", "rewards", "dones")
    }
    terminal_reason = "max_steps"
    stage_complete = False
    for _ in range(max_steps):
        effective_mask = np.logical_and(info["action_mask"], curriculum_mask)
        action = (
            teacher.choose_action_with_diagnostics(
                effective_mask,
                obs["spatial"],
                obs["global"],
                env.squash_targetable_lanes()
                if hasattr(env, "squash_targetable_lanes")
                else None,
            )[0]
            if teacher is not None
            else choose_action(effective_mask, profile, obs["spatial"])
        )
        rows["spatial"].append(np.array(obs["spatial"], dtype=np.float32, copy=True))
        rows["global"].append(np.array(obs["global"], dtype=np.float32, copy=True))
        rows["masks"].append(np.array(effective_mask, dtype=bool, copy=True))
        rows["actions"].append(action)
        obs, reward, done, truncated, info = env.step(action)
        # pi-lens-ignore: unchecked-throwing-call-python
        rows["rewards"].append(float(reward))
        rows["dones"].append(bool(done or truncated))
        if done or truncated:
            stage_complete = bool(info.get("stage_complete", False))
            terminal_reason = (
                "stage_complete"
                if stage_complete
                else "lost"
                if info.get("lost", False)
                else "truncated"
            )
            break
    return {
        "spatial": np.stack(rows["spatial"]),
        "global": np.stack(rows["global"]),
        "masks": np.stack(rows["masks"]),
        "actions": np.asarray(rows["actions"], dtype=np.int64),
        "rewards": np.asarray(rows["rewards"], dtype=np.float32),
        "dones": np.asarray(rows["dones"], dtype=bool),
        "terminal_reason": terminal_reason,
        "stage_complete": stage_complete,
    }


def split_episode_paths(
    paths: Iterable[str | Path], validation_fraction: float = 0.2
) -> tuple[list[Path], list[Path]]:
    """Split whole episodes deterministically; no timestep appears in both splits."""
    episode_paths = sorted(Path(path) for path in paths)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    validation_count = (
        max(1, round(len(episode_paths) * validation_fraction)) if episode_paths else 0
    )
    return episode_paths[validation_count:], episode_paths[:validation_count]
