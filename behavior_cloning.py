"""Supervised behavior-cloning training from episode-sharded teacher data."""

import math
import random
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from teacher_data import load_episode_shard, split_episode_paths

WAIT_ACTION = 0


def _label_weights(data: dict, background_weight: float) -> np.ndarray:
    """Per-step loss weights; Wait-labeled steps are the background class."""
    imitation_mask = data.get("imitation_mask")
    if imitation_mask is None:
        if background_weight <= 0.0:
            return np.ones(len(data["actions"]), dtype=np.float32)
        # Shards without an explicit mask: derive background = Wait labels.
        imitation_mask = data["actions"] != WAIT_ACTION
    return np.where(imitation_mask, 1.0, background_weight).astype(np.float32)


def _score_episodes(
    agent, datasets: list[dict], weights: list[np.ndarray], device
) -> dict[str, float]:
    """Full-episode weighted NLL/accuracy (matches deployment recurrence)."""
    total_steps = 0.0
    for weights_array in weights:
        total_steps += weights_array.sum().item()
    if total_steps == 0:
        raise ValueError("behavior cloning has no selected imitation labels")
    total_nll = correct = 0.0
    with torch.no_grad():
        for data, weights_array in zip(datasets, weights, strict=True):
            spatial = torch.from_numpy(data["spatial"]).unsqueeze(0).to(device)
            global_ = torch.from_numpy(data["global"]).unsqueeze(0).to(device)
            masks = torch.from_numpy(data["masks"]).unsqueeze(0).to(device)
            actions = torch.from_numpy(data["actions"]).unsqueeze(0).to(device)
            step_weights = torch.from_numpy(weights_array).unsqueeze(0).to(device)
            _, logprob, _, _ = agent.get_action_and_value(
                spatial, global_, masks, action=actions
            )
            total_nll += -(logprob * step_weights).sum().item()
            logits = agent._apply_action_mask(
                agent.actor(agent.forward_sequence(spatial, global_)), masks
            )
            predicted = logits.argmax(dim=-1)
            correct += ((predicted == actions) * step_weights).sum().item()
    return {"nll": total_nll / total_steps, "accuracy": correct / total_steps}


def train_behavior_cloning(
    agent,
    shard_paths: Iterable[str | Path] = (),
    epochs: int = 20,
    lr: float = 1e-4,
    *,
    train_paths: Iterable[str | Path] | None = None,
    validation_paths: Iterable[str | Path] | None = None,
    background_weight: float = 0.0,
    seq_len: int = 64,
    batch_size: int = 16,
):
    """Minibatch sequence training with class-balanced Wait weighting.

    Episodes are chunked into non-overlapping sequences of seq_len steps
    (recurrence restarts at chunk boundaries); every minibatch is one
    optimizer update, so `epochs` no longer means `epochs` gradient steps.
    Wait labels receive `background_weight` loss weight (0 disables balancing;
    a shard's explicit imitation_mask, when present, overrides the derivation).
    """
    if (train_paths is None) != (validation_paths is None):
        raise ValueError("provide both explicit train_paths and validation_paths")
    if not 0.0 <= background_weight <= 1.0:
        raise ValueError("background_weight must be in [0, 1]")
    if seq_len < 1 or batch_size < 1:
        raise ValueError("seq_len and batch_size must be positive")
    if train_paths is None:
        train_paths, validation_paths = split_episode_paths(shard_paths)
    else:
        assert validation_paths is not None
        train_paths = list(train_paths)
        validation_paths = list(validation_paths)
    if not train_paths or not validation_paths:
        raise ValueError(
            "behavior cloning requires non-empty train and validation shards"
        )
    train_paths = [Path(path).resolve() for path in train_paths]
    validation_paths = [Path(path).resolve() for path in validation_paths]
    if set(train_paths) & set(validation_paths):
        raise ValueError("training and validation shards must be episode-disjoint")
    device = next(agent.parameters()).device
    trainable = [
        parameter for parameter in agent.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable, lr=lr)

    train_datasets = [load_episode_shard(path) for path in train_paths]
    train_weights = [_label_weights(data, background_weight) for data in train_datasets]
    validation_datasets = [load_episode_shard(path) for path in validation_paths]
    validation_weights = [
        _label_weights(data, background_weight) for data in validation_datasets
    ]

    groups: dict[int, list[tuple[np.ndarray, ...]]] = {}
    for data, weights_array in zip(train_datasets, train_weights, strict=True):
        for start in range(0, len(data["actions"]), seq_len):
            end = min(start + seq_len, len(data["actions"]))
            groups.setdefault(end - start, []).append(
                (
                    np.ascontiguousarray(data["spatial"][start:end]),
                    np.ascontiguousarray(data["global"][start:end]),
                    np.ascontiguousarray(data["masks"][start:end]),
                    np.ascontiguousarray(data["actions"][start:end]),
                    weights_array[start:end],
                )
            )
    train_total_steps = 0.0
    for weights_array in train_weights:
        train_total_steps += weights_array.sum().item()
    if train_total_steps == 0:
        raise ValueError("behavior cloning has no selected imitation labels")

    shuffler = random.Random(0)
    updates = 0
    best_state = None
    best_validation = math.inf
    for _ in range(epochs):
        agent.train()
        for group in groups.values():
            shuffler.shuffle(group)
            for batch_start in range(0, len(group), batch_size):
                batch = group[batch_start : batch_start + batch_size]
                spatial = torch.from_numpy(np.stack([chunk[0] for chunk in batch])).to(
                    device
                )
                global_ = torch.from_numpy(np.stack([chunk[1] for chunk in batch])).to(
                    device
                )
                masks = torch.from_numpy(np.stack([chunk[2] for chunk in batch])).to(
                    device
                )
                actions = torch.from_numpy(np.stack([chunk[3] for chunk in batch])).to(
                    device
                )
                step_weights = torch.from_numpy(
                    np.stack([chunk[4] for chunk in batch])
                ).to(device)
                optimizer.zero_grad()
                _, logprob, _, _ = agent.get_action_and_value(
                    spatial, global_, masks, action=actions
                )
                loss = -(logprob * step_weights).sum() / step_weights.sum().clamp_min(
                    1e-8
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 0.5)
                optimizer.step()
                updates += 1
        agent.eval()
        validation = _score_episodes(
            agent, validation_datasets, validation_weights, device
        )
        if validation["nll"] < best_validation:
            best_validation = validation["nll"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in agent.state_dict().items()
            }
    agent.load_state_dict(best_state)
    agent.eval()
    return {
        "train": _score_episodes(agent, train_datasets, train_weights, device),
        "validation": _score_episodes(
            agent, validation_datasets, validation_weights, device
        ),
        "updates": updates,
        "chunks": sum(len(group) for group in groups.values()),
        "background_weight": background_weight,
    }
