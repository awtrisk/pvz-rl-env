"""Minimal PPO trainer for PvZ: Survival Endless.

Uses the PvZActorCritic network with a Mamba memory layer and action masking.
Rollouts are collected from a PvZVecEnv, then updated with vanilla PPO.
"""

import copy
import os
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from curriculum import (
    NATIVE_STAGE_COMPLETION_WAVE,
    WAVES_TO_SURVIVE,
    CurriculumManager,
)
from evaluation import fixed_evaluation_suites, schedule_checkpoint_evaluations
from network import PvZActorCritic
from teacher_policy import choose_actions


def _generalizing_selection(summaries, cfg, incumbent=None):
    """Return the target only when hardened gates do not regress the incumbent.

    Phase 4 gates: every required suite (selection_suite plus
    cfg['required_suites']) must be present with at least min_selection_episodes
    episodes, the selected suite must clear min_selection_completion_rate
    (truncation counts as non-completion by construction of completion_rate),
    and the incumbent may not regress in mean wave, median wave, or completion
    rate. Missing suites are missing evidence and never promote.
    """
    by_name = {summary["suite"]["name"]: summary for summary in summaries}
    required_names = [cfg.get("selection_suite", "zero_eval")]
    required_names += [
        name for name in cfg.get("required_suites", []) if name not in required_names
    ]
    min_episodes = cfg.get("min_selection_episodes", 20)
    required = []
    for name in required_names:
        summary = by_name.get(name)
        if summary is None:
            return None
        # Summaries built outside write_evaluation_results may omit episodes;
        # treat the absence as unknown rather than zero.
        if summary.get("episodes", min_episodes) < min_episodes:
            return None
        required.append(summary)
    selected = required[0]
    if selected.get("completion_rate", 1.0) < cfg.get(
        "min_selection_completion_rate", 0.0
    ):
        return None
    bootstrap = by_name.get("bootstrap_eval")
    transfer = by_name.get("transfer_eval")
    if (
        bootstrap is not None
        and bootstrap["mean_max_wave"] < cfg.get("min_bootstrap_mean_wave", 20.0)
    ) or (
        transfer is not None
        and transfer["mean_max_wave"] < cfg.get("min_transfer_mean_wave", 20.0)
    ):
        return None

    if incumbent:
        mean_tolerance = cfg.get("selection_regression_tolerance", 0.0)
        median_tolerance = cfg.get("median_regression_tolerance", 1.0)
        completion_tolerance = cfg.get("completion_regression_tolerance", 0.05)
        for name, previous in incumbent.items():
            current = by_name.get(name)
            if current is None:
                continue
            if current["mean_max_wave"] < previous["mean_max_wave"] - mean_tolerance:
                return None
            if (
                previous.get("median_max_wave") is not None
                and current.get("median_max_wave") is not None
                and current["median_max_wave"]
                < previous["median_max_wave"] - median_tolerance
            ):
                return None
            previous_completion = previous.get("completion_rate")
            if (
                previous_completion is not None
                and current.get("completion_rate", 1.0)
                < previous_completion - completion_tolerance
            ):
                return None
    return selected


def _evaluation_metadata(summaries):
    """Keep only the small metric record needed to protect an incumbent."""
    return {
        summary["suite"]["name"]: {
            "mean_max_wave": summary["mean_max_wave"],
            "median_max_wave": summary.get("median_max_wave"),
            "loss_rate": summary.get("loss_rate"),
            "completion_rate": summary.get("completion_rate"),
            "episodes": summary.get("episodes"),
        }
        for summary in summaries
    }


class RewardNormalizer:
    """Optional running mean/std reward normalization.

    Disabled by default in this configuration because the reward scale is fixed
    and bounded in the C++ engine.
    """

    def __init__(self, num_envs, clip=10.0, eps=1e-8, enabled=False):
        self.num_envs = num_envs
        self.clip = clip
        self.eps = eps
        self.enabled = enabled
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, reward):
        if not self.enabled:
            return
        reward = np.asarray(reward, dtype=np.float32).reshape(-1)
        # pi-lens-ignore: unchecked-throwing-call-python
        batch_mean = float(reward.mean())
        # pi-lens-ignore: unchecked-throwing-call-python
        batch_var = float(reward.var())
        batch_count = reward.size

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta**2) * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.mean = new_mean
        self.count = total_count

    def normalize(self, reward, update=True):
        if not self.enabled:
            return np.asarray(reward, dtype=np.float32)
        if update:
            self.update(reward)
        std = np.sqrt(self.var) + self.eps
        reward = np.asarray(reward, dtype=np.float32)
        return reward / std

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state):
        # pi-lens-ignore: unchecked-throwing-call-python
        self.mean = float(state.get("mean", self.mean))
        # pi-lens-ignore: unchecked-throwing-call-python
        self.var = float(state.get("var", self.var))
        # pi-lens-ignore: unchecked-throwing-call-python
        self.count = float(state.get("count", self.count))


def _set_rollout_cooldowns(env, cfg):
    """Apply an 80/20 current/preceding-domain worker split when configured."""
    current = cfg.get("cooldown_scale", 1.0)
    previous = cfg.get("previous_cooldown_scale")
    if previous is None:
        env.set_cooldown_scale(current)
        return
    if env.num_envs < 5:
        raise ValueError(
            "mixed-domain rollouts require at least five workers for the 80/20 split"
        )
    previous_count = env.num_envs // 5
    env.set_cooldown_scales(
        [previous] * previous_count + [current] * (env.num_envs - previous_count)
    )


def _set_rollout_suns(env, cfg, current_sun):
    """Apply the configured current/auxiliary starting-sun worker split."""
    auxiliary_sun = cfg.get("auxiliary_start_sun")
    if auxiliary_sun is None:
        env.set_start_sun(current_sun)
        return
    if env.num_envs < 5:
        raise ValueError("mixed-domain rollouts require at least five workers")
    auxiliary_count = env.num_envs // 5
    env.set_start_suns(
        [auxiliary_sun] * auxiliary_count
        + [current_sun] * (env.num_envs - auxiliary_count)
    )


def collect_rollout(
    agent,
    env,
    cfg,
    device,
    curriculum_manager,
    obs=None,
    infos=None,
    reward_normalizer=None,
    use_bc=False,
    bc_epsilon=0.0,
    caches=None,
    anchor_agent=None,
    anchor_caches=None,
):
    """Collect one rollout and return episode outcomes for the curriculum."""
    num_steps = cfg["steps_per_update"]

    fresh_sequence = caches is None
    if obs is None or infos is None:
        obs, infos = env.reset()
        caches = None
        fresh_sequence = True
    num_envs = obs["spatial"].shape[0]
    if reward_normalizer is None:
        reward_normalizer = RewardNormalizer(
            num_envs, enabled=cfg.get("use_reward_norm", False)
        )
    if caches is None:
        caches = agent.init_caches(num_envs, device)
    initial_caches = [
        (h.detach().clone(), inputs.detach().clone()) for h, inputs in caches
    ]
    if anchor_agent is not None and anchor_caches is None:
        anchor_caches = anchor_agent.init_caches(num_envs, device)
    anchor_initial_caches = (
        None
        if anchor_caches is None
        else [
            (h.detach().clone(), inputs.detach().clone()) for h, inputs in anchor_caches
        ]
    )

    rollout = {
        "obs_spatial": np.zeros(
            (num_steps, num_envs, *agent.spatial_shape), dtype=np.float32
        ),
        "obs_global": np.zeros(
            (num_steps, num_envs, agent.global_size), dtype=np.float32
        ),
        "actions": np.zeros((num_steps, num_envs), dtype=np.int64),
        "logprobs": np.zeros((num_steps, num_envs), dtype=np.float32),
        "values": np.zeros((num_steps, num_envs), dtype=np.float32),
        "rewards": np.zeros((num_steps, num_envs), dtype=np.float32),
        "dones": np.zeros((num_steps, num_envs), dtype=np.float32),
        "truncated": np.zeros((num_steps, num_envs), dtype=np.float32),
        "terminal_values": np.zeros((num_steps, num_envs), dtype=np.float32),
        "episode_starts": np.zeros((num_steps, num_envs), dtype=np.float32),
        "masks": np.zeros((num_steps, num_envs, agent.num_actions), dtype=np.float32),
        "deployment_masks": np.zeros(
            (num_steps, num_envs, agent.num_actions), dtype=np.float32
        ),
        "curriculum_gated": np.zeros((num_steps, num_envs), dtype=np.float32),
        "waves": np.zeros((num_steps, num_envs), dtype=np.float32),
        "suns": np.zeros((num_steps, num_envs), dtype=np.float32),
        "zombie_min_x": np.zeros((num_steps, num_envs), dtype=np.float32),
        "optimization_mask": np.zeros((num_steps, num_envs), dtype=np.float32),
        "initial_caches": initial_caches,
        "anchor_initial_caches": anchor_initial_caches,
    }

    frontier_wave = cfg.get("frontier_wave")
    episode_max_waves = np.zeros(num_envs, dtype=np.float32)
    episode_target_waves = np.full(
        num_envs, curriculum_manager.target_wave, dtype=np.float32
    )
    episode_outcomes = []
    episode_phases = np.full(num_envs, curriculum_manager.phase_index, dtype=np.int64)
    next_episode_starts = np.full(
        num_envs, 1.0 if fresh_sequence else 0.0, dtype=np.float32
    )
    auxiliary_count = num_envs // 5 if cfg.get("auxiliary_start_sun") is not None else 0

    for step in range(num_steps):
        spatial = torch.from_numpy(obs["spatial"]).to(device)
        global_vec = torch.from_numpy(obs["global"]).to(device)
        env_mask = _stack_masks(infos)
        deployment_mask = np.logical_and(
            env_mask.astype(bool),
            curriculum_manager.seed_mask(agent.num_actions).astype(bool),
        ).astype(np.float32)
        action_mask = curriculum_manager.action_mask(env_mask, obs["spatial"])
        mask = torch.from_numpy(action_mask).to(device)

        with torch.no_grad():
            if use_bc:
                action = torch.from_numpy(choose_actions(action_mask)).to(device)
                _, logprob, _, value, caches = agent.forward_step(
                    spatial, global_vec, mask, caches, action=action
                )
            else:
                action, logprob, _, value, caches = agent.forward_step(
                    spatial, global_vec, mask, caches
                )
            if anchor_agent is not None:
                _, _, anchor_caches = anchor_agent.step_logits(
                    spatial, global_vec, mask, anchor_caches
                )

        actions = action.cpu().numpy()
        rollout["obs_spatial"][step] = obs["spatial"]
        rollout["obs_global"][step] = obs["global"]
        rollout["episode_starts"][step] = next_episode_starts
        rollout["actions"][step] = actions
        rollout["logprobs"][step] = logprob.cpu().numpy()
        rollout["values"][step] = value.squeeze(-1).cpu().numpy()
        rollout["masks"][step] = action_mask
        rollout["deployment_masks"][step] = deployment_mask
        rollout["curriculum_gated"][step] = np.any(
            action_mask != deployment_mask, axis=1
        )
        rollout["optimization_mask"][step] = (
            1.0
            if frontier_wave is None
            else np.fromiter(
                (info.get("wave", 0) >= frontier_wave for info in infos),
                dtype=np.float32,
                count=num_envs,
            )
        )

        previous_waves = np.fromiter(
            (info.get("wave", 0) for info in infos),
            dtype=np.float32,
            count=num_envs,
        )
        obs, reward, done, truncated, infos = env.step(actions)
        truncated = np.logical_and(truncated, np.logical_not(done)).astype(np.float32)
        if cfg.get("frontier_sparse_reward", False):
            current_waves = np.fromiter(
                (info.get("wave", 0) for info in infos),
                dtype=np.float32,
                count=num_envs,
            )
            reward = np.maximum(current_waves - previous_waves, 0.0)
            reward += np.fromiter(
                (
                    1
                    if info.get("stage_complete", False)
                    else -1
                    if info.get("lost", False)
                    else 0
                    for info in infos
                ),
                dtype=np.float32,
                count=num_envs,
            )
        normalized_reward = reward_normalizer.normalize(reward, update=True)
        rollout["rewards"][step] = normalized_reward
        for i, info in enumerate(infos):
            wave = info.get("wave", 0)
            rollout["waves"][step, i] = wave
            rollout["suns"][step, i] = info.get("sun", 0)
            rollout["zombie_min_x"][step, i] = info.get("zombie_min_x", 0)
            episode_max_waves[i] = max(episode_max_waves[i], wave)
            # Curriculum targets must change the learning horizon, not merely
            # label an otherwise identical full episode after it ends. Targets
            # at or beyond the native stage-completion wave (absolute wave 20)
            # must NOT synthesize done: "wave >= 20" becomes true the moment
            # the final wave spawns, which would end the episode before the
            # final assault instead of letting the env's native stage_complete
            # terminal (or a genuine loss) end it.
            if (
                frontier_wave is None
                and episode_target_waves[i] < NATIVE_STAGE_COMPLETION_WAVE
                and wave >= episode_target_waves[i]
                and not done[i]
                and not truncated[i]
            ):
                done[i] = 1.0

        rollout["dones"][step] = done
        rollout["truncated"][step] = truncated
        truncated_indices = np.where(truncated)[0]
        if truncated_indices.size:
            # Bootstrap from the terminal observation without advancing the
            # live rollout cache. The next action must see that observation
            # exactly once, and unrelated envs must not be contaminated when
            # only one worker truncates.
            terminal_indices = torch.as_tensor(truncated_indices, device=device)
            terminal_caches = [
                (
                    h.index_select(0, terminal_indices),
                    inputs.index_select(0, terminal_indices),
                )
                for h, inputs in caches
            ]
            with torch.no_grad():
                terminal_spatial = (
                    torch.from_numpy(obs["spatial"])
                    .to(device)
                    .index_select(0, terminal_indices)
                )
                terminal_global = (
                    torch.from_numpy(obs["global"])
                    .to(device)
                    .index_select(0, terminal_indices)
                )
                terminal_mask = (
                    torch.from_numpy(_stack_masks(infos))
                    .to(device)
                    .index_select(0, terminal_indices)
                )
                _, _, _, terminal_value, _ = agent.forward_step(
                    terminal_spatial, terminal_global, terminal_mask, terminal_caches
                )
            rollout["terminal_values"][step, truncated_indices] = (
                terminal_value.squeeze(-1).cpu().numpy()
            )
        terminated = np.where(np.logical_or(done, truncated))[0]
        next_episode_starts = np.zeros(num_envs, dtype=np.float32)
        for i in terminated:
            # A native loss terminal must not count as a curriculum win even
            # if max_wave already touched the target (e.g. the final-wave
            # assault was reached but lost).
            lost_terminal = bool(infos[i].get("lost", False))
            win = episode_max_waves[i] >= episode_target_waves[i] and not lost_terminal
            episode_outcomes.append(
                {
                    # pi-lens-ignore: unchecked-throwing-call-python
                    "phase_index": int(episode_phases[i]),
                    "win": bool(win),
                    "promotion_eligible": bool(i >= auxiliary_count),
                }
            )
            episode_max_waves[i] = 0
            episode_target_waves[i] = curriculum_manager.target_wave
            episode_phases[i] = curriculum_manager.phase_index
            next_episode_starts[i] = 1.0

        if terminated.size > 0:
            if hasattr(env, "reset_at"):
                reset_results = env.reset_at(terminated)
                for idx, i in enumerate(terminated):
                    obs["spatial"][i] = reset_results[idx][0]["spatial"]
                    obs["global"][i] = reset_results[idx][0]["global"]
                    infos[i] = reset_results[idx][1]
            for rollout_caches in (caches, anchor_caches):
                if rollout_caches is None:
                    continue
                for h, inputs in rollout_caches:
                    for i in terminated:
                        h[i].zero_()
                        inputs[i].zero_()

    rollout["last_obs"] = obs
    rollout["last_infos"] = infos
    # Keep the state produced for the last acted-on observation. The bootstrap
    # value below uses a copy so the next rollout does not encode last_obs twice.
    rollout["last_caches"] = caches
    rollout["anchor_last_caches"] = anchor_caches

    with torch.no_grad():
        spatial = torch.from_numpy(obs["spatial"]).to(device)
        global_vec = torch.from_numpy(obs["global"]).to(device)
        action_mask = curriculum_manager.action_mask(
            _stack_masks(infos), obs["spatial"]
        )
        mask = torch.from_numpy(action_mask).to(device)
        value_caches = [(h.clone(), inputs.clone()) for h, inputs in caches]
        _, _, _, next_value, _ = agent.forward_step(
            spatial, global_vec, mask, value_caches
        )
        next_value = next_value.squeeze(-1).cpu().numpy()

    return rollout, next_value, episode_outcomes, reward_normalizer


def _stack_masks(infos):
    return np.stack([info["action_mask"] for info in infos])


def compute_gae(rollout, next_value, cfg):
    """Compute GAE advantages and returns."""
    gamma = cfg["gamma"]
    gae_lambda = cfg["gae_lambda"]
    rewards = rollout["rewards"]
    values = rollout["values"]
    dones = rollout["dones"]
    truncated = rollout["truncated"]
    terminal_values = rollout.get("terminal_values", np.zeros_like(values))
    num_steps = rewards.shape[0]

    nonterminal = 1.0 - dones
    boundary = np.clip(dones + truncated, 0.0, 1.0)
    trace_continue = 1.0 - boundary

    advantages = np.zeros_like(rewards)
    last_gae = np.zeros(rewards.shape[1], dtype=np.float32)
    is_trunc = truncated > 0.0
    for t in reversed(range(num_steps)):
        following_value = next_value if t == num_steps - 1 else values[t + 1]
        nextnonterminal = np.where(is_trunc[t], 1.0, nonterminal[t])
        next_v = np.where(is_trunc[t], terminal_values[t], following_value)
        delta = rewards[t] + gamma * next_v * nextnonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * trace_continue[t] * last_gae
        advantages[t] = last_gae

    rollout["advantages"] = advantages
    rollout["returns"] = advantages + values
    return rollout


def ppo_update(
    agent, optimizer, rollout, cfg, update_step, use_bc=False, anchor_agent=None
):
    """One PPO update over the full rollout."""
    device = next(agent.parameters()).device

    obs_spatial = torch.from_numpy(rollout["obs_spatial"].transpose(1, 0, 2, 3, 4)).to(
        device
    )
    obs_global = torch.from_numpy(rollout["obs_global"].transpose(1, 0, 2)).to(device)
    actions = torch.from_numpy(rollout["actions"].transpose(1, 0)).to(device)
    old_logprobs = torch.from_numpy(rollout["logprobs"].transpose(1, 0)).to(device)
    masks = torch.from_numpy(rollout["masks"].transpose(1, 0, 2)).to(device)
    deployment_masks = torch.from_numpy(
        rollout["deployment_masks"].transpose(1, 0, 2)
    ).to(device)
    curriculum_gated = torch.from_numpy(rollout["curriculum_gated"].transpose(1, 0)).to(
        device
    )
    episode_starts = torch.from_numpy(rollout["episode_starts"].transpose(1, 0)).to(
        device
    )
    advantages = torch.from_numpy(rollout["advantages"].transpose(1, 0)).to(device)
    returns = torch.from_numpy(rollout["returns"].transpose(1, 0)).to(device)
    optimization_mask = torch.from_numpy(
        rollout["optimization_mask"].transpose(1, 0)
    ).to(device)
    initial_caches = rollout["initial_caches"]
    anchor_initial_caches = rollout.get("anchor_initial_caches")
    optimization_weight = optimization_mask.sum()
    if optimization_weight.item() == 0:
        raise ValueError("rollout contains no transitions at or beyond frontier_wave")

    selected_advantages = advantages[optimization_mask.bool()]
    advantages = (advantages - selected_advantages.mean()) / (
        selected_advantages.std(unbiased=False) + 1e-8
    )

    schedule_updates = cfg.get("schedule_num_updates", cfg["num_updates"])
    schedule_start = cfg.get("schedule_start_update", 0)
    progress = (update_step - schedule_start) / max(
        1, schedule_updates - schedule_start - 1
    )
    progress = min(1.0, max(0.0, progress))
    lr = cfg["lr"] * max(0.0, 1.0 - progress)
    if progress < cfg.get("ent_flat_fraction", 0.5):
        ent_coef = cfg["ent_coef_start"]
    else:
        ent_flat_fraction = cfg.get("ent_flat_fraction", 0.5)
        if ent_flat_fraction >= 1.0 or ent_flat_fraction <= 0.0:
            ent_coef = cfg["ent_coef_end"]
        else:
            ent_progress = (progress - ent_flat_fraction) / (1.0 - ent_flat_fraction)
            ent_coef = cfg["ent_coef_start"] + ent_progress * (
                cfg["ent_coef_end"] - cfg["ent_coef_start"]
            )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    clip_coef = cfg["clip_coef"]
    vf_coef = cfg["vf_coef"]

    anchor_kl = torch.zeros((), device=device)
    distill_loss = torch.zeros((), device=device)
    # pi-lens-ignore: unchecked-throwing-call-python
    anchor_kl_coef = float(cfg.get("anchor_kl_coef", 0.0))
    # pi-lens-ignore: unchecked-throwing-call-python
    distill_coef = float(cfg.get("curriculum_distill_coef", 0.0))
    policy_loss = torch.zeros((), device=device)
    value_loss = torch.zeros((), device=device)
    entropy_loss = torch.zeros((), device=device)
    for _ in range(cfg["num_epochs"]):
        _, new_logprob, entropy, new_value = agent.get_action_and_value(
            obs_spatial,
            obs_global,
            masks,
            action=actions,
            episode_starts=episode_starts,
            initial_caches=initial_caches,
        )
        if use_bc:
            policy_loss = -(new_logprob * optimization_mask).sum() / optimization_weight
        else:
            ratio = torch.exp(new_logprob - old_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantages
            policy_loss = (
                -(torch.min(surr1, surr2) * optimization_mask).sum()
                / optimization_weight
            )
        value_loss = (
            (new_value.squeeze(-1) - returns) ** 2 * optimization_mask
        ).sum() / optimization_weight
        entropy_loss = (entropy * optimization_mask).sum() / optimization_weight
        current_latent = None
        distill_mask = curriculum_gated * optimization_mask
        if distill_coef > 0.0 and distill_mask.any():
            current_latent = agent.forward_sequence(
                obs_spatial, obs_global, episode_starts, initial_caches
            )
            deployment_logits = agent._apply_action_mask(
                agent.actor(current_latent), deployment_masks
            )
            deployment_logprob = torch.distributions.Categorical(
                logits=deployment_logits
            ).log_prob(actions)
            distill_loss = (
                -(deployment_logprob * distill_mask).sum() / distill_mask.sum()
            )
        else:
            distill_loss = torch.zeros((), device=device)
        if anchor_agent is not None and anchor_kl_coef > 0.0:
            if current_latent is None:
                current_latent = agent.forward_sequence(
                    obs_spatial, obs_global, episode_starts, initial_caches
                )
            current_logits = agent._apply_action_mask(
                agent.actor(current_latent), masks
            )
            if anchor_initial_caches is None:
                raise ValueError("anchor initial caches are required for anchor KL")
            with torch.no_grad():
                anchor_latent = anchor_agent.forward_sequence(
                    obs_spatial,
                    obs_global,
                    episode_starts,
                    anchor_initial_caches,
                )
                anchor_logits = anchor_agent._apply_action_mask(
                    anchor_agent.actor(anchor_latent), masks
                )
            legal = masks.to(torch.bool)
            current_log_probs = F.log_softmax(current_logits, dim=-1).masked_fill(
                ~legal, 0.0
            )
            anchor_log_probs = F.log_softmax(anchor_logits, dim=-1).masked_fill(
                ~legal, 0.0
            )
            anchor_probs = anchor_log_probs.exp().masked_fill(~legal, 0.0)
            anchor_kl = (
                (anchor_probs * (anchor_log_probs - current_log_probs))
                .sum(dim=-1)
                .mean()
            )
        else:
            anchor_kl = torch.zeros((), device=device)
        loss = (
            policy_loss
            + vf_coef * value_loss
            - ent_coef * entropy_loss
            + anchor_kl_coef * anchor_kl
            + distill_coef * distill_loss
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), cfg["max_grad_norm"])
        optimizer.step()

    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy_loss.item(),
        "anchor_kl": anchor_kl.item(),
        "distill_loss": distill_loss.item(),
        "lr": lr,
        "ent_coef": ent_coef,
        "optimization_fraction": optimization_mask.mean().item(),
    }


def train(cfg: dict[str, Any]) -> None:
    """Run PPO training, resuming the whole state when a full checkpoint exists."""
    if cfg.get("sparse_reward") and not cfg.get("frontier_sparse_reward"):
        # Sparse wave-advance reward without frontier termination: reuse the
        # frontier sparse-reward channel (reward computation only; frontier_wave
        # stays None so episodes terminate natively at the stage boundary).
        cfg["frontier_sparse_reward"] = True
    from vec_env import PvZVecEnv

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    training_seed = cfg.get("training_seed")
    if training_seed is not None:
        # pi-lens-ignore: unchecked-throwing-call-python
        np.random.seed(int(training_seed))
        # pi-lens-ignore: unchecked-throwing-call-python
        torch.manual_seed(int(training_seed))
        if torch.cuda.is_available():
            # pi-lens-ignore: unchecked-throwing-call-python
            torch.cuda.manual_seed_all(int(training_seed))
    agent = PvZActorCritic().to(device)
    checkpoint = None
    resume_state = False
    start_update = 0

    if cfg.get("checkpoint_path"):
        checkpoint_path = cfg["checkpoint_path"]
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(
                cfg.get("checkpoint_dir", ""), checkpoint_path
            )
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        agent.load_state_dict(checkpoint["agent_state"])
        resume_state = checkpoint.get("resume_state_version", 0) >= 1
        if resume_state:
            # pi-lens-ignore: unchecked-throwing-call-python
            start_update = int(checkpoint.get("update", -1)) + 1
            cfg["schedule_num_updates"] = (
                cfg["num_updates"]
                if cfg.get("extend_schedule", False)
                # pi-lens-ignore: unchecked-throwing-call-python
                else int(checkpoint.get("schedule_num_updates", cfg["num_updates"]))
            )
            cfg["schedule_start_update"] = (
                start_update
                if cfg.get("restart_schedule", False)
                # pi-lens-ignore: unchecked-throwing-call-python
                else int(checkpoint.get("schedule_start_update", 0))
            )
        if checkpoint.get("selection_protocol_version") == 1:
            if "best_mean_max_wave" in checkpoint:
                # pi-lens-ignore: unchecked-throwing-call-python
                cfg["best_mean_max_wave"] = float(checkpoint["best_mean_max_wave"])
            elif checkpoint.get("selection_summary"):
                # pi-lens-ignore: unchecked-throwing-call-python
                cfg["best_mean_max_wave"] = float(
                    checkpoint["selection_summary"].get("mean_max_wave", -1)
                )
            if "best_selection_completion" in checkpoint:
                # pi-lens-ignore: unchecked-throwing-call-python
                cfg["best_selection_completion"] = float(
                    checkpoint["best_selection_completion"]
                )
        else:
            # Legacy scores used unseeded evaluations and are not comparable.
            cfg.pop("best_mean_max_wave", None)

    anchor_agent = None
    # pi-lens-ignore: unchecked-throwing-call-python
    anchor_kl_coef = float(cfg.get("anchor_kl_coef", 0.0))
    if anchor_kl_coef > 0.0:
        if checkpoint is None:
            raise ValueError("anchor_kl_coef requires a loaded checkpoint")
        anchor_agent = copy.deepcopy(agent).to(device).eval()
        for parameter in anchor_agent.parameters():
            parameter.requires_grad_(False)
    if anchor_kl_coef > 0.0 and anchor_agent is None:
        raise RuntimeError("anchored PPO requires a frozen anchor agent")

    memory_only = bool(cfg.get("memory_only", False))
    adaptation_only = bool(cfg.get("adaptation_only", False))
    bootstrap_only = bool(cfg.get("bootstrap_only", False))
    residual_only = bool(cfg.get("residual_only", False))
    residual_policy_only = bool(cfg.get("residual_policy_only", False))
    factorized_only = bool(cfg.get("factorized_only", False))
    exclusive_modes = sum(
        (
            memory_only,
            adaptation_only,
            bootstrap_only,
            residual_only,
            residual_policy_only,
            factorized_only,
        )
    )
    if exclusive_modes > 1:
        raise ValueError("training modes are mutually exclusive")
    if exclusive_modes:
        for parameter in agent.parameters():
            parameter.requires_grad_(False)
        if memory_only or adaptation_only or bootstrap_only:
            agent.memory_gate.requires_grad_(True)
            for parameter in agent.mamba.parameters():
                parameter.requires_grad_(True)
            for parameter in agent.state_adapter.parameters():
                parameter.requires_grad_(True)
        if adaptation_only or bootstrap_only:
            for parameter in agent.actor.parameters():
                parameter.requires_grad_(True)
        if bootstrap_only:
            for parameter in agent.latent_proj.parameters():
                parameter.requires_grad_(True)
            for parameter in agent.critic.parameters():
                parameter.requires_grad_(True)
        if residual_only or residual_policy_only:
            for parameter in agent.actor.residual.parameters():
                parameter.requires_grad_(True)
            if residual_only:
                for parameter in agent.critic.parameters():
                    parameter.requires_grad_(True)
        if factorized_only:
            for head in (
                agent.actor.mode_head,
                agent.actor.seed_head,
                agent.actor.row_head,
                agent.actor.col_head,
            ):
                for parameter in head.parameters():
                    parameter.requires_grad_(True)
            for parameter in agent.critic.parameters():
                parameter.requires_grad_(True)
    trainable_parameters = [
        parameter for parameter in agent.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable_parameters, lr=cfg["lr"])
    optimizer_mode = (
        "residual_policy_only"
        if residual_policy_only
        else (
            "residual_only"
            if residual_only
            else (
                "factorized_only"
                if factorized_only
                else (
                    "bootstrap_only"
                    if bootstrap_only
                    else (
                        "adaptation_only"
                        if adaptation_only
                        else ("memory_only" if memory_only else "all")
                    )
                )
            )
        )
    )
    if (
        resume_state
        and checkpoint is not None
        and checkpoint.get("optimizer_mode") == optimizer_mode
        and not cfg.get("frontier_sparse_reward", False)
    ):
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        except ValueError as exc:
            # Older checkpoints can claim the same mode while carrying a
            # different parameter subset. Keep the weights and schedule, but
            # start Adam cleanly instead of aborting the continuation.
            print(f"optimizer state ignored: {exc}")

    curriculum_manager = CurriculumManager(
        initial_start_wave=cfg.get("start_wave"),
        promotion_window=cfg.get("promotion_window", 50),
        promotion_rate=cfg.get("promotion_rate", 0.9),
        demotion_window=cfg.get("demotion_window", 20),
        demotion_rate=cfg.get("demotion_rate", 0.6),
        waves_to_survive=cfg.get("waves_to_survive", WAVES_TO_SURVIVE),
        phases=cfg.get("phases"),
    )
    if (
        checkpoint
        and checkpoint.get("curriculum_state")
        and not cfg.get("reset_curriculum_state", False)
    ):
        curriculum_manager.load_state_dict(checkpoint["curriculum_state"])

    if start_update >= cfg["num_updates"]:
        raise ValueError(
            f"checkpoint resumes at update {start_update}, but configured num_updates "
            f"{cfg['num_updates']} leaves no updates; increase num_updates"
        )

    env_args = (
        cfg["num_envs"],
        cfg["resdir"],
        cfg["savedir"],
        curriculum_manager.start_wave,
    )
    env = (
        PvZVecEnv(*env_args, seed=training_seed)
        if training_seed is not None
        else PvZVecEnv(*env_args)
    )
    env.set_start_wave(curriculum_manager.start_wave)
    env.set_start_sun(curriculum_manager.start_sun)
    _set_rollout_cooldowns(env, cfg)
    if cfg.get("chain_stages"):
        # Endless mode: episodes chain past stage end and end on genuine loss.
        env.set_chain_stages(True)
    if cfg.get("deck"):
        # Custom deck (e.g. Coffee Bean in the Garlic slot): slot meanings
        # remap at the next reset, before any rollout is collected.
        env.set_deck(cfg["deck"])
    obs, infos = env.reset()
    caches = agent.init_caches(cfg["num_envs"], device)
    anchor_caches = (
        anchor_agent.init_caches(cfg["num_envs"], device)
        if anchor_agent is not None
        else None
    )
    reward_normalizer = RewardNormalizer(
        cfg["num_envs"], enabled=cfg.get("use_reward_norm", False)
    )
    if (
        checkpoint
        and checkpoint.get("reward_normalizer_state")
        and not cfg.get("frontier_sparse_reward", False)
    ):
        reward_normalizer.load_state_dict(checkpoint["reward_normalizer_state"])

    if (
        training_seed is None
        and resume_state
        and checkpoint is not None
        and checkpoint.get("torch_rng_state") is not None
    ):
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
            )

    last_selection_summary = checkpoint.get("selection_summary") if checkpoint else None
    last_incumbent_evaluations = (
        checkpoint.get("incumbent_evaluations") if checkpoint else None
    )

    def checkpoint_payload(update, selection_summary=None, incumbent_evaluations=None):
        payload = {
            "resume_state_version": 1,
            "selection_protocol_version": 1,
            "architecture_version": 2,
            "observation_schema_version": 2,
            "spatial_shape": tuple(agent.spatial_shape),
            "training_config": {
                key: value
                for key, value in cfg.items()
                if isinstance(value, (str, int, float, bool, type(None)))
            },
            "agent_state": agent.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "optimizer_mode": optimizer_mode,
            "curriculum_state": curriculum_manager.state_dict(),
            "reward_normalizer_state": reward_normalizer.state_dict(),
            "update": update,
            "schedule_num_updates": cfg.get("schedule_num_updates", cfg["num_updates"]),
            "schedule_start_update": cfg.get("schedule_start_update", 0),
            "best_mean_max_wave": cfg.get("best_mean_max_wave", -1),
            "best_selection_completion": cfg.get("best_selection_completion", -1.0),
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        selected = (
            selection_summary
            if selection_summary is not None
            else last_selection_summary
        )
        if selected is not None:
            payload["selection_summary"] = selected
        incumbent_state = (
            incumbent_evaluations
            if incumbent_evaluations is not None
            else last_incumbent_evaluations
        )
        if incumbent_state is not None:
            payload["incumbent_evaluations"] = incumbent_state
        return payload

    use_wandb = bool(cfg.get("wandb_project"))
    wandb: Any = None
    if use_wandb:
        try:
            import wandb  # type: ignore[import-not-found]
        except ImportError:
            print("wandb is unavailable; continuing without logging")
            use_wandb = False

    if use_wandb:
        if os.environ.get("WANDB_API_KEY"):
            wandb.login(key=os.environ["WANDB_API_KEY"])
        wandb_config = {key: value for key, value in cfg.items() if not callable(value)}
        wandb.init(
            project=cfg["wandb_project"],
            name=cfg.get("wandb_run_name"),
            config=wandb_config,
        )

    print(
        f"device={device} envs={cfg['num_envs']} steps={cfg['steps_per_update']} updates={cfg['num_updates']} start_update={start_update}"
    )
    trainable_count = sum(
        parameter.numel() for parameter in agent.parameters() if parameter.requires_grad
    )
    print(
        f"params={agent.count_parameters():,} trainable={trainable_count:,} mode={optimizer_mode}"
    )
    print(
        f"curriculum phase={curriculum_manager.phase_index}: start_wave={curriculum_manager.start_wave} "
        f"target_wave={curriculum_manager.target_wave} start_sun={curriculum_manager.start_sun} "
        f"allowed_seeds={curriculum_manager.current().get('allowed_seeds', 'all')} "
        f"frontier_wave={cfg.get('frontier_wave')}"
    )

    try:
        for update in range(start_update, cfg["num_updates"]):
            plant_scale = cfg.get("plant_count_scale", 0.3)
            sun_penalty_scale = cfg.get("sun_penalty_scale", 0.0)
            env.set_plant_bonus_scale(plant_scale)
            env.set_sun_penalty_scale(sun_penalty_scale)
            _set_rollout_cooldowns(env, cfg)
            _set_rollout_suns(env, cfg, curriculum_manager.start_sun)
            env.set_start_wave(curriculum_manager.start_wave)

            use_bc = update < cfg.get("bc_updates", 0)
            bc_epsilon = cfg.get("bc_epsilon", 0.5)

            start = time.time()
            rollout, next_value, outcomes, reward_normalizer = collect_rollout(
                agent,
                env,
                cfg,
                device,
                curriculum_manager,
                obs=obs,
                infos=infos,
                reward_normalizer=reward_normalizer,
                use_bc=use_bc,
                bc_epsilon=bc_epsilon,
                caches=caches,
                anchor_agent=anchor_agent,
                anchor_caches=anchor_caches,
            )
            obs, infos = rollout["last_obs"], rollout["last_infos"]
            caches = rollout["last_caches"]
            anchor_caches = rollout["anchor_last_caches"]
            phase_changed = False
            for outcome in outcomes:
                if (
                    outcome["promotion_eligible"]
                    and outcome["phase_index"] == curriculum_manager.phase_index
                    and curriculum_manager.record(outcome["win"])
                ):
                    phase_changed = True
                    break  # Never leak old-phase outcomes into the new phase.
            if phase_changed:
                env.set_start_wave(curriculum_manager.start_wave)
                _set_rollout_suns(env, cfg, curriculum_manager.start_sun)
                obs, infos = env.reset()
                caches = agent.init_caches(cfg["num_envs"], device)
                anchor_caches = (
                    anchor_agent.init_caches(cfg["num_envs"], device)
                    if anchor_agent is not None
                    else None
                )
            rollout = compute_gae(rollout, next_value, cfg)
            metrics = ppo_update(
                agent,
                optimizer,
                rollout,
                cfg,
                update,
                use_bc=use_bc,
                anchor_agent=anchor_agent,
            )
            elapsed = time.time() - start

            mean_reward = rollout["rewards"].mean()
            total_reward = rollout["rewards"].sum()
            explained_var = 1.0 - np.var(rollout["returns"] - rollout["values"]) / (
                np.var(rollout["returns"]) + 1e-8
            )

            print(
                f"update {update:4d} | reward {mean_reward:.4f} sum {total_reward:.2f} | "
                f"pl {metrics['policy_loss']:.4f} vl {metrics['value_loss']:.4f} | "
                f"ent {metrics['entropy']:.4f} anchor_kl {metrics['anchor_kl']:.5f} "
                f"distill {metrics['distill_loss']:.4f} | "
                f"frontier {metrics['optimization_fraction']:.3f} | "
                f"lr {metrics['lr']:.2e} ent_coef {metrics['ent_coef']:.4f} | "
                f"exp_var {explained_var:.3f} | sps {cfg['num_envs'] * cfg['steps_per_update'] / elapsed:.1f}"
            )

            actions_flat = rollout["actions"].reshape(-1)
            total_actions = actions_flat.size
            wait_ratio = (actions_flat == 0).sum() / total_actions
            plant_ratio = (actions_flat >= 46).sum() / total_actions
            shovel_ratio = (
                (actions_flat > 0) & (actions_flat < 46)
            ).sum() / total_actions
            invalid_ratio = (
                (1 - rollout["masks"].reshape(-1, agent.num_actions).max(axis=1))
                * (actions_flat > 0)
            ).sum() / total_actions
            sun_spent_per_step = (actions_flat >= 46).sum() / (
                rollout["masks"].shape[0] * rollout["masks"].shape[1]
            )
            num_plant_actions = rollout["masks"][:, :, 46:].sum() / (
                rollout["masks"].shape[0] * rollout["masks"].shape[1]
            )

            if use_wandb:
                log_data = {
                    "update": update,
                    "train/lr": metrics["lr"],
                    "train/ent_coef": metrics["ent_coef"],
                    "train/Action_Entropy": metrics["entropy"],
                    "train/anchor_kl": metrics["anchor_kl"],
                    "loss/policy": metrics["policy_loss"],
                    "loss/value": metrics["value_loss"],
                    "train/explained_variance": explained_var,
                    "rollout/mean_reward": mean_reward,
                    "rollout/total_reward": total_reward,
                    "rollout/action_wait_ratio": wait_ratio,
                    "rollout/action_plant_ratio": plant_ratio,
                    "rollout/action_shovel_ratio": shovel_ratio,
                    "rollout/action_invalid_ratio": invalid_ratio,
                    "rollout/sun_spent_per_step": sun_spent_per_step,
                    "rollout/num_plant_actions": num_plant_actions,
                    "rollout/Sun_Float_Average": rollout["suns"].mean(),
                    "rollout/Max_Wave_Reached": rollout["waves"].max(),
                    "rollout/Average_Zombie_Max_X": rollout["zombie_min_x"].max(),
                    "train/plant_count_scale": plant_scale,
                    "train/sun_penalty_scale": sun_penalty_scale,
                    "train/bc_active": float(use_bc),
                    "train/optimization_fraction": metrics["optimization_fraction"],
                    "curriculum/phase_index": curriculum_manager.phase_index,
                    "curriculum/start_wave": curriculum_manager.start_wave,
                    "curriculum/start_sun": curriculum_manager.start_sun,
                    "curriculum/target_wave": curriculum_manager.target_wave,
                    "curriculum/win_rate": curriculum_manager.win_rate(),
                    "curriculum/demotion_rate": curriculum_manager.demotion_rate(),
                    "curriculum/episodes_recorded": curriculum_manager.episodes_recorded(),
                    "curriculum/allowed_seeds": str(
                        curriculum_manager.current().get("allowed_seeds", "all")
                    ),
                    "train/reward_norm_mean": reward_normalizer.mean
                    if reward_normalizer.enabled
                    else 0.0,
                    "train/reward_norm_std": np.sqrt(reward_normalizer.var)
                    if reward_normalizer.enabled
                    else 1.0,
                }
                wandb.log(log_data, step=update)

            if (update + 1) % cfg.get("checkpoint_interval", 200) == 0 or update == cfg[
                "num_updates"
            ] - 1:
                checkpoint_prefix = cfg.get("checkpoint_prefix", "checkpoint")
                path = os.path.join(
                    cfg.get("checkpoint_dir", cfg.get("savedir", ".")),
                    f"{checkpoint_prefix}_{update:06d}.pt",
                )
                checkpoint_dir = os.path.dirname(path)
                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(checkpoint_payload(update), path)
                if use_wandb:
                    wandb.save(path)
                evaluator = cfg.get("checkpoint_evaluator")
                if evaluator is not None:
                    suites = cfg.get(
                        "evaluation_suites",
                        fixed_evaluation_suites(
                            episodes=cfg.get("evaluation_episodes", 20),
                            max_steps=cfg.get("evaluation_max_steps", 2000),
                        ),
                    )
                    summaries = schedule_checkpoint_evaluations(path, suites, evaluator)
                    print(
                        f"checkpoint {os.path.basename(path)} evaluations: {summaries}",
                        flush=True,
                    )
                    selected = _generalizing_selection(
                        summaries, cfg, last_incumbent_evaluations
                    )
                    if selected is not None and (
                        selected.get("completion_rate", 0.0),
                        selected["mean_max_wave"],
                    ) > (
                        cfg.get("best_selection_completion", -1.0),
                        cfg.get("best_mean_max_wave", -1.0),
                    ):
                        cfg["best_selection_completion"] = selected.get(
                            "completion_rate", 0.0
                        )
                        cfg["best_mean_max_wave"] = selected["mean_max_wave"]
                        last_selection_summary = selected
                        last_incumbent_evaluations = _evaluation_metadata(summaries)
                        best_path = os.path.join(
                            os.path.dirname(path), f"{checkpoint_prefix}_best.pt"
                        )
                        torch.save(
                            checkpoint_payload(
                                update, selected, last_incumbent_evaluations
                            ),
                            best_path,
                        )
                        print(
                            f"new best {cfg.get('selection_suite', 'zero_eval')}: mean_wave={selected['mean_max_wave']:.3f} -> {best_path}",
                            flush=True,
                        )
    finally:
        env.close()
        if use_wandb:
            wandb.finish()
