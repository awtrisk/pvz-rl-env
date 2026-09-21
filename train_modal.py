"""Modal training app for PvZ: Survival Endless PPO.

Usage:
    modal run train_modal.py::train --num-envs 8 --num-updates 1000 --start-wave 1 --wandb-project pvz-ppo
    modal run train_modal.py::smoke --start-wave 1

Before running, upload your legally purchased game assets:
    modal volume put pvz-assets main.pak /main.pak
    modal volume put pvz-assets properties /properties --recursive
"""

import os
import sys
from pathlib import Path

import modal
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent

app = modal.App("pvz-ppo")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "build-essential",
        "cmake",
        "ninja-build",
        "libogg-dev",
        "libjpeg-dev",
        "libopenmpt-dev",
        "libpng-dev",
        "libvorbis-dev",
        "libmpg123-dev",
        "libsdl2-dev",
        "pybind11-dev",
        "python3-dev",
    )
    .pip_install(
        "torch==2.13.0",
        "mambapy==1.2.0",
        "numpy==2.5.1",
        "pybind11==3.0.4",
        "gymnasium==1.3.0",
        "wandb",
        "imageio",
        "pillow",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .add_local_dir(
        REPO_ROOT,
        "/opt/pvz",
        copy=True,
        ignore=[
            "venv",
            ".git",
            ".ruff_cache",
            ".artifacts",
            ".pi-subagents",
            "__pycache__",
            "*.pyc",
            "savedata",
            "cache64",
            "cache32",
            "*.pyd",
            "*.so",
            "build",
            "build-new",
            "pvz-portable/build",
            "pvz-portable/build-new",
        ],
    )
    .run_commands(
        "echo FORCE_REBUILD_HASH_v46 && rm -rf /opt/pvz/pvz-portable/build && "
        "cd /opt/pvz/pvz-portable && "
        'PYBIND11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())") && '
        "cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dpybind11_DIR=$PYBIND11_DIR && "
        "cmake --build build -j8"
    )
)

assets = modal.Volume.from_name("pvz-assets", create_if_missing=True)
checkpoints = modal.Volume.from_name("pvz-checkpoints", create_if_missing=True)


def validate_checkpoint_name(value: str) -> str:
    if (
        not value
        or value in (".", "..")
        or "/" in value
        or "\\" in value
        or not value.endswith(".pt")
    ):
        raise ValueError("checkpoint name must be a safe .pt basename")
    return value


def validate_replay_volume_path(value: str) -> str:
    from pathlib import PurePosixPath

    if not value or "\\" in value or value.startswith("/") or value.endswith("/"):
        raise ValueError("replay path must be a normalized relative JSON path")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("replay path contains an unsafe segment")
    path = PurePosixPath(value)
    if path.parts[0] != "replay" or path.suffix != ".json":
        raise ValueError("replay path must be a JSON file below replay/")
    return path.as_posix()


def resolve_replay_volume_path(root: str, value: str):
    from pathlib import Path, PurePosixPath

    safe = validate_replay_volume_path(value)
    root_path = Path(root).resolve()
    candidate = root_path.joinpath(*PurePosixPath(safe).parts)
    parent = candidate.parent.resolve()
    if parent != root_path and root_path not in parent.parents:
        raise ValueError("replay path escapes its allowed filesystem root")
    return parent / candidate.name


def reserve_collector_manifest(reserve, reread, persist_abort, reservation):
    """Reserve once; abort an ambiguous write only after proving ownership."""
    try:
        reserve(reservation)
        return
    # pi-lens-ignore: no-boolean-in-except
    except FileExistsError:
        raise FileExistsError("collector_prefix_collision") from None
    # pi-lens-ignore: no-boolean-in-except
    except OSError:
        try:
            current = reread()
        except Exception:  # noqa: BLE001
            current = None
        if not isinstance(current, dict) or current.get(
            "reservation_token"
        ) != reservation.get("reservation_token"):
            raise RuntimeError("reservation_io_unowned") from None
        failure = {
            **reservation,
            "status": "aborted",
            "abort_reason": "reservation_io",
        }
        try:
            persist_abort(failure)
        except Exception:  # noqa: BLE001
            raise RuntimeError("reservation_abort_persistence_failed") from None
        raise RuntimeError("reservation_io_owned_aborted") from None


def collector_prefix_entries(root):
    """List every entry without following or skipping symlink directories."""
    import os
    from pathlib import Path

    root = Path(root)
    if not os.path.lexists(root):
        return []
    entries = [str(root)]
    if root.is_symlink() or not root.is_dir():
        return entries
    for entry in os.scandir(root):
        entries.extend(collector_prefix_entries(entry.path))
    return entries


def resolve_collector_destination(root, relative):
    """Confine a new publication below the owned collector prefix."""
    from pathlib import Path, PurePosixPath

    safe = validate_replay_volume_path(relative)
    parts = PurePosixPath(safe).parts
    if parts[:2] != ("replay", "collector_v7"):
        raise ValueError("collector destination is outside its frozen prefix")
    root = Path(root).resolve()
    candidate = root.joinpath(*parts)
    current = root
    for part in parts[:-1]:
        current /= part
        if current.is_symlink():
            raise ValueError("collector destination has a symlink parent")
    parent = candidate.parent.resolve()
    if parent != root and root not in parent.parents:
        raise ValueError("collector destination escapes its filesystem root")
    return candidate


def validate_replay_worker_receipt(value, *, private: bool = False):
    required = {
        "trace_version",
        "checkpoint_sha256",
        "build_identity",
        "run_identity",
        "reset_options",
        "root",
        "pre_root_digest",
        "outcome",
        "trace_sha256",
    }
    if private:
        required = required | {"trace_bytes"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("replay worker returned incomplete receipt")
    if value["trace_version"] != 1 or not value["trace_sha256"]:
        raise ValueError("replay worker returned invalid receipt identity")
    if not isinstance(value["build_identity"], dict) or not isinstance(
        value["run_identity"], dict
    ):
        raise ValueError("replay worker returned invalid identity maps")
    return value


secrets = (
    [modal.Secret.from_name("wandb")] if hasattr(modal.Secret, "from_name") else []
)


def build_training_config(
    *,
    num_envs: int,
    num_updates: int,
    start_wave: int,
    steps_per_update: int,
    promotion_window: int,
    promotion_rate: float,
    demotion_window: int,
    demotion_rate: float,
    waves_to_survive: int,
    checkpoint_path: str,
    wandb_project: str,
    wandb_run_name: str,
    checkpoint_interval: int,
    cooldown_scale: float,
    previous_cooldown_scale: float | None = None,
    checkpoint_prefix: str = "checkpoint",
    phase: str = "default",
    auxiliary_start_sun: int | None = None,
    evaluation_episodes: int = 20,
    lr: float = 1e-4,
    anchor_kl_coef: float = 0.0,
    extend_schedule: bool = False,
    restart_schedule: bool = False,
    memory_only: bool = False,
    adaptation_only: bool = False,
    bootstrap_only: bool = False,
    residual_only: bool = False,
    residual_policy_only: bool = False,
    factorized_only: bool = False,
    training_seed: int | None = None,
    curriculum_distill_coef: float | None = None,
    reset_curriculum_state: bool = False,
    gae_lambda: float | None = None,
    plant_count_scale: float = 0.3,
    sun_penalty_scale: float = 0.0,
    frontier_wave: int | None = None,
    frontier_sparse_reward: bool = False,
    sparse_reward: bool = False,
    selection_suite: str = "zero_eval",
    evaluation_suite_names: str = "",
    evaluation_seed_start: int | None = None,
    required_suites: str = "",
    min_selection_completion_rate: float = 0.0,
    chain_stages: bool = False,
    deck: str = "",
):
    """Build the serializable trainer configuration used by Modal."""
    from curriculum import (
        SEED_FUMESHROOM,
        SEED_GLOOMSHROOM,
        SEED_MELONPULT,
        SEED_PUMPKINSHELL,
        SEED_SQUASH,
        SEED_SUNFLOWER,
        SEED_TWINSUNFLOWER,
        SEED_WINTERMELON,
        ZERO_DEFENSE_PHASES,
        ZERO_MEMORY_PHASES,
        ZERO_OFFENSE_RAMP_PHASES,
        ZERO_SUN_RAMP_PHASES,
    )
    from evaluation import fixed_evaluation_suites, resolve_evaluation_suites

    phases = None
    if phase == "zero_defense":
        phases = ZERO_DEFENSE_PHASES
    elif phase == "zero_ramp":
        phases = ZERO_SUN_RAMP_PHASES
    elif phase == "zero_offense_ramp":
        phases = ZERO_OFFENSE_RAMP_PHASES
    elif phase == "zero_memory":
        phases = ZERO_MEMORY_PHASES
    elif phase.startswith("bcwm") and phase[4:].isdecimal():
        # Endless-deck experiment (bcwm750, ...): the BC five-seed mask plus
        # Fume-shroom, Gloom-shroom and Winter Melon — the classic endless
        # staples (lane AoE + slow field) the descent lineage never trained
        # because the teacher mask carried through every phase. Garlic and
        # Jalapeno stay masked.
        start_sun = int(phase[4:])  # pi-lens-ignore: unchecked-throwing-call-python
        phases = [
            {
                "start_wave": 1,
                "start_sun": start_sun,
                "allowed_seeds": [
                    SEED_SUNFLOWER,
                    SEED_TWINSUNFLOWER,
                    SEED_MELONPULT,
                    SEED_WINTERMELON,
                    SEED_GLOOMSHROOM,
                    SEED_FUMESHROOM,
                    SEED_PUMPKINSHELL,
                    SEED_SQUASH,
                ],
                "waves_to_survive": max(waves_to_survive, 19),
            }
        ]
    elif phase.startswith("bcfull") and phase[6:].isdecimal():
        # Coffee-deck exploration rung (bcfull750, ...): every deck slot
        # unmasked so entropy explores all ten and the sparse reward narrows
        # (explore-then-narrow); slot meanings come from the deck parameter
        # (Coffee Bean replacing Garlic in the rung's deck).
        start_sun = int(phase[6:])  # pi-lens-ignore: unchecked-throwing-call-python
        phases = [
            {
                "start_wave": 1,
                "start_sun": start_sun,
                "allowed_seeds": list(range(10)),
                "waves_to_survive": max(waves_to_survive, 19),
            }
        ]
    elif phase.startswith("bcnarrow") and phase[8:].isdecimal():
        # Masked consolidation rung (bcnarrow750, ...): the Phase 8 explore
        # rung's self-discovered usage mask — Twin Sunflower and Pumpkin
        # were driven to exactly zero use — removed from the action space.
        start_sun = int(phase[8:])  # pi-lens-ignore: unchecked-throwing-call-python
        phases = [
            {
                "start_wave": 1,
                "start_sun": start_sun,
                "allowed_seeds": [0, 2, 3, 4, 5, 7, 8, 9],
                "waves_to_survive": max(waves_to_survive, 19),
            }
        ]
    elif phase.startswith("bc") and phase[2:].isdecimal():
        # Phase 5 sun descent (bc1500, bc1000, bc500, ...): sun from the name.
        start_sun = int(phase[2:])  # pi-lens-ignore: unchecked-throwing-call-python
        phases = [
            {
                "start_wave": 1,
                "start_sun": start_sun,
                # Match the BC training mask (curriculum phase-0 seeds) so the
                # anchored policy trains on its imitation distribution.
                "allowed_seeds": [
                    SEED_SUNFLOWER,
                    SEED_TWINSUNFLOWER,
                    SEED_MELONPULT,
                    SEED_PUMPKINSHELL,
                    SEED_SQUASH,
                ],
                # Target 20 keeps episode termination native (stage_complete
                # or a genuine loss) instead of a synthetic stage boundary.
                "waves_to_survive": max(waves_to_survive, 19),
            }
        ]
    elif phase in {"zero", "hard", "medium", "easy"}:
        start_suns = {"zero": 0, "hard": 50, "medium": 250, "easy": 1000}
        phases = [
            {
                "start_wave": 1,
                "start_sun": start_suns[phase],
                "allowed_seeds": list(range(10)),
                # A direct zero-sun run must expose the full stage. The old
                # default of five waves silently truncated training at wave 6.
                "waves_to_survive": max(waves_to_survive, 19)
                if phase == "zero"
                else waves_to_survive,
            }
        ]
    elif phase != "default":
        raise ValueError(
            "phase must be default, zero_memory, zero_ramp, zero_offense_ramp, zero_defense, zero, hard, medium, easy, or bc<sun>/bcwm<sun>/bcfull<sun>/bcnarrow<sun> (e.g. bc1500)"
        )

    if gae_lambda is None:
        gae_lambda = (
            0.99
            if phase
            in {
                "zero",
                "zero_memory",
                "zero_ramp",
                "zero_offense_ramp",
                "zero_defense",
                "hard",
                "bc1500",
                "bc1000",
            }
            else 0.95
        )
    if curriculum_distill_coef is None:
        # Keep the deployment-mask distillation opt-in. A gated rollout can
        # otherwise impose a large cross-domain actor gradient before the
        # policy has earned a stable phase transition.
        curriculum_distill_coef = 0.0
    if not 0.0 < gae_lambda <= 1.0:
        raise ValueError(f"gae_lambda must be in (0, 1], got {gae_lambda}")
    if frontier_wave is not None and frontier_wave < 1:
        raise ValueError(f"frontier_wave must be positive, got {frontier_wave}")
    if frontier_sparse_reward and frontier_wave is None:
        raise ValueError("frontier_sparse_reward requires frontier_wave")
    return {
        "num_envs": num_envs,
        "steps_per_update": steps_per_update,
        "num_updates": num_updates,
        "start_wave": start_wave,
        "phases": phases,
        "promotion_window": promotion_window,
        "promotion_rate": promotion_rate,
        "demotion_window": demotion_window,
        "demotion_rate": demotion_rate,
        "waves_to_survive": waves_to_survive,
        "lr": lr,
        "anchor_kl_coef": anchor_kl_coef,
        "extend_schedule": extend_schedule,
        "restart_schedule": restart_schedule,
        "memory_only": memory_only,
        "adaptation_only": adaptation_only,
        "bootstrap_only": bootstrap_only,
        "residual_only": residual_only,
        "residual_policy_only": residual_policy_only,
        "factorized_only": factorized_only,
        "training_seed": training_seed,
        "curriculum_distill_coef": curriculum_distill_coef,
        "reset_curriculum_state": reset_curriculum_state,
        "gamma": 0.999,
        "gae_lambda": gae_lambda,
        "clip_coef": 0.1,
        "vf_coef": 0.05,
        "ent_coef_start": 0.003,
        "ent_coef_end": 0.0003,
        "plant_count_scale": plant_count_scale,
        "sun_penalty_scale": sun_penalty_scale,
        "frontier_wave": frontier_wave,
        "frontier_sparse_reward": frontier_sparse_reward,
        "sparse_reward": sparse_reward,
        "cooldown_scale": cooldown_scale,
        "previous_cooldown_scale": previous_cooldown_scale,
        "previous_domain_fraction": 0.2,
        "auxiliary_start_sun": auxiliary_start_sun,
        "chain_stages": chain_stages,
        "deck": [int(s) for s in deck.split(",") if s.strip()],  # pi-lens-ignore: unchecked-throwing-call-python
        "selection_suite": selection_suite,
        "required_suites": [name for name in required_suites.split(",") if name],
        "min_selection_completion_rate": min_selection_completion_rate,
        "min_bootstrap_mean_wave": 20.0,
        "min_transfer_mean_wave": 20.0,
        "bc_updates": 0,
        "bc_epsilon": 0.5,
        "max_grad_norm": 0.5,
        "num_epochs": 2,
        "device": "cuda",
        "use_reward_norm": not (frontier_sparse_reward or sparse_reward),
        "checkpoint_interval": checkpoint_interval,
        "resdir": "/pvz-assets",
        "savedir": "/pvz-assets/savedata",
        "checkpoint_path": checkpoint_path,
        "checkpoint_dir": "/pvz-checkpoints",
        "checkpoint_prefix": checkpoint_prefix,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name or "pvz-ppo",
        "evaluation_suites": (
            resolve_evaluation_suites(
                [name for name in evaluation_suite_names.split(",") if name],
                episodes=evaluation_episodes,
                seed_start=evaluation_seed_start,
            )
            if evaluation_suite_names
            else fixed_evaluation_suites(episodes=evaluation_episodes)
        ),
        "evaluation_episodes": evaluation_episodes,
        "evaluation_max_steps": 2000,
    }


@app.function(
    image=image,
    gpu="T4",
    # PvZVecEnv runs one engine process per num_envs plus this training
    # process; without this they all share Modal's 1-core default and env
    # stepping serializes (~220 sps). T4 hosts expose 8 vCPUs.
    cpu=8,
    volumes={"/pvz-assets": assets, "/pvz-checkpoints": checkpoints},
    secrets=secrets,
    timeout=86400,
)
def train(
    num_envs: int = 8,
    chain_stages: bool = False,
    deck: str = "",
    num_updates: int = 1000,
    start_wave: int = 1,
    steps_per_update: int = 2048,
    promotion_window: int = 50,
    promotion_rate: float = 0.9,
    demotion_window: int = 20,
    demotion_rate: float = 0.6,
    waves_to_survive: int = 5,
    checkpoint_path: str = "",
    checkpoint_interval: int = 200,
    wandb_project: str = "",
    wandb_run_name: str = "",
    cooldown_scale: float = 0.13,
    previous_cooldown_scale: float | None = None,
    checkpoint_prefix: str = "checkpoint",
    phase: str = "default",
    auxiliary_start_sun: int | None = None,
    evaluation_episodes: int = 20,
    lr: float = 1e-4,
    anchor_kl_coef: float = 0.0,
    extend_schedule: bool = False,
    restart_schedule: bool = False,
    memory_only: bool = False,
    adaptation_only: bool = False,
    bootstrap_only: bool = False,
    residual_only: bool = False,
    residual_policy_only: bool = False,
    factorized_only: bool = False,
    training_seed: int | None = None,
    curriculum_distill_coef: float | None = None,
    reset_curriculum_state: bool = False,
    gae_lambda: float | None = None,
    plant_count_scale: float = 0.3,
    sun_penalty_scale: float = 0.0,
    frontier_wave: int | None = None,
    frontier_sparse_reward: bool = False,
    sparse_reward: bool = False,
    selection_suite: str = "zero_eval",
    evaluation_suite_names: str = "",
    evaluation_seed_start: int | None = None,
    required_suites: str = "",
    min_selection_completion_rate: float = 0.0,
):
    """Train the PPO agent on PvZ: Survival Endless."""
    sys_path = "/opt/pvz"
    if sys_path not in __import__("sys").path:
        __import__("sys").path.insert(0, sys_path)
    __import__("sys").path.insert(0, "/opt/pvz/pvz-portable/build")

    import ppo  # noqa: E402
    from curriculum import PHASES  # noqa: E402

    print("=" * 60, flush=True)
    print(
        f"[TRAIN STARTED] num_envs={num_envs} steps_per_update={steps_per_update} num_updates={num_updates}",
        flush=True,
    )
    print(
        f"               start_wave={start_wave} waves_to_survive={waves_to_survive} phase={phase}",
        flush=True,
    )
    print("=" * 60, flush=True)

    if not os.path.exists("/pvz-assets/main.pak"):
        raise FileNotFoundError(
            "Game assets not mounted. Upload main.pak and properties/ to the 'pvz-assets' Modal Volume."
        )

    if start_wave not in {p["start_wave"] for p in PHASES}:
        raise ValueError(
            f"start_wave must be one of {[p['start_wave'] for p in PHASES]}, got {start_wave}"
        )

    cfg = build_training_config(
        num_envs=num_envs,
        num_updates=num_updates,
        start_wave=start_wave,
        steps_per_update=steps_per_update,
        promotion_window=promotion_window,
        promotion_rate=promotion_rate,
        demotion_window=demotion_window,
        demotion_rate=demotion_rate,
        waves_to_survive=waves_to_survive,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=checkpoint_interval,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        cooldown_scale=cooldown_scale,
        previous_cooldown_scale=previous_cooldown_scale,
        checkpoint_prefix=checkpoint_prefix,
        phase=phase,
        auxiliary_start_sun=auxiliary_start_sun,
        chain_stages=chain_stages,
        deck=deck,
        evaluation_episodes=evaluation_episodes,
        lr=lr,
        anchor_kl_coef=anchor_kl_coef,
        extend_schedule=extend_schedule,
        restart_schedule=restart_schedule,
        memory_only=memory_only,
        adaptation_only=adaptation_only,
        bootstrap_only=bootstrap_only,
        residual_only=residual_only,
        residual_policy_only=residual_policy_only,
        factorized_only=factorized_only,
        training_seed=training_seed,
        curriculum_distill_coef=curriculum_distill_coef,
        reset_curriculum_state=reset_curriculum_state,
        gae_lambda=gae_lambda,
        plant_count_scale=plant_count_scale,
        sun_penalty_scale=sun_penalty_scale,
        frontier_wave=frontier_wave,
        frontier_sparse_reward=frontier_sparse_reward,
        sparse_reward=sparse_reward,
        selection_suite=selection_suite,
        evaluation_suite_names=evaluation_suite_names,
        evaluation_seed_start=evaluation_seed_start,
        required_suites=required_suites,
        min_selection_completion_rate=min_selection_completion_rate,
    )
    print(
        f"               effective_gae_lambda={cfg['gae_lambda']} "
        f"curriculum_distill_coef={cfg['curriculum_distill_coef']} "
        f"plant_count_scale={cfg['plant_count_scale']} "
        f"sun_penalty_scale={cfg['sun_penalty_scale']} "
        f"frontier_wave={cfg['frontier_wave']} "
        f"frontier_sparse_reward={cfg['frontier_sparse_reward']} "
        f"sparse_reward={cfg['sparse_reward']} "
        f"selection_suite={cfg['selection_suite']}",
        flush=True,
    )

    def checkpoint_evaluator(checkpoint_path, suite):
        """Run every seeded episode in an isolated native process."""
        from evaluation import write_evaluation_results

        checkpoints.commit()
        calls = []
        for episode in range(suite["episodes"]):
            episode_suite = {
                **suite,
                "name": f"{suite['name']}_seed_{suite['seed_start'] + episode}",
                "episodes": 1,
                "seed_start": suite["seed_start"] + episode,
            }
            calls.append(
                evaluate_checkpoint.spawn(
                    checkpoint_name=Path(checkpoint_path).name,
                    suite=episode_suite,
                )
            )
        rows = [call.get()["results"][0] for call in calls]
        summary = write_evaluation_results(checkpoint_path, suite, rows)
        checkpoints.commit()
        summary["results"] = rows
        return summary

    cfg["checkpoint_evaluator"] = checkpoint_evaluator

    ppo.train(cfg)


def _danger_rows(obs):
    """Return rows with zombie HP in the two columns nearest the house."""
    pressure = obs["spatial"][:, :2, 3:].sum(axis=(1, 2))
    return [index for index, hp in enumerate(pressure) if hp > 0]


def _terminal_board_snapshot(obs):
    """Summarize terminal plant coverage and zombie pressure by row."""
    spatial = obs["spatial"]
    return {
        "plant_occupancy_by_row": (spatial[:, :, 0] > 0).sum(axis=1).tolist(),
        "zombie_hp_by_row": spatial[:, :, 3:].sum(axis=(1, 2)).tolist(),
    }


def _spatial_action(mask):
    """Select the shared deterministic structured teacher action."""
    from teacher_policy import choose_action

    return choose_action(mask)


def _wait_offense_override_action(
    action: int, action_mask: np.ndarray, logits: np.ndarray
) -> int:
    """Replace an eligible Wait using logits from the same policy step."""
    legal = np.asarray(action_mask, dtype=bool)
    offense = np.arange(46 + 2 * 45, 46 + 4 * 45)
    emergency = slice(46 + 8 * 45, 46 + 10 * 45)
    offense_legal = legal[offense]
    if action != 0 or legal[emergency].any() or not offense_legal.any():
        return action
    candidates = offense[offense_legal]
    # pi-lens-ignore: unchecked-throwing-call-python
    return int(candidates[np.asarray(logits)[candidates].argmax()])


def _evaluation_policy_step(
    agent, spatial, global_vec, mask, caches, *, deterministic: bool, override: bool
):
    """Run inference once, optionally replacing Wait from the resulting logits."""
    import torch

    logits, _, caches = agent.step_logits(spatial, global_vec, mask, caches)
    action = (
        # pi-lens-ignore: unchecked-throwing-call-python
        int(logits[0].argmax().item())
        if deterministic
        # pi-lens-ignore: unchecked-throwing-call-python
        else int(torch.distributions.Categorical(logits=logits).sample().item())
    )
    selected = (
        _wait_offense_override_action(
            action, mask[0].cpu().numpy(), logits[0].detach().cpu().numpy()
        )
        if override
        else action
    )
    return selected, caches, selected != action, logits


@app.function(
    image=image,
    volumes={"/pvz-assets": assets},
    timeout=900,
)
def smoke_mask(
    start_wave: int = 1,
    max_steps: int = 500,
    num_episodes: int = 3,
    cooldown_scale: float = 0.13,
):
    __import__("sys").path.insert(0, "/opt/pvz")
    __import__("sys").path.insert(0, "/opt/pvz/pvz-portable/build")
    import pvz_env  # noqa: E402

    env = pvz_env.PvZEnv("/pvz-assets", "/pvz-assets/savedata")  # noqa: F821
    env.set_cooldown_scale(cooldown_scale)

    replay_seed = 272000
    env.reset(start_wave, replay_seed)
    env.set_sun_money(3000)
    before_replay = np.array(env.get_obs()["spatial"], copy=True)
    env.step(0)
    env.reset(start_wave, replay_seed)
    env.set_sun_money(3000)
    after_replay = np.asarray(env.get_obs()["spatial"])
    if not np.array_equal(before_replay, after_replay):
        raise RuntimeError("seeded reset replay did not restore the board state")
    print("seeded reset replay = OK")

    print(f"action_space_size = {pvz_env.PvZEnv.action_space_size}")  # noqa: F821
    for ep in range(num_episodes):
        env.reset(start_wave)
        env.set_sun_money(3000)
        plant_actions = 0
        wait_actions = 0
        shovel_actions = 0
        invalid_actions = 0
        steps_taken = 0
        total_reward = 0.0
        final_sun = 0
        final_wave = 0
        end_reason = "running"
        plant_log = []

        for step in range(max_steps):
            mask = np.asarray(env.get_action_mask(), dtype=bool)
            action = _spatial_action(mask)
            raw_obs, m, reward, done, truncated, info = env.step(action)
            steps_taken += 1
            # pi-lens-ignore: unchecked-throwing-call-python
            total_reward += float(reward)
            # Diagnostic: count plants and zombies in obs
            spatial = raw_obs["spatial"]
            # pi-lens-ignore: unchecked-throwing-call-python
            n_plants = int((spatial[:, :, 0] > 0).sum())
            # pi-lens-ignore: unchecked-throwing-call-python
            z_hp = float(spatial[:, :, 3:].sum())
            # pi-lens-ignore: unchecked-throwing-call-python
            final_wave = max(final_wave, int(info.get("wave", 0)))
            # pi-lens-ignore: unchecked-throwing-call-python
            final_sun = int(info.get("sun", 0))
            if action == 0:
                wait_actions += 1
            elif action < 46:
                shovel_actions += 1
            else:
                plant_actions += 1
                seed = (action - 46) // 45
                row = ((action - 46) % 45) // 9
                col = (action - 46) % 9
                seed_name = [
                    "sun",
                    "twin_sun",
                    "melon",
                    "winter",
                    "gloom",
                    "fume",
                    "pumpkin",
                    "garlic",
                    "squash",
                    "jalapeno",
                ][seed]
                plant_log.append(f"{seed_name[0]}r{row}c{col}")

            if action >= 46 and not mask[action]:
                invalid_actions += 1

            if step % 30 == 0 or step == max_steps - 1:
                seed_mask_counts = []
                for s in range(10):
                    start = 46 + s * 45
                    end = start + 45
                    # pi-lens-ignore: unchecked-throwing-call-python
                    seed_mask_counts.append(int(mask[start:end].sum()))
                print(
                    # pi-lens-ignore: unchecked-throwing-call-python
                    f"  step {step:3d}: mask_by_seed={seed_mask_counts} sun={int(info.get('sun', 0))} wave={int(info.get('wave', 0))} plants={n_plants} z_hp={z_hp:.0f}"
                )
            if done:
                end_reason = "lost" if info.get("lost", False) else "stage_complete"
                break
            if truncated:
                end_reason = "truncated"
                break

        seed_counts = {}
        for p in plant_log:
            seed_name = p[0]
            seed_counts[seed_name] = seed_counts.get(seed_name, 0) + 1
        print(
            f"\n=== EP {ep} ===\n"
            f"  steps: {steps_taken}\n"
            f"  plant: {plant_actions} wait: {wait_actions} shovel: {shovel_actions} invalid: {invalid_actions}\n"
            f"  plant_ratio: {plant_actions / max(steps_taken, 1):.3f}\n"
            f"  seed_counts: {seed_counts}\n"
            f"  final_wave: {final_wave}\n"
            f"  final_sun: {final_sun}\n"
            f"  end: {end_reason}\n"
            f"  TOTAL_REWARD: {total_reward:.2f}  MEAN_REWARD: {total_reward / max(steps_taken, 1):.4f}\n"
            f"  first_30_plants: {' '.join(plant_log[:30])}"
        )


@app.function(
    image=image,
    volumes={"/pvz-assets": assets, "/pvz-checkpoints": checkpoints},
    timeout=3600,
    single_use_containers=True,
)
def probe_counterfactual_checkpoint(
    checkpoint_name: str = "plan_zero_best.pt",
    episodes: int = 5,
    max_steps: int = 250,
    start_sun: int = 0,
    horizon: int = 10,
):
    """Measure same-state one-step and short-horizon values for feasible actions."""
    import sys

    import torch

    sys.path.insert(0, "/opt/pvz")
    sys.path.insert(0, "/opt/pvz/pvz-portable/build")
    from network import PvZActorCritic
    from pvz_portable_gym import PvZGymEnv

    checkpoints.reload()
    agent = PvZActorCritic().to("cpu").eval()
    checkpoint = torch.load(
        f"/pvz-checkpoints/{checkpoint_name}", map_location="cpu", weights_only=True
    )
    agent.load_state_dict(checkpoint["agent_state"])
    env = PvZGymEnv(resdir="/pvz-assets", savedir="/pvz-assets/savedata")
    env.set_cooldown_scale(1.0)

    offense = np.arange(46 + 2 * 45, 46 + 4 * 45)
    emergency = np.arange(46 + 8 * 45, 46 + 10 * 45)
    totals = {
        "states": 0,
        "offense_only_states": 0,
        "both_states": 0,
        "repeat_checks": 0,
        "repeat_matches": 0,
        "offense_minus_wait": [],
        "offense_minus_emergency": [],
        "horizon_offense_minus_wait": [],
        "horizon_offense_minus_emergency": [],
    }
    try:
        for episode in range(episodes):
            episode_seed = 272000 + episode
            obs, info = env.reset(
                options={"wave": 1, "sun": start_sun, "seed": episode_seed}
            )
            caches = agent.init_caches(1, torch.device("cpu"))
            history = []

            def restore_root(
                episode_seed=episode_seed, history=history, start_sun=start_sun
            ):
                root_obs, root_info = env.reset(
                    options={"wave": 1, "sun": start_sun, "seed": episode_seed}
                )
                for replay_action in history:
                    root_obs, _, done, truncated, root_info = env.step(replay_action)
                    if done or truncated:
                        raise RuntimeError(
                            "seeded replay terminated before the probe root"
                        )
                return root_obs, root_info

            def replayed_step(candidate, root_obs=obs, root_info=info):
                replay_obs, replay_info = restore_root()
                if not (
                    np.array_equal(replay_obs["spatial"], root_obs["spatial"])
                    and np.array_equal(replay_obs["global"], root_obs["global"])
                    and np.array_equal(
                        replay_info["action_mask"], root_info["action_mask"]
                    )
                ):
                    raise RuntimeError("seeded replay did not reproduce the probe root")
                try:
                    return env.step(candidate)
                finally:
                    restore_root()

            for _ in range(max_steps):
                spatial = torch.from_numpy(obs["spatial"]).unsqueeze(0)
                global_vec = torch.from_numpy(obs["global"]).unsqueeze(0)
                mask = torch.from_numpy(
                    np.asarray(info["action_mask"], dtype=bool)
                ).unsqueeze(0)
                with torch.no_grad():
                    features = agent._encode_features(spatial, global_vec)
                    recurrent, caches = agent.mamba.step(features, caches)
                    latent = agent.latent_proj(
                        features
                        + torch.tanh(agent.memory_gate) * (recurrent - features)
                    )
                    logits = agent._apply_action_mask(agent.actor(latent), mask)[0]
                legal = mask[0].numpy().astype(bool)
                offense_legal = legal[offense]
                emergency_legal = legal[emergency]
                if not offense_legal.any():
                    action = int(logits.argmax())
                    history.append(action)
                    obs, _, done, truncated, info = env.step(action)
                    if done or truncated:
                        break
                    continue

                totals["states"] += 1
                if emergency_legal.any():
                    totals["both_states"] += 1
                else:
                    totals["offense_only_states"] += 1
                candidates = [
                    0,
                    int(
                        offense[offense_legal][logits[offense[offense_legal]].argmax()]
                    ),
                ]
                if emergency_legal.any():
                    candidates.append(
                        int(
                            emergency[emergency_legal][
                                logits[emergency[emergency_legal]].argmax()
                            ]
                        )
                    )
                probes = {}
                for candidate in dict.fromkeys(candidates):
                    probes[candidate] = replayed_step(candidate)

                def branch_return(
                    candidate, root_obs=obs, root_info=info, root_caches=caches
                ):
                    branch_obs, branch_info = restore_root()
                    if not (
                        np.array_equal(branch_obs["spatial"], root_obs["spatial"])
                        and np.array_equal(branch_obs["global"], root_obs["global"])
                        and np.array_equal(
                            branch_info["action_mask"], root_info["action_mask"]
                        )
                    ):
                        raise RuntimeError(
                            "seeded replay did not reproduce the probe root"
                        )
                    branch_caches = [
                        (hidden.clone(), inputs.clone())
                        for hidden, inputs in root_caches
                    ]
                    total = 0.0
                    try:
                        for depth in range(horizon):
                            if depth == 0:
                                branch_action = candidate
                            else:
                                branch_spatial = torch.from_numpy(
                                    branch_obs["spatial"]
                                ).unsqueeze(0)
                                branch_global = torch.from_numpy(
                                    branch_obs["global"]
                                ).unsqueeze(0)
                                branch_mask = torch.from_numpy(
                                    np.asarray(branch_info["action_mask"], dtype=bool)
                                ).unsqueeze(0)
                                with torch.no_grad():
                                    (
                                        branch_action_tensor,
                                        _,
                                        _,
                                        _,
                                        branch_caches,
                                    ) = agent.forward_step(
                                        branch_spatial,
                                        branch_global,
                                        branch_mask,
                                        branch_caches,
                                        deterministic=True,
                                    )
                                branch_action = int(branch_action_tensor.item())
                            (
                                branch_obs,
                                branch_reward,
                                branch_done,
                                branch_truncated,
                                branch_info,
                            ) = env.step(branch_action)
                            total += float(branch_reward)
                            if branch_done or branch_truncated:
                                break
                    finally:
                        restore_root()
                    return total

                wait_reward = probes[0][1]
                offense_reward = probes[candidates[1]][1]
                offense_return = branch_return(candidates[1])
                wait_return = branch_return(0)
                if not emergency_legal.any():
                    totals["offense_minus_wait"].append(offense_reward - wait_reward)
                    totals["horizon_offense_minus_wait"].append(
                        offense_return - wait_return
                    )
                else:
                    emergency_reward = probes[candidates[2]][1]
                    emergency_return = branch_return(candidates[2])
                    totals["offense_minus_emergency"].append(
                        offense_reward - emergency_reward
                    )
                    totals["horizon_offense_minus_emergency"].append(
                        offense_return - emergency_return
                    )
                if totals["repeat_checks"] < 20:
                    first = probes[candidates[1]]
                    second = replayed_step(candidates[1])
                    totals["repeat_checks"] += 1
                    if (
                        np.array_equal(first[0]["spatial"], second[0]["spatial"])
                        and np.array_equal(first[0]["global"], second[0]["global"])
                        and first[1] == second[1]
                    ):
                        totals["repeat_matches"] += 1

                action = int(logits.argmax())
                history.append(action)
                obs, _, done, truncated, info = env.step(action)
                if done or truncated:
                    break
    finally:
        env.close()

    summary = {
        key: value
        if not isinstance(value, list)
        else {
            "count": len(value),
            "mean": sum(value) / len(value) if value else None,
            "min": min(value) if value else None,
            "max": max(value) if value else None,
        }
        for key, value in totals.items()
    }
    print(summary, flush=True)
    return summary


@app.local_entrypoint()
def probe_counterfactual(
    checkpoint_name: str = "plan_zero_best.pt",
    episodes: int = 5,
    max_steps: int = 250,
    start_sun: int = 0,
    horizon: int = 10,
):
    print(
        probe_counterfactual_checkpoint.remote(
            checkpoint_name=checkpoint_name,
            episodes=episodes,
            max_steps=max_steps,
            start_sun=start_sun,
            horizon=horizon,
        )
    )


@app.local_entrypoint()
def smoke(
    start_wave: int = 1,
    max_steps: int = 500,
    num_episodes: int = 3,
    cooldown_scale: float = 0.13,
):
    smoke_mask.remote(
        start_wave=start_wave,
        max_steps=max_steps,
        num_episodes=num_episodes,
        cooldown_scale=cooldown_scale,
    )


@app.function(
    image=image,
    volumes={"/pvz-assets": assets, "/pvz-checkpoints": checkpoints},
    timeout=3600,
    single_use_containers=True,
)
def evaluate_checkpoint(
    checkpoint_name: str,
    start_wave: int = 1,
    start_sun: int = 50,
    cooldown_scale: float = 1.0,
    episodes: int = 10,
    max_steps: int = 2000,
    full_deck: bool = True,
    suite: dict | None = None,
):
    """Evaluate one checkpoint with the suite's explicit action-selection mode."""
    import sys

    import torch

    sys.path.insert(0, "/opt/pvz")
    sys.path.insert(0, "/opt/pvz/pvz-portable/build")
    from curriculum import (
        NUM_PLANT_ACTIONS,
        NUM_SHOVEL_ACTIONS,
        PLANT_ACTION_OFFSET,
        WAIT_ACTION,
        CurriculumManager,
    )
    from evaluation import write_evaluation_results
    from network import PvZActorCritic
    from pvz_portable_gym import PvZGymEnv

    if suite is None:
        suite = {
            "name": "manual_eval",
            "start_wave": start_wave,
            "start_sun": start_sun,
            "cooldown_scale": cooldown_scale,
            "episodes": episodes,
            "max_steps": max_steps,
            "full_deck": full_deck,
        }
    else:
        suite = dict(suite)

    # This function runs in a fresh container; refresh its volume mount after
    # the trainer committed the just-written checkpoint.
    checkpoints.reload()

    checkpoint_path = f"/pvz-checkpoints/{checkpoint_name}"
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    agent = PvZActorCritic().to(device)
    agent.load_state_dict(checkpoint["agent_state"])
    agent.eval()

    # Non-full-deck suites always use the phase-0 seed restriction.
    curriculum = CurriculumManager()
    curriculum_mask = torch.from_numpy(curriculum.seed_mask(agent.num_actions)).to(
        device
    )
    if suite.get("allowed_seeds") is not None:
        # Narrow-deck suite: honor the suite's explicit slot mask instead of
        # the phase-0 five-seed default (same construction as seed_mask).
        narrow_mask = torch.zeros(agent.num_actions, dtype=torch.float32)
        narrow_mask[WAIT_ACTION] = 1.0
        narrow_mask[1 : 1 + NUM_SHOVEL_ACTIONS] = 1.0
        for seed in suite["allowed_seeds"]:
            start = PLANT_ACTION_OFFSET + seed * NUM_PLANT_ACTIONS
            narrow_mask[start : start + NUM_PLANT_ACTIONS] = 1.0
        curriculum_mask = narrow_mask.to(device)
    results = []
    env = PvZGymEnv(resdir="/pvz-assets", savedir="/pvz-assets/savedata", chdir=True)
    env.set_cooldown_scale(suite["cooldown_scale"])
    if suite.get("chain_stages"):
        # Endless mode: the stage boundary chains instead of terminating, so
        # max_wave becomes an absolute stage*20+wave survival measure.
        env.set_chain_stages(True)
    if suite.get("deck"):
        # Custom deck (Coffee Bean rung): slot meanings remap at reset.
        env.set_deck(suite["deck"])
    try:
        for episode in range(suite["episodes"]):
            episode_seed = (
                None
                if suite.get("seed_start") is None
                else suite["seed_start"] + episode
            )
            reset_options = {"wave": suite["start_wave"], "sun": suite["start_sun"]}
            if episode_seed is not None:
                reset_options["seed"] = episode_seed
            obs, info = env.reset(options=reset_options)
            caches = agent.init_caches(1, device)
            max_wave = int(info.get("wave", suite["start_wave"]))
            total_reward = 0.0
            reward_components = dict.fromkeys(
                (
                    "reward_damage",
                    "reward_wave",
                    "reward_alive",
                    "reward_plant",
                    "reward_shovel",
                    "reward_sun",
                    "reward_danger",
                    "reward_mower",
                    "reward_death",
                    "reward_time",
                ),
                0.0,
            )
            action_counts = {
                "wait": 0,
                "shovel": 0,
                "plant": 0,
                "wait_offense_overrides": 0,
                "seeds": [0] * 10,
                "plant_rows": [0] * 5,
                "plant_columns": [0] * 9,
                "early_plant_rows": [0] * 5,
                "early_plant_columns": [0] * 9,
            }
            terminal_reason = "max_steps"
            first_danger_step = None
            first_danger_rows = []
            step = -1

            for step in range(suite["max_steps"]):
                spatial = torch.from_numpy(obs["spatial"]).unsqueeze(0).to(device)
                global_vec = torch.from_numpy(obs["global"]).unsqueeze(0).to(device)
                mask = torch.from_numpy(info["action_mask"]).unsqueeze(0).to(device)
                if not suite["full_deck"]:
                    mask = mask * curriculum_mask
                with torch.no_grad():
                    action_id, caches, overridden, _ = _evaluation_policy_step(
                        agent,
                        spatial,
                        global_vec,
                        mask,
                        caches,
                        deterministic=bool(suite.get("deterministic", True)),
                        override=bool(suite.get("wait_offense_override", False)),
                    )
                    action_counts["wait_offense_overrides"] += int(overridden)
                if action_id == 0:
                    action_counts["wait"] += 1
                elif action_id < 46:
                    action_counts["shovel"] += 1
                else:
                    action_counts["plant"] += 1
                    plant_cell = (action_id - 46) % 45
                    action_counts["seeds"][(action_id - 46) // 45] += 1
                    action_counts["plant_rows"][plant_cell // 9] += 1
                    action_counts["plant_columns"][plant_cell % 9] += 1
                    if step < 100:
                        action_counts["early_plant_rows"][plant_cell // 9] += 1
                        action_counts["early_plant_columns"][plant_cell % 9] += 1

                obs, reward, done, truncated, info = env.step(action_id)
                total_reward += float(reward)
                for name in reward_components:
                    reward_components[name] += float(info.get(name, 0.0))
                max_wave = max(max_wave, int(info.get("wave", max_wave)))
                danger_rows = _danger_rows(obs)
                if first_danger_step is None and danger_rows:
                    first_danger_step = step + 1
                    first_danger_rows = danger_rows
                if done:
                    terminal_reason = (
                        "lost" if info.get("lost", False) else "stage_complete"
                    )
                    break
                if truncated:
                    terminal_reason = "truncated"
                    break

            result = {
                "checkpoint": checkpoint_name,
                "suite": suite,
                "episode": episode,
                "seed": episode_seed,
                "steps": step + 1,
                "max_wave": max_wave,
                "total_reward": total_reward,
                "reward_components": reward_components,
                "terminal_reason": terminal_reason,
                "lost": bool(info.get("lost", False)),
                "stage_complete": bool(info.get("stage_complete", False)),
                "action_counts": action_counts,
                "terminal_board": _terminal_board_snapshot(obs),
                "first_danger_step": first_danger_step,
                "first_danger_rows": first_danger_rows,
            }
            results.append(result)
            print(result, flush=True)
    finally:
        env.close()

    summary = write_evaluation_results(checkpoint_path, suite, results)
    summary["results"] = results
    print(summary, flush=True)
    return summary


@app.local_entrypoint()
def evaluate(
    checkpoint_name: str,
    start_wave: int = 1,
    start_sun: int = 50,
    cooldown_scale: float = 1.0,
    episodes: int = 10,
    full_deck: bool = True,
    suite_name: str = "manual_eval",
    deterministic: bool = True,
    seed_start: int | None = None,
    wait_offense_override: bool = False,
    chain_stages: bool = False,
    deck: str = "",
    allowed_seeds: str = "",
):
    suite = {
        "name": suite_name,
        "start_wave": start_wave,
        "start_sun": start_sun,
        "cooldown_scale": cooldown_scale,
        "episodes": episodes,
        "max_steps": 2000,
        "full_deck": full_deck,
        "deterministic": deterministic,
        "seed_start": seed_start,
        "wait_offense_override": wait_offense_override,
        "chain_stages": chain_stages,
        "deck": [int(s) for s in deck.split(",") if s.strip()],  # pi-lens-ignore: unchecked-throwing-call-python
    }
    if allowed_seeds:
        # Narrow-deck paired panels: explicit slot mask; omitted keeps the
        # phase-0 five-seed default for non-full-deck suites.
        suite["allowed_seeds"] = [int(s) for s in allowed_seeds.split(",") if s.strip()]  # pi-lens-ignore: unchecked-throwing-call-python
    if seed_start is None:
        evaluate_checkpoint.remote(checkpoint_name=checkpoint_name, suite=suite)
        return

    # Each seeded episode gets a new native process: the engine retains static
    # state across board resets, so batching would invalidate paired rollouts.
    rows = []
    for episode in range(episodes):
        episode_suite = {
            **suite,
            "name": f"{suite_name}_seed_{seed_start + episode}",
            "episodes": 1,
            "seed_start": seed_start + episode,
        }
        rows.append(
            evaluate_checkpoint.remote(
                checkpoint_name=checkpoint_name, suite=episode_suite
            )["results"][0]
        )
    print({"suite": suite, "results": rows})


@app.local_entrypoint()
def compare_wait_offense_override(
    checkpoint_name: str = "plan_zero_best.pt",
    episodes: int = 100,
    seed_start: int = 272000,
    workers: int = 20,
    start_sun: int = 0,
    cooldown_scale: float = 1.0,
    full_deck: bool = True,
    suite_name: str = "zero",
):
    """Run paired isolated evaluations of the narrow Wait override."""
    from concurrent.futures import ThreadPoolExecutor

    from evaluation import paired_teacher_comparison

    def run(seed: int, override: bool):
        suite = {
            "name": f"wait_offense_{suite_name}_{'candidate' if override else 'baseline'}_seed_{seed}",
            "start_wave": 1,
            "start_sun": start_sun,
            "cooldown_scale": cooldown_scale,
            "episodes": 1,
            "max_steps": 2000,
            "full_deck": full_deck,
            "deterministic": True,
            "seed_start": seed,
            "wait_offense_override": override,
        }
        return evaluate_checkpoint.remote(checkpoint_name=checkpoint_name, suite=suite)[
            "results"
        ][0]

    seeds = list(range(seed_start, seed_start + episodes))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        baseline = list(executor.map(lambda seed: run(seed, False), seeds))
        candidate = list(executor.map(lambda seed: run(seed, True), seeds))
    comparison = paired_teacher_comparison(baseline, candidate)
    comparison["baseline_mean_max_wave"] = (
        sum(row["max_wave"] for row in baseline) / episodes
    )
    comparison["candidate_mean_max_wave"] = (
        sum(row["max_wave"] for row in candidate) / episodes
    )
    comparison["override_count"] = sum(
        row["action_counts"]["wait_offense_overrides"] for row in candidate
    )
    print(comparison)


def _replay_branch_arm_impl(
    checkpoint_name: str,
    seed: int,
    arm: str,
    prefix_name: str | None = None,
    output_name: str = "replay/arm.json",
    horizon: int = 50,
    max_prefix_steps: int = 500,
    max_episode_steps: int = 2000,
    exclusive_output: bool = False,
    private_output: bool = False,
    prefix_bytes: bytes | None = None,
):
    """Run one bounded trace or branch arm in a fresh native process."""
    import glob
    import json
    import shutil
    import subprocess
    import tempfile

    checkpoint_name = validate_checkpoint_name(checkpoint_name)
    output_name = validate_replay_volume_path(output_name)
    prefix_name = (
        validate_replay_volume_path(prefix_name) if prefix_name is not None else None
    )
    if prefix_name is not None and prefix_bytes is not None:
        raise ValueError("prefix_name and prefix_bytes are mutually exclusive")
    if prefix_bytes is not None and not isinstance(prefix_bytes, bytes):
        raise TypeError("prefix_bytes must be bytes")
    checkpoints.reload()
    native_modules = glob.glob("/opt/pvz/pvz-portable/build/pvz_env*.so")
    if len(native_modules) != 1:
        raise RuntimeError(f"expected one native module, found {native_modules}")
    with tempfile.TemporaryDirectory(prefix="pvz-replay-") as savedir:
        if private_output:
            output_path = os.path.join(savedir, "private_trace.json")
        else:
            output_path = str(
                resolve_replay_volume_path("/pvz-checkpoints", output_name)
            )
            # pi-lens-ignore: unchecked-throwing-call-python
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_path = str(
                resolve_replay_volume_path("/pvz-checkpoints", output_name)
            )
        bootstrap = "/pvz-assets/savedata/registry.regemu"
        if not os.path.isfile(bootstrap):
            raise RuntimeError("missing replay profile bootstrap registry.regemu")
        shutil.copy2(bootstrap, os.path.join(savedir, "registry.regemu"))
        if prefix_bytes is not None:
            Path(savedir, "prefix.json").write_bytes(prefix_bytes)
        # pi-lens-ignore: unchecked-throwing-call-python
        os.makedirs(os.path.join(savedir, "userdata"))
        command = [
            sys.executable,
            "/opt/pvz/scripts/replay_branch_worker.py",
            "--checkpoint",
            f"/pvz-checkpoints/{checkpoint_name}",
            "--native-module",
            native_modules[0],
            "--resdir",
            "/pvz-assets",
            "--savedir",
            savedir,
            "--seed",
            str(seed),
            "--arm",
            arm,
            "--output",
            output_path,
            "--horizon",
            str(horizon),
            "--max-prefix-steps",
            str(max_prefix_steps),
            "--max-episode-steps",
            str(max_episode_steps),
        ]
        if exclusive_output:
            command.append("--exclusive-output")
        if prefix_bytes is not None:
            command.extend(["--prefix", str(Path(savedir, "prefix.json"))])
        elif prefix_name is not None:
            prefix_path = resolve_replay_volume_path("/pvz-checkpoints", prefix_name)
            command.extend(["--prefix", str(prefix_path)])
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode:
            if private_output:
                diagnostic = Path(savedir) / "worker_diagnostic.txt"
                diagnostic.write_text(
                    completed.stdout + "\n" + completed.stderr, encoding="utf-8"
                )
                return {
                    "private_error": {
                        "code": "worker_subprocess_failed",
                        "error_class": "subprocess",
                        "diagnostic_path": "private-local://worker/diagnostic",
                    }
                }
            raise RuntimeError(
                f"replay worker failed ({completed.returncode}): "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        try:
            result = json.loads(completed.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as error:
            if private_output:
                return {
                    "private_error": {
                        "code": "worker_invalid_receipt",
                        "error_class": "protocol",
                        "diagnostic_path": "private-local://worker/diagnostic",
                    }
                }
            raise RuntimeError("replay worker returned invalid JSON") from error
        if private_output:
            result["trace_bytes"] = Path(output_path).read_bytes()
        try:
            result = validate_replay_worker_receipt(result, private=private_output)
        except ValueError as error:
            raise RuntimeError(str(error)) from error
    if not private_output:
        checkpoints.commit()
    return result


def run_private_worker_safely(call, private_output):
    """Prevent private worker exception text from crossing the process boundary."""
    try:
        return call()
    except Exception:  # noqa: BLE001
        if not private_output:
            raise
        return {
            "private_error": {
                "code": "worker_invocation_failed",
                "error_class": "infrastructure",
                "diagnostic_path": "private-local://worker/diagnostic",
            }
        }


@app.function(
    image=image,
    volumes={"/pvz-assets": assets, "/pvz-checkpoints": checkpoints},
    timeout=3600,
    retries=0,
    single_use_containers=True,
)
def replay_branch_arm(
    checkpoint_name: str,
    seed: int,
    arm: str,
    prefix_name: str | None = None,
    output_name: str = "replay/arm.json",
    horizon: int = 50,
    max_prefix_steps: int = 500,
    max_episode_steps: int = 2000,
    exclusive_output: bool = False,
    private_output: bool = False,
    prefix_bytes: bytes | None = None,
):
    return run_private_worker_safely(
        lambda: _replay_branch_arm_impl(
            checkpoint_name,
            seed,
            arm,
            prefix_name,
            output_name,
            horizon,
            max_prefix_steps,
            max_episode_steps,
            exclusive_output,
            private_output,
            prefix_bytes,
        ),
        private_output,
    )


@app.function(image=image, volumes={"/pvz-checkpoints": checkpoints})
def checkpoint_volume_hashes():
    """Hash the exact champion files mounted from the checkpoint volume."""
    import hashlib

    checkpoints.reload()
    hashes = {}
    for name in ("plan_zero_best.pt", "plan_wave20_best.pt"):
        path = f"/pvz-checkpoints/{name}"
        if not os.path.isfile(path):
            hashes[name] = None
            continue
        digest = hashlib.sha256()
        # pi-lens-ignore: unchecked-throwing-call-python
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
    return hashes


@app.local_entrypoint()
def attest_checkpoint_hashes():
    print(checkpoint_volume_hashes.remote())


@app.function(image=image, volumes={"/pvz-checkpoints": checkpoints})
def replay_collector_prefix_objects(prefix: str = "replay/collector_v7"):
    """Return every entry, including the prefix and symlink directories."""
    checkpoints.reload()
    safe = validate_replay_volume_path(prefix + "/probe.json").rsplit("/", 1)[0]
    return collector_prefix_entries(f"/pvz-checkpoints/{safe}")


@app.function(
    image=image,
    volumes={"/pvz-assets": assets, "/pvz-checkpoints": checkpoints},
    timeout=3600,
)
def evaluate_teacher_transfer(
    episodes: int = 20,
    max_steps: int = 2000,
    start_sun: int = 3000,
    suite_name: str = "teacher_v2_transfer_eval",
    checkpoint_name: str = "teacher-v2-normal-cooldown.pt",
    teacher_profile: str = "structured-v2",
    seed_start: int | None = None,
):
    """Evaluate the structured teacher at normal cooldown and persist every episode."""
    import sys

    sys.path.insert(0, "/opt/pvz")
    sys.path.insert(0, "/opt/pvz/pvz-portable/build")
    from curriculum import CurriculumManager
    from evaluation import teacher_transfer_gate, write_evaluation_results
    from pvz_portable_gym import PvZGymEnv
    from teacher_policy import StatefulTeacher, choose_action

    suite = {
        "name": suite_name,
        "start_wave": 1,
        "start_sun": start_sun,
        "cooldown_scale": 1.0,
        "episodes": episodes,
        "max_steps": max_steps,
        "full_deck": False,
        "teacher": teacher_profile,
        "seed_start": seed_start,
    }
    curriculum = CurriculumManager()
    curriculum_mask = curriculum.seed_mask(496).astype(bool)
    results = []
    env = PvZGymEnv(resdir="/pvz-assets", savedir="/pvz-assets/savedata", chdir=True)
    env.set_cooldown_scale(suite["cooldown_scale"])
    try:
        for episode in range(episodes):
            episode_seed = None if seed_start is None else seed_start + episode
            teacher = (
                StatefulTeacher(teacher_profile)
                if teacher_profile
                in {
                    "structured-v4-sun1000",
                    "structured-v5-sun1000-cooldown",
                }
                else None
            )
            reset_options = {"wave": suite["start_wave"], "sun": suite["start_sun"]}
            if episode_seed is not None:
                reset_options["seed"] = episode_seed
            obs, info = env.reset(options=reset_options)
            if teacher is not None:
                teacher.reset()
            max_wave = int(info.get("wave", suite["start_wave"]))
            total_reward = 0.0
            action_counts = {
                "wait": 0,
                "shovel": 0,
                "plant": 0,
                "seeds": [0] * 10,
                "lanes": [0] * 5,
                "cells": [[0] * 9 for _ in range(5)],
            }
            terminal_reason = "max_steps"
            decision_diagnostics = []
            step = -1
            for step in range(max_steps):
                effective_mask = np.logical_and(info["action_mask"], curriculum_mask)
                if teacher is not None:
                    action_id, diagnostic = teacher.choose_action_with_diagnostics(
                        effective_mask,
                        obs["spatial"],
                        obs["global"],
                        env.squash_targetable_lanes(),
                    )
                    diagnostic["step"] = step
                    diagnostic["normalized_sun"] = float(obs["global"][0])
                    diagnostic["wave"] = int(info.get("wave", max_wave))
                    diagnostic["triggered_mowers"] = int(
                        info.get("triggered_mowers", 0)
                    )
                    decision_diagnostics.append(diagnostic)
                else:
                    action_id = choose_action(
                        effective_mask, teacher_profile, obs["spatial"]
                    )
                if not effective_mask[action_id]:
                    raise AssertionError(
                        "teacher selected an action outside its effective mask"
                    )
                if action_id == 0:
                    action_counts["wait"] += 1
                elif action_id < 46:
                    action_counts["shovel"] += 1
                else:
                    action_counts["plant"] += 1
                    seed, cell = divmod(action_id - 46, 45)
                    row, col = divmod(cell, 9)
                    action_counts["seeds"][seed] += 1
                    action_counts["lanes"][row] += 1
                    action_counts["cells"][row][col] += 1
                obs, reward, done, truncated, info = env.step(action_id)
                total_reward += float(reward)
                max_wave = max(max_wave, int(info.get("wave", max_wave)))
                if done:
                    terminal_reason = (
                        "lost" if info.get("lost", False) else "stage_complete"
                    )
                    break
                if truncated:
                    terminal_reason = "truncated"
                    break
            commitment_outcomes = (
                teacher.close_episode(terminal_reason) if teacher is not None else []
            )
            result = {
                "checkpoint": checkpoint_name,
                "suite": suite,
                "episode": episode,
                "seed": episode_seed,
                "steps": step + 1,
                "max_wave": max_wave,
                "total_reward": total_reward,
                "terminal_reason": terminal_reason,
                "lost": bool(info.get("lost", False)),
                "stage_complete": bool(info.get("stage_complete", False)),
                "action_counts": action_counts,
                "decision_diagnostics": decision_diagnostics,
                "squash_commitment_outcomes": commitment_outcomes,
            }
            results.append(result)
            print(result, flush=True)
    finally:
        env.close()

    checkpoint_path = f"/pvz-checkpoints/{checkpoint_name}"
    summary = write_evaluation_results(checkpoint_path, suite, results)
    summary["transfer_gate"] = teacher_transfer_gate(summary, results)
    checkpoints.commit()
    print(summary, flush=True)
    return {**summary, "results": results}


@app.function(
    image=image,
    volumes={"/pvz-checkpoints": checkpoints},
    timeout=300,
)
def analyze_squash_commitments(checkpoint_name: str):
    """Summarize cross-lane emergencies by the danger present at commitment."""
    import json
    from collections import Counter

    trace_path = Path("/pvz-checkpoints") / (
        f"{Path(checkpoint_name).with_suffix('').name}."
        "structured_v4_sun1000_squash_commitment_diagnostics_eval.jsonl"
    )
    rows = [
        # pi-lens-ignore: unchecked-throwing-call-python
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    cooldown_events = []
    first_conflict_age = {}
    rank_counts = Counter()
    commitment_dangers = []
    commitment_danger_ratios = []
    different_lane_by_terminal = Counter()
    for row in rows:
        for diagnostic in row["decision_diagnostics"]:
            if (
                diagnostic["emergency_lane"] is None
                or diagnostic["emergency_squash_legal"]
                or diagnostic["squash_seed_cooldown"] <= 0.0
            ):
                continue
            cooldown_events.append(diagnostic)
            if (
                diagnostic["emergency_squash_history_classification"]
                != "different_lane_conflict"
            ):
                continue
            commitment = diagnostic["last_squash_commitment_danger"]
            if commitment is None:
                continue
            emergency_lane = diagnostic["emergency_lane"]
            # pi-lens-ignore: unchecked-throwing-call-python
            commitment_danger = float(commitment[emergency_lane])
            rank = (
                None
                if commitment_danger <= 0.0
                else 1 + sum(value > commitment_danger for value in commitment)
            )
            rank_counts["zero" if rank is None else str(rank)] += 1
            different_lane_by_terminal[row["terminal_reason"]] += 1
            if commitment_danger > 0.0:
                commitment_dangers.append(commitment_danger)
                commitment_danger_ratios.append(
                    # pi-lens-ignore: unchecked-throwing-call-python
                    commitment_danger / max(float(value) for value in commitment)
                )
            commitment_key = (
                row["episode"],
                diagnostic["last_squash_commitment_decision"],
            )
            age = diagnostic["decisions_since_last_squash"]
            first_conflict_age[commitment_key] = min(
                first_conflict_age.get(commitment_key, age), age
            )
    different_lane = [
        diagnostic
        for diagnostic in cooldown_events
        if diagnostic["emergency_squash_history_classification"]
        == "different_lane_conflict"
    ]
    with_danger_at_commitment = [
        diagnostic
        for diagnostic in different_lane
        if diagnostic["emergency_lane_had_danger_when_squash_committed"]
    ]

    def median(values):
        ordered = sorted(values)
        midpoint = len(ordered) // 2
        return (
            ordered[midpoint]
            if len(ordered) % 2
            else (ordered[midpoint - 1] + ordered[midpoint]) / 2
        )

    summary = {
        "checkpoint": checkpoint_name,
        "trace_path": str(trace_path),
        "cooldown_masked_emergencies": len(cooldown_events),
        "different_lane_conflicts": len(different_lane),
        "different_lane_conflicts_with_danger_at_commitment": len(
            with_danger_at_commitment
        ),
        "different_lane_danger_at_commitment_rate": (
            len(with_danger_at_commitment) / len(different_lane)
            if different_lane
            else 0.0
        ),
        "different_lane_danger_rank_counts": dict(sorted(rank_counts.items())),
        "different_lane_conflicts_by_terminal_reason": dict(
            sorted(different_lane_by_terminal.items())
        ),
        "nonzero_danger_at_commitment": {
            "median": median(commitment_dangers) if commitment_dangers else None,
            "mean": (
                sum(commitment_dangers) / len(commitment_dangers)
                if commitment_dangers
                else None
            ),
            "median_fraction_of_committed_lane_danger": (
                median(commitment_danger_ratios) if commitment_danger_ratios else None
            ),
        },
        "commitments_with_later_different_lane_emergency": len(first_conflict_age),
        "first_different_lane_emergency_age": {
            "min": min(first_conflict_age.values()) if first_conflict_age else None,
            "mean": (
                sum(first_conflict_age.values()) / len(first_conflict_age)
                if first_conflict_age
                else None
            ),
            "max": max(first_conflict_age.values()) if first_conflict_age else None,
        },
    }
    summary_path = trace_path.with_suffix(".commitment_analysis.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checkpoints.commit()
    return summary


@app.function(
    image=image,
    volumes={"/pvz-checkpoints": checkpoints},
    timeout=300,
)
def analyze_squash_commitment_outcomes(
    checkpoint_name: str,
    suite_name: str = "structured_v4_sun1000_squash_commitment_outcomes_diagnostics_eval",
):
    """Summarize every selected Squash by its first subsequent emergency."""
    import json
    from collections import Counter

    trace_path = Path("/pvz-checkpoints") / (
        f"{Path(checkpoint_name).with_suffix('').name}.{suite_name}.jsonl"
    )
    rows = [
        # pi-lens-ignore: unchecked-throwing-call-python
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    outcomes = [
        outcome for row in rows for outcome in row.get("squash_commitment_outcomes", [])
    ]

    def summarize(group):
        categories = Counter(outcome["outcome_category"] for outcome in group)
        ages = [
            # pi-lens-ignore: unchecked-throwing-call-python
            int(outcome["first_emergency_age"])
            for outcome in group
            if outcome["first_emergency_age"] is not None
        ]
        cleared = sum(
            bool(outcome["committed_lane_cleared_before_first_emergency"])
            for outcome in group
        )
        seen_ages = [
            # pi-lens-ignore: unchecked-throwing-call-python
            int(outcome["first_squash_visible_age"])
            for outcome in group
            if outcome.get("first_squash_visible_age") is not None
        ]
        seen_count = sum(
            bool(outcome.get("squash_seen_after_commitment")) for outcome in group
        )
        first_post_commit_state_by_column = Counter(
            f"{outcome.get('nearest_zombie_column_at_commitment')}:"
            f"{outcome.get('first_post_commit_squash_state')}"
            for outcome in group
        )
        return {
            "commitments": len(group),
            "outcome_categories": dict(sorted(categories.items())),
            "committed_lane_clear_rate": cleared / len(group) if group else 0.0,
            "first_post_commit_state_by_commitment_column": dict(
                sorted(first_post_commit_state_by_column.items())
            ),
            "squash_visibility": {
                "seen_count": seen_count,
                "seen_rate": seen_count / len(group) if group else 0.0,
                "first_seen_age": {
                    "count": len(seen_ages),
                    "min": min(seen_ages) if seen_ages else None,
                    "median": (
                        sorted(seen_ages)[len(seen_ages) // 2]
                        if len(seen_ages) % 2
                        else (
                            sorted(seen_ages)[len(seen_ages) // 2 - 1]
                            + sorted(seen_ages)[len(seen_ages) // 2]
                        )
                        / 2
                        if seen_ages
                        else None
                    ),
                    "mean": sum(seen_ages) / len(seen_ages) if seen_ages else None,
                    "max": max(seen_ages) if seen_ages else None,
                },
            },
            "first_emergency_age": {
                "count": len(ages),
                "min": min(ages) if ages else None,
                "median": (
                    sorted(ages)[len(ages) // 2]
                    if len(ages) % 2
                    else (
                        sorted(ages)[len(ages) // 2 - 1] + sorted(ages)[len(ages) // 2]
                    )
                    / 2
                    if ages
                    else None
                ),
                "mean": sum(ages) / len(ages) if ages else None,
                "max": max(ages) if ages else None,
            },
        }

    by_terminal_reason = {
        terminal_reason: summarize(
            [
                outcome
                for row in rows
                if row["terminal_reason"] == terminal_reason
                for outcome in row.get("squash_commitment_outcomes", [])
            ]
        )
        for terminal_reason in sorted({row["terminal_reason"] for row in rows})
    }
    summary = {
        "checkpoint": checkpoint_name,
        "trace_path": str(trace_path),
        "overall": summarize(outcomes),
        "completed_episodes": summarize(
            [
                outcome
                for row in rows
                if row["terminal_reason"] == "stage_complete"
                for outcome in row.get("squash_commitment_outcomes", [])
            ]
        ),
        "lost_episodes": summarize(
            [
                outcome
                for row in rows
                if row["terminal_reason"] == "lost"
                for outcome in row.get("squash_commitment_outcomes", [])
            ]
        ),
        "by_terminal_reason": by_terminal_reason,
    }
    summary_path = trace_path.with_suffix(".commitment_outcome_analysis.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checkpoints.commit()
    return summary


@app.local_entrypoint()
def teacher_transfer(episodes: int = 20, max_steps: int = 2000):
    """Run the teacher-only normal-cooldown transfer measurement."""
    print(evaluate_teacher_transfer.remote(episodes=episodes, max_steps=max_steps))


@app.local_entrypoint()
def teacher_sun1000(episodes: int = 20, max_steps: int = 2000):
    """Qualify the low-sun teacher without changing any learner artifact."""
    print(
        evaluate_teacher_transfer.remote(
            episodes=episodes,
            max_steps=max_steps,
            start_sun=1000,
            suite_name="teacher_v3_sun1000_r3_eval",
            checkpoint_name="teacher-v3-sun1000-r3.pt",
            teacher_profile="structured-v3-sun1000-r3",
        )
    )


@app.local_entrypoint()
def teacher_squash_history_diagnostics(episodes: int = 20, max_steps: int = 2000):
    """Measure the unchanged 1000-sun teacher with persisted Squash history."""
    print(
        evaluate_teacher_transfer.remote(
            episodes=episodes,
            max_steps=max_steps,
            start_sun=1000,
            suite_name="structured_v4_sun1000_squash_history_diagnostics_eval",
            checkpoint_name="structured_v4_sun1000_squash_history_diagnostics.pt",
            teacher_profile="structured-v4-sun1000",
        )
    )


@app.local_entrypoint()
def teacher_squash_commitment_diagnostics(episodes: int = 20, max_steps: int = 2000):
    """Measure unchanged low-sun teaching with Squash commitment attribution."""
    print(
        evaluate_teacher_transfer.remote(
            episodes=episodes,
            max_steps=max_steps,
            start_sun=1000,
            suite_name="structured_v4_sun1000_squash_commitment_diagnostics_eval",
            checkpoint_name="structured_v4_sun1000_squash_commitment_diagnostics.pt",
            teacher_profile="structured-v4-sun1000",
        )
    )


@app.local_entrypoint()
def analyze_teacher_squash_commitments(
    checkpoint_name: str = "structured_v4_sun1000_squash_commitment_diagnostics.pt",
):
    """Analyze the persisted commitment diagnostic gate without rerunning it."""
    print(analyze_squash_commitments.remote(checkpoint_name))


@app.local_entrypoint()
def teacher_squash_commitment_outcomes_diagnostics(
    episodes: int = 20, max_steps: int = 2000
):
    """Measure unchanged low-sun teaching with commitment-outcome attribution."""
    print(
        evaluate_teacher_transfer.remote(
            episodes=episodes,
            max_steps=max_steps,
            start_sun=1000,
            suite_name="structured_v4_sun1000_squash_commitment_outcomes_diagnostics_eval",
            checkpoint_name="structured_v4_sun1000_squash_commitment_outcomes_diagnostics.pt",
            teacher_profile="structured-v4-sun1000",
        )
    )


@app.local_entrypoint()
def analyze_teacher_squash_commitment_outcomes(
    checkpoint_name: str = "structured_v4_sun1000_squash_commitment_outcomes_diagnostics.pt",
):
    """Analyze the persisted commitment-outcome gate without rerunning it."""
    print(analyze_squash_commitment_outcomes.remote(checkpoint_name))


@app.local_entrypoint()
def teacher_squash_commitment_lifecycle_diagnostics(
    episodes: int = 20, max_steps: int = 2000
):
    """Measure unchanged low-sun teaching with observed Squash target lifecycle."""
    print(
        evaluate_teacher_transfer.remote(
            episodes=episodes,
            max_steps=max_steps,
            start_sun=1000,
            suite_name="structured_v4_sun1000_squash_commitment_lifecycle_diagnostics_eval",
            checkpoint_name="structured_v4_sun1000_squash_commitment_lifecycle_diagnostics.pt",
            teacher_profile="structured-v4-sun1000",
        )
    )


@app.local_entrypoint()
def analyze_teacher_squash_commitment_lifecycle(
    checkpoint_name: str = "structured_v4_sun1000_squash_commitment_lifecycle_diagnostics.pt",
):
    """Analyze the persisted lifecycle gate without rerunning it."""
    print(
        analyze_squash_commitment_outcomes.remote(
            checkpoint_name,
            "structured_v4_sun1000_squash_commitment_lifecycle_diagnostics_eval",
        )
    )


@app.local_entrypoint()
def teacher_squash_commitment_lifecycle_corrected_diagnostics(
    episodes: int = 20, max_steps: int = 2000
):
    """Run the corrected plant-ID lifecycle attribution gate once."""
    print(
        evaluate_teacher_transfer.remote(
            episodes=episodes,
            max_steps=max_steps,
            start_sun=1000,
            suite_name="structured_v4_sun1000_squash_commitment_lifecycle_corrected_diagnostics_eval",
            checkpoint_name="structured_v4_sun1000_squash_commitment_lifecycle_corrected_diagnostics.pt",
            teacher_profile="structured-v4-sun1000",
        )
    )


@app.local_entrypoint()
def analyze_teacher_squash_commitment_lifecycle_corrected(
    checkpoint_name: str = "structured_v4_sun1000_squash_commitment_lifecycle_corrected_diagnostics.pt",
):
    """Analyze the corrected lifecycle gate without rerunning it."""
    print(
        analyze_squash_commitment_outcomes.remote(
            checkpoint_name,
            "structured_v4_sun1000_squash_commitment_lifecycle_corrected_diagnostics_eval",
        )
    )


@app.local_entrypoint()
def teacher_squash_commitment_geometry_diagnostics(
    episodes: int = 20, max_steps: int = 2000
):
    """Measure unchanged teaching with commitment-time zombie geometry."""
    print(
        evaluate_teacher_transfer.remote(
            episodes=episodes,
            max_steps=max_steps,
            start_sun=1000,
            suite_name="structured_v4_sun1000_squash_commitment_geometry_diagnostics_eval",
            checkpoint_name="structured_v4_sun1000_squash_commitment_geometry_diagnostics.pt",
            teacher_profile="structured-v4-sun1000",
        )
    )


@app.local_entrypoint()
def analyze_teacher_squash_commitment_geometry(
    checkpoint_name: str = "structured_v4_sun1000_squash_commitment_geometry_diagnostics.pt",
):
    """Analyze geometry attribution without rerunning the teacher."""
    print(
        analyze_squash_commitment_outcomes.remote(
            checkpoint_name,
            "structured_v4_sun1000_squash_commitment_geometry_diagnostics_eval",
        )
    )


@app.local_entrypoint()
def teacher_squash_geometry_repair(episodes: int = 20, max_steps: int = 2000):
    """Qualify the single forward-placement Squash repair once."""
    print(
        evaluate_teacher_transfer.remote(
            episodes=episodes,
            max_steps=max_steps,
            start_sun=1000,
            suite_name="structured_v4_sun1000_squash_geometry_repair_eval",
            checkpoint_name="structured_v4_sun1000_squash_geometry_repair.pt",
            teacher_profile="structured-v4-sun1000",
        )
    )


@app.local_entrypoint()
def teacher_budget(
    teacher_profile: str = "structured-v2",
    start_sun: int = 1000,
    episodes: int = 20,
    max_steps: int = 2000,
    run_name: str | None = None,
    seed_start: int | None = None,
):
    """Run one bounded deterministic teacher budget qualification."""
    label = run_name or f"{teacher_profile.replace('-', '_')}_sun{start_sun}"
    print(
        evaluate_teacher_transfer.remote(
            episodes=episodes,
            max_steps=max_steps,
            start_sun=start_sun,
            suite_name=f"{label}_eval",
            checkpoint_name=f"{label}.pt",
            teacher_profile=teacher_profile,
            seed_start=seed_start,
        )
    )


# ── Visualization: record a BC-heuristic episode and save a GIF to a volume ──
# Python-side renderer because the OpenGL headless render is broken.
# We draw the obs tensor directly: plants (colored squares) and zombies
# (red-tinted squares with HP bars).
_PLANT_COLORS_RGB = {
    0: (255, 220, 50),  # sunflower: yellow
    1: (255, 180, 30),  # twin-sunflower: orange-yellow
    2: (180, 70, 50),  # melon-pult: red-brown
    3: (130, 30, 30),  # winter-melon: dark red
    4: (90, 30, 110),  # gloom-shroom: purple
    5: (160, 110, 80),  # fume-shroom: brown
    6: (220, 140, 40),  # pumpkin-shell: orange
    7: (240, 240, 200),  # garlic: pale yellow
    8: (200, 80, 80),  # squash: pink-red
    9: (240, 50, 30),  # jalapeno: bright red
}
_ZOMBIE_COLOR = (90, 90, 90)  # dark gray


def _render_frame(obs, info, frame_w=180, frame_h=140):
    """Render the obs tensor into an RGB frame using simple colored squares.

    Grid: 5 rows x 9 cols. Plants in their cell, zombies overlaid.
    Top bar: wave + sun + step.
    """
    import numpy as _np

    frame = _np.full((frame_h, frame_w, 3), 60, dtype=_np.uint8)  # dark green grass
    spatial = obs["spatial"]
    global_ = obs["global"]

    # Layout: top 14 px = info bar, bottom = grid (5x9 cells)
    grid_top = 14
    cell_h = (frame_h - grid_top) // 5
    cell_w = frame_w // 9

    # Draw info bar
    # pi-lens-ignore: unchecked-throwing-call-python
    sun = int(global_[0] * 9990)
    # pi-lens-ignore: unchecked-throwing-call-python
    wave = int(global_[1] * 100)
    # Simple info bar: just paint colored strip
    frame[0:grid_top, :, 0] = 30
    frame[0:grid_top, :, 1] = 30
    frame[0:grid_top, :, 2] = 80

    # Draw plants
    for r in range(5):
        for c in range(9):
            # pi-lens-ignore: unchecked-throwing-call-python
            plant_type = int(spatial[r, c, 0]) - 1
            if plant_type < 0:
                continue
            color = _PLANT_COLORS_RGB.get(plant_type, (200, 200, 200))
            y0 = grid_top + r * cell_h + cell_h // 5
            y1 = grid_top + (r + 1) * cell_h - cell_h // 5
            x0 = c * cell_w + cell_w // 5
            x1 = (c + 1) * cell_w - cell_w // 5
            frame[y0:y1, x0:x1, 0] = color[0]
            frame[y0:y1, x0:x1, 1] = color[1]
            frame[y0:y1, x0:x1, 2] = color[2]
    for r in range(5):
        for c in range(9):
            # pi-lens-ignore: unchecked-throwing-call-python
            z_hp = float(spatial[r, c, 3:].sum())
            if z_hp <= 0:
                continue
            # Zombie in left half of cell, bright magenta with HP-based shade
            hp_norm = min(1.0, z_hp / 2000.0)
            # Magenta-purple gradient: (R=200+55*hp, G=0, B=180+75*hp)
            # pi-lens-ignore: unchecked-throwing-call-python
            r_int = int(150 + 100 * hp_norm)
            # pi-lens-ignore: unchecked-throwing-call-python
            g_int = int(20 + 30 * hp_norm)
            # pi-lens-ignore: unchecked-throwing-call-python
            b_int = int(120 + 100 * hp_norm)
            y0 = grid_top + r * cell_h + cell_h // 5
            y1 = grid_top + (r + 1) * cell_h - cell_h // 5
            x0 = c * cell_w + cell_w // 6
            # pi-lens-ignore: unchecked-throwing-call-python
            x1 = c * cell_w + int(cell_w * 0.5)
            frame[y0:y1, x0:x1, 0] = r_int
            frame[y0:y1, x0:x1, 1] = g_int
            frame[y0:y1, x0:x1, 2] = b_int
    # Add text overlay using PIL
    from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-not-found]

    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)
    # pi-lens-ignore: unchecked-throwing-call-python
    sun = int(global_[0] * 9990)
    # pi-lens-ignore: unchecked-throwing-call-python
    wave = int(global_[1] * 100)
    step = info.get("step", 0) if info else 0
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text(
        (2, 1), f"wave={wave} sun={sun} step={step}", fill=(255, 255, 255), font=font
    )
    return np.array(pil_img)


@app.function(
    image=image,
    volumes={
        "/pvz-assets": assets,
        "/pvz-gifs": modal.Volume.from_name("pvz-gifs", create_if_missing=True),
    },
    timeout=900,
)
def record_episode_gif(
    start_wave: int = 1,
    max_steps: int = 1500,
    cooldown_scale: float = 0.13,
    frame_stride: int = 10,
    downsample: int = 1,
):
    """Run the BC heuristic and save a GIF to the pvz-gifs Modal volume.

    Uses Python-side obs rendering (the OpenGL headless render is broken).
    Usage:
        modal run train_modal.py::record --start-wave 1 --cooldown-scale 0.13
    Then download with:
        modal volume get pvz-gifs pvz_episode.gif .
    """
    __import__("sys").path.insert(0, "/opt/pvz")
    __import__("sys").path.insert(0, "/opt/pvz/pvz-portable/build")
    import imageio.v2 as imageio  # noqa: E402  # type: ignore[import-not-found]
    import pvz_env  # noqa: E402

    env = pvz_env.PvZEnv("/pvz-assets", "/pvz-assets/savedata")
    env.set_cooldown_scale(cooldown_scale)
    env.reset(start_wave)
    env.set_sun_money(3000)

    frames = []
    plant_log = []
    final_wave = 0
    final_sun = 0
    end_reason = "running"
    total_reward = 0.0

    for step in range(max_steps):
        mask = np.asarray(env.get_action_mask(), dtype=bool)
        action = _spatial_action(mask)
        obs, m, reward, done, truncated, info = env.step(action)
        # pi-lens-ignore: unchecked-throwing-call-python
        total_reward += float(reward)

        if action >= 46:
            seed = (action - 46) // 45
            row = ((action - 46) % 45) // 9
            col = (action - 46) % 9
            plant_log.append(f"s{seed}r{row}c{col}")

        if step % frame_stride == 0:
            frame = _render_frame(obs, info)
            frames.append(frame)

        # pi-lens-ignore: unchecked-throwing-call-python
        final_wave = max(final_wave, int(info.get("wave", 0)))
        # pi-lens-ignore: unchecked-throwing-call-python
        final_sun = int(info.get("sun", 0))
        if done:
            end_reason = "lost" if info.get("lost", False) else "stage_complete"
            break
        if truncated:
            end_reason = "truncated"
            break

    out_path = "/pvz-gifs/pvz_episode.gif"
    imageio.mimsave(out_path, frames, fps=10)
    size_kb = os.path.getsize(out_path) / 1024
    print(
        f"[record_episode_gif] saved {len(frames)} frames to {out_path} ({size_kb:.1f} KB)"
    )
    print(
        f"[record_episode_gif] final_wave={final_wave} final_sun={final_sun} end={end_reason}"
    )
    print(
        f"[record_episode_gif] total_reward={total_reward:.2f} plants_logged={len(plant_log)}"
    )
    print(f"[record_episode_gif] plant_count_by_seed: {_count_seeds(plant_log)}")
    return out_path


def _count_seeds(plant_log):
    counts = {}
    for entry in plant_log:
        seed = entry[1]
        counts[seed] = counts.get(seed, 0) + 1
    seed_names = {
        "0": "sun",
        "1": "twin",
        "2": "melon",
        "3": "winter",
        "4": "gloom",
        "5": "fume",
        "6": "pumpkin",
        "7": "garlic",
        "8": "squash",
        "9": "jalapeno",
    }
    return {seed_names.get(k, k): v for k, v in counts.items()}


@app.local_entrypoint()
def record(
    start_wave: int = 1,
    max_steps: int = 1500,
    cooldown_scale: float = 0.13,
    frame_stride: int = 10,
):
    """Record an episode and save to pvz-gifs volume. Download with `modal volume get pvz-gifs pvz_episode.gif .`"""
    path = record_episode_gif.remote(
        start_wave=start_wave,
        max_steps=max_steps,
        cooldown_scale=cooldown_scale,
        frame_stride=frame_stride,
    )
    print(f"GIF saved at: {path}")
    print("Download: modal volume get pvz-gifs pvz_episode.gif .")


@app.function(
    image=image,
    volumes={"/pvz-assets": assets, "/pvz-checkpoints": checkpoints},
    timeout=7200,
)
def collect_teacher_data(
    episodes: int = 50,
    max_steps: int = 2000,
    cooldown_scale: float = 0.13,
    start_sun: int = 3000,
    suite_name: str = "bootstrap_eval",
    output_dir: str = "/pvz-checkpoints/teacher/bootstrap",
    teacher_name: str = "structured-v1",
    teacher_profile: str = "structured-v2",
    seed_start: int | None = None,
    require_stage_complete: bool = False,
    max_attempts: int | None = None,
):
    """Collect deterministic teacher trajectories under one fixed evaluation condition."""

    sys.path.insert(0, "/opt/pvz")
    sys.path.insert(0, "/opt/pvz/pvz-portable/build")
    from curriculum import CurriculumManager
    from pvz_portable_gym import PvZGymEnv
    from teacher_data import collect_teacher_episode, save_episode_shard

    curriculum = CurriculumManager()
    env = PvZGymEnv(resdir="/pvz-assets", savedir="/pvz-assets/savedata", chdir=True)
    env.set_cooldown_scale(cooldown_scale)
    mask = curriculum.seed_mask(496).astype(bool)
    paths = []
    attempts = 0
    attempt_limit = max_attempts or (
        episodes * 3 if require_stage_complete else episodes
    )
    try:
        while len(paths) < episodes:
            if attempts >= attempt_limit:
                raise RuntimeError(
                    f"collected {len(paths)}/{episodes} successful teacher episodes "
                    f"after {attempts} attempts"
                )
            episode_seed = None if seed_start is None else seed_start + attempts
            reset_options = {"wave": 1, "sun": start_sun}
            if episode_seed is not None:
                reset_options["seed"] = episode_seed
            trajectory = collect_teacher_episode(
                env,
                mask,
                max_steps,
                reset_options=reset_options,
                profile=teacher_profile,
            )
            attempts += 1
            if require_stage_complete and not trajectory["stage_complete"]:
                continue
            path = f"{output_dir}/episode_{len(paths):05d}.npz"
            save_episode_shard(
                path,
                trajectory,
                {
                    "suite": suite_name,
                    "start_wave": 1,
                    "start_sun": start_sun,
                    "cooldown_scale": cooldown_scale,
                    "allowed_seeds": curriculum.current()["allowed_seeds"],
                    "teacher": teacher_name,
                    "seed": episode_seed,
                    "attempt": attempts,
                    "terminal_reason": trajectory["terminal_reason"],
                    "stage_complete": trajectory["stage_complete"],
                },
            )
            paths.append(path)
    finally:
        env.close()
    checkpoints.commit()
    return paths


@app.local_entrypoint()
def collect_teacher(
    episodes: int = 50,
    max_steps: int = 2000,
    cooldown_scale: float = 0.13,
    start_sun: int = 3000,
    suite_name: str = "bootstrap_eval",
    output_dir: str = "/pvz-checkpoints/teacher/bootstrap",
    teacher_name: str = "structured-v1",
    teacher_profile: str = "structured-v2",
    seed_start: int | None = None,
    require_stage_complete: bool = False,
    max_attempts: int | None = None,
):
    print(
        collect_teacher_data.remote(
            episodes=episodes,
            max_steps=max_steps,
            cooldown_scale=cooldown_scale,
            start_sun=start_sun,
            suite_name=suite_name,
            output_dir=output_dir,
            teacher_name=teacher_name,
            teacher_profile=teacher_profile,
            seed_start=seed_start,
            require_stage_complete=require_stage_complete,
            max_attempts=max_attempts,
        )
    )


@app.local_entrypoint()
def collect_teacher_normal_cooldown(episodes: int = 50, max_steps: int = 2000):
    print(
        collect_teacher_data.remote(
            episodes=episodes,
            max_steps=max_steps,
            cooldown_scale=1.0,
            suite_name="teacher_v2_transfer_eval",
            output_dir="/pvz-checkpoints/teacher/normal-cooldown-v2",
            teacher_name="structured-v2",
            teacher_profile="structured-v2",
        )
    )


@app.function(
    image=image,
    gpu="T4",
    volumes={"/pvz-checkpoints": checkpoints},
    timeout=7200,
)
def train_bc(
    epochs: int = 20,
    source_dir: str = "/pvz-checkpoints/teacher/bootstrap",
    checkpoint_name: str = "bc_bootstrap.pt",
    background_weight: float = 0.0,
    seq_len: int = 64,
    batch_size: int = 16,
    init_checkpoint: str = "",
):
    """Train and save a behavior-cloning baseline from one shard directory."""
    import sys

    import torch

    sys.path.insert(0, "/opt/pvz")
    from behavior_cloning import train_behavior_cloning
    from network import PvZActorCritic

    paths = sorted(Path(source_dir).glob("episode_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no teacher shards found in {source_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = PvZActorCritic().to(device)
    init_source = ""
    if init_checkpoint:
        init_path = Path("/pvz-checkpoints") / validate_checkpoint_name(init_checkpoint)
        if not init_path.is_file():
            raise FileNotFoundError(f"init checkpoint not found: {init_path}")
        checkpoint = torch.load(init_path, map_location=device, weights_only=True)
        agent.load_state_dict(checkpoint["agent_state"])
        init_source = str(init_path)
    metrics = train_behavior_cloning(
        agent,
        paths,
        epochs=epochs,
        background_weight=background_weight,
        seq_len=seq_len,
        batch_size=batch_size,
    )
    path = f"/pvz-checkpoints/{checkpoint_name}"
    torch.save(
        {
            "agent_state": agent.state_dict(),
            "bc_metrics": metrics,
            "source_shards": [str(p) for p in paths],
            "init_checkpoint": init_source,
        },
        path,
    )
    checkpoints.commit()
    return {"checkpoint": path, "init_checkpoint": init_source, **metrics}


@app.local_entrypoint()
def train_teacher_bc(
    epochs: int = 20,
    source_dir: str = "/pvz-checkpoints/teacher/bootstrap",
    checkpoint_name: str = "bc_bootstrap.pt",
    background_weight: float = 0.0,
    seq_len: int = 64,
    batch_size: int = 16,
    init_checkpoint: str = "",
):
    print(
        train_bc.remote(
            epochs=epochs,
            source_dir=source_dir,
            checkpoint_name=checkpoint_name,
            background_weight=background_weight,
            seq_len=seq_len,
            batch_size=batch_size,
            init_checkpoint=init_checkpoint,
        )
    )


@app.function(
    image=image,
    volumes={"/pvz-checkpoints": checkpoints},
    timeout=300,
)
def publish_success_distill_report(run_name: str, report: dict):
    """Persist the paired Modal evaluation report beside its sample shards."""
    import json

    if not run_name or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for char in run_name
    ):
        raise ValueError("run_name must be a safe basename")
    path = Path("/pvz-checkpoints/success_distill") / run_name / "evaluation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    checkpoints.commit()
    return str(path)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/pvz-checkpoints": checkpoints},
    timeout=7200,
)
def fine_tune_bc_normal_cooldown(epochs: int = 20):
    """Fine-tune the bootstrap BC actor on normal-cooldown demonstrations."""
    import sys

    import torch

    sys.path.insert(0, "/opt/pvz")
    from behavior_cloning import train_behavior_cloning
    from network import PvZActorCritic
    from teacher_data import split_episode_paths

    checkpoints.reload()
    bootstrap_checkpoint = "/pvz-checkpoints/bc_bootstrap.pt"
    normal_paths = sorted(
        Path("/pvz-checkpoints/teacher/normal-cooldown-v2").glob("episode_*.npz")
    )
    bootstrap_paths = sorted(
        Path("/pvz-checkpoints/teacher/bootstrap").glob("episode_*.npz")
    )
    if not bootstrap_paths:
        raise FileNotFoundError("bootstrap teacher shards are required for retention")
    if not normal_paths:
        raise FileNotFoundError(
            "normal-cooldown teacher shards are required for transfer"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = PvZActorCritic().to(device)
    checkpoint = torch.load(
        bootstrap_checkpoint, map_location=device, weights_only=True
    )
    agent.load_state_dict(checkpoint["agent_state"])
    normal_train_paths, normal_validation_paths = split_episode_paths(normal_paths)
    metrics = train_behavior_cloning(
        agent,
        epochs=epochs,
        train_paths=[*bootstrap_paths, *normal_train_paths],
        validation_paths=normal_validation_paths,
    )
    path = "/pvz-checkpoints/bc_normal_cooldown.pt"
    torch.save(
        {
            "agent_state": agent.state_dict(),
            "bc_metrics": metrics,
            "source_shards": [str(item) for item in [*bootstrap_paths, *normal_paths]],
            "normal_train_episodes": len(normal_train_paths),
            "normal_validation_episodes": len(normal_validation_paths),
        },
        path,
    )
    checkpoints.commit()
    return {
        "checkpoint": path,
        "bootstrap_episodes": len(bootstrap_paths),
        "normal_train_episodes": len(normal_train_paths),
        "normal_validation_episodes": len(normal_validation_paths),
        **metrics,
    }


@app.local_entrypoint()
def bc_normal_cooldown(epochs: int = 20):
    print(fine_tune_bc_normal_cooldown.remote(epochs=epochs))


@app.function(image=image, volumes={"/pvz-assets": assets}, timeout=1800)
def probe_chain_remote():
    """Runtime-verify endless stage chaining on the rebuilt cloud engine (v44+)."""
    import sys

    sys.path.insert(0, "/opt/pvz/pvz-portable/build")
    import pvz_env

    wait = 0
    spatial_sunflower = 2.0

    def plant_action(seed: int, row: int, col: int) -> int:
        return 46 + seed * 45 + row * 9 + col

    env = pvz_env.PvZEnv("/pvz-assets", "/pvz-assets/savedata")
    env.reset(1, 283010)
    env.set_chain_stages(True)

    obs, _, _, _, _, _ = env.step(plant_action(0, 0, 0))
    assert (
        float(obs["spatial"][0][0][0]) == spatial_sunflower  # pi-lens-ignore: unchecked-throwing-call-python
    ), "sunflower planted"
    env.debug_trigger_stage_end()
    obs, _, _, done, truncated, info = env.step(wait)
    assert bool(info["stage_complete"]) and not bool(done) and not bool(truncated)
    assert (
        int(info["stage"]) == 1  # pi-lens-ignore: unchecked-throwing-call-python
        and int(info["wave"]) == 21  # pi-lens-ignore: unchecked-throwing-call-python
    )
    assert (
        int(info["num_waves"]) == 20  # pi-lens-ignore: unchecked-throwing-call-python
    )
    assert (
        float(obs["spatial"][0][0][0]) == spatial_sunflower  # pi-lens-ignore: unchecked-throwing-call-python
    ), "plants persist"

    spawned = False
    for _ in range(80):
        obs, _, _, done, _, info = env.step(wait)
        if (
            float(info["zombie_min_x"])  # pi-lens-ignore: unchecked-throwing-call-python
            < 700.0
        ):
            spawned = True
            break
    assert spawned, "stage-2 zombie spawn"

    env.reset(1, 283010)
    env.set_chain_stages(False)
    obs, _, _, _, _, _ = env.step(plant_action(0, 0, 0))
    env.debug_trigger_stage_end()
    _, _, _, done, _, info = env.step(wait)
    assert bool(info["stage_complete"]) and bool(done), "default mode still terminates"
    return "chain probe OK: advance, persist, spawn, default-terminate"


@app.local_entrypoint()
def probe_chain():
    print(probe_chain_remote.remote())


@app.function(image=image, volumes={"/pvz-assets": assets}, timeout=1800)
def probe_env_v2_remote():
    """Runtime-verify obs v2 channels/globals and deck config on v45+."""
    import sys

    sys.path.insert(0, "/opt/pvz/pvz-portable/build")
    import pvz_env

    env = pvz_env.PvZEnv("/pvz-assets", "/pvz-assets/savedata")
    env.set_obs_version(2)
    env.set_deck([0, 30] + [1] * 8)  # slot 0 peashooter, slot 1 pumpkin
    env.reset(1, 283020)

    obs = env.get_obs()
    assert len(obs["spatial"][0][0]) == 38, "v2 spatial channels"
    assert len(obs["global"]) == 26, "v2 globals"
    # absolute wave 1 of 21, stage 0
    assert abs(float(obs["global"][24]) - 1.0 / 21.0) < 1e-4  # pi-lens-ignore: unchecked-throwing-call-python
    assert float(obs["global"][25]) == 0.0  # pi-lens-ignore: unchecked-throwing-call-python

    # custom deck: slot 0 plants peashooter (seed 0 -> channel value 1.0)
    obs, _, _, _, _, _ = env.step(46)
    assert float(obs["spatial"][0][0][0]) == 1.0  # pi-lens-ignore: unchecked-throwing-call-python

    # v2 normal layer: peashooter revealed beneath the pumpkin overlay
    obs, _, _, _, _, _ = env.step(46 + 45)  # pumpkin slot 1 on the same cell
    assert float(obs["spatial"][0][0][36]) == 1.0  # pi-lens-ignore: unchecked-throwing-call-python
    assert float(obs["spatial"][0][0][37]) > 0.0  # pi-lens-ignore: unchecked-throwing-call-python

    try:
        env.set_deck([99] + [0] * 9)
        raise AssertionError("out-of-range deck accepted")
    except ValueError:
        pass

    return "env v2 probe OK: shapes, globals, deck, stacking layer"


@app.local_entrypoint()
def probe_env_v2():
    print(probe_env_v2_remote.remote())
