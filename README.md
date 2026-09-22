# PvZ RL Environment

[![CI](https://github.com/awtrisk/pvz-rl-env/actions/workflows/ci.yml/badge.svg)](https://github.com/awtrisk/pvz-rl-env/actions/workflows/ci.yml)

A Gymnasium environment for Plants vs. Zombies survival endless, with a
pybind11 C++ bridge to a headless engine port, a recurrent-PPO training stack,
and bridge-level regression tests guarding engine semantics.

**Bring your own game copy.** PvZ assets are PopCap IP and cannot ship with
this repo. Point the environment at your legally-owned copy's `main.pak` and
`properties/` folder (see [Assets](#assets)).

## Features

- Masked `Discrete(496)` action space (Wait / shovel / plant seed × cell),
  legality computed by the engine's own rules — illegal actions are masked out
  rather than punished
- `Dict` observation: `(5, 9, 38)` spatial tensor (plant IDs/HP, per-zombie-type
  HP, hidden-base-plant layer) + 26 globals (sun, wave/stage, cooldowns, mower
  readiness, per-lane threat)
- Configurable 10-seed deck (any legal seed combination; costs, cooldowns and
  masks follow the seed bank generically)
- Endless mode: chain survival stages with plants/sun/cooldowns persisting and
  zombie HP scaling forever — `set_chain_stages(True)`
- Honest reward modes: dense `legacy` (historical comparability) or `sparse`
  (wave advance +1/stage −1/loss) as a first-class constructor option
- Gymnasium-registered (`PvZ-v0`), passes `check_env`
- 122-test gate including bridge-semantics regressions (mowers spawn and fire,
  upgrades consume base plants, cooldown encoding, no board leaks)

## Build

Requires Python 3.10+ (tested on 3.14), CMake, Ninja, and a C++20 compiler
(gcc 13+ / clang / MSVC; on Windows, [MSYS2 UCRT64](https://www.msys2.org/)
with `mingw-w64-ucrt-x86_64-gcc`, `ninja`, `cmake`, `pybind11` installed).

```bash
python -m pip install -e ".[build]"   # gymnasium, numpy, torch, mambapy, pybind11
cmake -S pvz-portable -B pvz-portable/build -G Ninja -DCMAKE_POSITION_INDEPENDENT_CODE=ON
ninja -C pvz-portable/build
```

The compiled module (`pvz_env*.pyd` / `pvz_env*.so`) is imported from
`pvz-portable/build/` automatically, or place a copy at the repo root.

## Assets

Copy from your legally-owned PvZ installation into `pvz-portable/`:

```
pvz-portable/main.pak
pvz-portable/properties/
```

Save data is written under `pvz-portable/savedata/` (created automatically).

## Checkpoints

A trained recurrent-PPO agent (the endless champion: mean 49.7 / median 50.5 absolute waves at
3000 sun under its training deck, and 72.1 mean with a coffee-deck swap) is hosted at
<https://huggingface.co/awtrisk/pvz-rl-agent>. It's a single `{"agent_state": state_dict}`
payload for `network.PvZActorCritic`:

```python
import torch
from network import PvZActorCritic

agent = PvZActorCritic()
agent.load_state_dict(torch.load("pvz_endless_champion.pt", weights_only=True)["agent_state"])
agent.eval()
```

## Quickstart

```python
import numpy as np
import pvz_portable_gym

env = pvz_portable_gym.PvZGymEnv(
    reward_mode="sparse",
    obs_version=2,
    deck=[1, 41, 39, 44, 42, 10, 30, 36, 17, 20],
)
env.set_chain_stages(True)  # endless mode
obs, info = env.reset(seed=0)

done = truncated = False
while not (done or truncated):
    legal = np.flatnonzero(info["action_mask"])
    obs, reward, done, truncated, info = env.step(int(env.np_random.choice(legal)))

print(f"fell at absolute wave {info['wave']}, stage {info['stage']}")
```

A runnable version is at `examples/quickstart.py`.

## Using your own algorithm

The env is plain Gymnasium — nothing from this repo's training stack is required. Bring any
framework (sb3-contrib, CleanRL, Tianshou, your own). Four things to know:

1. **Legality mask**: `info["action_mask"]` is recomputed every step over the `Discrete(496)`
   space. Use any masked-discrete method; unmasked agents will slam into illegal actions.
2. **One engine per process**: the bridge holds a singleton `gLawnApp`, so vectorize with
   subprocesses (gymnasium's `SubprocVecEnv`, or this repo's `vec_env.PvZVecEnv`), never
   in-process lists.
3. **Partial observability**: plant age, wave rhythm and shovel history are not in the frame —
   go recurrent or frame-stack.
4. **Reward**: start with `reward_mode="sparse"` (survival duration + stage bonuses). The dense
   `"legacy"` shaping exists for comparability with historical runs and needs per-stage tuning.

`examples/sb3_maskable_ppo.py` shows the full hookup with `sb3-contrib`'s `MaskablePPO`, and the
[HuggingFace champion](https://huggingface.co/awtrisk/pvz-rl-agent) loads into any
`network.PvZActorCritic` via its plain `agent_state` dict if you want a sparring partner.

## API

### `PvZGymEnv(render_mode=None, chdir=True, resdir="pvz-portable/", savedir="pvz-portable/savedata/", reward_mode="legacy", max_steps=2000, obs_version=1, deck=None)`

| Parameter | Meaning |
|---|---|
| `reward_mode` | `"legacy"` dense shaped reward (historical comparability) or `"sparse"` = `max(wave advance, 0) + 1·stage_complete − 1·loss` per step; the legacy value remains available as `info["reward_legacy"]` |
| `max_steps` | Wrapper-level truncation (ORed with the bridge's own cap); override per-episode via `reset(options={"max_steps": n})` |
| `obs_version` | `1` = `(5,9,36)` spatial + 24 globals (all existing checkpoints). `2` = `(5,9,38)` + 26 globals, adding the normal-plant layer (the plant hidden under a Pumpkin shell) and absolute wave + survival stage |
| `deck` | List of exactly 10 seed-type ints (0–44); applied on the next `reset` |

### Action space — `Discrete(496)`

| Range | Action |
|---|---|
| `0` | Wait |
| `1 + row*9 + col` | Shovel cell (row 0–4, col 0–8) |
| `46 + seed*45 + row*9 + col` | Plant seed (0–9) at cell |

`info["action_mask"]` (`bool[496]`) marks legal actions from reset and step.

### Observation

`spatial` `(5, 9, 36|38)`: channels 0–2 plant ID (`seedType+1`), plant HP
(0–1), plant state; 3–35 per-zombie-type aggregate HP by column; 36–37 (v2)
normal plant beneath an upgrade/overlay (`seedType+1`, HP fraction).

`global` `(24|26,)`: sun/9990 · wave fraction · 10 seed cooldowns (0=ready) ·
5 mower readiness · 5 nearest-zombie X · sky-sun countdown · zombie countdown
· (v2) absolute wave/(20+1) · survival stage.

### `info` keys (selected)

`wave` (absolute, stage-aware), `stage`, `sun`, `stage_complete`, `lost`,
`reward_legacy`, `action_mask`, `triggered_mowers`, `first_danger_step`.

### Reset contract and process model

- `reset(seed=s, options={"wave": w})` starts from wave `w`; options may also
  carry an explicit engine seed for paired rollouts.
- **One engine per process** (singleton). Multiple `PvZGymEnv` instances
  share it; per-instance `obs_version`/`deck` are applied on each instance's
  own reset.
- The engine retains minor static state across in-process resets. For
  decisive measurements, evaluate in fresh processes (the included evaluation
  harness does exactly this — one process per episode).

## Endless mode

`env.set_chain_stages(True)` — on stage completion the board persists (plants,
sun, cooldowns), the wave state machine resets, the survival stage increments,
and zombie HP scaling continues. Episodes then end only on genuine loss (or
the step cap). `info["wave"]` is the absolute wave (stage × 20 + wave).

## Tests

```bash
python -m unittest discover -s tests   # 122 tests, ~45 s
```

CI (`.github/workflows/ci.yml`) builds the engine and runs the asset-free
subset on every push; the full gate needs your own `main.pak` and `properties/`
and runs locally.

Includes `tests/test_engine_semantics.py`: bridge-level regressions for the
mower initialization, upgrade base-plant consumption, cooldown encoding,
safe-delete draining, stage chaining, obs v2 and deck configuration.

## Training stack

- `ppo.py` — recurrent PPO: sequence-chunked GAE, anchor-KL trust region,
  sparse-reward mode, mixed-difficulty rollouts, hardened selection gates
- `behavior_cloning.py` — minibatch sequence BC with class-balanced weighting
- `vec_env.py` — synchronous vector env (one engine process per worker)
- `curriculum.py`, `evaluation.py` — phase presets and seed-disciplined suites
- `train_modal.py` — [Modal](https://modal.com) cloud entrypoints for
  collection, training and paired evaluation

## Engine / upstream

The `pvz-portable/` directory is a git subtree of
[awtrisk/PvZ-Portable](https://github.com/awtrisk/PvZ-Portable) branch `rl-bridge`
— a fork of [wszqkzqk/PvZ-Portable](https://github.com/wszqkzqk/PvZ-Portable)
carrying the headless pybind11 bridge and the engine fixes (mower init, upgrade
base-plant consumption, cooldown encoding, safe-delete draining, endless stage
chaining, deterministic RNG capture). To pick up upstream engine changes: merge
`wszqkzqk/PvZ-Portable` main into the fork's `rl-bridge`, then:

```bash
git subtree pull --prefix=pvz-portable https://github.com/awtrisk/PvZ-Portable.git rl-bridge --squash
```

## License / IP

This repository is licensed under the GPL-3.0 (see `LICENSE`). The vendored
engine under `pvz-portable/` is
[PvZ-Portable](https://github.com/wszqkzqk/PvZ-Portable) by wszqkzqk,
licensed LGPL-3.0 (see `pvz-portable/LICENSE`); our bridge modifications to
it are released under the same terms. Game assets (`main.pak`,
`properties/`) are PopCap IP and must be supplied by the user from their own
legally-owned copy. Plants vs. Zombies is © PopCap Games.
