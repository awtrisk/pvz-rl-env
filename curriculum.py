"""Reverse curriculum manager for PvZ: Survival Endless.

The curriculum is now a staged action-space + difficulty curriculum:

    Phase 0: sunflower + fumeshroom, wave 1, 3000 sun  (economy + basic defense)
    Phase 1: + pumpkin,                wave 1, 2000 sun  (defensive support)
    Phase 2: + melonpult,             wave 1, 1500 sun  (heavy attacker)
    Phase 3: + wintermelon + garlic,  wave 1, 1500 sun  (slow + lane control)
    Phase 4: all seeds,               wave 1, 1000 sun  (full strategic deck)
    Phase 5: all seeds,              wave 10, 1500 sun
    Phase 6: all seeds,              wave 20, 3000 sun
    Phase 7: all seeds,              wave 30, 6000 sun
    Phase 8: all seeds,              wave 50, 9990 sun
    Phase 9: all seeds,               wave 1,   50 sun  (true from-scratch)

The early phases restrict the action space to a small subset of seeds so the
agent can discover economy and defense before it must reason about the full
deck. Wait and shovel are always allowed. The later phases mirror the original
wave-based reverse curriculum, ending with a true from-scratch condition.

Promotion:  rolling window of 50 episodes, win rate >= 90%
Demotion:   rolling window of 20 episodes, win rate < 60%

A win is defined as surviving WAVES_TO_SURVIVE waves beyond the phase's
starting wave.
"""

from collections import deque
from typing import Any

import numpy as np

NUM_SEEDS = 10
GRID_ROWS = 5
GRID_COLS = 9
NUM_PLANT_ACTIONS = GRID_ROWS * GRID_COLS
NUM_SHOVEL_ACTIONS = GRID_ROWS * GRID_COLS
WAIT_ACTION = 0
PLANT_ACTION_OFFSET = 1 + NUM_SHOVEL_ACTIONS

# Seed indices in the PyBridge action space.
SEED_SUNFLOWER = 0
SEED_TWINSUNFLOWER = 1
SUNFLOWER_SPATIAL_ID = 2
TWINSUNFLOWER_SPATIAL_ID = 42
SEED_MELONPULT = 2
SEED_WINTERMELON = 3
SEED_GLOOMSHROOM = 4
SEED_FUMESHROOM = 5
SEED_PUMPKINSHELL = 6
SEED_GARLIC = 7
SEED_SQUASH = 8
SEED_JALAPENO = 9


# Staged action-space + difficulty curriculum.
# Phase 0-2 progressively unlock defense and emergency plants so the agent can
# handle realistic Survival Endless waves (bucketheads, pole vaulters, etc).
# Mushrooms (gloom, fume) are excluded — they're night-only and Survival Endless
# is a day level, so the C++ action mask never allows them regardless.
PHASES = [
    {
        "start_wave": 1,
        "start_sun": 3000,
        "allowed_seeds": [
            SEED_SUNFLOWER,
            SEED_TWINSUNFLOWER,
            SEED_MELONPULT,
            SEED_PUMPKINSHELL,
            SEED_SQUASH,
        ],
    },
    {
        "start_wave": 1,
        "start_sun": 2000,
        "allowed_seeds": [
            SEED_SUNFLOWER,
            SEED_TWINSUNFLOWER,
            SEED_MELONPULT,
            SEED_WINTERMELON,
            SEED_PUMPKINSHELL,
            SEED_SQUASH,
            SEED_JALAPENO,
        ],
    },
    {
        "start_wave": 1,
        "start_sun": 1500,
        "allowed_seeds": [
            SEED_SUNFLOWER,
            SEED_TWINSUNFLOWER,
            SEED_MELONPULT,
            SEED_WINTERMELON,
            SEED_PUMPKINSHELL,
            SEED_SQUASH,
            SEED_JALAPENO,
            SEED_GARLIC,
        ],
    },
    {"start_wave": 1, "start_sun": 1000, "allowed_seeds": list(range(NUM_SEEDS))},
    {"start_wave": 10, "start_sun": 1500, "allowed_seeds": list(range(NUM_SEEDS))},
    {"start_wave": 20, "start_sun": 3000, "allowed_seeds": list(range(NUM_SEEDS))},
    {"start_wave": 30, "start_sun": 6000, "allowed_seeds": list(range(NUM_SEEDS))},
    {"start_wave": 50, "start_sun": 9990, "allowed_seeds": list(range(NUM_SEEDS))},
    {"start_wave": 1, "start_sun": 50, "allowed_seeds": list(range(NUM_SEEDS))},
]

# Zero-sun defense curriculum. Every phase has the full deck, but early phases
# gate offense until the board has a small economy. Emergency seeds remain
# available so the gate cannot strand the policy with no defense.
ZERO_DEFENSE_PHASES = [
    {
        "start_wave": 1,
        "start_sun": 0,
        "allowed_seeds": list(range(NUM_SEEDS)),
        "economy_floor": 3,
        "waves_to_survive": 1,
    },
    {
        "start_wave": 1,
        "start_sun": 0,
        "allowed_seeds": list(range(NUM_SEEDS)),
        "economy_floor": 5,
        "waves_to_survive": 3,
    },
    {
        "start_wave": 1,
        "start_sun": 0,
        "allowed_seeds": list(range(NUM_SEEDS)),
        "economy_floor": 5,
        "waves_to_survive": 6,
    },
    {
        "start_wave": 1,
        "start_sun": 0,
        "allowed_seeds": list(range(NUM_SEEDS)),
        "economy_floor": 5,
        "waves_to_survive": 11,
    },
    {
        "start_wave": 1,
        "start_sun": 0,
        "allowed_seeds": list(range(NUM_SEEDS)),
        "waves_to_survive": 19,
    },
]

# Same full-deck policy, but lower starting sun only after the learned high-sun
# policy has demonstrated each intermediate budget. The numeric phase positions
# intentionally keep index 3 near the old 1000-sun curriculum checkpoint.
ZERO_SUN_RAMP_PHASES = [
    {
        "start_wave": 1,
        "start_sun": 3000,
        "allowed_seeds": list(range(NUM_SEEDS)),
        "waves_to_survive": 5,
    },
    {
        "start_wave": 1,
        "start_sun": 1500,
        "allowed_seeds": list(range(NUM_SEEDS)),
        "waves_to_survive": 5,
    },
    {
        "start_wave": 1,
        "start_sun": 1000,
        "allowed_seeds": list(range(NUM_SEEDS)),
        "waves_to_survive": 5,
    },
    {
        "start_wave": 1,
        "start_sun": 750,
        "allowed_seeds": list(range(NUM_SEEDS)),
        "waves_to_survive": 5,
    },
    {
        "start_wave": 1,
        "start_sun": 500,
        "allowed_seeds": list(range(NUM_SEEDS)),
        "waves_to_survive": 5,
    },
    {
        "start_wave": 1,
        "start_sun": 300,
        "allowed_seeds": list(range(NUM_SEEDS)),
        "waves_to_survive": 5,
    },
    {
        "start_wave": 1,
        "start_sun": 150,
        "allowed_seeds": list(range(NUM_SEEDS)),
        "waves_to_survive": 5,
    },
    {
        "start_wave": 1,
        "start_sun": 50,
        "allowed_seeds": list(range(NUM_SEEDS)),
        "waves_to_survive": 5,
    },
    {
        "start_wave": 1,
        "start_sun": 0,
        "allowed_seeds": list(range(NUM_SEEDS)),
        "waves_to_survive": 19,
    },
]

# The ramp above changes only the resource distribution. This companion ramp
# also gives PPO a short, observable build target: establish economy, then build
# sustained Melon/Wintermelon offense before returning to the full action space.
ZERO_MEMORY_PHASES = [
    {
        "start_wave": 1,
        "start_sun": 0,
        "allowed_seeds": list(range(NUM_SEEDS)),
        "economy_floor": 5,
        "waves_to_survive": 6,
        "one_way": True,
    },
    {
        "start_wave": 1,
        "start_sun": 0,
        "allowed_seeds": list(range(NUM_SEEDS)),
        "economy_floor": 5,
        "waves_to_survive": 9,
        "one_way": True,
    },
    {
        "start_wave": 1,
        "start_sun": 0,
        "allowed_seeds": list(range(NUM_SEEDS)),
        "economy_floor": 5,
        "waves_to_survive": 12,
        "one_way": True,
    },
    {
        "start_wave": 1,
        "start_sun": 0,
        "allowed_seeds": list(range(NUM_SEEDS)),
        "economy_floor": 5,
        "waves_to_survive": 15,
        "one_way": True,
    },
    {
        "start_wave": 1,
        "start_sun": 0,
        "allowed_seeds": list(range(NUM_SEEDS)),
        "economy_floor": 5,
        "waves_to_survive": 19,
        "one_way": True,
    },
]

ZERO_OFFENSE_RAMP_PHASES = [
    {
        "start_wave": 1,
        "start_sun": 3000,
        "allowed_seeds": list(range(NUM_SEEDS)),
        "economy_floor": 3,
        "offense_floor": 6,
        "waves_to_survive": 5,
    },
    {
        "start_wave": 1,
        "start_sun": 1500,
        "allowed_seeds": list(range(NUM_SEEDS)),
        "economy_floor": 4,
        "offense_floor": 6,
        "waves_to_survive": 5,
    },
    {
        "start_wave": 1,
        "start_sun": 1000,
        "allowed_seeds": list(range(NUM_SEEDS)),
        "economy_floor": 5,
        "offense_floor": 6,
        "waves_to_survive": 5,
    },
    {
        "start_wave": 1,
        "start_sun": 750,
        "allowed_seeds": list(range(NUM_SEEDS)),
        "economy_floor": 5,
        "offense_floor": 6,
        "waves_to_survive": 5,
    },
    {
        "start_wave": 1,
        "start_sun": 500,
        "allowed_seeds": list(range(NUM_SEEDS)),
        "economy_floor": 5,
        "offense_floor": 6,
        "waves_to_survive": 5,
    },
    {
        "start_wave": 1,
        "start_sun": 300,
        "allowed_seeds": list(range(NUM_SEEDS)),
        "economy_floor": 5,
        "offense_floor": 6,
        "waves_to_survive": 5,
    },
    {
        "start_wave": 1,
        "start_sun": 150,
        "allowed_seeds": list(range(NUM_SEEDS)),
        "economy_floor": 5,
        "offense_floor": 6,
        "waves_to_survive": 5,
    },
    {
        "start_wave": 1,
        "start_sun": 50,
        "allowed_seeds": list(range(NUM_SEEDS)),
        "economy_floor": 5,
        "offense_floor": 6,
        "waves_to_survive": 5,
    },
    {
        "start_wave": 1,
        "start_sun": 0,
        "allowed_seeds": list(range(NUM_SEEDS)),
        "economy_floor": 5,
        "offense_floor": 6,
        "waves_to_survive": 19,
    },
]

PROMOTION_WINDOW = 50
PROMOTION_RATE = 0.90
DEMOTION_WINDOW = 20
DEMOTION_RATE = 0.60
# 5 waves beyond start = wave 6 from phase 0. The BC heuristic reliably
# reaches wave 15+, so this is achievable but non-trivial for PPO to learn.
WAVES_TO_SURVIVE = 5

# Absolute wave at which the engine natively ends a survival stage (the
# bridge reports done with stage_complete once the final wave is cleared).
# Curriculum targets at or beyond this wave must not synthesize an early
# done — the env terminates the episode natively.
NATIVE_STAGE_COMPLETION_WAVE = 20


class CurriculumManager:
    """Tracks per-episode outcomes and selects the starting wave and seed set.

    Args:
        phases: list of dicts with ``start_wave``, ``start_sun``, and
                ``allowed_seeds``. Defaults to the staged action-space curriculum.
        promotion_window: number of episodes required to evaluate promotion.
        promotion_rate: win rate required to promote.
        demotion_window: number of episodes required to evaluate demotion.
        demotion_rate: win rate below which to demote.
        waves_to_survive: waves beyond the start wave that count as a win.
        initial_start_wave: if given, resume at the phase with this start_wave.
    """

    def __init__(
        self,
        phases=None,
        promotion_window: int = PROMOTION_WINDOW,
        promotion_rate: float = PROMOTION_RATE,
        demotion_window: int = DEMOTION_WINDOW,
        demotion_rate: float = DEMOTION_RATE,
        waves_to_survive: int = WAVES_TO_SURVIVE,
        initial_start_wave=None,
    ):
        if phases is None:
            phases = PHASES
        self.phases = list(phases)
        self.phase_index = 0
        if initial_start_wave is not None:
            for i, phase in enumerate(self.phases):
                if phase["start_wave"] == initial_start_wave:
                    self.phase_index = i
                    break
        self.promotion_window = promotion_window
        self.promotion_threshold = promotion_rate
        self.demotion_window = demotion_window
        self.demotion_threshold = demotion_rate
        self.waves_to_survive = waves_to_survive

        self._window = deque(maxlen=promotion_window)
        self._demotion_window = deque(maxlen=demotion_window)
        self._promotions = 0
        self._demotions = 0

    def current(self) -> dict[str, Any]:
        """Return the current phase dict."""
        return self.phases[self.phase_index]

    @property
    def start_wave(self) -> int:
        return self.current()["start_wave"]

    @property
    def start_sun(self) -> int:
        return self.current()["start_sun"]

    @property
    def target_wave(self) -> int:
        # pi-lens-ignore: unchecked-throwing-call-python
        return self.start_wave + int(
            self.current().get("waves_to_survive", self.waves_to_survive)
        )

    def seed_mask(self, num_actions: int) -> np.ndarray:
        """Return a 1-D action mask disabling disallowed seeds for this phase.

        Wait and shovel are always allowed. Plant actions for seeds not in the
        current phase are disabled.
        """
        allowed = set(self.current().get("allowed_seeds", list(range(NUM_SEEDS))))
        mask = np.zeros(num_actions, dtype=np.float32)
        mask[WAIT_ACTION] = 1.0
        mask[1 : 1 + NUM_SHOVEL_ACTIONS] = 1.0
        for seed in allowed:
            start = PLANT_ACTION_OFFSET + seed * NUM_PLANT_ACTIONS
            end = start + NUM_PLANT_ACTIONS
            mask[start:end] = 1.0
        return mask

    def action_mask(self, env_mask: np.ndarray, spatial: np.ndarray) -> np.ndarray:
        """Combine legal actions with the phase's temporary training subgoals."""
        env_mask = np.asarray(env_mask, dtype=bool)
        if (
            env_mask.ndim != 2
            or env_mask.shape[1] != PLANT_ACTION_OFFSET + NUM_SEEDS * NUM_PLANT_ACTIONS
        ):
            raise ValueError("env_mask must have shape (batch, 496)")
        base = np.logical_and(
            env_mask,
            self.seed_mask(env_mask.shape[1]).astype(bool),
        )
        # pi-lens-ignore: unchecked-throwing-call-python
        economy_floor = int(self.current().get("economy_floor", 0))
        # pi-lens-ignore: unchecked-throwing-call-python
        offense_floor = int(self.current().get("offense_floor", 0))
        if economy_floor <= 0 and offense_floor <= 0:
            return base.astype(np.float32)

        board = np.asarray(spatial, dtype=np.float32)
        if board.ndim == 3:
            board = board[None, ...]
        if board.ndim != 4 or board.shape[1:3] != (GRID_ROWS, GRID_COLS):
            raise ValueError("spatial must have shape (batch, 5, 9, channels)")

        sunflower_actions = np.array(
            [
                PLANT_ACTION_OFFSET
                + SEED_SUNFLOWER * NUM_PLANT_ACTIONS
                + row * GRID_COLS
                for row in range(GRID_ROWS)
            ],
            dtype=np.int64,
        )
        economy_actions = np.concatenate(
            [
                sunflower_actions,
                np.array(
                    [
                        PLANT_ACTION_OFFSET
                        + SEED_TWINSUNFLOWER * NUM_PLANT_ACTIONS
                        + row * GRID_COLS
                        for row in range(GRID_ROWS)
                    ],
                    dtype=np.int64,
                ),
            ]
        )
        offense_actions = np.concatenate(
            [
                np.arange(
                    PLANT_ACTION_OFFSET + seed * NUM_PLANT_ACTIONS,
                    PLANT_ACTION_OFFSET + (seed + 1) * NUM_PLANT_ACTIONS,
                    dtype=np.int64,
                )
                for seed in (SEED_MELONPULT, SEED_WINTERMELON)
            ]
        )

        for batch in range(board.shape[0]):
            # pi-lens-ignore: unchecked-throwing-call-python
            sunflowers = int(
                np.count_nonzero(board[batch, :, :, 0] == SUNFLOWER_SPATIAL_ID)
            )
            # pi-lens-ignore: unchecked-throwing-call-python
            twin_sunflowers = int(
                np.count_nonzero(board[batch, :, :, 0] == TWINSUNFLOWER_SPATIAL_ID)
            )
            economy_score = sunflowers + 2 * twin_sunflowers
            # pi-lens-ignore: unchecked-throwing-call-python
            offense = int(
                np.count_nonzero(np.isin(board[batch, :, :, 0], (40.0, 45.0)))
            )
            zombie_hp = board[batch, :, :, 3:].sum(axis=2)
            danger = []
            for lane in zombie_hp:
                occupied = np.flatnonzero(lane > 0)
                # pi-lens-ignore: unchecked-throwing-call-python
                danger.append(
                    # pi-lens-ignore: unchecked-throwing-call-python
                    float(lane.sum() * (GRID_COLS - occupied.min()))
                    if occupied.size
                    else 0.0
                )
            emergency = max(danger, default=0.0) >= 700.0

            gate = np.zeros(base.shape[1], dtype=bool)
            if emergency:
                # Emergency defense takes precedence over economy/offense shaping.
                gate[WAIT_ACTION] = True
                gate[1 : 1 + NUM_SHOVEL_ACTIONS] = True
                for row in range(GRID_ROWS):
                    for seed in (SEED_SQUASH, SEED_JALAPENO):
                        gate[
                            PLANT_ACTION_OFFSET
                            + seed * NUM_PLANT_ACTIONS
                            + row * GRID_COLS
                            + 6
                        ] = True
            elif economy_score < economy_floor:
                if sunflowers == 0 and np.any(base[batch, sunflower_actions]):
                    gate[sunflower_actions] = True
                elif np.any(base[batch, economy_actions]):
                    gate[economy_actions] = True
                else:
                    gate[WAIT_ACTION] = True
                    gate[1 : 1 + NUM_SHOVEL_ACTIONS] = True
            elif offense < offense_floor:
                if np.any(base[batch, offense_actions]):
                    gate[offense_actions] = True
                else:
                    gate[WAIT_ACTION] = True
                    gate[1 : 1 + NUM_SHOVEL_ACTIONS] = True
            else:
                continue
            base[batch] &= gate
        return base.astype(np.float32)

    def is_win(self, max_wave_reached: int) -> bool:
        """Return True if the episode reached the target wave."""
        return max_wave_reached >= self.target_wave

    def record(self, outcome: bool) -> bool:
        """Record one episode outcome and update phase if thresholds are met.

        Returns:
            True if the phase index actually changed.
        """
        self._window.append(bool(outcome))
        self._demotion_window.append(bool(outcome))

        # Compare the index before and after: _advance() is a no-op at the
        # first/last phase (it never clears the windows there), so reporting
        # "changed" from the threshold check alone made ppo.train reset all
        # live workers on every episode once the final phase was reached.
        previous_index = self.phase_index
        if self._should_promote():
            self._advance(1)
        elif not self.current().get("one_way", False) and self._should_demote():
            self._advance(-1)
        return self.phase_index != previous_index

    def _advance(self, delta: int):
        """Move to the next or previous phase and reset statistics."""
        new_index = max(0, min(len(self.phases) - 1, self.phase_index + delta))
        if new_index == self.phase_index:
            return
        self.phase_index = new_index
        self._window.clear()
        self._demotion_window.clear()
        if delta > 0:
            self._promotions += 1
        else:
            self._demotions += 1

    def state_dict(self) -> dict[str, Any]:
        """Return serializable state for checkpointing."""
        return {
            "phase_index": self.phase_index,
            "window": list(self._window),
            "demotion_window": list(self._demotion_window),
            "promotions": self._promotions,
            "demotions": self._demotions,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state from a checkpoint."""
        self.phase_index = max(
            0, min(len(self.phases) - 1, state.get("phase_index", 0))
        )
        self._window.clear()
        self._window.extend(state.get("window", []))
        self._demotion_window.clear()
        self._demotion_window.extend(state.get("demotion_window", []))
        self._promotions = state.get("promotions", 0)
        self._demotions = state.get("demotions", 0)

    def _should_promote(self) -> bool:
        return len(self._window) >= self.promotion_window and (
            sum(self._window) / len(self._window) >= self.promotion_threshold
        )

    def _should_demote(self) -> bool:
        return len(self._demotion_window) >= self.demotion_window and (
            sum(self._demotion_window) / len(self._demotion_window)
            < self.demotion_threshold
        )

    def win_rate(self) -> float:
        return (sum(self._window) / len(self._window)) if self._window else 0.0

    def demotion_rate(self) -> float:
        return (
            (sum(self._demotion_window) / len(self._demotion_window))
            if self._demotion_window
            else 0.0
        )

    def episodes_recorded(self) -> int:
        return len(self._window)

    def stats(self) -> dict[str, Any]:
        """Return current statistics for logging."""
        return {
            "phase_index": self.phase_index,
            "start_wave": self.start_wave,
            "start_sun": self.start_sun,
            "target_wave": self.target_wave,
            "allowed_seeds": self.current().get(
                "allowed_seeds", list(range(NUM_SEEDS))
            ),
            "promotion_window_size": len(self._window),
            "demotion_window_size": len(self._demotion_window),
            "win_rate": self.win_rate(),
            "demotion_rate": self.demotion_rate(),
            "promotions": self._promotions,
            "demotions": self._demotions,
        }
