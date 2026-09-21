"""Deterministic structured teacher for bootstrap PvZ trajectories."""

import numpy as np

PLANT_ACTION_OFFSET = 46
GRID_CELLS = 45


def _plant_action(seed: int, row: int, col: int) -> int:
    return PLANT_ACTION_OFFSET + seed * GRID_CELLS + row * 9 + col


def preferred_actions(profile: str = "structured-v2") -> tuple[int, ...]:
    """Return the fixed placement order for one named deterministic teacher."""
    layouts = {
        "structured-v2": (
            (0, 0),  # sunflower economy
            (8, 6),  # squash in front of each lane before slow attackers are ready
            (2, 3), (2, 4),  # two melon-pult firing columns
            (6, 3), (6, 4),  # protect the firing columns with pumpkins
            (1, 0),  # twin sunflower upgrades
            (3, 3), (3, 4),  # winter melon upgrades
        ),
        "structured-v3-sun1000": (
            (0, 0), (0, 1),  # two-column sunflower economy
            (8, 6),  # squash emergency line after the economy is established
            (2, 3), (2, 4),  # two melon-pult firing columns
            (6, 3), (6, 4),  # protect the firing columns with pumpkins
            (1, 0),  # twin sunflower upgrades
            (3, 3), (3, 4),  # winter melon upgrades
        ),
    }
    try:
        layout = layouts[profile]
    except KeyError as error:
        raise ValueError(f"unknown teacher profile: {profile}") from error
    return tuple(_plant_action(seed, row, col) for seed, col in layout for row in range(5))



def _profile_stages(profile: str) -> tuple[tuple[int, ...], ...] | None:
    if profile != "structured-v3-sun1000-r3":
        return None
    return (
        tuple(_plant_action(0, row, 0) for row in range(5)),
        tuple(_plant_action(8, row, 6) for row in range(5)),
        tuple(_plant_action(2, row, 3) for row in range(5)),
        tuple(_plant_action(0, row, 1) for row in range(5)),
        tuple(_plant_action(6, row, 3) for row in range(5)),
        tuple(_plant_action(2, row, 4) for row in range(5)),
        tuple(_plant_action(6, row, 4) for row in range(5)),
    )

def _target_is_empty(spatial: np.ndarray, action: int) -> bool:
    _, cell = divmod(action - PLANT_ACTION_OFFSET, GRID_CELLS)
    row, col = divmod(cell, 9)
    return spatial[row, col, 0] <= 0


class StatefulTeacher:
    """Low-sun teacher whose build milestones survive later plant destruction."""

    MIN_ECONOMY = 5
    SQUASH_SEED = 8
    # Deck position is 8, but the bridge emits underlying SeedType + 1 (17 + 1).
    SQUASH_SPATIAL_ID = 18
    SQUASH_COLUMN = 6
    MELON_SEED = 2
    MELON_COLUMNS = (3, 4)
    SUNFLOWER_SEED = 0
    SUNFLOWER_COLUMN = 0
    # 700 HP-distance is roughly one basic zombie two cells from the house.
    # At 100-frame decisions, Squash must fire before the firing line is exposed.
    EMERGENCY_DANGER = 700.0
    # Arm a ready Squash once a threat has crossed half the emergency score.
    # This leaves enough time for its long cooldown before danger reaches 700.
    PRE_EMERGENCY_DANGER = 350.0
    SQUASH_COST = 50
    SUN_NORMALIZER = 9990.0
    GLOBAL_SUN_INDEX = 0
    GLOBAL_COOLDOWN_OFFSET = 2

    def __init__(self, profile: str):
        if profile not in {
            "structured-v4-sun1000",
            "structured-v5-sun1000-cooldown",
        }:
            raise ValueError(f"unknown stateful teacher profile: {profile}")
        self.profile = profile
        self.reset()

    def reset(self) -> None:
        """Clear all per-episode teacher state before a fresh environment reset."""
        self.milestones: set[int] = set()
        self._decision_index = 0
        self._last_squash_lane: int | None = None
        self._last_squash_decision: int | None = None
        self._last_squash_commitment_danger: np.ndarray | None = None
        self._open_squash_commitment: dict[str, object] | None = None
        self._closed_squash_commitments: list[dict[str, object]] = []

    @staticmethod
    def _squash_lane(action: int) -> int | None:
        """Return the placement lane when an action plants Squash."""
        if action < PLANT_ACTION_OFFSET:
            return None
        seed, cell = divmod(action - PLANT_ACTION_OFFSET, GRID_CELLS)
        if seed != StatefulTeacher.SQUASH_SEED:
            return None
        lane, _ = divmod(cell, 9)
        return lane

    @staticmethod
    def _lane_danger(spatial: np.ndarray) -> np.ndarray:
        """Score zombie HP by proximity; column zero is closest to the house."""
        zombie_hp = np.asarray(spatial, dtype=np.float32)[:, :, 3:].sum(axis=2)
        danger = np.zeros(5, dtype=np.float32)
        for lane, lane_hp in enumerate(zombie_hp):
            occupied = np.flatnonzero(lane_hp > 0)
            if occupied.size:
                nearest_col = int(occupied.min())
                danger[lane] = float(lane_hp.sum()) * (9 - nearest_col)
        return danger

    @staticmethod
    def _nearest_zombie_columns(spatial: np.ndarray) -> list[int | None]:
        """Return the house-side-most occupied zombie column in each lane."""
        zombie_hp = np.asarray(spatial, dtype=np.float32)[:, :, 3:].sum(axis=2)
        return [
            int(occupied.min()) if (occupied := np.flatnonzero(lane_hp > 0)).size else None
            for lane_hp in zombie_hp
        ]

    @staticmethod
    def _legal(mask: np.ndarray, action: int) -> bool:
        return action < mask.size and bool(mask[action])

    def _choose_and_latch(self, mask: np.ndarray, action: int) -> int | None:
        if self._legal(mask, action):
            self.milestones.add(action)
            return action
        return None

    def _observe_open_squash_commitment(
        self,
        board: np.ndarray,
        danger: np.ndarray,
        nearest_zombie_columns: list[int | None],
        emergency_lane: int | None,
    ) -> None:
        """Record the current decision against the active Squash commitment."""
        commitment = self._open_squash_commitment
        if commitment is None:
            return
        commitment_lane = int(commitment["lane"])
        action_age = self._decision_index - int(commitment["decision"])
        target = board[commitment_lane, self.SQUASH_COLUMN]
        squash_visible = bool(target[0] == self.SQUASH_SPATIAL_ID)
        commitment["last_observed_danger"] = float(danger[commitment_lane])
        commitment["last_squash_visible_at_target"] = squash_visible
        commitment["last_squash_state_at_target"] = float(target[2]) if squash_visible else None
        commitment["last_squash_health_at_target"] = float(target[1]) if squash_visible else None
        if squash_visible and commitment["first_squash_visible_age"] is None:
            commitment["first_squash_visible_age"] = action_age
        if not squash_visible and commitment["first_squash_absent_age"] is None:
            commitment["first_squash_absent_age"] = action_age
        if action_age == 1:
            commitment["first_post_commit_squash_state"] = (
                float(target[2]) if squash_visible else None
            )
            commitment["first_post_commit_nearest_zombie_column"] = (
                nearest_zombie_columns[commitment_lane]
            )
        if commitment["first_emergency_lane"] is None:
            if float(danger[commitment_lane]) == 0.0:
                commitment["committed_lane_cleared_before_first_emergency"] = True
            if emergency_lane is not None:
                commitment["first_emergency_lane"] = emergency_lane
                commitment["first_emergency_age"] = action_age
                commitment["first_emergency_lane_danger_at_commitment"] = float(
                    commitment["danger_at_commitment"][emergency_lane]
                )
                commitment["committed_lane_danger_at_first_emergency"] = float(
                    danger[commitment_lane]
                )

    @staticmethod
    def _commitment_outcome_category(commitment: dict[str, object]) -> str:
        """Classify a closed commitment using only recorded observations."""
        emergency_lane = commitment["first_emergency_lane"]
        if emergency_lane is None:
            return "no_subsequent_emergency"
        if int(emergency_lane) == int(commitment["lane"]):
            return "same_lane_recurrence"
        if float(commitment["first_emergency_lane_danger_at_commitment"]) > 0.0:
            return "known_competing_lane_emergence"
        return "zero_danger_lane_emergence"

    def _close_open_squash_commitment(self, close_reason: str) -> dict[str, object] | None:
        """Finalize the active commitment exactly once."""
        commitment = self._open_squash_commitment
        if commitment is None:
            return None
        outcome = {
            **commitment,
            "close_reason": close_reason,
            "close_decision": self._decision_index,
            "outcome_category": self._commitment_outcome_category(commitment),
            "squash_seen_after_commitment": commitment["first_squash_visible_age"]
            is not None,
        }
        self._closed_squash_commitments.append(outcome)
        self._open_squash_commitment = None
        return outcome

    def _start_squash_commitment(
        self,
        lane: int,
        danger: np.ndarray,
        nearest_zombie_columns: list[int | None],
        action: int,
    ) -> None:
        """Open a diagnostic-only commitment snapshot after selecting Squash."""
        self._open_squash_commitment = {
            "lane": lane,
            "decision": self._decision_index,
            "danger_at_commitment": danger.astype(float).tolist(),
            "nearest_zombie_column_at_commitment": nearest_zombie_columns[lane],
            "selected_action": action,
            "first_emergency_lane": None,
            "first_emergency_age": None,
            "first_emergency_lane_danger_at_commitment": None,
            "committed_lane_danger_at_first_emergency": None,
            "committed_lane_cleared_before_first_emergency": False,
            "last_observed_danger": float(danger[lane]),
            "last_squash_visible_at_target": None,
            "last_squash_state_at_target": None,
            "last_squash_health_at_target": None,
            "first_squash_visible_age": None,
            "first_squash_absent_age": None,
            "first_post_commit_squash_state": None,
            "first_post_commit_nearest_zombie_column": None,
        }

    def close_episode(self, close_reason: str) -> list[dict[str, object]]:
        """Close the final commitment and return this episode's outcomes."""
        self._close_open_squash_commitment(close_reason)
        return [dict(outcome) for outcome in self._closed_squash_commitments]

    def choose_action(self, action_mask: np.ndarray, spatial: np.ndarray) -> int:
        """Choose a legal scarcity-first action for one environment episode."""
        action, _ = self.choose_action_with_diagnostics(action_mask, spatial)
        return action

    def choose_action_with_diagnostics(
        self,
        action_mask: np.ndarray,
        spatial: np.ndarray,
        global_observation: np.ndarray | None = None,
        squash_targetable_lanes: np.ndarray | None = None,
    ) -> tuple[int, dict[str, object]]:
        """Choose an action and expose the decision inputs used by the teacher."""
        mask = np.asarray(action_mask, dtype=bool)
        board = np.asarray(spatial)
        global_state = (
            None
            if global_observation is None
            else np.asarray(global_observation, dtype=np.float32)
        )
        targetable_lanes = (
            None
            if squash_targetable_lanes is None
            else np.asarray(squash_targetable_lanes, dtype=bool)
        )
        if mask.ndim != 1 or mask.shape[0] < PLANT_ACTION_OFFSET:
            raise ValueError("action_mask must be a one-dimensional flattened action mask")
        if board.shape[:2] != (5, 9) or board.ndim != 3 or board.shape[2] < 4:
            raise ValueError("spatial observations must have shape (5, 9, channels >= 4)")
        if global_state is not None and global_state.shape != (24,):
            raise ValueError("global observations must have shape (24,)")
        if targetable_lanes is not None and targetable_lanes.shape != (5,):
            raise ValueError("squash_targetable_lanes must have shape (5,)")

        danger = self._lane_danger(board)
        nearest_zombie_columns = self._nearest_zombie_columns(board)
        threatened_lanes = np.flatnonzero(danger >= self.EMERGENCY_DANGER)
        emergency_lane = (
            int(threatened_lanes[np.argmax(danger[threatened_lanes])])
            if threatened_lanes.size
            else None
        )
        emergency_action = (
            _plant_action(self.SQUASH_SEED, emergency_lane, self.SQUASH_COLUMN)
            if emergency_lane is not None
            else None
        )
        emergency_squash_legal = (
            self._legal(mask, emergency_action)
            if emergency_action is not None
            else None
        )
        emergency_melon_legal = (
            any(
                self._legal(mask, _plant_action(self.MELON_SEED, emergency_lane, column))
                for column in self.MELON_COLUMNS
            )
            if emergency_lane is not None
            else None
        )
        available_sun = (
            float(global_state[self.GLOBAL_SUN_INDEX] * self.SUN_NORMALIZER)
            if global_state is not None and emergency_lane is not None
            else None
        )
        squash_seed_cooldown = (
            float(global_state[self.GLOBAL_COOLDOWN_OFFSET + self.SQUASH_SEED])
            if global_state is not None and emergency_lane is not None
            else None
        )
        squash_affordable = (
            available_sun >= self.SQUASH_COST if available_sun is not None else None
        )
        squash_target_occupied = (
            bool(board[emergency_lane, self.SQUASH_COLUMN, 0] > 0)
            if emergency_lane is not None
            else None
        )

        last_squash_lane = self._last_squash_lane
        decisions_since_last_squash = (
            self._decision_index - self._last_squash_decision
            if self._last_squash_decision is not None
            else None
        )
        emergency_squash_history_classification = (
            None
            if emergency_lane is None
            else (
                "no_recent_squash"
                if last_squash_lane is None
                else (
                    "same_lane_repeat"
                    if last_squash_lane == emergency_lane
                    else "different_lane_conflict"
                )
            )
        )
        last_squash_commitment_danger = (
            self._last_squash_commitment_danger.astype(float).tolist()
            if self._last_squash_commitment_danger is not None
            else None
        )
        emergency_lane_danger_at_last_squash = (
            float(self._last_squash_commitment_danger[emergency_lane])
            if emergency_lane is not None and self._last_squash_commitment_danger is not None
            else None
        )
        emergency_lane_had_danger_when_squash_committed = (
            emergency_lane_danger_at_last_squash > 0.0
            if emergency_lane_danger_at_last_squash is not None
            else None
        )
        self._observe_open_squash_commitment(
            board, danger, nearest_zombie_columns, emergency_lane
        )
        active_commitment = self._open_squash_commitment
        active_commitment_lane = (
            int(active_commitment["lane"]) if active_commitment is not None else None
        )
        committed_lane_current_danger = (
            float(active_commitment["last_observed_danger"])
            if active_commitment is not None
            else None
        )
        committed_lane_cleared_before_first_emergency = (
            bool(active_commitment["committed_lane_cleared_before_first_emergency"])
            if active_commitment is not None
            else None
        )
        first_emergency_lane_since_squash = (
            active_commitment["first_emergency_lane"]
            if active_commitment is not None
            else None
        )
        first_emergency_age_since_squash = (
            active_commitment["first_emergency_age"]
            if active_commitment is not None
            else None
        )
        committed_squash_visible_at_target = (
            active_commitment["last_squash_visible_at_target"]
            if active_commitment is not None
            else None
        )
        committed_squash_state_at_target = (
            active_commitment["last_squash_state_at_target"]
            if active_commitment is not None
            else None
        )
        committed_squash_health_at_target = (
            active_commitment["last_squash_health_at_target"]
            if active_commitment is not None
            else None
        )

        def finish(action: int) -> tuple[int, dict[str, object]]:
            diagnostic = {
                "lane_danger": danger.astype(float).tolist(),
                "nearest_zombie_columns": nearest_zombie_columns,
                "squash_targetable_lanes": (
                    targetable_lanes.tolist() if targetable_lanes is not None else None
                ),
                "emergency_lane": emergency_lane,
                "emergency_squash_legal": emergency_squash_legal,
                "emergency_melon_legal": emergency_melon_legal,
                "squash_seed_cooldown": squash_seed_cooldown,
                "available_sun": available_sun,
                "squash_affordable": squash_affordable,
                "squash_target_occupied": squash_target_occupied,
                "last_squash_lane": last_squash_lane,
                "decisions_since_last_squash": decisions_since_last_squash,
                "emergency_squash_history_classification": emergency_squash_history_classification,
                "last_squash_commitment_lane": last_squash_lane,
                "last_squash_commitment_decision": self._last_squash_decision,
                "last_squash_commitment_danger": last_squash_commitment_danger,
                "emergency_lane_danger_at_last_squash": emergency_lane_danger_at_last_squash,
                "emergency_lane_had_danger_when_squash_committed": emergency_lane_had_danger_when_squash_committed,
                "active_squash_commitment_lane": active_commitment_lane,
                "committed_lane_current_danger": committed_lane_current_danger,
                "committed_lane_cleared_before_first_emergency": committed_lane_cleared_before_first_emergency,
                "first_emergency_lane_since_squash": first_emergency_lane_since_squash,
                "first_emergency_age_since_squash": first_emergency_age_since_squash,
                "committed_squash_visible_at_target": committed_squash_visible_at_target,
                "committed_squash_state_at_target": committed_squash_state_at_target,
                "committed_squash_health_at_target": committed_squash_health_at_target,
                "selected_action": action,
            }
            selected_squash_lane = self._squash_lane(action)
            if selected_squash_lane is not None:
                closed_commitment = self._close_open_squash_commitment("next_squash")
                diagnostic["closed_squash_commitment"] = closed_commitment
                self._last_squash_lane = selected_squash_lane
                self._last_squash_decision = self._decision_index
                self._last_squash_commitment_danger = danger.copy()
                self._start_squash_commitment(
                    selected_squash_lane, danger, nearest_zombie_columns, action
                )
            self._decision_index += 1
            return action, diagnostic

        if threatened_lanes.size:
            ordered_threatened_lanes = threatened_lanes[
                np.argsort(danger[threatened_lanes])[::-1]
            ]
            for lane in ordered_threatened_lanes:
                if targetable_lanes is not None and not targetable_lanes[lane]:
                    continue
                action = _plant_action(self.SQUASH_SEED, int(lane), self.SQUASH_COLUMN)
                chosen = self._choose_and_latch(mask, action)
                if chosen is not None:
                    return finish(chosen)

        if self.profile == "structured-v5-sun1000-cooldown":
            pre_emergency_lanes = np.flatnonzero(
                (danger >= self.PRE_EMERGENCY_DANGER)
                & (danger < self.EMERGENCY_DANGER)
            )
            for lane in pre_emergency_lanes[np.argsort(danger[pre_emergency_lanes])[::-1]]:
                if targetable_lanes is not None and not targetable_lanes[lane]:
                    continue
                action = _plant_action(self.SQUASH_SEED, int(lane), self.SQUASH_COLUMN)
                chosen = self._choose_and_latch(mask, action)
                if chosen is not None:
                    return finish(chosen)

        economy = tuple(
            _plant_action(self.SUNFLOWER_SEED, lane, self.SUNFLOWER_COLUMN)
            for lane in range(5)
        )
        if sum(action in self.milestones for action in economy) < self.MIN_ECONOMY:
            for action in economy:
                if action not in self.milestones:
                    chosen = self._choose_and_latch(mask, action)
                    if chosen is not None:
                        return finish(chosen)


        for column in self.MELON_COLUMNS:
            for lane in np.argsort(danger)[::-1]:
                action = _plant_action(self.MELON_SEED, int(lane), column)
                chosen = self._choose_and_latch(mask, action)
                if chosen is not None:
                    return finish(chosen)

        # A destroyed Sunflower is an economy repair, not a minimum-economy
        # milestone. Defer it until no lane contains zombies.
        if np.any(board[:, :, 3:] > 0):
            if mask[0]:
                return finish(0)
            raise ValueError("effective action mask has no legal wait action")

        for action in economy:
            if action in self.milestones and _target_is_empty(board, action):
                chosen = self._choose_and_latch(mask, action)
                if chosen is not None:
                    return finish(chosen)

        if mask[0]:
            return finish(0)
        raise ValueError("effective action mask has no legal wait action")

PREFERRED_ACTIONS = preferred_actions()


def choose_action(
    action_mask: np.ndarray,
    profile: str = "structured-v2",
    spatial: np.ndarray | None = None,
) -> int:
    """Choose one legal action from the effective environment × curriculum mask."""
    mask = np.asarray(action_mask, dtype=bool)
    if mask.ndim != 1 or mask.shape[0] < PLANT_ACTION_OFFSET:
        raise ValueError("action_mask must be a one-dimensional flattened action mask")
    stages = _profile_stages(profile)
    if stages is not None:
        if spatial is None:
            raise ValueError(f"{profile} requires spatial observations for staged action selection")
        board = np.asarray(spatial)
        if board.shape[:2] != (5, 9):
            raise ValueError("spatial observations must begin with shape (5, 9)")
        for stage in stages:
            missing = tuple(action for action in stage if _target_is_empty(board, action))
            if not missing:
                continue
            for action in missing:
                if mask[action]:
                    return action
            return 0
    for action in preferred_actions(profile):
        if action < mask.size and mask[action]:
            return action
    if mask[0]:
        return 0
    raise ValueError("effective action mask has no legal wait action")


def choose_actions(
    action_masks: np.ndarray,
    profile: str = "structured-v2",
    spatials: np.ndarray | None = None,
) -> np.ndarray:
    """Choose a deterministic legal action for every vector-environment slot."""
    masks = np.asarray(action_masks, dtype=bool)
    if masks.ndim != 2:
        raise ValueError("action_masks must have shape (num_envs, num_actions)")
    if spatials is not None and len(spatials) != masks.shape[0]:
        raise ValueError("spatials must contain one observation per action mask")
    return np.fromiter(
        (
            choose_action(mask, profile, None if spatials is None else spatials[index])
            for index, mask in enumerate(masks)
        ),
        dtype=np.int64,
        count=masks.shape[0],
    )
