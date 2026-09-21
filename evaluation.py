"""Checkpoint evaluation contracts and JSONL result serialization."""

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

EVALUATION_SUITES = (
    {
        "name": "bootstrap_eval",
        "start_wave": 1,
        "start_sun": 3000,
        "cooldown_scale": 0.13,
        "full_deck": False,
        "deterministic": True,
    },
    {
        "name": "transfer_eval",
        "start_wave": 1,
        "start_sun": 3000,
        "cooldown_scale": 1.0,
        "full_deck": False,
        "deterministic": True,
    },
    {
        "name": "zero_eval",
        "start_wave": 1,
        "start_sun": 0,
        "cooldown_scale": 1.0,
        "full_deck": True,
        "deterministic": True,
    },
    {
        "name": "target_eval",
        "start_wave": 1,
        "start_sun": 50,
        "cooldown_scale": 1.0,
        "full_deck": True,
        "deterministic": True,
    },
)


def fixed_evaluation_suites(
    episodes: int = 20, max_steps: int = 2000, seed_start: int | None = 272000
) -> list[dict[str, Any]]:
    """Return fixed-condition suites over the same reproducible seed set."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if seed_start is not None and seed_start < 0:
        raise ValueError("seed_start must be non-negative or None")
    common = {"episodes": episodes, "max_steps": max_steps}
    if seed_start is not None:
        common["seed_start"] = seed_start
    return [{**suite, **common} for suite in EVALUATION_SUITES]


# Deployment conditions beyond the fixed quartet, selectable by name for
# runs whose training regime is not one of the four historical suites.
EXTRA_SUITES = {
    f"bc{sun}_eval": {
        "name": f"bc{sun}_eval",
        "start_wave": 1,
        "start_sun": sun,
        "cooldown_scale": 1.0,
        "full_deck": False,
        "deterministic": True,
    }
    for sun in (1500, 1000, 750, 500, 200, 50, 0)
}

# Endless survival: stage boundaries chain, so max_wave is the absolute
# stage*20+wave endurance measure. Selection gates on mean wave (completion
# is never terminal under chaining).
EXTRA_SUITES["endless_eval"] = {
    "name": "endless_eval",
    "start_wave": 1,
    "start_sun": 3000,
    "cooldown_scale": 1.0,
    "full_deck": False,
    "deterministic": True,
    "chain_stages": True,
}

# Full-deck endless: identical regime without the phase-0 five-seed mask,
# so Winter-Melon/Gloom-trained policies can play their full deck at
# evaluation (evaluate_checkpoint applies the curriculum mask whenever
# full_deck is false).
EXTRA_SUITES["endless_eval_full"] = {
    "name": "endless_eval_full",
    "start_wave": 1,
    "start_sun": 3000,
    "cooldown_scale": 1.0,
    "full_deck": True,
    "deterministic": True,
    "chain_stages": True,
}

# Coffee-deck endless: the corrected deck with Coffee Bean (seed 35) in the
# Garlic slot, so wake-combo policies play their native deck at evaluation.
EXTRA_SUITES["endless_eval_coffee"] = {
    "name": "endless_eval_coffee",
    "start_wave": 1,
    "start_sun": 3000,
    "cooldown_scale": 1.0,
    "full_deck": True,
    "deterministic": True,
    "chain_stages": True,
    "deck": [1, 41, 39, 44, 42, 10, 30, 35, 17, 20],
}

# Narrow-deck endless: the Phase 8 self-discovered mask (Twin Sunflower and
# Pumpkin were driven to exactly zero use) so a consolidation rung evaluates
# in the same narrowed action space it trains with.
EXTRA_SUITES["endless_eval_narrow"] = {
    "name": "endless_eval_narrow",
    "start_wave": 1,
    "start_sun": 3000,
    "cooldown_scale": 1.0,
    "full_deck": False,
    "deterministic": True,
    "chain_stages": True,
    "deck": [1, 41, 39, 44, 42, 10, 30, 35, 17, 20],
    "allowed_seeds": [0, 2, 3, 4, 5, 7, 8, 9],
}


def resolve_evaluation_suites(
    names: Iterable[str],
    episodes: int,
    max_steps: int = 2000,
    seed_start: int | None = None,
) -> list[dict[str, Any]]:
    """Build suites by name from the fixed quartet plus EXTRA_SUITES."""
    catalog = {suite["name"]: suite for suite in EVALUATION_SUITES}
    catalog.update(EXTRA_SUITES)
    resolved = []
    for name in names:
        if name not in catalog:
            raise ValueError(
                f"unknown evaluation suite {name!r}; known: {sorted(catalog)}"
            )
        suite = dict(catalog[name])
        suite["episodes"] = episodes
        suite["max_steps"] = max_steps
        if seed_start is not None:
            suite["seed_start"] = seed_start
        resolved.append(suite)
    return resolved


def checkpoint_result_paths(checkpoint_path: str, suite_name: str) -> tuple[Path, Path]:
    """Return JSONL episode and JSON summary paths beside a checkpoint."""
    checkpoint = Path(checkpoint_path)
    stem = checkpoint.with_suffix("")
    return (
        stem.with_name(f"{stem.name}.{suite_name}.jsonl"),
        stem.with_name(f"{stem.name}.{suite_name}.summary.json"),
    )


def write_evaluation_results(
    checkpoint_path: str,
    suite: dict[str, Any],
    results: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Persist one complete suite's per-episode rows and machine-readable summary."""
    rows = list(results)
    expected_episodes = suite["episodes"]
    if len(rows) != expected_episodes:
        raise ValueError(
            f"expected {expected_episodes} evaluation rows, got {len(rows)}"
        )

    jsonl_path, summary_path = checkpoint_result_paths(checkpoint_path, suite["name"])
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    max_waves = [row["max_wave"] for row in rows]
    losses = [bool(row["lost"]) for row in rows]
    completions = [
        bool(row.get("stage_complete", row.get("terminal_reason") == "stage_complete"))
        for row in rows
    ]
    summary = {
        "checkpoint": Path(checkpoint_path).name,
        "suite": dict(suite),
        "episodes": len(rows),
        "mean_max_wave": sum(max_waves) / len(max_waves),
        "median_max_wave": sorted(max_waves)[len(max_waves) // 2]
        if len(max_waves) % 2
        else (
            sorted(max_waves)[len(max_waves) // 2 - 1]
            + sorted(max_waves)[len(max_waves) // 2]
        )
        / 2,
        "p75_max_wave": sorted(max_waves)[
            min(
                len(max_waves) - 1, int(0.75 * len(max_waves))  # pi-lens-ignore: unchecked-throwing-call-python
            )
        ],
        "max_max_wave": max(max_waves),
        "loss_rate": sum(losses) / len(losses),
        "completion_rate": sum(completions) / len(completions),
        "results_path": str(jsonl_path),
    }
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return summary


def teacher_transfer_gate(
    summary: dict[str, Any], results: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """Classify a normal-cooldown teacher evaluation against the fixed transfer gate."""
    rows = list(results)
    if len(rows) != summary["episodes"]:
        raise ValueError("teacher rows must match the evaluation summary episode count")
    successful_stage_completions = sum(
        # pi-lens-ignore: unchecked-throwing-call-python
        bool(row["stage_complete"]) and int(row["max_wave"]) >= 20
        for row in rows
    )
    success_rate = successful_stage_completions / len(rows)
    return {
        "successful_stage_completions": successful_stage_completions,
        "success_rate": success_rate,
        "median_max_wave": summary["median_max_wave"],
        "passed": success_rate >= 0.80 and summary["median_max_wave"] >= 20,
    }


def paired_teacher_comparison(
    baseline: Iterable[dict[str, Any]], candidate: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """Compare teacher outcomes matched by explicit game seed."""
    # pi-lens-ignore: unchecked-throwing-call-python
    baseline_by_seed = {int(row["seed"]): row for row in baseline}
    # pi-lens-ignore: unchecked-throwing-call-python
    candidate_by_seed = {int(row["seed"]): row for row in candidate}
    if baseline_by_seed.keys() != candidate_by_seed.keys():
        raise ValueError("paired evaluations must contain identical seed sets")
    pairs = []
    for seed in sorted(baseline_by_seed):
        before, after = baseline_by_seed[seed], candidate_by_seed[seed]
        before_complete = before["terminal_reason"] == "stage_complete"
        after_complete = after["terminal_reason"] == "stage_complete"
        pairs.append(
            {
                "seed": seed,
                "baseline_terminal_reason": before["terminal_reason"],
                "candidate_terminal_reason": after["terminal_reason"],
                # pi-lens-ignore: unchecked-throwing-call-python
                "baseline_max_wave": int(before["max_wave"]),
                # pi-lens-ignore: unchecked-throwing-call-python
                "candidate_max_wave": int(after["max_wave"]),
                # pi-lens-ignore: unchecked-throwing-call-python
                "max_wave_delta": int(after["max_wave"]) - int(before["max_wave"]),
                # pi-lens-ignore: unchecked-throwing-call-python
                "completion_delta": int(after_complete) - int(before_complete),
            }
        )
    return {
        "episodes": len(pairs),
        "completion_wins": sum(pair["completion_delta"] > 0 for pair in pairs),
        "completion_losses": sum(pair["completion_delta"] < 0 for pair in pairs),
        "completion_ties": sum(pair["completion_delta"] == 0 for pair in pairs),
        "max_wave_wins": sum(pair["max_wave_delta"] > 0 for pair in pairs),
        "max_wave_losses": sum(pair["max_wave_delta"] < 0 for pair in pairs),
        "mean_max_wave_delta": (
            sum(pair["max_wave_delta"] for pair in pairs) / len(pairs) if pairs else 0.0
        ),
        "pairs": pairs,
    }


def schedule_checkpoint_evaluations(
    checkpoint_path: str,
    suites: Iterable[dict[str, Any]],
    evaluator: Callable[[str, dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run checkpoint suites in order; each caller decides process isolation."""
    return [evaluator(checkpoint_path, dict(suite)) for suite in suites]
