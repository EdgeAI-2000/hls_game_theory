"""Audit strict-FFR E2/E3 cells before they can be treated as formal evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from experiment_trace import write_json


METHOD = {
    "L0": "llm_base",
    "L1": "llm_reactive_z",
    "L1_z_bphi": "llm_reactive_z_bphi",
}


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    actual = path if path.is_file() else path.with_suffix(path.suffix + ".gz")
    opener = gzip.open if actual.suffix == ".gz" else open
    with opener(actual, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    cells: dict[str, Any] = {}
    seed_sets: dict[tuple[str, int], set[int]] = {}
    artifact_hashes: dict[str, dict[str, set[str]]] = {}

    manifest = json.loads(args.curriculum_manifest.read_text(encoding="utf-8"))
    manifest_sha256 = hashlib.sha256(args.curriculum_manifest.read_bytes()).hexdigest()
    if args.curriculum_sha256 and manifest_sha256 != args.curriculum_sha256:
        failures.append("curriculum manifest SHA mismatch")
    expected_gate_sha256 = {}
    for item in args.expected_gate_sha256:
        if "=" not in item:
            failures.append(f"invalid expected gate SHA entry: {item}")
            continue
        variant, expected_sha = item.split("=", 1)
        expected_gate_sha256[variant] = expected_sha
    expected_opponents = {
        str(item["id"])
        for item in manifest.get("opponents", [])
        if item.get("split") == args.curriculum_split
    }

    for variant in args.variants:
        method = METHOD.get(variant, "llm_mask")
        artifact_hashes[variant] = {}
        for outer_seed in args.seeds:
            cell = args.root / variant / f"seed_{outer_seed}"
            label = f"{variant}/seed_{outer_seed}"
            summary_path = cell / "gate1_summary.json"
            if not summary_path.is_file():
                failures.append(f"{label}: missing gate1_summary.json")
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            row = (summary.get("E2_ladder") or {}).get(method)
            if not isinstance(row, dict):
                failures.append(f"{label}: missing E2_ladder.{method}")
                continue

            games = list(iter_jsonl(cell / "gate1_games.jsonl"))
            game_seeds = {int(game["seed"]) for game in games}
            expected_game_seeds = {
                outer_seed * 1_000_000 + game_index
                for game_index in range(args.games)
            }
            seed_sets[(variant, outer_seed)] = game_seeds
            unsettled = sum(
                game.get("end_reason") == "max_steps_or_unsettled" for game in games
            )
            max_primary_opportunities = max(
                (int(game.get("false_fold_opportunities", 0)) for game in games),
                default=0,
            )
            if len(games) != args.games:
                failures.append(f"{label}: {len(games)} games, expected {args.games}")
            if len(game_seeds) != args.games:
                failures.append(f"{label}: game seeds are not unique")
            if game_seeds != expected_game_seeds:
                failures.append(f"{label}: game seeds do not match the frozen schedule")
            if unsettled:
                failures.append(f"{label}: {unsettled} max_steps_or_unsettled games")
            if max_primary_opportunities > 3:
                failures.append(
                    f"{label}: primary FFR opportunities exceed three per game"
                )

            ffr_events = list(iter_jsonl(cell / "gate1_ffr_events.jsonl"))
            false_folds = list(iter_jsonl(cell / "gate1_false_folds.jsonl"))
            counts = row.get("FFR_counts") or {}
            expected_opps = int(counts.get("false_fold_opportunities", 0))
            expected_false = int(counts.get("false_folds", 0))
            if counts.get("unit") != "first eligible decision per opponent-game":
                failures.append(f"{label}: unexpected strict FFR unit")
            if bool(counts.get("repeated_opportunities_within_game", True)):
                failures.append(f"{label}: primary FFR still allows repeated opportunities")
            if len(ffr_events) != expected_opps:
                failures.append(
                    f"{label}: FFR event/count mismatch {len(ffr_events)} != {expected_opps}"
                )
            if len(false_folds) != expected_false:
                failures.append(
                    f"{label}: false-fold event/count mismatch {len(false_folds)} != {expected_false}"
                )

            pair_keys = [
                (int(event.get("seed", -1)), int(event.get("defender_pid", -1)))
                for event in ffr_events
            ]
            if len(pair_keys) != len(set(pair_keys)):
                failures.append(f"{label}: repeated primary FFR opponent-game key")
            reconstructed_by_opponent: dict[str, dict[str, int]] = {}
            for event in ffr_events:
                if int(event.get("own_shanten", 99)) > 1:
                    failures.append(f"{label}: FFR event has own_shanten>1")
                    break
                if int(event.get("p0_shanten", -99)) <= 0:
                    failures.append(f"{label}: FFR event has dangerous/tenpai P0")
                    break
                if event.get("false_fold") and float(
                    event.get("threat_at_fold", -1.0)
                ) < float(event.get("threat_threshold", 0.0)):
                    failures.append(f"{label}: false fold is below the defensive threshold")
                    break
                if not str(event.get("push_action", "")).startswith("d "):
                    failures.append(f"{label}: non-discard push entered strict FFR")
                    break
                if not str(event.get("chosen_action", "")).startswith("d "):
                    failures.append(f"{label}: non-discard choice entered strict FFR")
                    break
                if event.get("false_fold") and not event.get("progress_loss"):
                    failures.append(f"{label}: false fold without progress loss")
                    break
                opponent_id = str(event.get("opponent_id") or "unknown")
                if opponent_id not in expected_opponents:
                    failures.append(f"{label}: unexpected opponent id {opponent_id}")
                    break
                opponent_counts = reconstructed_by_opponent.setdefault(
                    opponent_id, {"opportunities": 0, "false_folds": 0}
                )
                opponent_counts["opportunities"] += 1
                opponent_counts["false_folds"] += int(bool(event.get("false_fold")))

            reported_by_opponent = (
                (summary.get("FFR_attribution") or {}).get("by_opponent") or {}
            )
            if set(reported_by_opponent) != set(reconstructed_by_opponent):
                failures.append(f"{label}: opponent-level FFR keys do not match events")
            else:
                for opponent_id, reconstructed in reconstructed_by_opponent.items():
                    reported = reported_by_opponent[opponent_id]
                    if any(
                        int(reported.get(field, -1)) != value
                        for field, value in reconstructed.items()
                    ):
                        failures.append(
                            f"{label}: opponent-level FFR counts mismatch for {opponent_id}"
                        )
                        break

            hashes = summary.get("runtime_artifact_sha256") or {}
            for name, value in hashes.items():
                artifact_hashes[variant].setdefault(name, set()).add(str(value))
            if args.belief_sha256 and hashes.get("belief_public_model") != args.belief_sha256:
                failures.append(f"{label}: belief artifact SHA mismatch")
            expected_gate_sha = expected_gate_sha256.get(variant)
            if expected_gate_sha and hashes.get("gate_value_model") != expected_gate_sha:
                failures.append(f"{label}: gate checkpoint SHA mismatch")

            cells[label] = {
                "games": len(games),
                "unsettled": unsettled,
                "strict_ffr": row.get("FFR"),
                "strict_ffr_counts": counts,
                "strict_event_ffr": row.get("strict_FFR_event_diagnostic"),
                "defensive_switch_rate": row.get(
                    "conditional_defensive_switch_rate"
                ),
                "max_primary_opportunities_per_game": max_primary_opportunities,
                "strict_ffr_by_opponent": reported_by_opponent,
            }

        for name, values in artifact_hashes[variant].items():
            if len(values) != 1:
                failures.append(
                    f"{variant}: artifact {name} changed across outer seeds"
                )

    for outer_seed in args.seeds:
        reference = seed_sets.get((args.variants[0], outer_seed), set())
        for variant in args.variants[1:]:
            if seed_sets.get((variant, outer_seed), set()) != reference:
                failures.append(
                    f"seed_{outer_seed}: game seed set differs for {variant}"
                )

    result = {
        "passed": not failures,
        "root": str(args.root),
        "variants": args.variants,
        "outer_seeds": args.seeds,
        "games_per_cell": args.games,
        "expected_opponents": sorted(expected_opponents),
        "curriculum_manifest_sha256": manifest_sha256,
        "expected_gate_sha256": expected_gate_sha256,
        "artifact_hashes": {
            variant: {name: sorted(values) for name, values in by_name.items()}
            for variant, by_name in artifact_hashes.items()
        },
        "cells": cells,
        "failures": failures,
    }
    write_json(args.output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--curriculum-manifest", type=Path, required=True)
    parser.add_argument("--curriculum-split", choices=["train", "holdout"], default="holdout")
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--belief-sha256")
    parser.add_argument("--curriculum-sha256")
    parser.add_argument("--expected-gate-sha256", action="append", default=[])
    args = parser.parse_args()

    result = audit(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
