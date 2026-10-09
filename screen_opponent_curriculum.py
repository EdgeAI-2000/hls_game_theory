"""Screen generated opponents for legality, responsiveness, strength and diversity."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from experiment_trace import write_json
from opponent_curriculum import (
    curriculum_fingerprint,
    defender_config,
    load_curriculum,
    normalize_spec,
    response_vector_distance,
)
from run_gate1_experiments import ResponsiveDefender, init_game
from tiles import Suit, Tile


REPO_DIR = Path(__file__).resolve().parent


def _public_signal_game(seed: int, signal: str):
    """Build paired public-information probes on one dealt state."""
    game, _, _ = init_game(seed, "greedy", f"course_probe_{seed}_{signal}")
    game = copy.deepcopy(game)
    p0 = game.players[0]
    suit = next(candidate for candidate in Suit if candidate != p0.missing_suit)
    numbers = (1, 9, 1, 9, 1, 9) if signal == "low" else (4, 5, 6, 4, 5, 6)
    p0.discarded_tiles.extend(Tile(suit, number) for number in numbers)
    return game


def behavioral_inference_probe(
    spec: Dict[str, Any], probe_seeds: List[int], samples: int
) -> Dict[str, Any]:
    """Measure actual defender response to paired low/high public-action signals."""
    cfg = defender_config(spec)
    rows = []
    response_vector: List[float] = []
    for seed in probe_seeds:
        low_game = _public_signal_game(seed, "low")
        high_game = _public_signal_game(seed, "high")
        crn_seeds = [seed * 1000 + offset for offset in range(4)]

        def score(game, suffix: int) -> float:
            defender = ResponsiveDefender(
                observer_pid=1,
                threat_threshold=cfg["threat_threshold"],
                oracle_samples=cfg["oracle_samples"],
                beta=cfg["beta"],
                danger_threshold=cfg["danger_threshold"],
                ffr_hand_shanten=cfg["ffr_hand_shanten"],
                rng=random.Random(seed * 100 + suffix),
                threat_model=cfg["threat_model"],
                tell_weight=cfg["tell_weight"],
                tell_window=cfg["tell_window"],
            )
            return defender.threat_crn(game, crn_seeds, samples)

        low = score(low_game, 1)
        high = score(high_game, 2)
        low_fold = low >= cfg["threat_threshold"]
        high_fold = high >= cfg["threat_threshold"]
        rows.append({
            "seed": seed,
            "low_threat": low,
            "high_threat": high,
            "delta": high - low,
            "low_fold": low_fold,
            "high_fold": high_fold,
        })
        response_vector.extend([low, high, float(low_fold), float(high_fold)])

    mean_delta = sum(row["delta"] for row in rows) / len(rows)
    switch_rate = sum(
        bool(row["high_fold"]) and not bool(row["low_fold"]) for row in rows
    ) / len(rows)
    delta_score = max(0.0, min(1.0, mean_delta / 0.50))
    inference_score = 0.7 * delta_score + 0.3 * switch_rate
    return {
        "inference_score": inference_score,
        "mean_paired_threat_delta": mean_delta,
        "fold_switch_rate": switch_rate,
        "response_vector": response_vector,
        "paired_probes": rows,
    }


def assign_behavior_holdout(
    opponents: List[Dict[str, Any]], holdout_ratio: float
) -> List[Dict[str, Any]]:
    """Assign the final split after screening in behavioral-response space."""
    if len(opponents) < 6:
        raise RuntimeError(
            f"behavioral screening needs at least 6 opponents, got {len(opponents)}"
        )
    holdout_count = max(2, min(len(opponents) - 4, round(len(opponents) * holdout_ratio)))
    vectors = [item["screening"]["response_vector"] for item in opponents]
    dimensions = len(vectors[0])
    centroid = [
        sum(float(vector[index]) for vector in vectors) / len(vectors)
        for index in range(dimensions)
    ]
    first = max(
        range(len(opponents)),
        key=lambda index: response_vector_distance(vectors[index], centroid),
    )
    chosen = [first]
    while len(chosen) < holdout_count:
        candidates = [index for index in range(len(opponents)) if index not in chosen]
        chosen.append(max(
            candidates,
            key=lambda index: min(
                response_vector_distance(vectors[index], vectors[other])
                for other in chosen
            ),
        ))
    holdout = set(chosen)
    for index, opponent in enumerate(opponents):
        opponent["split"] = "holdout" if index in holdout else "train"
    return opponents


def evaluate_spec(args: argparse.Namespace, spec: Dict[str, Any]) -> Dict[str, Any]:
    output = args.work_dir / spec["id"]
    summary_path = output / "gate1_summary.json"
    signature_path = output / "screen.command.sha256"
    command = [
        str(args.python), "run_gate1_experiments.py",
        "--methods", "llm_base",
        "--games", str(args.games),
        "--seed", str(args.seed),
        "--opponent-style", "curriculum",
        "--curriculum-manifest", str(args.input.resolve()),
        "--curriculum-split", "all",
        "--curriculum-spec-id", spec["id"],
        "--backend", "heuristic_fallback",
        "--sample-every", "0",
        "--auto-hu", "common",
        "--snapshot-oracle-samples", "1",
        "--snapshot-crn-seeds", "1",
        "--output-dir", str(output),
    ]
    signature = hashlib.sha256(json.dumps({
        "command": command,
        "spec": spec,
        "manifest_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
    }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    reusable = (
        summary_path.is_file()
        and signature_path.is_file()
        and signature_path.read_text(encoding="utf-8").strip() == signature
    )
    if args.force or not reusable:
        output.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.setdefault("PYTHONHASHSEED", "0")
        with (output / "screen.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                cwd=REPO_DIR,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        signature_path.write_text(signature + "\n", encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary["E2_ladder"]["llm_base"]
    probe = behavioral_inference_probe(spec, args.probe_seeds, args.probe_samples)
    inference_score = float(probe["inference_score"])
    accepted = (
        args.min_agent_net <= float(metrics["avg_net"]) <= args.max_agent_net
        and inference_score >= args.min_inference_score
    )
    return {
        "accepted": accepted,
        "inference_score": inference_score,
        "mean_paired_threat_delta": probe["mean_paired_threat_delta"],
        "fold_switch_rate": probe["fold_switch_rate"],
        "response_vector": probe["response_vector"],
        "paired_probes": probe["paired_probes"],
        "agent_avg_net": float(metrics["avg_net"]),
        "agent_hu_rate": float(metrics["hu_rate"]),
        "agent_dealin_rate": float(metrics["dealin_rate"]),
        "ffr_opportunities": int(metrics["FFR_counts"]["false_fold_opportunities"]),
        "ffr": float(metrics["FFR"]),
        "games": int(metrics["games"]),
        "legality": "engine-completed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path("py10/bin/python3"))
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026081101)
    parser.add_argument("--min-agent-net", type=float, default=-250.0)
    parser.add_argument("--max-agent-net", type=float, default=250.0)
    parser.add_argument("--min-inference-score", type=float, default=0.25)
    parser.add_argument("--min-distance", type=float, default=0.05)
    parser.add_argument("--holdout-ratio", type=float, default=1.0 / 3.0)
    parser.add_argument(
        "--probe-seeds", nargs="+", type=int,
        default=[202608111, 202608112, 202608113, 202608114, 202608115, 202608116],
    )
    parser.add_argument("--probe-samples", type=int, default=32)
    parser.add_argument("--parallel-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = load_curriculum(args.input, split="all")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    normalized = [normalize_spec(raw, index=i) for i, raw in enumerate(manifest["opponents"])]
    evaluated: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.parallel_workers)) as pool:
        futures = {pool.submit(evaluate_spec, args, spec): spec for spec in normalized}
        for future in as_completed(futures):
            spec = futures[future]
            evaluated[spec["id"]] = future.result()
            print(
                f"[course] evaluated {spec['id']} net={evaluated[spec['id']]['agent_avg_net']:.2f}",
                flush=True,
            )

    accepted: List[Dict[str, Any]] = []
    screening = []
    for spec in normalized:
        metrics = evaluated[spec["id"]]
        if metrics["accepted"] and accepted:
            distance = min(
                response_vector_distance(
                    metrics["response_vector"], other["screening"]["response_vector"]
                )
                for other in accepted
            )
            metrics["nearest_accepted_distance"] = distance
            metrics["accepted"] = distance >= args.min_distance
        else:
            metrics["nearest_accepted_distance"] = None
        screening.append({"id": spec["id"], **metrics})
        if metrics["accepted"]:
            accepted.append({**spec, "screening": metrics})
        print(f"[course] {spec['id']} accepted={metrics['accepted']} net={metrics['agent_avg_net']:.2f}", flush=True)

    accepted = assign_behavior_holdout(accepted, args.holdout_ratio)
    train_count = sum(item.get("split") == "train" for item in accepted)
    holdout_count = sum(item.get("split") == "holdout" for item in accepted)
    if train_count < 4 or holdout_count < 2:
        raise RuntimeError(
            f"screening left an unusable split: train={train_count}, holdout={holdout_count}"
        )
    output = {
        **{key: value for key, value in manifest.items() if key not in {"opponents", "path"}},
        "schema_version": "mask_opponent_curriculum_v1_screened",
        "screening_config": {
            "games_per_opponent": args.games,
            "agent_net_window": [args.min_agent_net, args.max_agent_net],
            "min_inference_score": args.min_inference_score,
            "min_behavior_distance": args.min_distance,
            "behavior_distance_space": "paired_public_signal_response_vector_rms",
            "probe_seeds": args.probe_seeds,
            "probe_samples": args.probe_samples,
            "holdout_ratio": args.holdout_ratio,
            "split_space": "paired_public_signal_response_vector_farthest_point",
        },
        "screening": screening,
        "opponents": accepted,
    }
    output["fingerprint"] = curriculum_fingerprint(accepted)
    write_json(args.output, output)
    print(json.dumps({"output": str(args.output), "accepted": len(accepted), "train": train_count, "holdout": holdout_count, "fingerprint": output["fingerprint"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
