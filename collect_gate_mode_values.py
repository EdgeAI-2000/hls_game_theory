"""Collect paired CRN settlement values for distinct MASK mode actions."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from collect_neural_gate_teacher import neural_available_modes
from environment_rollout import (
    capture_rollout_snapshot,
    mask_continuation_policy,
    rollout_candidate,
)
from experiment_trace import ensure_deterministic_hashing, write_json
from gate_value_policy import (
    GATE_VALUE_FEATURE_NAMES,
    GateValuePolicy,
    build_gate_value_features,
)
from llm_backends import build_llm_callable
from mask_llm import MASKLLMAgent
from prompt_builder import get_legal_actions
from public_belief_model import PublicBeliefModel, sha256_file
from run_gate1_experiments import (
    execute_action,
    init_game,
    load_belief_calibration_steps,
    resolve_responses,
)


MODES = ("exploit", "safe", "deceive")


def make_agent(
    seed: int,
    llm,
    gate_policy: str,
    neural_gate=None,
    auto_hu: str = "common",
    belief_calibration_steps=None,
    mc_oracle_samples: int = 60,
    mc_beta: float = 2.0,
    mc_danger_threshold: int = 0,
    belief_public_model=None,
) -> MASKLLMAgent:
    return MASKLLMAgent(
        player_id=0,
        decision_llm=llm,
        belief_llm=llm,
        neural_gate=neural_gate,
        gate_policy=gate_policy,
        auto_hu=auto_hu,
        mc_seed=seed * 13 + 1,
        mc_oracle_samples=mc_oracle_samples,
        mc_beta=mc_beta,
        mc_danger_threshold=mc_danger_threshold,
        belief_public_model=belief_public_model,
        belief_calibration_steps=belief_calibration_steps,
        dir_ready_threshold=0,
        deceive_threat_ceiling=0.5,
        forced_deceive="off",
        deceive_style="threat",
        threat_allow_break_ready=False,
        threat_max_result_shanten=0,
        threat_max_shanten_regret=0,
        threat_min_ukeire_ratio=1.0,
        threat_gate_threshold=0.4,
        threat_gate_margin=0.12,
        threat_min_delta=0.03,
        threat_response_model="tell",
        threat_response_tell_weight=1.0,
        threat_tell_window=6,
        threat_max_start_shanten=3,
        threat_require_non_exploit=True,
        threat_require_real_target=False,
    )


def lower_tail(values: Sequence[float], alpha: float) -> float:
    count = max(1, int(math.ceil(len(values) * alpha)))
    return sum(sorted(values)[:count]) / count


def distinct_mode_actions(
    agent: MASKLLMAgent,
    game,
    legal_actions: List[str],
    beliefs: Dict[str, Any],
) -> Tuple[List[str], Dict[str, str], Dict[str, Any]]:
    _, diagnostics = neural_available_modes(agent, game, legal_actions, beliefs)
    raw_actions = diagnostics.get("mode_actions") or {}
    available: List[str] = []
    actions: Dict[str, str] = {}
    seen_actions = set()
    for mode in MODES:
        action = raw_actions.get(mode)
        if not isinstance(action, str) or action not in legal_actions or action in seen_actions:
            continue
        available.append(mode)
        actions[mode] = action
        seen_actions.add(action)
    diagnostics["raw_mode_actions"] = raw_actions
    diagnostics["mode_actions"] = actions
    diagnostics["distinct_available_modes"] = available
    return available, actions, diagnostics


def utility_from_outcomes(outcomes, args: argparse.Namespace) -> Tuple[float, Dict[str, float]]:
    returns = [float(outcome.continuation_return) for outcome in outcomes]
    cvar = lower_tail(returns, args.cvar_alpha)
    hua_rate = sum(outcome.agent_hua_zhu for outcome in outcomes) / len(outcomes)
    mean_return = sum(returns) / len(returns)
    utility = (
        mean_return
        - args.tail_weight * max(0.0, -cvar)
        - args.hua_zhu_penalty * hua_rate
    )
    return utility, {
        "mean_return": mean_return,
        "cvar_return": cvar,
        "hua_zhu_rate": hua_rate,
        "hu_rate": sum(outcome.agent_hu for outcome in outcomes) / len(outcomes),
        "dealin_rate": sum(outcome.agent_dealin for outcome in outcomes) / len(outcomes),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_deterministic_hashing()
    random.seed(args.seed)
    repo_dir = Path(__file__).resolve().parent
    llm = build_llm_callable(
        backend="local_qwen",
        repo_dir=repo_dir,
        model_path=args.model_path,
        adapter_path=args.adapter_path or None,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
    )
    student_policy = (
        GateValuePolicy.load(args.student_gate_value_model_path, device=args.gate_device)
        if args.rollout_policy == "student"
        else None
    )
    if args.rollout_policy == "student" and student_policy is None:
        raise ValueError("student rollout requires --student-gate-value-model-path")
    belief_public_model = (
        PublicBeliefModel.load(args.belief_public_model)
        if args.belief_public_model else None
    )
    belief_calibration_steps = load_belief_calibration_steps(
        args.belief_calibration_summary,
        oracle_samples=args.mask_oracle_samples,
        oracle_beta=args.mask_oracle_beta,
        danger_threshold=args.mask_danger_threshold,
        b_phi_source=("public_model" if belief_public_model else "mc"),
        belief_public_model_sha256=(
            sha256_file(args.belief_public_model)
            if args.belief_public_model else ""
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "gate_mode_values.jsonl"
    ordinary_target = max(0, args.target_states - args.target_deceive_states)
    ordinary_rows = deceive_rows = rows = games = p0_states = rollout_calls = 0
    skipped: Counter[str] = Counter()
    best_modes: Counter[str] = Counter()
    belief_bins: Counter[str] = Counter()

    def done() -> bool:
        return ordinary_rows >= ordinary_target and deceive_rows >= args.target_deceive_states

    defender_cfg = {
        "neural_model_path": args.neural_opponent_model_path,
        "neural_device": "cpu",
        "danger_threshold": 1,
        "ffr_hand_shanten": 1,
        "curriculum_manifest": args.curriculum_manifest,
        "curriculum_split": args.curriculum_split,
    }
    with rows_path.open("w", encoding="utf-8") as handle:
        for game_index in range(args.max_games):
            if done():
                break
            game_seed = args.seed + game_index
            game, opponent_funcs, defenders = init_game(
                game_seed, args.opponent_style, f"GateModeValue_{game_seed}", defender_cfg
            )
            episode_start_balance = float(game.players[0].balance)
            trajectory = make_agent(
                game_seed,
                llm,
                "neural_value_gate" if student_policy is not None else "continuous_v2",
                neural_gate=student_policy,
                auto_hu=args.auto_hu,
                belief_calibration_steps=belief_calibration_steps,
                mc_oracle_samples=args.mask_oracle_samples,
                mc_beta=args.mask_oracle_beta,
                mc_danger_threshold=args.mask_danger_threshold,
                belief_public_model=belief_public_model,
            )
            candidate_agent = make_agent(
                game_seed,
                llm,
                "rule",
                auto_hu=args.auto_hu,
                belief_calibration_steps=belief_calibration_steps,
                mc_oracle_samples=args.mask_oracle_samples,
                mc_beta=args.mask_oracle_beta,
                mc_danger_threshold=args.mask_danger_threshold,
                belief_public_model=belief_public_model,
            )
            games += 1
            steps = 0
            skip_draw = True
            last_p0_state: Dict[str, Any] = {}
            deceive_active_until = -1

            while not game.is_game_over and steps < args.max_steps and not done():
                steps += 1
                if game.deck.remaining_count() == 0 or sum(p.is_hu for p in game.players) >= 3:
                    game.check_game_over()
                    break
                pid = game.current_player_id
                player = game.players[pid]
                if player.is_hu:
                    game.next_player()
                    skip_draw = False
                    continue
                drawn_tile = None
                if not skip_draw:
                    drawn_tile = game.draw_tile(pid)
                    if drawn_tile is None:
                        game.check_game_over()
                        break
                else:
                    skip_draw = False

                if pid == 0:
                    legal_actions = get_legal_actions(game, 0)
                    action = trajectory.decide(game, legal_actions)
                    trajectory_state = dict(trajectory.last_decision)
                    p0_states += 1
                    should_sample = (
                        "h" not in legal_actions
                        and any(item.startswith("d ") for item in legal_actions)
                        and p0_states % max(1, args.sample_stride) == 0
                    )
                    if should_sample:
                        available, mode_actions, diagnostics = distinct_mode_actions(
                            candidate_agent,
                            game,
                            legal_actions,
                            trajectory_state.get("beliefs", {}),
                        )
                        has_deceive = "deceive" in available
                        quota_open = (
                            has_deceive and deceive_rows < args.target_deceive_states
                        ) or (not has_deceive and ordinary_rows < ordinary_target)
                        if len(available) < 2:
                            skipped["fewer_than_two_distinct_modes"] += 1
                        elif not quota_open:
                            skipped["quota_full"] += 1
                        else:
                            snapshot = capture_rollout_snapshot(
                                game,
                                opponent_funcs,
                                defenders,
                                trajectory,
                                steps,
                                args.max_steps,
                                episode_start_balance,
                                drawn_tile=drawn_tile,
                                last_p0_state=last_p0_state,
                                deceive_active_until=deceive_active_until,
                            )
                            mode_values: Dict[str, float] = {}
                            mode_rollouts: Dict[str, Any] = {}
                            mode_outcomes: Dict[str, Any] = {}
                            for mode in available:
                                outcomes = []
                                for crn_index in range(args.crn_seeds):
                                    rollout_seed = (
                                        game_seed * 1000003 + steps * 101 + crn_index
                                    ) & 0x7FFFFFFF
                                    outcomes.append(
                                        rollout_candidate(
                                            snapshot,
                                            mode_actions[mode],
                                            rollout_seed,
                                            continuation_policy=mask_continuation_policy,
                                            use_agent_for_responses=True,
                                            initial_mode=mode,
                                        )
                                    )
                                    rollout_calls += 1
                                utility, outcome_summary = utility_from_outcomes(outcomes, args)
                                mode_values[mode] = utility
                                mode_outcomes[mode] = outcomes
                                penalized_returns = [
                                    float(item.continuation_return)
                                    - args.hua_zhu_penalty * float(item.agent_hua_zhu)
                                    for item in outcomes
                                ]
                                standard_error = (
                                    statistics.stdev(penalized_returns)
                                    / math.sqrt(len(penalized_returns))
                                    if len(penalized_returns) > 1
                                    else 0.0
                                )
                                mode_rollouts[mode] = {
                                    "action": mode_actions[mode],
                                    "utility": utility,
                                    "returns": [float(item.continuation_return) for item in outcomes],
                                    "hua_zhu_flags": [bool(item.agent_hua_zhu) for item in outcomes],
                                    "standard_error": standard_error,
                                    **outcome_summary,
                                }
                            pairwise_standard_errors: Dict[str, float] = {}
                            for left in available:
                                for right in available:
                                    if left == right:
                                        continue
                                    paired_differences = [
                                        (
                                            float(left_outcome.continuation_return)
                                            - args.hua_zhu_penalty * float(left_outcome.agent_hua_zhu)
                                        )
                                        - (
                                            float(right_outcome.continuation_return)
                                            - args.hua_zhu_penalty * float(right_outcome.agent_hua_zhu)
                                        )
                                        for left_outcome, right_outcome in zip(
                                            mode_outcomes[left], mode_outcomes[right]
                                        )
                                    ]
                                    pairwise_standard_errors[f"{left}_vs_{right}"] = (
                                        statistics.stdev(paired_differences)
                                        / math.sqrt(len(paired_differences))
                                        if len(paired_differences) > 1
                                        else 0.0
                                    )
                            best_mode = max(available, key=lambda mode: mode_values[mode])
                            best_modes[best_mode] += 1
                            features = build_gate_value_features(
                                game,
                                0,
                                trajectory_state.get("z_state", {}),
                                trajectory_state.get("beliefs", {}),
                                trajectory_state.get("gate", {}),
                                available,
                                mode_actions,
                            )
                            belief_max = float(
                                features[GATE_VALUE_FEATURE_NAMES.index("belief_max")]
                            )
                            if belief_max < 0.80:
                                belief_bin = "lt_0.80"
                            elif belief_max < 0.90:
                                belief_bin = "0.80_0.90"
                            elif belief_max < 0.95:
                                belief_bin = "0.90_0.95"
                            else:
                                belief_bin = "ge_0.95"
                            belief_bins[belief_bin] += 1
                            row = {
                                "state_id": f"g{game_index:05d}_s{steps:03d}",
                                "game_seed": game_seed,
                                "step": steps,
                                "rollout_policy": args.rollout_policy,
                                "opponent_distribution": {
                                    "style": args.opponent_style,
                                    "curriculum_split": (
                                        args.curriculum_split
                                        if args.opponent_style == "curriculum"
                                        else None
                                    ),
                                    "curriculum_ids": [
                                        str(getattr(defender, "curriculum_spec", {}).get("id"))
                                        for defender in defenders.values()
                                        if getattr(defender, "curriculum_spec", None)
                                    ],
                                },
                                "features": features,
                                "belief_max": belief_max,
                                "belief_bin": belief_bin,
                                "available_modes": available,
                                "mode_actions": mode_actions,
                                "mode_values": mode_values,
                                "mode_rollouts": mode_rollouts,
                                "pairwise_standard_errors": pairwise_standard_errors,
                                "target_mode": best_mode,
                                "trajectory_mode": trajectory_state.get("mode", "exploit"),
                                "trajectory_action": action,
                                "diagnostics": diagnostics,
                            }
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                            handle.flush()
                            rows += 1
                            if has_deceive:
                                deceive_rows += 1
                            else:
                                ordinary_rows += 1
                            if rows == 1 or rows % 20 == 0:
                                print(
                                    f"[GateModeValue] rows={rows}/{args.target_states} "
                                    f"ordinary={ordinary_rows}/{ordinary_target} "
                                    f"deceive={deceive_rows}/{args.target_deceive_states} "
                                    f"games={games} rollouts={rollout_calls} best={dict(best_modes)}",
                                    flush=True,
                                )
                    actual_mode = str(trajectory_state.get("mode", "exploit"))
                    last_p0_state = {
                        "step": steps,
                        "mode": actual_mode,
                        "action": action,
                        "reason": trajectory_state.get("reason", ""),
                    }
                    if actual_mode == "deceive":
                        deceive_active_until = steps + 4
                elif pid in defenders:
                    action = defenders[pid].turn(
                        player,
                        game,
                        step=steps,
                        last_p0_state=last_p0_state,
                        in_deceive_window=deceive_active_until >= steps,
                    )
                else:
                    action = opponent_funcs[pid][0](player, game)

                discarded_tile = execute_action(game, pid, action, drawn_tile)
                if game.is_game_over:
                    break
                if discarded_tile is None:
                    if action == "g":
                        skip_draw = True
                    else:
                        game.next_player()
                        skip_draw = False
                    continue
                response = resolve_responses(game, discarded_tile, pid, trajectory, defenders)
                if game.is_game_over:
                    break
                if response["responded"]:
                    skip_draw = True
                else:
                    game.next_player()
                    skip_draw = False

    summary = {
        "rows": rows,
        "ordinary_rows": ordinary_rows,
        "deceive_available_rows": deceive_rows,
        "games": games,
        "p0_states": p0_states,
        "rollout_calls": rollout_calls,
        "best_modes": dict(best_modes),
        "belief_bins": dict(belief_bins),
        "skipped": dict(skipped),
        "rows_path": str(rows_path),
        "config": vars(args),
        "target_complete": done(),
    }
    write_json(args.output_dir / "gate_mode_value_collection_summary.json", summary)
    if args.require_target and not done():
        raise RuntimeError(
            "collection exhausted --max-games before reaching both ordinary and "
            f"deceive quotas: ordinary={ordinary_rows}/{ordinary_target}, "
            f"deceive={deceive_rows}/{args.target_deceive_states}"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--rollout-policy", choices=["continuous", "student"], default="continuous")
    parser.add_argument("--student-gate-value-model-path", default="")
    parser.add_argument("--gate-device", default="cpu")
    parser.add_argument("--target-states", type=int, default=700)
    parser.add_argument("--target-deceive-states", type=int, default=180)
    parser.add_argument("--max-games", type=int, default=6000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--crn-seeds", type=int, default=8)
    parser.add_argument("--cvar-alpha", type=float, default=0.25)
    parser.add_argument("--tail-weight", type=float, default=0.10)
    parser.add_argument("--hua-zhu-penalty", type=float, default=256.0)
    parser.add_argument("--auto-hu", choices=["common", "off"], default="common")
    parser.add_argument("--seed", type=int, default=2026073001)
    parser.add_argument(
        "--neural-opponent-model-path",
        default="Neural_opponent_model/neural_opponent_policy.pth",
    )
    parser.add_argument(
        "--opponent-style",
        choices=["neural", "responsive", "mixed", "curriculum"],
        default="neural",
    )
    parser.add_argument("--curriculum-manifest", default=None)
    parser.add_argument("--curriculum-split", choices=["train", "holdout", "all"], default="train")
    parser.add_argument("--belief-calibration-summary", default=None)
    parser.add_argument("--belief-public-model", default=None)
    parser.add_argument("--mask-oracle-samples", type=int, default=60)
    parser.add_argument("--mask-oracle-beta", type=float, default=2.0)
    parser.add_argument("--mask-danger-threshold", type=int, default=0)
    parser.add_argument("--require-target", action="store_true")
    args = parser.parse_args()
    if args.target_deceive_states < 0 or args.target_deceive_states > args.target_states:
        raise ValueError("--target-deceive-states must be in [0, target-states]")
    if args.crn_seeds < 2:
        raise ValueError("--crn-seeds must be at least 2")
    if args.rollout_policy == "student" and not args.student_gate_value_model_path:
        raise ValueError("student rollout requires --student-gate-value-model-path")
    if args.opponent_style == "curriculum" and not args.curriculum_manifest:
        raise ValueError("curriculum collection requires --curriculum-manifest")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
