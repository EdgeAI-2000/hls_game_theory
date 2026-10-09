"""Collect on-policy hard states with CRN settlement values per action."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

from action_value_policy import ValueResidualPolicy, build_action_candidates
from environment_rollout import capture_rollout_snapshot, rollout_candidate
from experiment_trace import ensure_deterministic_hashing, write_json
from mask_llm import MASKLLMAgent
from ppo_features import extract_state_features
from prompt_builder import get_legal_actions
from run_gate1_experiments import execute_action, init_game, resolve_responses


def make_agent(seed: int, gate_policy: str, action_policy=None) -> MASKLLMAgent:
    return MASKLLMAgent(
        player_id=0,
        decision_llm=None,
        belief_llm=None,
        action_value_policy=action_policy,
        gate_policy=gate_policy,
        auto_hu="off",
        mc_seed=seed * 13 + 1,
        mc_oracle_samples=30,
        mc_beta=2.0,
        mc_danger_threshold=1,
        dir_ready_threshold=0,
        deceive_style="threat",
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
    )


def lower_tail(values: Sequence[float], alpha: float) -> float:
    count = max(1, int(math.ceil(len(values) * alpha)))
    return sum(sorted(values)[:count]) / count


def candidate_mode(action: str, teacher_state: Dict[str, Any]) -> str:
    if action == "h":
        return "exploit"
    q_base = teacher_state.get("candidate_q_base") or {}
    delta = teacher_state.get("candidate_delta_shape") or {}
    if q_base:
        best_q = max(q_base, key=lambda item: float(q_base[item]))
        if action != best_q and float(delta.get(action, 0.0)) > 0.0:
            return "deceive"
    gate = teacher_state.get("gate") or {}
    if float(gate.get("risk_budget", 0.0)) > 0.75 or gate.get("mode_hint") == "safe":
        return "safe"
    return "exploit"


def select_rollout_actions(
    candidates: List[Dict[str, Any]],
    student_scores: Dict[str, float],
    teacher_scores: Dict[str, float],
    student_action: str,
    teacher_action: str,
    max_candidates: int,
) -> List[Dict[str, Any]]:
    by_action = {str(item["action"]): item for item in candidates}
    priority = [student_action, teacher_action]
    priority.extend(sorted(student_scores, key=student_scores.get, reverse=True))
    priority.extend(sorted(teacher_scores, key=teacher_scores.get, reverse=True))
    selected: List[Dict[str, Any]] = []
    seen = set()
    for action in priority:
        if action in seen or action not in by_action:
            continue
        selected.append(by_action[action])
        seen.add(action)
        if len(selected) >= max_candidates:
            break
    return selected


def run(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_deterministic_hashing()
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "value_residual_rollouts.jsonl"
    bootstrap = ValueResidualPolicy.load(
        args.base_action_value_model_path,
        args.bootstrap_residual_model_path,
        device=args.policy_device,
    ) if args.bootstrap_residual_model_path else None
    if bootstrap is None:
        from action_value_policy import ActionValuePolicy

        student_policy = ActionValuePolicy.load(
            args.base_action_value_model_path, device=args.policy_device
        )
        student_gate_policy = "neural_action_value"
    else:
        student_policy = bootstrap
        student_gate_policy = "neural_value_residual"

    defender_cfg = {
        "neural_model_path": args.neural_opponent_model_path,
        "neural_device": "cpu",
        "danger_threshold": 1,
        "ffr_hand_shanten": 1,
    }
    rows = games = p0_states = hard_states = rollout_calls = 0
    hard_reasons: Counter[str] = Counter()
    target_actions: Counter[str] = Counter()
    utility_gains: List[float] = []

    with rows_path.open("w", encoding="utf-8") as handle:
        for game_index in range(args.max_games):
            if rows >= args.target_states:
                break
            game_seed = args.seed + game_index
            game, opponent_funcs, defenders = init_game(
                game_seed, "neural", f"ValueResidual_{game_seed}", defender_cfg
            )
            episode_start_balance = float(game.players[0].balance)
            teacher = make_agent(game_seed, "continuous_v2")
            student = make_agent(game_seed, student_gate_policy, student_policy)
            games += 1
            skip_draw = True
            steps = 0
            last_p0_state: Dict[str, Any] = {}
            deceive_active_until = -1

            while not game.is_game_over and steps < args.max_steps and rows < args.target_states:
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
                    student_action = student.decide(game, legal_actions)
                    student_state = dict(student.last_decision)
                    teacher_action = teacher.decide(game, legal_actions)
                    teacher_state = dict(teacher.last_decision)
                    p0_states += 1
                    student_scores = {
                        str(key): float(value)
                        for key, value in (
                            (student_state.get("action_value") or {}).get("candidate_scores") or {}
                        ).items()
                    }
                    teacher_scores = {
                        str(key): float(value)
                        for key, value in (teacher_state.get("candidate_scores") or {}).items()
                    }
                    sorted_student = sorted(student_scores.values(), reverse=True)
                    margin = (
                        sorted_student[0] - sorted_student[1]
                        if len(sorted_student) >= 2
                        else float("inf")
                    )
                    reasons = []
                    if student_action != teacher_action:
                        reasons.append("student_teacher_disagree")
                    if margin <= args.hard_margin:
                        reasons.append("low_student_margin")
                    candidates = build_action_candidates(
                        game, 0, legal_actions, max_shanten_regret=0, min_ukeire_ratio=1.0
                    )
                    candidate_actions = {str(item["action"]) for item in candidates}
                    eligible = (
                        len(candidates) >= 2
                        and student_action in candidate_actions
                        and teacher_action in candidate_actions
                        and bool(reasons)
                    )
                    if eligible and p0_states % max(1, args.sample_stride) == 0:
                        hard_states += 1
                        for reason in reasons:
                            hard_reasons[reason] += 1
                        selected = select_rollout_actions(
                            candidates,
                            student_scores,
                            teacher_scores,
                            student_action,
                            teacher_action,
                            args.max_candidates,
                        )
                        if len(selected) >= 2:
                            snapshot = capture_rollout_snapshot(
                                game,
                                opponent_funcs,
                                defenders,
                                None,
                                steps,
                                args.max_steps,
                                episode_start_balance,
                                drawn_tile=drawn_tile,
                                last_p0_state=last_p0_state,
                                deceive_active_until=deceive_active_until,
                            )
                            valued = []
                            for item in selected:
                                action = str(item["action"])
                                outcomes = []
                                for crn_index in range(args.crn_seeds):
                                    rollout_seed = (
                                        game_seed * 1000003 + steps * 101 + crn_index
                                    ) & 0x7FFFFFFF
                                    outcome = rollout_candidate(
                                        snapshot,
                                        action,
                                        rollout_seed,
                                        use_agent_for_responses=False,
                                        initial_mode=candidate_mode(action, teacher_state),
                                    )
                                    rollout_calls += 1
                                    outcomes.append(outcome)
                                returns = [float(outcome.continuation_return) for outcome in outcomes]
                                cvar = lower_tail(returns, args.cvar_alpha)
                                hua_rate = sum(outcome.agent_hua_zhu for outcome in outcomes) / len(outcomes)
                                utility = (
                                    sum(returns) / len(returns)
                                    - args.tail_weight * max(0.0, -cvar)
                                    - args.hua_zhu_penalty * hua_rate
                                )
                                valued.append(
                                    {
                                        "action": action,
                                        "features": item["features"],
                                        "base_score": float(student_scores.get(action, 0.0)),
                                        "teacher_score": float(teacher_scores.get(action, 0.0)),
                                        "mode": candidate_mode(action, teacher_state),
                                        "mean_return": sum(returns) / len(returns),
                                        "cvar_return": cvar,
                                        "hua_zhu_rate": hua_rate,
                                        "utility": utility,
                                        "returns": returns,
                                        "hu_rate": sum(outcome.agent_hu for outcome in outcomes) / len(outcomes),
                                        "dealin_rate": sum(outcome.agent_dealin for outcome in outcomes) / len(outcomes),
                                    }
                                )
                            best = max(valued, key=lambda item: item["utility"])
                            student_value = next(
                                item["utility"] for item in valued if item["action"] == student_action
                            )
                            utility_gains.append(float(best["utility"] - student_value))
                            target_actions[best["action"]] += 1
                            state_features = extract_state_features(
                                game,
                                0,
                                student_state.get("z_state", {}),
                                student_state.get("beliefs", {}),
                                student_state.get("gate", {}),
                            )
                            row = {
                                "state_id": f"g{game_index:05d}_s{steps:03d}",
                                "game_seed": game_seed,
                                "step": steps,
                                "hard_reasons": reasons,
                                "student_action": student_action,
                                "teacher_action": teacher_action,
                                "student_margin": margin,
                                "state_features": state_features,
                                "candidates": valued,
                                "target_action": best["action"],
                                "target_utility": best["utility"],
                            }
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                            rows += 1
                            if rows == 1 or rows % 25 == 0:
                                print(
                                    f"[ValueResidual] rows={rows}/{args.target_states} "
                                    f"games={games} rollouts={rollout_calls} "
                                    f"avg_gain={sum(utility_gains) / len(utility_gains):.2f}",
                                    flush=True,
                                )

                    actual_mode = str(student_state.get("mode", "exploit"))
                    action = student_action
                    last_p0_state = {
                        "step": steps,
                        "mode": actual_mode,
                        "action": action,
                        "reason": student_state.get("reason", ""),
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
                response = resolve_responses(game, discarded_tile, pid, student, defenders)
                if game.is_game_over:
                    break
                if response["responded"]:
                    skip_draw = True
                else:
                    game.next_player()
                    skip_draw = False

    summary = {
        "rows": rows,
        "games": games,
        "p0_states": p0_states,
        "hard_states": hard_states,
        "hard_reasons": dict(hard_reasons),
        "rollout_calls": rollout_calls,
        "target_actions": dict(target_actions),
        "avg_oracle_gain_over_student": (
            sum(utility_gains) / len(utility_gains) if utility_gains else 0.0
        ),
        "rows_path": str(rows_path),
        "config": vars(args),
    }
    write_json(args.output_dir / "value_residual_collection_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-action-value-model-path", required=True)
    parser.add_argument("--bootstrap-residual-model-path", default="")
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--target-states", type=int, default=1000)
    parser.add_argument("--max-games", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--hard-margin", type=float, default=0.10)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--crn-seeds", type=int, default=4)
    parser.add_argument("--cvar-alpha", type=float, default=0.25)
    parser.add_argument("--tail-weight", type=float, default=0.10)
    parser.add_argument("--hua-zhu-penalty", type=float, default=256.0)
    parser.add_argument("--seed", type=int, default=2026072903)
    parser.add_argument(
        "--neural-opponent-model-path",
        default="Neural_opponent_model/neural_opponent_policy.pth",
    )
    args = parser.parse_args()
    if args.crn_seeds < 2:
        raise ValueError("--crn-seeds must be at least 2 for paired rollout labels")
    if args.max_candidates < 2:
        raise ValueError("--max-candidates must be at least 2")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
