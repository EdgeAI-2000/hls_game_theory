"""Collect per-action continuous-v2 scores on teacher/student mixed trajectories."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from action_value_policy import build_action_candidates
from experiment_trace import ensure_deterministic_hashing, write_json
from llm_backends import build_llm_callable
from mask_llm import MASKLLMAgent
from neural_gate_policy import NeuralGatePolicy
from ppo_features import extract_state_features
from prompt_builder import get_legal_actions
from run_gate1_experiments import execute_action, init_game, resolve_responses


def make_agent(seed: int, llm, gate_policy: str, neural_gate=None) -> MASKLLMAgent:
    return MASKLLMAgent(
        player_id=0,
        decision_llm=llm,
        belief_llm=llm,
        neural_gate=neural_gate,
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
        neural_gate_deceive_margin=0.05,
        neural_gate_safe_margin=0.05,
        neural_gate_safe_risk_threshold=0.30,
    )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_deterministic_hashing()
    random.seed(args.seed)
    repo_dir = Path(__file__).resolve().parent
    llm = build_llm_callable(
        backend=args.backend,
        repo_dir=repo_dir,
        model_path=args.model_path,
        adapter_path=None,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
    )
    bootstrap_gate = (
        NeuralGatePolicy.load(args.student_neural_gate_model_path, device="cpu")
        if args.student_fraction > 0.0
        else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "action_value_teacher.jsonl"
    rows = games = attempts = 0
    teacher_modes: Counter[str] = Counter()
    rollout_policies: Counter[str] = Counter()
    action_disagreements = 0

    defender_cfg = {
        "neural_model_path": args.neural_opponent_model_path,
        "neural_device": "cpu",
        "danger_threshold": 1,
        "ffr_hand_shanten": 1,
    }
    with rows_path.open("w", encoding="utf-8") as handle:
        for game_index in range(args.max_games):
            if rows >= args.target_states:
                break
            game_seed = args.seed + game_index
            game, opponent_funcs, defenders = init_game(
                game_seed,
                "neural",
                f"ActionValueTeacher_{game_seed}",
                defender_cfg,
            )
            teacher = make_agent(game_seed, llm, "continuous_v2")
            use_student = bool(
                bootstrap_gate is not None
                and random.Random(game_seed * 1009).random() < args.student_fraction
            )
            rollout_agent = (
                make_agent(game_seed, llm, "neural", neural_gate=bootstrap_gate)
                if use_student
                else teacher
            )
            rollout_name = "student_neural_gate" if use_student else "continuous_v2_teacher"
            rollout_policies[rollout_name] += 1
            games += 1
            skip_draw = True
            steps = 0
            last_p0_state: Dict[str, Any] = {}
            deceive_active_until = -1

            while not game.is_game_over and steps < args.max_steps and rows < args.target_states:
                steps += 1
                if game.deck.remaining_count() == 0 or sum(player.is_hu for player in game.players) >= 3:
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
                    teacher_action = teacher.decide(game, legal_actions)
                    teacher_state = dict(teacher.last_decision)
                    action = rollout_agent.decide(game, legal_actions) if use_student else teacher_action
                    rollout_state = dict(rollout_agent.last_decision)
                    if action != teacher_action:
                        action_disagreements += 1
                    candidate_scores = teacher_state.get("candidate_scores") or {}
                    if len(candidate_scores) >= 2 and steps % max(1, args.sample_stride) == 0:
                        attempts += 1
                        candidate_rows = build_action_candidates(
                            game,
                            0,
                            legal_actions,
                            max_shanten_regret=0,
                            min_ukeire_ratio=1.0,
                        )
                        candidate_rows = [
                            item for item in candidate_rows if item["action"] in candidate_scores
                        ]
                        if len(candidate_rows) >= 2 and teacher_action in {
                            item["action"] for item in candidate_rows
                        }:
                            state_features = extract_state_features(
                                game,
                                0,
                                teacher_state.get("z_state", {}),
                                teacher_state.get("beliefs", {}),
                                teacher_state.get("gate", {}),
                            )
                            row = {
                                "state_id": f"g{game_index:05d}_s{steps:03d}",
                                "game_seed": game_seed,
                                "step": steps,
                                "rollout_policy": rollout_name,
                                "state_features": state_features,
                                "legal_actions": legal_actions,
                                "candidates": [
                                    {
                                        "action": item["action"],
                                        "features": item["features"],
                                        "teacher_score": float(candidate_scores[item["action"]]),
                                        "teacher_q_base": float(
                                            (teacher_state.get("candidate_q_base") or {}).get(item["action"], 0.0)
                                        ),
                                        "teacher_delta_shape": float(
                                            (teacher_state.get("candidate_delta_shape") or {}).get(item["action"], 0.0)
                                        ),
                                    }
                                    for item in candidate_rows
                                ],
                                "teacher_action": teacher_action,
                                "teacher_mode": str(teacher_state.get("mode", "exploit")),
                                "teacher_alpha": float(teacher_state.get("alpha", 0.0) or 0.0),
                                "student_action": action if use_student else None,
                                "student_mode": str(rollout_state.get("mode", "exploit")) if use_student else None,
                            }
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                            rows += 1
                            teacher_modes[row["teacher_mode"]] += 1
                            if rows == 1 or rows % 100 == 0:
                                print(
                                    f"[ActionValueTeacher] rows={rows}/{args.target_states} "
                                    f"games={games} modes={dict(teacher_modes)}",
                                    flush=True,
                                )
                    actual_mode = str(rollout_state.get("mode", "exploit"))
                    last_p0_state = {
                        "step": steps,
                        "mode": actual_mode,
                        "action": action,
                        "reason": rollout_state.get("reason", ""),
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
                response = resolve_responses(game, discarded_tile, pid, rollout_agent, defenders)
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
        "attempts": attempts,
        "teacher_modes": dict(teacher_modes),
        "rollout_policies": dict(rollout_policies),
        "action_disagreements": action_disagreements,
        "rows_path": str(rows_path),
        "config": vars(args),
    }
    write_json(args.output_dir / "action_value_teacher_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-states", type=int, default=6000)
    parser.add_argument("--max-games", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--student-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026072901)
    parser.add_argument("--backend", choices=["heuristic_fallback", "local_qwen"], default="local_qwen")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--student-neural-gate-model-path", required=True)
    parser.add_argument("--neural-opponent-model-path", default="Neural_opponent_model/neural_opponent_policy.pth")
    args = parser.parse_args()
    if not 0.0 <= args.student_fraction <= 1.0:
        raise ValueError("--student-fraction must be in [0, 1]")
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
