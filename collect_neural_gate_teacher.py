"""Collect online-feature neural gate distillation data from a rule teacher.

The teacher chooses the actual action/mode, usually continuous_v2.  The collected
row uses the same feature extractor and mode availability constraints as the
deployed neural gate, so train/serve feature and action-set drift is visible and
measurable.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from experiment_trace import ensure_deterministic_hashing, write_json
from llm_backends import build_llm_callable
from mask_llm import MASKLLMAgent
from policy_metrics import discard_progress_metrics
from ppo_features import extract_state_features
from prompt_builder import get_legal_actions
from run_gate1_experiments import execute_action, init_game, resolve_responses


MODES = ("exploit", "safe", "deceive")


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def defender_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "threat_threshold": args.threat_fold_threshold,
        "oracle_samples": args.oracle_samples,
        "beta": args.oracle_beta,
        "danger_threshold": args.danger_threshold,
        "ffr_hand_shanten": args.ffr_hand_shanten,
        "threat_model": args.defender_threat_model,
        "tell_weight": args.defender_tell_weight,
        "tell_window": args.defender_tell_window,
        "learned_model_path": args.defender_learned_model_path,
        "neural_model_path": args.neural_opponent_model_path,
        "neural_device": args.neural_opponent_device,
    }


def make_agent(
    seed: int,
    llm,
    args: argparse.Namespace,
    gate_policy: str,
    neural_gate=None,
) -> MASKLLMAgent:
    return MASKLLMAgent(
        player_id=0,
        decision_llm=llm,
        belief_llm=llm,
        mc_seed=seed * 13 + 1,
        mc_oracle_samples=args.mask_oracle_samples,
        mc_beta=args.mask_oracle_beta,
        mc_danger_threshold=args.mask_danger_threshold,
        dir_ready_threshold=args.mask_dir_ready_threshold,
        deceive_threat_ceiling=args.mask_deceive_threat_ceiling,
        forced_deceive=args.mask_forced_deceive,
        deceive_style=args.mask_deceive_style,
        threat_allow_break_ready=args.mask_threat_allow_break_ready,
        threat_max_result_shanten=args.mask_threat_max_result_shanten,
        threat_max_shanten_regret=args.mask_threat_max_shanten_regret,
        threat_min_ukeire_ratio=args.mask_threat_min_ukeire_ratio,
        threat_gate_threshold=args.mask_threat_gate_threshold,
        threat_gate_margin=args.mask_threat_gate_margin,
        threat_min_delta=args.mask_threat_min_delta,
        threat_gate_mode=args.mask_threat_gate_mode,
        threat_response_model=args.mask_threat_response_model,
        threat_response_tell_weight=args.mask_threat_response_tell_weight,
        threat_tell_window=args.mask_threat_tell_window,
        threat_max_start_shanten=args.mask_threat_max_start_shanten,
        threat_require_non_exploit=not args.mask_threat_allow_exploit_overlap,
        threat_require_real_target=args.mask_threat_require_real_target,
        threat_target_max_shanten=args.mask_threat_target_max_shanten,
        threat_target_signal=args.mask_threat_target_signal,
        threat_target_prob_threshold=args.mask_threat_target_prob_threshold,
        log_counterfactual=False,
        gate_policy=gate_policy,
        neural_gate=neural_gate,
        neural_gate_deceive_margin=args.student_deceive_margin,
        neural_gate_safe_margin=args.student_safe_margin,
        neural_gate_safe_risk_threshold=args.student_safe_risk_threshold,
        auto_hu=args.auto_hu,
    )


def neural_available_modes(
    agent: MASKLLMAgent,
    game,
    legal_actions: List[str],
    beliefs: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    exploit_action, exploit_reason = agent._exploit_action(game, legal_actions)
    safe_action = agent._safe_discard(game, legal_actions, fallback_action=exploit_action)
    actions: Dict[str, str] = {"exploit": exploit_action}
    available = ["exploit"]
    if any(action.startswith("d ") for action in legal_actions):
        available.append("safe")
        actions["safe"] = safe_action

    rng_state = agent.rng.getstate()
    try:
        deceive_action = agent._deceptive_discard(
            game,
            legal_actions,
            beliefs,
            relaxed_gate=True,
        )
    finally:
        agent.rng.setstate(rng_state)

    can_deceive = bool(deceive_action and deceive_action.startswith("d ") and deceive_action in legal_actions)
    block_reason: Optional[str] = None
    if not deceive_action:
        block_reason = (agent._last_deceive_signal or {}).get("blocked_reason") or "candidate_unavailable"
    elif not deceive_action.startswith("d "):
        block_reason = "candidate_not_discard"
    if can_deceive and agent.deceive_style == "threat" and agent.threat_require_non_exploit:
        if deceive_action == exploit_action:
            can_deceive = False
            block_reason = "same_as_exploit"
        elif not exploit_action.startswith("d "):
            can_deceive = False
            block_reason = "counterfactual_exploit_not_discard"
    if can_deceive:
        available.append("deceive")
        actions["deceive"] = deceive_action
        block_reason = None

    return available, {
        "mode_actions": actions,
        "exploit_reason": exploit_reason,
        "deceive_action": deceive_action,
        "can_deceive": can_deceive,
        "deceive_block_reason": block_reason,
        "deceive_signal": agent._last_deceive_signal,
    }


def _action_tile(game, player_id: int, action: str):
    if not action.startswith("d "):
        return None
    tile_text = action[2:]
    return next((tile for tile in game.players[player_id].hand_tiles if str(tile) == tile_text), None)


def _hand_after_action(game, player_id: int, action: str) -> Optional[List[Any]]:
    tile = _action_tile(game, player_id, action)
    if tile is None:
        return None
    hand = game.players[player_id].hand_tiles.copy()
    hand.remove(tile)
    return hand


def _public_safe_score(game, player_id: int, action: str) -> float:
    tile = _action_tile(game, player_id, action)
    if tile is None:
        return 0.0
    public_discards = {
        (discard.suit, discard.number)
        for player in game.players
        for discard in player.discarded_tiles
    }
    if (tile.suit, tile.number) in public_discards:
        return 1.0
    if tile.number in (1, 9):
        return 0.5
    return 0.0


def value_delta_mode_rewards(
    agent: MASKLLMAgent,
    game,
    available_modes: List[str],
    diagnostics: Dict[str, Any],
    beliefs: Dict[str, Any],
    gate: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[Dict[str, float], str, Dict[str, Any]]:
    """Short-horizon proxy value for v3 neural-gate labels.

    This is deliberately local and low-variance.  It scores the actual candidate
    action produced for each mode at the same state, then only permits a
    deceive label when its score clears the best exploit/safe alternative by a
    positive margin.  Otherwise can-deceive states become explicit negative
    examples for deceive.
    """
    actions = diagnostics.get("mode_actions") or {}
    unique_discards = {
        action
        for mode in available_modes
        for action in [actions.get(mode)]
        if isinstance(action, str) and action.startswith("d ")
    }
    progress = {
        action: value
        for action in unique_discards
        for value in [discard_progress_metrics(game, 0, action)]
        if value is not None
    }
    if not progress:
        target = str(diagnostics.get("teacher_mode") or "exploit")
        return {mode: (args.teacher_reward_margin if mode == target else 0.0) for mode in available_modes}, target, {
            "fallback": "no_discard_progress",
        }

    best_shanten = min(value["shanten"] for value in progress.values())
    best_ukeire_same_shanten = max(
        (value["effective_copies"] for value in progress.values() if value["shanten"] == best_shanten),
        default=0,
    )
    player = game.players[0]
    tell_before = agent._discard_tell_threat(game)
    avg_belief_conf = 0.0
    if beliefs:
        confs = [float(value.get("tenpai_confidence", 0.0)) for value in beliefs.values()]
        avg_belief_conf = _clip(sum(confs) / len(confs)) if confs else 0.0
    risk_budget = _clip(float(gate.get("risk_budget", 0.0))) if isinstance(gate, dict) else 0.0
    signal = diagnostics.get("deceive_signal") or {}
    response_delta = float(signal.get("response_delta", signal.get("tell_delta", 0.0)) or 0.0)

    rewards: Dict[str, float] = {}
    details: Dict[str, Any] = {
        "best_shanten": best_shanten,
        "best_ukeire_same_shanten": best_ukeire_same_shanten,
        "risk_budget": round(risk_budget, 4),
        "avg_belief_conf": round(avg_belief_conf, 4),
        "response_delta": round(response_delta, 4),
        "mode_actions": actions,
        "mode_details": {},
    }

    for mode in available_modes:
        action = actions.get(mode)
        metrics = progress.get(action)
        if not isinstance(action, str) or metrics is None:
            rewards[mode] = -100.0
            details["mode_details"][mode] = {"action": action, "score": rewards[mode], "reason": "no_progress"}
            continue

        hand_after = _hand_after_action(game, 0, action)
        potential_fan = 0
        shape_direction = 0.0
        if hand_after is not None:
            if metrics["shanten"] == 0:
                potential_fan = int(MASKLLMAgent._candidate_potential_fan(player, hand_after))
            shape_direction = float(
                MASKLLMAgent._hand_shape_fan_direction(hand_after, player.open_melds, player.missing_suit)
            )
        tile = _action_tile(game, 0, action)
        tell_after = agent._discard_tell_threat(game, extra_tile=tile) if tile is not None else tell_before
        tell_delta = tell_after - tell_before
        shanten_regret = max(0, int(metrics["shanten"]) - int(best_shanten))
        ukeire_loss = max(0, int(best_ukeire_same_shanten) - int(metrics["effective_copies"]))
        safe_score = _public_safe_score(game, 0, action)

        score = (
            -args.v3_w_shanten * float(metrics["shanten"])
            + args.v3_w_ukeire * float(metrics["effective_copies"])
            + args.v3_w_fan * float(potential_fan)
            + args.v3_w_shape * shape_direction
        )
        if mode == "safe":
            score += args.v3_w_safe * risk_budget * safe_score
        if mode == "deceive":
            deceive_gain = (
                args.v3_w_deceive_delta * max(0.0, response_delta)
                + args.v3_w_deceive_tell * max(0.0, tell_delta) * (1.0 - avg_belief_conf)
            )
            deceive_cost = (
                args.v3_w_deceive_shanten_cost * shanten_regret
                + args.v3_w_deceive_ukeire_cost * (ukeire_loss / max(1.0, float(best_ukeire_same_shanten)))
                + args.v3_w_deceive_low_value_cost * max(0.0, 1.0 - min(1.0, potential_fan / 2.0))
            )
            score += deceive_gain - deceive_cost

        rewards[mode] = float(score)
        details["mode_details"][mode] = {
            "action": action,
            "score": round(float(score), 4),
            "shanten": int(metrics["shanten"]),
            "effective_copies": int(metrics["effective_copies"]),
            "potential_fan": int(potential_fan),
            "shape_direction": round(shape_direction, 4),
            "tell_delta": round(tell_delta, 4),
            "safe_score": safe_score,
            "shanten_regret": shanten_regret,
            "ukeire_loss": ukeire_loss,
        }

    teacher_mode = str(diagnostics.get("teacher_mode") or "exploit")
    exploit_action = actions.get("exploit")
    safe_action = actions.get("safe")
    exploit_safe_same_action = bool(
        isinstance(exploit_action, str)
        and isinstance(safe_action, str)
        and exploit_action == safe_action
    )
    details["teacher_mode"] = teacher_mode
    details["exploit_safe_same_action"] = exploit_safe_same_action
    details["risk_budget"] = round(risk_budget, 4)

    if "deceive" in rewards:
        alt_modes = [mode for mode in ("exploit", "safe") if mode in rewards]
        if alt_modes:
            best_alt = max(alt_modes, key=lambda mode: rewards[mode])
            deceive_delta = rewards["deceive"] - rewards[best_alt]
            details["deceive_delta_vs_best_alt"] = round(deceive_delta, 4)
            details["best_alt_mode"] = best_alt
            if deceive_delta < args.v3_deceive_positive_margin:
                rewards["deceive"] = min(
                    rewards["deceive"],
                    rewards[best_alt] - args.v3_deceive_negative_gap,
                )
                details["deceive_forced_negative"] = True
            else:
                details["deceive_forced_negative"] = False

    target = max(sorted(rewards), key=lambda mode: rewards[mode])
    raw_value_target = target

    # Keep value-delta for high-confidence overrides, but do not let a tiny
    # local safe bonus rewrite the continuous teacher's exploit prior. If
    # exploit/safe produce the same discard, retain only a low-weight anchor.
    if target == "safe" and "exploit" in rewards:
        safe_gap = rewards["safe"] - rewards["exploit"]
        details["safe_delta_vs_exploit"] = round(safe_gap, 4)
        if exploit_safe_same_action or safe_gap < args.v31_safe_positive_margin:
            target = teacher_mode if teacher_mode in available_modes else "exploit"
            if target == "deceive" and "deceive" in rewards:
                target = "exploit"
            details["safe_override_rejected"] = True
        else:
            details["safe_override_rejected"] = False
    if target not in available_modes:
        target = "exploit" if "exploit" in available_modes else available_modes[0]

    mode_decision_weight = 1.0
    if exploit_safe_same_action and target in {"exploit", "safe"}:
        mode_decision_weight = args.v31_same_action_weight
    if rewards:
        best_non_target = max(
            (value for mode, value in rewards.items() if mode != target),
            default=rewards[target],
        )
        if rewards[target] <= best_non_target:
            rewards[target] = best_non_target + args.v31_target_reward_margin

    details["raw_value_target"] = raw_value_target
    details["final_target"] = target
    details["mode_decision_weight"] = mode_decision_weight
    return rewards, target, details


def aligned_v32_mode_rewards(
    agent: MASKLLMAgent,
    game,
    available_modes: List[str],
    diagnostics: Dict[str, Any],
    beliefs: Dict[str, Any],
    gate: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[Dict[str, float], str, Dict[str, Any]]:
    """Action-aligned labels for the second neural-gate distillation pass.

    A mode label is useful only when it changes the discard. Same-action
    exploit/safe rows are dropped unless they form a real deceive-vs-normal
    decision. Distinct safe labels additionally have to clear risk and hand
    progress constraints, so the student cannot learn "safe" as a generic
    replacement for exploit.
    """
    rewards, value_target, details = value_delta_mode_rewards(
        agent,
        game,
        available_modes,
        diagnostics,
        beliefs,
        gate,
        args,
    )
    actions = diagnostics.get("mode_actions") or {}
    exploit_action = actions.get("exploit")
    safe_action = actions.get("safe")
    deceive_action = actions.get("deceive")
    teacher_action = diagnostics.get("teacher_action")
    same_exploit_safe = bool(
        isinstance(exploit_action, str)
        and exploit_action == safe_action
    )
    deceive_is_distinct = bool(
        isinstance(deceive_action, str)
        and deceive_action not in {exploit_action, safe_action}
    )
    deceive_positive = bool(
        deceive_is_distinct
        and "deceive" in rewards
        and not details.get("deceive_forced_negative", False)
        and float(details.get("deceive_delta_vs_best_alt", -1e9))
        >= args.v3_deceive_positive_margin
    )

    details["aligned_v32"] = True
    details["teacher_action"] = teacher_action
    details["deceive_is_distinct"] = deceive_is_distinct

    if same_exploit_safe:
        if deceive_positive:
            target = "deceive"
            decision_type = "deceive_positive"
        elif deceive_is_distinct and "deceive" in rewards:
            target = "exploit"
            decision_type = "deceive_negative"
        else:
            details["mode_decision_weight"] = 0.0
            details["decision_type"] = "same_action_drop"
            details["final_target"] = "exploit"
            return rewards, "exploit", details
    else:
        matched_modes = [
            mode for mode in available_modes
            if actions.get(mode) == teacher_action
        ]
        if len(matched_modes) == 1:
            target = matched_modes[0]
            decision_type = "teacher_action_match"
        elif deceive_positive:
            target = "deceive"
            decision_type = "deceive_positive"
        else:
            target = value_target
            decision_type = "value_delta"

        if target == "deceive" and not deceive_positive:
            target = "exploit"
            decision_type = "deceive_negative"

        if target == "safe":
            mode_details = details.get("mode_details") or {}
            exploit_metrics = mode_details.get("exploit") or {}
            safe_metrics = mode_details.get("safe") or {}
            risk_budget = float(details.get("risk_budget", 0.0) or 0.0)
            safe_gain = float(safe_metrics.get("safe_score", 0.0) or 0.0) - float(
                exploit_metrics.get("safe_score", 0.0) or 0.0
            )
            exploit_shanten = int(exploit_metrics.get("shanten", 99))
            safe_shanten = int(safe_metrics.get("shanten", 99))
            exploit_ukeire = int(exploit_metrics.get("effective_copies", 0))
            safe_ukeire = int(safe_metrics.get("effective_copies", 0))
            progress_ok = bool(
                safe_shanten <= exploit_shanten + args.aligned_safe_max_shanten_regret
                and safe_ukeire >= exploit_ukeire * args.aligned_safe_min_ukeire_ratio
            )
            teacher_safe_match = bool(
                teacher_action == safe_action
                and teacher_action != exploit_action
            )
            safe_allowed = bool(
                safe_action != exploit_action
                and risk_budget >= args.aligned_safe_min_risk
                and (teacher_safe_match or safe_gain >= args.aligned_safe_min_gain)
                and progress_ok
            )
            details["aligned_safe_gain"] = round(safe_gain, 4)
            details["aligned_safe_progress_ok"] = progress_ok
            details["aligned_teacher_safe_match"] = teacher_safe_match
            details["aligned_safe_allowed"] = safe_allowed
            if not safe_allowed:
                target = "exploit"
                decision_type = "safe_rejected"

    if target not in available_modes:
        target = "exploit" if "exploit" in available_modes else available_modes[0]
    if target in rewards:
        best_other = max(
            (score for mode, score in rewards.items() if mode != target),
            default=rewards[target],
        )
        if rewards[target] <= best_other:
            rewards[target] = best_other + args.v31_target_reward_margin

    details["mode_decision_weight"] = 1.0
    details["decision_type"] = decision_type
    details["final_target"] = target
    return rewards, target, details


def run(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_deterministic_hashing()
    random.seed(args.seed)
    repo_dir = Path(__file__).resolve().parent
    llm = build_llm_callable(
        backend=args.backend,
        repo_dir=repo_dir,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
    )
    student_gate = None
    if args.rollout_policy == "student":
        if not args.student_neural_gate_model_path:
            raise ValueError("student rollout requires --student-neural-gate-model-path")
        from neural_gate_policy import NeuralGatePolicy

        student_gate = NeuralGatePolicy.load(
            args.student_neural_gate_model_path,
            device=args.student_neural_gate_device,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "teacher_gate_states.jsonl"
    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    attempts = 0
    rows = 0
    quotas = {
        "exploit": args.target_exploit,
        "safe": args.target_safe,
        "deceive": args.target_deceive,
    }
    quota_mode = any(value > 0 for value in quotas.values())

    def collection_done() -> bool:
        base_done = rows >= args.target_states
        if quota_mode:
            base_done = all(counts[mode] >= target for mode, target in quotas.items())
        negative_done = (
            args.target_deceive_negative <= 0
            or counts["deceive_negative"] >= args.target_deceive_negative
        )
        return base_done and negative_done

    with rows_path.open("w", encoding="utf-8") as handle:
        for game_index in range(args.max_games):
            if collection_done():
                break
            game_seed = args.seed + game_index
            game, opponent_funcs, defenders = init_game(
                game_seed,
                args.opponent_style,
                f"NeuralGateTeacher_{game_seed}",
                defender_config(args),
            )
            teacher = make_agent(game_seed, llm, args, args.teacher_gate_policy)
            candidate_agent = make_agent(game_seed, llm, args, "rule")
            student = (
                make_agent(game_seed, llm, args, "neural", neural_gate=student_gate)
                if student_gate is not None
                else None
            )
            skip_draw = True
            steps = 0
            last_p0_state: Dict[str, Any] = {}
            deceive_active_until = -1

            while not game.is_game_over and steps < args.max_steps and not collection_done():
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
                    decision_state = dict(teacher.last_decision)
                    teacher_mode = str(decision_state.get("mode", "exploit"))
                    if student is not None:
                        action = student.decide(game, legal_actions)
                        student_state = dict(student.last_decision)
                        actual_mode = str(student_state.get("mode", "exploit"))
                    else:
                        action = teacher_action
                        student_state = {}
                        actual_mode = teacher_mode
                    if (
                        "h" not in legal_actions
                        and any(item.startswith("d ") for item in legal_actions)
                        and steps % args.sample_stride == 0
                    ):
                        attempts += 1
                        available, diagnostics = neural_available_modes(
                            candidate_agent,
                            game,
                            legal_actions,
                            decision_state.get("beliefs", {}),
                        )
                        diagnostics["teacher_mode"] = teacher_mode
                        diagnostics["teacher_action"] = teacher_action
                        if args.label_policy == "aligned_v32":
                            mode_rewards, target_mode, value_diagnostics = aligned_v32_mode_rewards(
                                candidate_agent,
                                game,
                                available,
                                diagnostics,
                                decision_state.get("beliefs", {}),
                                decision_state.get("gate", {}),
                                args,
                            )
                        elif args.label_policy == "value_delta":
                            mode_rewards, target_mode, value_diagnostics = value_delta_mode_rewards(
                                candidate_agent,
                                game,
                                available,
                                diagnostics,
                                decision_state.get("beliefs", {}),
                                decision_state.get("gate", {}),
                                args,
                            )
                        else:
                            target_mode = teacher_mode
                            mode_rewards = {
                                mode: (args.teacher_reward_margin if mode == target_mode else 0.0)
                                for mode in available
                            }
                            value_diagnostics = {"label_policy": "teacher"}
                        is_deceive_negative = bool(
                            diagnostics.get("can_deceive") and target_mode != "deceive"
                        )
                        mode_decision_weight = float(
                            value_diagnostics.get("mode_decision_weight", 1.0) or 0.0
                        )
                        if mode_decision_weight <= 0.0:
                            skipped[str(value_diagnostics.get("decision_type") or "zero_weight")] += 1
                        elif target_mode not in available:
                            skipped[f"target_unavailable_{target_mode}"] += 1
                        elif (
                            quota_mode
                            and counts[target_mode] >= quotas[target_mode]
                            and not (
                                is_deceive_negative
                                and args.target_deceive_negative > 0
                                and counts["deceive_negative"] < args.target_deceive_negative
                            )
                        ):
                            skipped[f"quota_full_{target_mode}"] += 1
                        else:
                            features = extract_state_features(
                                game,
                                0,
                                decision_state.get("z_state", {}),
                                decision_state.get("beliefs", {}),
                                decision_state.get("gate", {}),
                            )
                            row = {
                                "state_id": f"g{game_index:04d}_s{steps:03d}",
                                "game_seed": game_seed,
                                "game_index": game_index,
                                "step": steps,
                                "teacher_gate_policy": args.teacher_gate_policy,
                                "label_policy": args.label_policy,
                                "target_mode": target_mode,
                                "teacher_mode": teacher_mode,
                                "student_mode": actual_mode if student is not None else None,
                                "student_action": action if student is not None else None,
                                "student_teacher_mode_disagree": bool(
                                    student is not None and actual_mode != teacher_mode
                                ),
                                "student_teacher_action_disagree": bool(
                                    student is not None and action != teacher_action
                                ),
                                "student_gate": student_state.get("learned_gate", {}),
                                "rule_mode": target_mode,
                                "available_modes": available,
                                "features": features,
                                "mode_rewards": mode_rewards,
                                "mode_decision_weight": mode_decision_weight,
                                "decision_type": value_diagnostics.get("decision_type"),
                                "value_diagnostics": value_diagnostics,
                                "teacher_action": teacher_action,
                                "teacher_reason": decision_state.get("reason", ""),
                                "teacher_alpha": decision_state.get("alpha"),
                                "teacher_value_gate": decision_state.get("value_gate"),
                                "teacher_q_base": decision_state.get("q_base"),
                                "teacher_delta_shape": decision_state.get("delta_shape"),
                                "legal_actions": legal_actions,
                                **diagnostics,
                            }
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                            rows += 1
                            counts[target_mode] += 1
                            if is_deceive_negative:
                                counts["deceive_negative"] += 1
                            if rows == 1 or rows % 25 == 0:
                                print(
                                    f"[NeuralGateTeacher] rows={rows}/{args.target_states} "
                                    f"attempts={attempts} counts={dict(counts)} skipped={dict(skipped)}",
                                    flush=True,
                                )

                    last_p0_state = {
                        "step": steps,
                        "mode": actual_mode,
                        "action": action,
                        "reason": decision_state.get("reason", ""),
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
                resolve_responses(
                    game,
                    discarded_tile,
                    pid,
                    student if student is not None else teacher,
                    defenders,
                )
                if game.is_game_over:
                    break

    summary = {
        "rows": rows,
        "attempts": attempts,
        "target_counts": dict(counts),
        "skipped": dict(skipped),
        "rows_path": str(rows_path),
        "config": {
            "teacher_gate_policy": args.teacher_gate_policy,
            "rollout_policy": args.rollout_policy,
            "student_neural_gate_model_path": args.student_neural_gate_model_path,
            "student_neural_gate_device": args.student_neural_gate_device,
            "student_deceive_margin": args.student_deceive_margin,
            "student_safe_margin": args.student_safe_margin,
            "student_safe_risk_threshold": args.student_safe_risk_threshold,
            "opponent_style": args.opponent_style,
            "backend": args.backend,
            "sample_stride": args.sample_stride,
            "target_states": args.target_states,
            "max_games": args.max_games,
            "label_policy": args.label_policy,
            "teacher_reward_margin": args.teacher_reward_margin,
            "target_deceive_negative": args.target_deceive_negative,
            "quotas": quotas,
            "quota_mode": quota_mode,
            "v3_value_delta": {
                "deceive_positive_margin": args.v3_deceive_positive_margin,
                "deceive_negative_gap": args.v3_deceive_negative_gap,
                "w_shanten": args.v3_w_shanten,
                "w_ukeire": args.v3_w_ukeire,
                "w_fan": args.v3_w_fan,
                "w_shape": args.v3_w_shape,
                "w_safe": args.v3_w_safe,
                "w_deceive_delta": args.v3_w_deceive_delta,
                "w_deceive_tell": args.v3_w_deceive_tell,
                "w_deceive_shanten_cost": args.v3_w_deceive_shanten_cost,
                "w_deceive_ukeire_cost": args.v3_w_deceive_ukeire_cost,
                "w_deceive_low_value_cost": args.v3_w_deceive_low_value_cost,
                "v31_safe_positive_margin": args.v31_safe_positive_margin,
                "v31_same_action_weight": args.v31_same_action_weight,
                "v31_target_reward_margin": args.v31_target_reward_margin,
                "aligned_safe_min_risk": args.aligned_safe_min_risk,
                "aligned_safe_min_gain": args.aligned_safe_min_gain,
                "aligned_safe_max_shanten_regret": args.aligned_safe_max_shanten_regret,
                "aligned_safe_min_ukeire_ratio": args.aligned_safe_min_ukeire_ratio,
            },
        },
    }
    write_json(args.output_dir / "teacher_gate_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-states", type=int, default=1000)
    parser.add_argument("--target-exploit", type=int, default=0)
    parser.add_argument("--target-safe", type=int, default=0)
    parser.add_argument("--target-deceive", type=int, default=0)
    parser.add_argument("--target-deceive-negative", type=int, default=0,
                        help="Extra quota for can_deceive=True rows whose value-delta target is exploit/safe.")
    parser.add_argument("--max-games", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026072301)
    parser.add_argument("--backend", default="heuristic_fallback")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--teacher-gate-policy", default="continuous_v2")
    parser.add_argument("--rollout-policy", choices=["teacher", "student"], default="teacher")
    parser.add_argument("--student-neural-gate-model-path")
    parser.add_argument("--student-neural-gate-device", default="cpu")
    parser.add_argument("--student-deceive-margin", type=float, default=0.05)
    parser.add_argument("--student-safe-margin", type=float, default=0.05)
    parser.add_argument("--student-safe-risk-threshold", type=float, default=0.30)
    parser.add_argument("--auto-hu", choices=["common", "off"], default="common")
    parser.add_argument(
        "--label-policy",
        choices=["teacher", "value_delta", "aligned_v32"],
        default="teacher",
    )
    parser.add_argument("--teacher-reward-margin", type=float, default=10.0)
    parser.add_argument("--v3-deceive-positive-margin", type=float, default=5.0,
                        help="Require V(deceive)-V(best exploit/safe) above this margin to label deceive.")
    parser.add_argument("--v3-deceive-negative-gap", type=float, default=3.0,
                        help="When deceive is below the positive margin, force it below best alt by this gap.")
    parser.add_argument("--v3-w-shanten", type=float, default=100.0)
    parser.add_argument("--v3-w-ukeire", type=float, default=1.0)
    parser.add_argument("--v3-w-fan", type=float, default=10.0)
    parser.add_argument("--v3-w-shape", type=float, default=3.0)
    parser.add_argument("--v3-w-safe", type=float, default=10.0)
    parser.add_argument("--v3-w-deceive-delta", type=float, default=80.0)
    parser.add_argument("--v3-w-deceive-tell", type=float, default=40.0)
    parser.add_argument("--v3-w-deceive-shanten-cost", type=float, default=120.0)
    parser.add_argument("--v3-w-deceive-ukeire-cost", type=float, default=30.0)
    parser.add_argument("--v3-w-deceive-low-value-cost", type=float, default=12.0)
    parser.add_argument("--v31-safe-positive-margin", type=float, default=4.0,
                        help="Safe overrides exploit only when V_safe - V_exploit clears this margin.")
    parser.add_argument("--v31-same-action-weight", type=float, default=0.15,
                        help="CE weight for exploit/safe labels when both modes choose the same discard.")
    parser.add_argument("--v31-target-reward-margin", type=float, default=2.0,
                        help="Reward margin injected when v3.1 hybrid relabeling overrides local value argmax.")
    parser.add_argument("--aligned-safe-min-risk", type=float, default=0.30,
                        help="aligned_v32: minimum risk budget for a safe label.")
    parser.add_argument("--aligned-safe-min-gain", type=float, default=0.5,
                        help="aligned_v32: minimum public safety-score gain over exploit.")
    parser.add_argument("--aligned-safe-max-shanten-regret", type=int, default=0,
                        help="aligned_v32: maximum safe-vs-exploit shanten regression.")
    parser.add_argument("--aligned-safe-min-ukeire-ratio", type=float, default=0.8,
                        help="aligned_v32: minimum safe/exploit ukeire ratio.")
    parser.add_argument("--opponent-style", choices=["greedy", "aggressive", "conservative", "random", "mixed", "responsive", "neural"], default="neural")
    parser.add_argument("--neural-opponent-model-path", default="Neural_opponent_model/neural_opponent_policy.pth")
    parser.add_argument("--neural-opponent-device", default="cpu")
    parser.add_argument("--threat-fold-threshold", type=float, default=0.7)
    parser.add_argument("--oracle-samples", type=int, default=30)
    parser.add_argument("--oracle-beta", type=float, default=2.0)
    parser.add_argument("--danger-threshold", type=int, default=1)
    parser.add_argument("--ffr-hand-shanten", type=int, default=1)
    parser.add_argument("--defender-threat-model", choices=["mc", "discard_tell", "blend", "learned"], default="blend")
    parser.add_argument("--defender-tell-weight", type=float, default=0.3)
    parser.add_argument("--defender-tell-window", type=int, default=6)
    parser.add_argument("--defender-learned-model-path", default="Defender_danger_model/danger_model.pth")
    parser.add_argument("--mask-oracle-samples", type=int, default=30)
    parser.add_argument("--mask-oracle-beta", type=float, default=2.0)
    parser.add_argument("--mask-danger-threshold", type=int, default=1)
    parser.add_argument("--mask-dir-ready-threshold", type=int, default=0)
    parser.add_argument("--mask-deceive-threat-ceiling", type=float, default=0.5)
    parser.add_argument("--mask-forced-deceive", choices=["off", "eligible", "always"], default="off")
    parser.add_argument("--mask-deceive-style", choices=["safe", "threat"], default="threat")
    parser.add_argument("--mask-threat-allow-break-ready", action="store_true")
    parser.add_argument("--mask-threat-max-result-shanten", type=int, default=0)
    parser.add_argument("--mask-threat-max-shanten-regret", type=int, default=0)
    parser.add_argument("--mask-threat-min-ukeire-ratio", type=float, default=1.0)
    parser.add_argument("--mask-threat-gate-threshold", type=float, default=0.4)
    parser.add_argument("--mask-threat-gate-margin", type=float, default=0.12)
    parser.add_argument("--mask-threat-min-delta", type=float, default=0.03)
    parser.add_argument("--mask-threat-gate-mode", choices=["cross", "delta_only"], default="cross")
    parser.add_argument("--mask-threat-response-model", choices=["tell", "blend"], default="tell")
    parser.add_argument("--mask-threat-response-tell-weight", type=float, default=1.0)
    parser.add_argument("--mask-threat-tell-window", type=int, default=6)
    parser.add_argument("--mask-threat-max-start-shanten", type=int, default=3)
    parser.add_argument("--mask-threat-allow-exploit-overlap", action="store_true")
    parser.add_argument("--mask-threat-require-real-target", action="store_true")
    parser.add_argument("--mask-threat-target-max-shanten", type=int, default=0)
    parser.add_argument("--mask-threat-target-signal", choices=["oracle", "mc"], default="oracle")
    parser.add_argument("--mask-threat-target-prob-threshold", type=float, default=0.5)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
