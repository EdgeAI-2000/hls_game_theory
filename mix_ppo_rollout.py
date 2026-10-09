"""Complete P0 trajectories using the P4 engine's action and settlement semantics."""

import random
from collections import Counter

from game import bot_decide_response
from mix_ppo_baseline import ACTIONS, public_planes, action_mask
from prompt_builder import get_legal_actions
from run_gate1_experiments import init_game, execute_action
from rule_engine import ShantenCalculator


def episode(seed, opponent_style="curriculum", manifest="Opponent_Curriculum_formal/screened.json",
            split="train", max_steps=300, reward_scale=100.0, context_inputs=False, prospective_inputs=False,
            search_inputs=False, reward_mode="net", opponent_checkpoint=None,
            privileged_inputs=False, privileged_scale=0.):
    """Yield (planes, mask, balance); receive an action index; return settled metrics.

    Each generator owns Python RNG state so interleaving episodes does not
    change walls or stochastic opponents. No global policy RNG is used here.
    """
    if reward_scale <= 0:
        raise ValueError("reward_scale must be positive")
    if split != "train" and privileged_scale != 0:
        raise ValueError("Privileged information is forbidden on holdout")
    cfg = {"curriculum_manifest": manifest, "curriculum_split": split}
    game, opponents, defenders = init_game(seed, opponent_style, f"mix_ppo_{seed}", cfg)
    start = game.players[0].balance
    own_rng = random.getstate()
    steps, decisions = 0, 0
    ron, dealt_in = False, False
    behavior = Counter()
    progress = []
    if reward_mode not in ("net", "paper_adapted"):
        raise ValueError("Unknown reward mode")

    def progress_state():
        p = game.players[0]
        return (ShantenCalculator.calculate_shanten(p.hand_tiles, p.missing_suit),
                p.calculate_potential_fan()[0])

    def decide(legal, response=False):
        nonlocal own_rng, decisions
        own_rng = random.getstate()
        obs = public_planes(game, 0, legal if context_inputs else None, response, prospective_inputs, search_inputs)
        if privileged_inputs:
            from mix_ppo_baseline import privileged_planes
            obs = privileged_planes(game, 0, obs, privileged_scale)
        if reward_mode == "paper_adapted":
            progress.append(progress_state())
        index = yield (obs, action_mask(legal), game.players[0].balance)
        random.setstate(own_rng)
        action = ACTIONS[index]
        if action not in legal:
            raise ValueError(f"Illegal P0 action {action}")
        decisions += 1
        phase = "response" if response else "turn"
        behavior[phase + "_decisions"] += 1
        behavior["action_" + ("discard" if action.startswith("d ") else action)] += 1
        if "h" in legal:
            behavior[phase + "_hu_available"] += 1
            behavior[phase + "_hu_taken"] += int(action == "h")
        return action

    skip_draw = True
    while not game.is_game_over and steps < max_steps:
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
        drawn = None
        if not skip_draw:
            drawn = game.draw_tile(pid)
            if drawn is None:
                game.check_game_over()
                break
        skip_draw = False
        if pid == 0:
            action = yield from decide(get_legal_actions(game, 0))
        elif opponent_checkpoint:
            from mix_ppo_selfplay import opponent_action
            action = opponent_action(opponent_checkpoint, game, pid, get_legal_actions(game, pid))
        elif pid in defenders:
            action = defenders[pid].turn(player, game, step=steps)
        else:
            action = opponents[pid][0](player, game)
        tile = execute_action(game, pid, action, drawn)
        if game.is_game_over:
            break
        if tile is None:
            if action == "g":
                skip_draw = True
            else:
                game.next_player()
            continue
        responded = False
        # P4 processes responders in engine insertion order and stops after
        # the first accepted response; preserve that convention for pairing.
        for rid, acts in game.check_responses(tile, pid).items():
            responder = game.players[rid]
            if rid == 0:
                response = yield from decide(get_legal_actions(game, rid, acts), response=True)
            elif opponent_checkpoint:
                from mix_ppo_selfplay import opponent_action
                response = opponent_action(opponent_checkpoint, game, rid, get_legal_actions(game, rid, acts), True)
            elif rid in defenders:
                response = defenders[rid].response(responder, acts, game)
            else:
                response = bot_decide_response(responder, acts)
            if response == "h" and "hu" in acts:
                game.hu(rid, tile, False, pid)
                game.check_game_over()
                ron |= rid == 0
                dealt_in |= pid == 0
                responded = True
            elif response == "g" and "gang" in acts:
                game.gang(rid, tile, pid)
                game.current_player_id = rid
                responded = True
            elif response == "p" and "peng" in acts:
                game.peng(rid, tile, pid)
                game.current_player_id = rid
                responded = True
            if responded:
                break
        if game.is_game_over:
            break
        if responded:
            skip_draw = True
        else:
            game.next_player()
    if not game.is_game_over:
        raise RuntimeError(f"Unsettled game {seed}, steps={steps}")
    learning_rewards = []
    if progress:
        from mix_ppo_search import progress_reward, paper_terminal_reward
        last = (-1, game.players[0].hu_fan) if game.players[0].is_hu else progress_state()
        for before, after in zip(progress, progress[1:] + [last]):
            learning_rewards.append(progress_reward(before[0], after[0], before[1], after[1]))
        learning_rewards[-1] += paper_terminal_reward(game.players[0].balance-start, game.base_score,
                                                       draw=not any(p.is_hu for p in game.players))
    return {"seed": seed, "net": game.players[0].balance - start,
            "normalized_net": (game.players[0].balance - start) / reward_scale,
            "learning_rewards": learning_rewards,
            "final_balance": game.players[0].balance, "start_balance": start,
            "ron": ron, "dealt_in": dealt_in, "hu": bool(game.players[0].is_hu),
            "behavior": dict(behavior),
            "final_shanten": None if game.players[0].is_hu else ShantenCalculator.calculate_shanten(
                game.players[0].hand_tiles, game.players[0].missing_suit),
            "steps": steps, "decisions": decisions, "max_steps_or_unsettled": 0}
