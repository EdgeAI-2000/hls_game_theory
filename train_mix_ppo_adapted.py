"""Train independent Mix-PPO-adapted with complete episodes and batched inference."""

import argparse
import hashlib
import json
import math
import time
import atexit
from collections import Counter
from pathlib import Path

import torch
from torch.distributions import Categorical, kl_divergence

from mix_ppo_baseline import (SichuanResNet101, FEATURE_VERSION, CONTEXT_FEATURE_VERSION,
                              PROSPECTIVE_FEATURE_VERSION, SEARCH_FEATURE_VERSION, ACTIONS, adapted_loss,
                              episode_gae, update_kl_coefficient)
from mix_ppo_rollout import episode


def collect(model, args, game_index):
    samples, games = [], []
    while len(samples) < args.rollout_steps:
        streams, pending, tracks = {}, {}, {}
        for slot in range(args.envs):
            seed = args.seed * 1000000 + game_index
            game_index += 1
            stream = episode(seed, args.opponents, args.manifest, "train", args.max_steps,
                             args.reward_scale, context_inputs=getattr(args, "context_inputs", False),
                             prospective_inputs=getattr(args, "prospective_inputs", False),
                             search_inputs=getattr(args, "search_inputs", False),
                             reward_mode=getattr(args, "reward_mode", "net"),
                             opponent_checkpoint=getattr(args, "opponent_checkpoint", None),
                             privileged_inputs=getattr(args, "privileged_until_steps", 0) > 0,
                             privileged_scale=getattr(args, "privileged_scale", 0.))
            streams[slot], tracks[slot] = stream, []
            try:
                pending[slot] = next(stream)
            except StopIteration as finished:
                # A settled game can have no P0 decisions; retain its metrics,
                # but there is no policy transition to train on.
                games.append(finished.value)
        while pending:
            slots = sorted(pending)
            obs = torch.stack([pending[s][0] for s in slots]).to(args.device)
            masks = torch.stack([pending[s][1] for s in slots]).to(args.device)
            with torch.no_grad():
                dist, values = model(obs, masks)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
            for position, slot in enumerate(slots):
                planes, mask, balance = pending[slot]
                tracks[slot].append({"obs": planes, "mask": mask,
                                     "action": actions[position].item(),
                                     "old_logits": dist.logits[position].cpu(),
                                     "old_log_prob": log_probs[position].item(),
                                     "value": values[position].item(), "balance": balance})
                try:
                    pending[slot] = streams[slot].send(actions[position].item())
                except StopIteration as finished:
                    result = finished.value
                    trajectory = tracks[slot]
                    balances = [row["balance"] for row in trajectory] + [result["final_balance"]]
                    rewards = [(b - a) / args.reward_scale for a, b in zip(balances, balances[1:])]
                    if abs(sum(rewards) * args.reward_scale - result["net"]) > 1e-5:
                        raise AssertionError("Reward does not account for the entire settlement")
                    if getattr(args, "reward_mode", "net") != "net":
                        rewards = result["learning_rewards"]
                    advantages, returns = episode_gae(rewards, [row["value"] for row in trajectory],
                                                      args.gamma, args.gae_lambda)
                    for row, advantage, target in zip(trajectory, advantages, returns):
                        row["advantage"], row["return"] = advantage, target
                    samples.extend(trajectory)
                    games.append(result)
                    del pending[slot]
    return samples, games, game_index


def optimize(model, optimizer, rows, args, coefficient):
    obs = torch.stack([r["obs"] for r in rows])
    masks = torch.stack([r["mask"] for r in rows])
    old_logits = torch.stack([r["old_logits"] for r in rows])
    actions = torch.tensor([r["action"] for r in rows])
    old_log_probs = torch.tensor([r["old_log_prob"] for r in rows])
    advantages = torch.stack([r["advantage"] for r in rows])
    returns = torch.stack([r["return"] for r in rows])
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    losses, value_losses, policy_objectives = [], [], []
    for _ in range(args.epochs):
        for indices in torch.randperm(len(rows)).split(args.minibatch):
            dist, values = model(obs[indices].to(args.device), masks[indices].to(args.device))
            loss, details = adapted_loss(dist, values, actions[indices].to(args.device),
                                   old_logits[indices].to(args.device), old_log_probs[indices].to(args.device),
                                   advantages[indices].to(args.device), returns[indices].to(args.device),
                                   coefficient, args.entropy_coef, args.clip, args.vf_coef)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss; no checkpoint published: {details}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.item())
            value_losses.append(float(details["value_loss"]))
            policy_objectives.append(float(details["policy_objective"]))
    total_kl, total_entropy = 0.0, 0.0
    with torch.no_grad():
        for indices in torch.arange(len(rows)).split(args.minibatch):
            dist, _ = model(obs[indices].to(args.device), masks[indices].to(args.device))
            old = Categorical(logits=old_logits[indices].to(args.device))
            total_kl += kl_divergence(old, dist).sum().item()
            total_entropy += dist.entropy().sum().item()
    mean_kl = total_kl / len(rows)
    if not math.isfinite(mean_kl) or not math.isfinite(total_entropy):
        raise FloatingPointError("Nonfinite final distribution")
    return {"loss": sum(losses) / len(losses), "value_loss": sum(value_losses) / len(value_losses),
            "policy_objective": sum(policy_objectives) / len(policy_objectives),
            "return_mean": float(returns.mean()), "return_std": float(returns.std(unbiased=False)),
            "kl": mean_kl,
            "entropy": total_entropy / len(rows), "delta_used": coefficient,
            "delta_next": update_kl_coefficient(coefficient, mean_kl, args.kl_target)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--context-inputs", action="store_true")
    parser.add_argument("--prospective-inputs", action="store_true")
    parser.add_argument("--search-inputs", action="store_true")
    parser.add_argument("--reward-mode", choices=["net", "paper_adapted"], default="net")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--goal-selfplay", action="store_true")
    parser.add_argument("--promotion-interval", type=int, default=50)
    parser.add_argument("--promotion-seed", type=int, default=2026092201)
    parser.add_argument("--privileged-until-steps", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026091601)
    parser.add_argument("--opponents", choices=["random", "mixed", "curriculum"], default="curriculum")
    parser.add_argument("--manifest", default="Opponent_Curriculum_formal/screened.json")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--reward-scale", type=float, default=100.0)
    parser.add_argument("--gamma", type=float, default=0.925)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--kl-coef", type=float, default=0.005)
    parser.add_argument("--kl-target", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    args = parser.parse_args()
    if min(args.updates, args.envs, args.rollout_steps, args.minibatch, args.epochs) <= 0:
        parser.error("All counts must be positive")
    if args.workers < 0 or (args.workers and args.workers != args.envs):
        parser.error("Use workers=envs for parallel collection, or workers=0 for serial diagnostics")
    if args.search_inputs and (not args.context_inputs or args.prospective_inputs):
        parser.error("Search requires context inputs and replaces prospective inputs")
    if args.goal_selfplay and (args.opponents != "random" or args.reward_mode != "net" or not args.workers):
        parser.error("V6 goal selfplay requires random start, linear net reward and parallel workers")
    if args.promotion_interval <= 0 or args.promotion_seed == args.seed or 2026082001 <= args.promotion_seed <= 2026082020:
        parser.error("Invalid promotion interval or validation seed")
    if args.privileged_until_steps < 0:
        parser.error("privileged-until-steps must be nonnegative")
    if 2026082001 <= args.seed <= 2026082020 or not 0 < args.seed:
        parser.error("Training seeds must exclude the P4 evaluation family")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error("Output directory is not empty; use --resume to continue")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if args.prospective_inputs and not args.context_inputs:
        parser.error("--prospective-inputs requires --context-inputs")
    config.update({"method": "Mix-PPO-adapted", "feature_version": SEARCH_FEATURE_VERSION if args.search_inputs else (PROSPECTIVE_FEATURE_VERSION if args.prospective_inputs else (CONTEXT_FEATURE_VERSION if args.context_inputs else FEATURE_VERSION)),
                   "actions": list(ACTIONS), "batch_norm": "frozen_running_stats_trainable_affine",
                   "sampling": "synchronous_process_workers_central_inference" if args.workers else "serial_interleaved_slots",
                   "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                   "torch": torch.__version__, "source_sha256": {
                       f: hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest()
                       for f in ("mix_ppo_baseline.py", "mix_ppo_rollout.py", "train_mix_ppo_adapted.py",
                                 "mix_ppo_parallel.py", "mix_ppo_search.py", "mix_ppo_selfplay.py", "tiles.py",
                                 "game.py", "prompt_builder.py", "run_gate1_experiments.py", "opponent_curriculum.py", "rule_engine.py")}})
    # Fixed BN statistics keep rollout and update distributions identical at
    # theta_old, regardless of minibatch composition. Affine weights still learn.
    model = SichuanResNet101(context_inputs=args.context_inputs,
                             prospective_inputs=args.prospective_inputs, search_inputs=args.search_inputs,
                             privileged_inputs=args.privileged_until_steps > 0).to(args.device).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    first, game_index, total_steps, coefficient = 0, 0, 0, args.kl_coef
    generation, opponent_path = 0, None
    checkpoint = args.output / "latest.pt"
    if args.resume:
        saved = torch.load(checkpoint, map_location=args.device, weights_only=False)
        for key, value in config.items():
            if key == "source_sha256" and saved["config"].get(key) != value:
                previous = saved["config"][key]
                permitted = dict(previous)
                if previous.get("train_mix_ppo_adapted.py") == "7afd6cfbd5a400df6d54a2cdf01db7825caed1d9aa78aae3bad4da99f2bb1fe7":
                    permitted["train_mix_ppo_adapted.py"] = value["train_mix_ppo_adapted.py"]
                    if permitted == value:
                        (args.output / "zero_decision_resume_audit.json").write_text(json.dumps({
                            "reason": "Handle completed episodes with zero P0 decisions",
                            "resumed_update": saved["update"], "previous_source_sha256": previous,
                            "current_source_sha256": value,
                            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                        }, indent=2) + "\n")
                        continue
            if key not in {"resume", "updates", "output"} and saved["config"].get(key) != value:
                raise ValueError(f"Resume configuration mismatch: {key}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        first, game_index, total_steps = saved["update"], saved["game_index"], saved["total_steps"]
        coefficient = saved["delta"]
        generation = saved.get("opponent_generation", 0)
        opponent_path = saved.get("opponent_checkpoint")
        if opponent_path and hashlib.sha256(Path(opponent_path).read_bytes()).hexdigest() != saved["opponent_sha256"]:
            raise ValueError("Opponent checkpoint changed")
        torch.set_rng_state(saved["torch_rng"].cpu())
        if args.device.startswith("cuda"):
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), args.device)
    else:
        (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    start = time.monotonic()
    pool = None
    if args.workers:
        from mix_ppo_parallel import EnvironmentPool, collect_parallel
        pool = EnvironmentPool(args.workers)
        atexit.register(pool.close)
    for iteration in range(first, args.updates):
        args.opponent_checkpoint = opponent_path
        args.privileged_scale = max(0., 1.-total_steps/args.privileged_until_steps) if args.privileged_until_steps else 0.
        rows, games, game_index = (collect_parallel(model, args, game_index, pool) if pool
                                    else collect(model, args, game_index))
        stats = optimize(model, optimizer, rows, args, coefficient)
        coefficient = stats["delta_next"]
        total_steps += len(rows)
        stats.update({"update": iteration + 1, "steps": len(rows), "total_steps": total_steps,
                      "games": len(games), "total_games": game_index,
                      "avg_net": sum(g["net"] for g in games) / len(games),
                      "max_steps_or_unsettled": 0, "elapsed_seconds": time.monotonic() - start})
        behavior = Counter()
        for game in games:
            behavior.update(game["behavior"])
        available = sum(behavior[k + "_hu_available"] for k in ("turn", "response"))
        taken = sum(behavior[k + "_hu_taken"] for k in ("turn", "response"))
        unfinished = [g["final_shanten"] for g in games if g["final_shanten"] is not None]
        stats.update({"behavior": dict(behavior), "hu_opportunities": available,
                      "hu_taken": taken, "missed_hu": available - taken,
                      "hu_game_rate": sum(g["hu"] for g in games) / len(games),
                      "ron_game_rate": sum(g["ron"] for g in games) / len(games),
                      "dealin_game_rate": sum(g["dealt_in"] for g in games) / len(games),
                      "mean_unwon_final_shanten": sum(unfinished) / len(unfinished) if unfinished else None})
        mean_reward = sum(g["normalized_net"] if args.reward_mode == "net" else sum(g["learning_rewards"]) for g in games) / len(games)
        stats["learning_reward_mean"] = mean_reward
        stats["privileged_scale"] = args.privileged_scale
        stats["opponent_generation_used"] = generation
        if args.goal_selfplay and (iteration + 1) % args.promotion_interval == 0 and args.privileged_scale == 0:
            from mix_ppo_selfplay import validate_promotion
            decision = validate_promotion(model, args, pool, iteration + 1)
            stats["promotion_validation"] = decision
            if decision["promote"]:
                generation += 1
                path = args.output / f"opponent_generation_{generation}.pt"
                if path.exists():
                    raise FileExistsError(path)
                torch.save({"model": model.state_dict(), "config": config}, path)
                opponent_path = str(path.resolve())
        stats["opponent_generation_next"] = generation
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "opponent_generation": generation, "opponent_checkpoint": opponent_path,
                   "opponent_sha256": hashlib.sha256(Path(opponent_path).read_bytes()).hexdigest() if opponent_path else None,
                   "config": config, "update": iteration + 1, "game_index": game_index,
                   "total_steps": total_steps, "delta": coefficient,
                   "torch_rng": torch.get_rng_state(), "cuda_rng":
                   torch.cuda.get_rng_state(args.device) if args.device.startswith("cuda") else None}
        temporary = args.output / "latest.tmp"
        torch.save(payload, temporary)
        temporary.replace(checkpoint)
        with (args.output / "metrics.jsonl").open("a") as out:
            out.write(json.dumps(stats, allow_nan=False) + "\n")
        with (args.output / "games.jsonl").open("a") as out:
            for game in games:
                out.write(json.dumps({"update": iteration + 1, **game}) + "\n")
        print(json.dumps(stats, allow_nan=False), flush=True)
    (args.output / "run_complete.json").write_text(json.dumps({
        "completed_updates": max(first, args.updates), "total_steps": total_steps,
        "total_games": game_index, "formal_evaluation": False,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}, indent=2) + "\n")
    if pool:
        atexit.unregister(pool.close)
        pool.close()


if __name__ == "__main__":
    main()
