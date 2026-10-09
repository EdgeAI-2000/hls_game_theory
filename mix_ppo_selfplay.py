"""Immutable opponent snapshots for train-only goal promotion."""
from functools import lru_cache
import torch


def promotion_decision(scores):
    """Development gate, not a paper hypothesis test or a convergence claim."""
    import numpy as np
    from scipy.stats import t
    values = np.asarray(scores, dtype=float)
    if len(values) < 200 or not np.isfinite(values).all():
        raise ValueError("Need at least 200 finite validation payoffs")
    mean = float(values.mean())
    lower = mean - float(t.ppf(.95, len(values)-1) * values.std(ddof=1) / len(values)**.5)
    return dict(games=len(values), mean_net=mean, lower_one_sided_95=lower,
                promote=bool(lower > 0))


def validate_promotion(model, args, pool, update):
    import copy
    from mix_ppo_parallel import collect_parallel

    class Greedy:
        def __call__(self, obs, mask):
            dist, value = model(obs, mask)
            class Distribution:
                logits = dist.logits
                def sample(self):
                    return dist.probs.argmax(dim=-1)
                def log_prob(self, actions):
                    return dist.log_prob(actions)
            return Distribution(), value

    probe = copy.copy(args)
    probe.seed = args.promotion_seed
    probe.privileged_scale = 0.
    probe.rollout_steps = 1
    # Disjoint validation blocks per check; never consume training RNG.
    index = update * 10000
    if index + 10000 >= 1000000:
        # Use a separate large seed family for arbitrary update budgets.
        probe.seed = args.promotion_seed * 100000 + update
        index = 0
    initial = index
    scores = []
    while len(scores) < 200:
        _, games, index = collect_parallel(Greedy(), probe, index, pool)
        scores.extend(g["net"] for g in games)
    return {**promotion_decision(scores), "seed_family": probe.seed,
            "first_game_index": initial, "next_game_index": index,
            "privileged_scale": 0.}


def goal_threshold(generation):
    if generation < 0:
        raise ValueError("generation must be nonnegative")
    return max(.1, 1.-.1*generation)


@lru_cache(maxsize=1)
def load_opponent(path):
    from mix_ppo_baseline import SichuanResNet101
    saved = torch.load(path, map_location="cpu", weights_only=False)
    config = saved["config"]
    model = SichuanResNet101(context_inputs=config.get("context_inputs", False),
        prospective_inputs=config.get("prospective_inputs", False),
        search_inputs=config.get("search_inputs", False),
        privileged_inputs=config.get("privileged_until_steps", 0) > 0).eval()
    model.load_state_dict(saved["model"])
    return model, config


def opponent_action(path, game, pid, legal, response=False):
    from mix_ppo_baseline import public_planes, action_mask, ACTIONS
    model, cfg = load_opponent(path)
    obs = public_planes(game, pid, legal if cfg.get("context_inputs") else None,
                        response, cfg.get("prospective_inputs", False), cfg.get("search_inputs", False))
    if cfg.get("privileged_until_steps", 0):
        from mix_ppo_baseline import privileged_planes
        obs = privileged_planes(game, pid, obs, 0.)
    with torch.no_grad():
        dist, _ = model(obs[None], action_mask(legal)[None])
    return ACTIONS[int(dist.probs.argmax())]
