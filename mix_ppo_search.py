"""Sichuan depth-limited public expectimax; explicit adaptation, not author code."""
from collections import Counter
from functools import lru_cache
from rule_engine import ShantenCalculator, FanCalculator
from tiles import Tile, Suit

SUITS = (Suit.WAN, Suit.TIAO, Suit.TONG)
TILES = tuple(Tile(s, n) for s in SUITS for n in range(1, 10))


def public_search(game, pid, legal, depth=3):
    """Root discard -> public unseen draw -> best next discard, then leaf.

    Leaves use a discounted public improvement probability times a payoff
    multiple. Beyond tenpai this is a heuristic, not a win probability.
    Shanten and ukeire remain separate input features.
    """
    if depth not in (1, 3):
        raise ValueError("Supported decision-ply depths are 1 and 3")
    player = game.players[pid]
    counts = Counter(player.hand_tiles)
    known = Counter(counts)
    for other in game.players:
        known.update(other.discarded_tiles)
        for meld in other.open_melds:
            known.update(meld)
    unseen = tuple(max(0, 4-known[t]) for t in TILES)
    start = tuple(counts[t] for t in TILES)
    missing = player.missing_suit

    def tiles(state):
        return [tile for tile, count in zip(TILES, state) for _ in range(count)]

    @lru_cache(maxsize=None)
    def shanten(state):
        return ShantenCalculator.calculate_shanten(tiles(state), missing)

    def after(state, i, delta):
        value = list(state)
        value[i] += delta
        return tuple(value)

    @lru_cache(maxsize=None)
    def winning_fan(state, i):
        fan, _ = FanCalculator.calculate_fan(tiles(state), TILES[i],
            open_melds=player.open_melds, is_self_drawn=False,
            gang_count=player.gang_count, concealed_kong_count=player.concealed_kong_count)
        return min(game.max_fan, max(0, fan))

    @lru_cache(maxsize=None)
    def leaf(state, availability):
        s = shanten(state)
        effective, fan_sum, wins, max_fan = 0, 0., 0, 0
        for i, copies in enumerate(availability):
            if copies <= 0:
                continue
            drawn = after(state, i, 1)
            ns = shanten(drawn)
            if ns < s:
                effective += copies
            if ns == -1:
                fan = winning_fan(drawn, i)
                fan_sum += copies * (2 ** fan)
                wins += copies
                max_fan = max(max_fan, fan)
        multiple = fan_sum / wins if wins else 1.
        value = progress_value(s, effective, sum(availability), multiple)
        return s, effective, max_fan, value

    def discards(state):
        has_missing = any(c and t.suit == missing for t, c in zip(TILES, state))
        return [i for i, c in enumerate(state) if c and (not has_missing or TILES[i].suit == missing)]

    results = {}
    for i, tile in enumerate(TILES):
        action = f"d {tile}"
        if action not in legal:
            continue
        state = after(start, i, -1)
        s, e, fan, value = leaf(state, unseen)
        if depth == 3:
            value = 0.
            total = sum(unseen)
            for j, copies in enumerate(unseen):
                if not copies:
                    continue
                drawn = after(state, j, 1)
                available = after(unseen, j, -1)
                if shanten(drawn) == -1:
                    best = float(2 ** winning_fan(drawn, j))
                else:
                    best = max(leaf(after(drawn, k, -1), available)[3] for k in discards(drawn))
                value += copies / max(1, total) * best
        results[action] = dict(shanten=s, ukeire=e, reachable_fan=fan, expectimax=value)
    return results


def paper_terminal_reward(net, base_score, draw=False):
    """Signed log2 payoff multiple; blood-battle multi-event adaptation."""
    import math
    if base_score <= 0:
        raise ValueError("base_score must be positive")
    if net == 0:
        return -1.0 if draw else 0.0
    return math.copysign(math.log2(max(1., abs(net)/base_score)), net)


def progress_value(shanten, effective, unseen, multiple=1.):
    """V6 heuristic: constant-rate improvement approximation, no offset bonus."""
    probability = effective / max(1, unseen)
    return probability ** max(1, shanten + 1) * multiple


def progress_reward(before_shanten, after_shanten, before_fan, after_fan):
    # Interpret Eq.6 using its stated intent: improvement earns positive reward.
    return .01 * (before_shanten-after_shanten+after_fan-before_fan)
