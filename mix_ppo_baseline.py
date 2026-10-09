"""Mix-PPO-adapted: independent Sichuan actor-critic, not an exact reproduction.

The actor-critic below adapts the paper's shared, pool-free ResNet101 idea.
Feature encoding and 27-tile action space are explicitly Sichuan adaptations.
The paper's inconsistent objective is exposed only for diagnostic tests.
"""

from collections import Counter

import torch
from torch import nn
from torch.distributions import Categorical
from torchvision.models import resnet101

from tiles import Suit, Tile
from rule_engine import ShantenCalculator


SUITS = (Suit.WAN, Suit.TIAO, Suit.TONG)
DISCARDS = tuple(f"d {number}{suit.value}" for suit in SUITS for number in range(1, 10))
ACTIONS = DISCARDS + ("h", "g", "p", "n")
FEATURE_VERSION = "mix_ppo_adapted_public_v2"
CONTEXT_FEATURE_VERSION = "mix_ppo_adapted_public_v3_decision_context"
PROSPECTIVE_FEATURE_VERSION = "mix_ppo_adapted_public_v4_one_step_prospective"
SEARCH_FEATURE_VERSION = "mix_ppo_adapted_public_v6_payoff_progress"
# Own hand, per-player discards/melds/history, missing suits, wins, balances, wall.
CHANNELS = 4 + 16 + 16 + 120 + 12 + 4 + 4 + 1
CONTEXT_CHANNELS = CHANNELS + len(ACTIONS) + 2 + 1 + 4
PROSPECTIVE_CHANNELS = CONTEXT_CHANNELS + 2 * len(ACTIONS)
SEARCH_CHANNELS = CONTEXT_CHANNELS + 4 * len(ACTIONS)
PRIVILEGED_CHANNELS = 12


def privileged_planes(game, pid, observation, scale=0.):
    """Explicit training-only channels; at zero do not access hidden hands."""
    if not 0 <= scale <= 1:
        raise ValueError("privileged scale must be in [0,1]")
    extra = torch.zeros(PRIVILEGED_CHANNELS, 3, 9)
    if scale:
        for offset in range(1, 4):
            counts = Counter(game.players[(pid+offset) % 4].hand_tiles)
            for level in range(4):
                for si, suit in enumerate(SUITS):
                    for n in range(1, 10):
                        extra[(offset-1)*4+level, si, n-1] = scale * (counts[Tile(suit, n)] > level)
    return torch.cat((observation, extra))


def _prospective_action_features(game, player_id, legal_actions):
    """Public one-step action features: post-discard shanten and ukeire."""
    player = game.players[player_id]
    own_counts = Counter((tile.suit, tile.number) for tile in player.hand_tiles)
    public_counts = Counter(own_counts)
    for other in game.players:
        public_counts.update((tile.suit, tile.number) for tile in other.discarded_tiles)
        public_counts.update((tile.suit, tile.number) for meld in other.open_melds for tile in meld)
    features = []
    for action in ACTIONS:
        if action.startswith("d "):
            text = action[2:]
            suit = next(s for s in SUITS if s.value in text)
            tile = Tile(suit, int(text[0]))
            if action not in legal_actions or tile not in player.hand_tiles:
                features.append((-1.0, 0.0))
                continue
            remaining = list(player.hand_tiles)
            remaining.remove(tile)
            post = ShantenCalculator.calculate_shanten(remaining, player.missing_suit)
            ukeire = 0
            for candidate_suit in SUITS:
                for number in range(1, 10):
                    candidate = Tile(candidate_suit, number)
                    if public_counts[candidate_suit, number] >= 4:
                        continue
                    next_shanten = ShantenCalculator.calculate_shanten(
                        remaining + [candidate], player.missing_suit)
                    if next_shanten < post:
                        ukeire += 4 - public_counts[candidate_suit, number]
            features.append((max(-1.0, min(1.0, -post / 8.0)), ukeire / 36.0))
        elif action == "h":
            features.append((1.0 if action in legal_actions else -1.0, 1.0 if action in legal_actions else 0.0))
        elif action in ("g", "p"):
            features.append((0.25 if action in legal_actions else -1.0, 0.0))
        else:
            features.append((0.0 if action in legal_actions else -1.0, 0.0))
    return features


def public_planes(game, player_id, legal_actions=None, response=False, prospective=False, search=False):
    """Read only own hand and public state, preserving per-player discard order."""
    planes = []

    def counts(tiles):
        counter = Counter((t.suit, t.number) for t in tiles)
        for level in range(1, 5):
            planes.append([[float(counter[suit, n] >= level) for n in range(1, 10)]
                           for suit in SUITS])

    def scalar(value):
        planes.append([[float(value)] * 9 for _ in SUITS])

    players = [game.players[(player_id + offset) % 4] for offset in range(4)]
    counts(players[0].hand_tiles)
    for player in players:
        counts(player.discarded_tiles)
    for player in players:
        counts(tile for meld in player.open_melds for tile in meld)
    for player in players:
        history = player.discarded_tiles[-30:]
        for index in range(30):
            tile = history[index] if index < len(history) else None
            planes.append([[float(tile is not None and tile.suit == suit and tile.number == n)
                            for n in range(1, 10)] for suit in SUITS])
    for player in players:
        for suit in SUITS:
            scalar(player.missing_suit == suit)
    for player in players:
        scalar(player.is_hu)
    for player in players:
        scalar((player.balance - players[0].balance) / 10000.0)
    scalar(game.deck.remaining_count() / 108.0)
    if legal_actions is not None:
        # Legality previously only masked output logits: the shared trunk and
        # critic could not condition on response phase or the available calls.
        for allowed in action_mask(legal_actions):
            scalar(allowed)
        scalar(not response)
        scalar(response)
        tile = game.last_discarded_tile if response else None
        planes.append([[float(tile is not None and tile.suit == suit and tile.number == n)
                        for n in range(1, 10)] for suit in SUITS])
        for offset in range(4):
            scalar(response and game.last_discard_player_id == (player_id + offset) % 4)
        if search:
            from mix_ppo_search import public_search
            looked = public_search(game, player_id, legal_actions)
            for name, scale in (("shanten", 8.), ("ukeire", 108.), ("reachable_fan", 8.), ("expectimax", 8.)):
                for action in ACTIONS:
                    scalar(looked.get(action, {}).get(name, 0.) / scale)
        elif prospective:
            features = _prospective_action_features(game, player_id, legal_actions)
            for value_index in range(2):
                for first, second in features:
                    scalar((first, second)[value_index])
    result = torch.tensor(planes, dtype=torch.float32)
    expected = (SEARCH_CHANNELS if search else (PROSPECTIVE_CHANNELS if prospective else CONTEXT_CHANNELS)) if legal_actions is not None else CHANNELS
    assert result.shape == (expected, 3, 9)
    return result


def discard_mask(legal_actions):
    """A discard-only actor must not silently select an action in a response phase."""
    mask = torch.tensor([a in legal_actions for a in DISCARDS], dtype=torch.bool)
    if not mask.any():
        raise ValueError("No legal discard: handle non-discard actions outside this actor")
    return mask


def action_mask(legal_actions):
    unknown = set(legal_actions) - set(ACTIONS)
    if unknown:
        raise ValueError(f"Unsupported actions: {unknown}")
    mask = torch.tensor([a in legal_actions for a in ACTIONS], dtype=torch.bool)
    if not mask.any():
        raise ValueError("No legal action")
    return mask


class SichuanResNet101(nn.Module):
    """Pool-free bottleneck stages [3,4,23,3]; no pretrained weights or LLM.

    All spatial strides are one to preserve the 3x9 Sichuan tile layout. The
    stem and heads are migration choices, not asserted exact paper settings.
    """

    def __init__(self, context_inputs=False, prospective_inputs=False, search_inputs=False, privileged_inputs=False):
        super().__init__()
        backbone = resnet101(weights=None, zero_init_residual=True)
        channels = SEARCH_CHANNELS if search_inputs else (PROSPECTIVE_CHANNELS if prospective_inputs else (CONTEXT_CHANNELS if context_inputs else CHANNELS))
        channels += PRIVILEGED_CHANNELS if privileged_inputs else 0
        backbone.conv1 = nn.Conv2d(channels, 64, 3, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        for module in backbone.modules():
            if isinstance(module, nn.Conv2d):
                module.stride = (1, 1)
        self.trunk = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu,
                                   backbone.layer1, backbone.layer2,
                                   backbone.layer3, backbone.layer4)
        self.actor = nn.Sequential(nn.Conv2d(2048, 1, 1), nn.Flatten(), nn.Linear(27, len(ACTIONS)))
        self.critic = nn.Sequential(nn.Conv2d(2048, 1, 1), nn.Flatten(), nn.Linear(27, 1))

    def forward(self, observations, masks):
        if masks.dtype != torch.bool or masks.shape != (len(observations), len(ACTIONS)):
            raise ValueError("Expected a boolean [batch,31] legal-action mask")
        if not masks.any(dim=-1).all():
            raise ValueError("Every observation must have at least one legal discard")
        features = self.trunk(observations)
        logits = self.actor(features).masked_fill(~masks, -torch.inf)
        return Categorical(logits=logits), self.critic(features).squeeze(-1)


def adapted_loss(distribution, values, actions, old_logits, old_log_probs,
                 advantages, returns, kl_coef, entropy_coef=0.01, clip=0.2, vf_coef=0.5):
    """Minimized loss: -L_CLIP + delta*KL(old||new) - alpha*H + c_v*MSE."""
    old = Categorical(logits=old_logits.detach())
    ratio = (distribution.log_prob(actions) - old_log_probs.detach()).exp()
    surrogate = torch.minimum(ratio * advantages,
                              ratio.clamp(1 - clip, 1 + clip) * advantages).mean()
    kl = torch.distributions.kl_divergence(old, distribution).mean()
    entropy = distribution.entropy().mean()
    value_loss = (values - returns).square().mean()
    loss = -surrogate + kl_coef * kl - entropy_coef * entropy + vf_coef * value_loss
    return loss, {"kl": kl.detach(), "entropy": entropy.detach(),
                  "policy_objective": surrogate.detach(), "value_loss": value_loss.detach()}


def update_kl_coefficient(coef, measured_kl, target):
    """Schulman et al. 2017 section 4, once per complete policy update."""
    if target <= 0 or coef <= 0:
        raise ValueError("KL target and coefficient must be positive")
    if measured_kl < target / 1.5:
        return coef / 2
    if measured_kl > target * 1.5:
        return coef * 2
    return coef


def episode_gae(rewards, values, gamma=0.925, lam=0.95):
    """Complete episodes only: bootstrap zero at actual settlement, not P0's hu."""
    if len(rewards) != len(values) or not rewards:
        raise ValueError("Expected a nonempty, aligned complete episode")
    advantages = torch.zeros(len(rewards), dtype=torch.float32)
    carry = 0.0
    for index in reversed(range(len(rewards))):
        next_value = values[index + 1] if index + 1 < len(values) else 0.0
        delta = rewards[index] + gamma * next_value - values[index]
        carry = delta + gamma * lam * carry
        advantages[index] = carry
    return advantages, advantages + torch.tensor(values, dtype=torch.float32)


def paper_equation4_diagnostic(ratio, advantage, entropy, kl,
                               entropy_coef=0.005, kl_coef=0.005, clip=0.2):
    """Literal Eq.4, interpreting J as the resampled advantage surrogate.

    Do NOT use as a validated training objective: maximizing the printed
    positive KL term can reward divergence. The definition of J is ambiguous.
    """
    return (ratio * advantage + entropy_coef * entropy
            + torch.maximum(kl_coef * kl, ratio.clamp(1 - clip, 1 + clip))).mean()


def paper_equation3_diagnostic(coef, kl, target, step):
    """Literal printed adaptive coefficient update, for auditing only."""
    return coef + step * (target - kl)
