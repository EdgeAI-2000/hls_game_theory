"""Pure reward helpers for shared-backbone GRPO.

No function in this module reads opponent beliefs, opponent identities, or
MASK gate labels.  Keeping rewards pure also makes them cheap to unit test.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


ACTION_RE = re.compile(r"^(d [1-9][万筒条]|h|g|p|n)$")


def extract_text(item: Any) -> str:
    if isinstance(item, list) and item and isinstance(item[-1], dict):
        return str(item[-1].get("content", ""))
    if isinstance(item, dict):
        return str(item.get("content", ""))
    return str(item)


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def extract_action(completion: Any) -> Tuple[str, bool]:
    text = extract_text(completion).strip()
    parsed = parse_json_object(text)
    if parsed is not None:
        return str(parsed.get("action", "")).strip(), True
    match = re.search(r"(?:^|[^a-z])(d\s*[1-9][万筒条]|[hgpn])(?:$|[^a-z])", text)
    if match:
        action = re.sub(r"\s+", " ", match.group(1)).strip()
        if action.startswith("d") and not action.startswith("d "):
            action = "d " + action[1:].strip()
        return action, False
    return text, False


def legal_actions_from_prompt(prompt_text: str) -> List[str]:
    match = re.search(r"合法动作[:：]\s*(.*?)(?:\n|$)", prompt_text)
    if not match:
        return []
    return [value.strip() for value in re.split(r"[,，]", match.group(1)) if value.strip()]


def optional_value(values: Any, index: int) -> Any:
    if isinstance(values, list) and index < len(values):
        return values[index]
    return None


def parse_action_values(raw: Any) -> Dict[str, Dict[str, float]]:
    if isinstance(raw, dict):
        value = raw
    else:
        try:
            value = json.loads(str(raw or "{}"))
        except json.JSONDecodeError:
            return {}
    if not isinstance(value, dict):
        return {}
    output: Dict[str, Dict[str, float]] = {}
    for action, metrics in value.items():
        if not isinstance(metrics, dict):
            continue
        try:
            output[str(action)] = {
                "shanten": float(metrics["shanten"]),
                "effective_copies": float(metrics.get("effective_copies", 0.0)),
                "effective_types": float(metrics.get("effective_types", 0.0)),
            }
        except (KeyError, TypeError, ValueError):
            continue
    return output


def format_reward_func(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        text = extract_text(completion).strip()
        rewards.append(2.0 if ACTION_RE.fullmatch(text) else -6.0)
    return rewards


def legal_action_reward_func(prompts, completions, **kwargs):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        legal = legal_actions_from_prompt(extract_text(prompt))
        action, _ = extract_action(completion)
        if not legal:
            rewards.append(0.0)
        else:
            rewards.append(6.0 if action in legal else -15.0)
    return rewards


def hu_decision_reward_func(prompts, completions, **kwargs):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        legal = legal_actions_from_prompt(extract_text(prompt))
        action, _ = extract_action(completion)
        if "h" in legal:
            rewards.append(20.0 if action == "h" else -20.0)
        elif action == "h":
            rewards.append(-20.0)
        else:
            rewards.append(0.0)
    return rewards


def progress_reward_func(prompts, completions, action_values_json=None, **kwargs):
    rewards = []
    for index, completion in enumerate(completions):
        action, _ = extract_action(completion)
        values = parse_action_values(optional_value(action_values_json, index))
        chosen = values.get(action)
        if not values or chosen is None:
            rewards.append(0.0)
            continue

        best_shanten = min(metrics["shanten"] for metrics in values.values())
        best_copies = max(
            metrics["effective_copies"]
            for metrics in values.values()
            if metrics["shanten"] == best_shanten
        )
        best_types = max(
            metrics["effective_types"]
            for metrics in values.values()
            if metrics["shanten"] == best_shanten
            and metrics["effective_copies"] == best_copies
        )
        shanten_regret = max(0.0, chosen["shanten"] - best_shanten)
        reward = 10.0 - 10.0 * shanten_regret
        if shanten_regret == 0:
            reward += max(-5.0, 0.5 * (chosen["effective_copies"] - best_copies))
            if chosen["effective_copies"] == best_copies:
                reward += max(-2.0, 0.25 * (chosen["effective_types"] - best_types))
        rewards.append(max(-20.0, min(12.0, reward)))
    return rewards


def teacher_anchor_reward_func(
    prompts,
    completions,
    reference_action=None,
    shared_base_source=None,
    **kwargs,
):
    rewards = []
    for index, completion in enumerate(completions):
        action, _ = extract_action(completion)
        reference = str(optional_value(reference_action, index) or "")
        source = str(optional_value(shared_base_source, index) or "")
        if not reference:
            rewards.append(0.0)
        elif action == reference:
            rewards.append(2.0 if "curriculum" in source else 1.0)
        elif reference == "h":
            rewards.append(-5.0)
        elif action.startswith("d ") and reference.startswith("d "):
            rewards.append(-0.5)
        else:
            rewards.append(-1.0)
    return rewards
