import json
import unittest

from shared_base_reward import (
    format_reward_func,
    hu_decision_reward_func,
    legal_action_reward_func,
    progress_reward_func,
)


def prompt(legal: str):
    return [{"role": "user", "content": f"合法动作: {legal}\n请只输出动作。"}]


class SharedBaseRewardTest(unittest.TestCase):
    def test_hu_is_rewarded_only_when_legal(self):
        prompts = [prompt("h, d 1万"), prompt("d 1万, d 2万")]
        completions = ["h", "h"]
        self.assertEqual(hu_decision_reward_func(prompts, completions), [20.0, -20.0])
        self.assertEqual(legal_action_reward_func(prompts, completions), [6.0, -15.0])

    def test_declining_legal_hu_is_penalized(self):
        self.assertEqual(
            hu_decision_reward_func([prompt("h, d 1万")], ["d 1万"]),
            [-20.0],
        )

    def test_progress_prefers_min_shanten_then_ukeire(self):
        values = json.dumps({
            "d 1万": {"shanten": 0, "effective_copies": 8, "effective_types": 2},
            "d 2万": {"shanten": 0, "effective_copies": 4, "effective_types": 1},
            "d 3万": {"shanten": 1, "effective_copies": 12, "effective_types": 4},
        }, ensure_ascii=False)
        rewards = progress_reward_func(
            [prompt("d 1万, d 2万, d 3万")] * 3,
            ["d 1万", "d 2万", "d 3万"],
            action_values_json=[values] * 3,
        )
        self.assertGreater(rewards[0], rewards[1])
        self.assertGreater(rewards[1], rewards[2])

    def test_format_reward_rejects_explanation(self):
        self.assertEqual(format_reward_func([], ["d 1万", "我选择 d 1万"]), [2.0, -6.0])


if __name__ == "__main__":
    unittest.main()
