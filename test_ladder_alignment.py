from __future__ import annotations

import unittest
from unittest.mock import patch

from mask_llm import MASKLLMAgent, PublicOpponentTracker, RiskGate
from prompt_builder import build_base_decision_prompt, build_reactive_decision_prompt, get_legal_actions
from rule_engine import ShantenCalculator
from run_gate1_experiments import choose_reactive_z_action, init_game


class CaptureLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        marker = "合法动作: "
        action = prompt.split(marker, 1)[1].splitlines()[0].split(",", 1)[0].strip()
        return action


def discard_result_shanten(game, action: str) -> int:
    player = game.players[0]
    tile = next(tile for tile in player.hand_tiles if str(tile) == action[2:])
    hand = player.hand_tiles.copy()
    hand.remove(tile)
    return int(ShantenCalculator.calculate_shanten(hand, player.missing_suit))


class LadderAlignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.game, _, _ = init_game(20260713, "greedy", "ladder_alignment")
        tracker = PublicOpponentTracker([1, 2, 3])
        self.z_state = tracker.update_from_game(self.game)
        self.gate = RiskGate().compute(self.game, 0, self.z_state, beliefs={})

    def test_reactive_prompt_keeps_known_state_schema(self):
        prompt = build_reactive_decision_prompt(
            self.game, 0, self.z_state, self.gate, get_legal_actions(self.game, 0)
        )
        self.assertIn("【局势分析】", prompt)
        self.assertNotIn("【公开对手漂移 z_j(t)】", prompt)
        self.assertNotIn("recent_actions", prompt)

    def test_l0_l1_l2_share_base_exploit_prompt(self):
        llm = CaptureLLM()
        agent = MASKLLMAgent(player_id=0, decision_llm=llm, mc_oracle_samples=2)
        legal = get_legal_actions(self.game, 0)
        agent._exploit_action(self.game, legal)
        expected = build_base_decision_prompt(self.game, 0, valid_actions=legal)
        self.assertEqual(llm.prompts[-1], expected)

    def test_reactive_belief_changes_gate_and_action(self):
        class BeliefGate:
            def compute(self, game, player_id, z_state, beliefs):
                confidence = max(
                    (value["tenpai_confidence"] for value in beliefs.values()),
                    default=0.0,
                )
                return {"mode_hint": "safe" if confidence > 0.5 else "exploit"}

        legal = [a for a in get_legal_actions(self.game, 0) if a.startswith("d ")]
        self.assertGreaterEqual(len(legal), 2)
        safe_action = legal[1]
        with patch("run_gate1_experiments.choose_public_safe_no_regret", return_value=safe_action):
            plain, plain_trace = choose_reactive_z_action(
                self.game, 0, PublicOpponentTracker([1, 2, 3]), BeliefGate(),
                lambda _: legal[0], legal, auto_hu="off",
            )
            guarded, guarded_trace = choose_reactive_z_action(
                self.game, 0, PublicOpponentTracker([1, 2, 3]), BeliefGate(),
                lambda _: legal[0], legal, auto_hu="off",
                beliefs={"P1": {"tenpai_confidence": 0.9}},
            )
        self.assertEqual(plain, legal[0])
        self.assertEqual(guarded, safe_action)
        self.assertEqual(plain_trace["beliefs"], {})
        self.assertEqual(guarded_trace["beliefs"]["P1"]["tenpai_confidence"], 0.9)

    def test_threat_discard_has_zero_shanten_regret(self):
        found = False
        for seed in range(20260700, 20260800):
            game, _, _ = init_game(seed, "greedy", f"shanten_guard_{seed}")
            legal = get_legal_actions(game, 0)
            agent = MASKLLMAgent(
                player_id=0,
                forced_deceive="eligible",
                deceive_style="threat",
                mc_danger_threshold=0,
                threat_gate_mode="delta_only",
                threat_min_delta=-1.0,
                threat_max_start_shanten=99,
                threat_require_real_target=False,
                threat_max_shanten_regret=0,
            )
            action = agent._threat_discard(game, legal)
            if not action:
                continue
            all_results = [
                discard_result_shanten(game, candidate)
                for candidate in legal
                if candidate.startswith("d ")
            ]
            self.assertEqual(discard_result_shanten(game, action), min(all_results))
            found = True
            break
        self.assertTrue(found, "test seeds did not produce an eligible threat discard")

    def test_no_risk_gate_forces_available_deceive_only(self):
        game, _, _ = init_game(20260714, "greedy", "no_risk_gate_deceive")
        legal = [action for action in get_legal_actions(game, 0) if action.startswith("d ")]
        self.assertGreaterEqual(len(legal), 3)
        agent = MASKLLMAgent(
            player_id=0,
            neural_gate=object(),
            gate_policy="neural_value_gate",
            risk_gate_ablation="constant_deceive",
            deceive_style="threat",
            mc_oracle_samples=2,
        )
        agent._exploit_action = lambda game, actions: (legal[0], "test exploit")
        agent._safe_discard = lambda game, actions, fallback_action=None: legal[1]
        agent._deceptive_discard = lambda game, actions, beliefs, relaxed_gate=False: legal[2]

        action = agent.decide(game, get_legal_actions(game, 0))

        self.assertEqual(action, legal[2])
        self.assertEqual(agent.last_decision["mode"], "deceive")
        self.assertEqual(
            agent.last_decision["learned_gate"]["policy"],
            "risk_ablation_constant_deceive",
        )

    def test_no_risk_gate_falls_back_to_exploit_not_safe(self):
        legal = [action for action in get_legal_actions(self.game, 0) if action.startswith("d ")]
        self.assertGreaterEqual(len(legal), 2)
        agent = MASKLLMAgent(
            player_id=0,
            neural_gate=object(),
            gate_policy="neural_value_gate",
            risk_gate_ablation="constant_deceive",
            deceive_style="threat",
            mc_oracle_samples=2,
        )
        agent._exploit_action = lambda game, actions: (legal[0], "test exploit")
        agent._safe_discard = lambda game, actions, fallback_action=None: legal[1]
        agent._deceptive_discard = lambda game, actions, beliefs, relaxed_gate=False: ""

        action = agent.decide(self.game, get_legal_actions(self.game, 0))

        self.assertEqual(action, legal[0])
        self.assertEqual(agent.last_decision["mode"], "exploit")
        self.assertTrue(agent.last_decision["learned_gate"]["fallback"])

    def test_mc_belief_calibration_is_applied_before_gate_features(self):
        agent = MASKLLMAgent(
            player_id=0,
            mc_oracle_samples=2,
            belief_calibration_steps=[(0.5, 0.1), (1.0, 0.3)],
        )
        beliefs = agent._mc_beliefs(self.game)
        self.assertTrue(beliefs)
        for value in beliefs.values():
            self.assertIn("raw_tenpai_confidence", value)
            self.assertLessEqual(value["tenpai_confidence"], 0.3)
            self.assertEqual(value["source"], "mc_posterior_isotonic")


if __name__ == "__main__":
    unittest.main()
