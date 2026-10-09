from analyze_e5_risk_frontier import metric
from run_e5_risk_frontier import SETTINGS


def test_deceive_margin_order_is_conservative_to_aggressive():
    assert SETTINGS["tight"]["deceive_margin"] > SETTINGS["default"]["deceive_margin"]
    assert SETTINGS["default"]["deceive_margin"] > SETTINGS["loose"]["deceive_margin"]
    assert SETTINGS["tight"]["safe_margin"] == SETTINGS["loose"]["safe_margin"] == 0.05


def test_metric_extracts_per_game_and_mode_rates():
    row = {
        "games": 100,
        "avg_net": 2.5,
        "dealin_rate": 0.1,
        "cvar_10pct": -20.0,
        "DIR_counts": {"induced_dealin": 4},
        "FFR_counts": {"false_folds": 3, "false_fold_opportunities": 12},
        "mode_counts": {"exploit": 5, "safe": 3, "deceive": 2},
    }
    row["ron_game_rate"] = 0.04
    assert metric(row, "ron_game_rate") == 0.04
    assert metric(row, "oracle_ffr") == 0.25
    assert metric(row, "deceive_rate") == 0.2


def test_metric_supports_compact_p3r_rows():
    row = {
        "games": 1000,
        "discard_hu_rate": 0.3,
        "FFR": 0.2,
        "exploit": 7,
        "safe": 2,
        "deceive": 1,
    }
    assert metric(row, "discard_hu_rate") == 0.3
    assert metric(row, "oracle_ffr") == 0.2
    assert metric(row, "deceive_rate") == 0.1
