"""Plot the formal E5 score and empirical tail-safety frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


LABELS = {
    "no_deceive": "No deceive",
    "tight": "Tight",
    "default": "Default",
    "loose": "Loose",
    "constant_deceive": "Constant deceive",
}
COLORS = {
    "no_deceive": "#6B7280",
    "tight": "#2563EB",
    "default": "#15803D",
    "loose": "#D97706",
    "constant_deceive": "#B91C1C",
}
OFFSETS = {
    "holdout": {
        "no_deceive": (5, 5),
        "tight": (5, -16),
        "default": (5, 7),
        "loose": (-48, 7),
        "constant_deceive": (5, 5),
    },
    "anti_deception": {
        "no_deceive": (5, 5),
        "tight": (5, -14),
        "default": (5, 7),
        "loose": (5, 7),
        "constant_deceive": (5, 5),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.analysis.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    for axis, (suite, title) in zip(
        axes,
        (("holdout", "Frozen curriculum holdout"), ("anti_deception", "Deception-resistant stress test")),
    ):
        rows = data["suites"][suite]["frontier"]
        for row in rows:
            setting = row["setting"]
            x = float(row["score_gain_vs_no_deceive"])
            y = float(row["cvar_change_vs_no_deceive"])
            size = 50 + 500 * float(row["deceive_rate"])
            axis.scatter(x, y, s=size, color=COLORS[setting], edgecolor="white", linewidth=0.8, zorder=3)
            axis.annotate(
                LABELS[setting],
                (x, y),
                xytext=OFFSETS[suite][setting],
                textcoords="offset points",
                fontsize=8,
            )
        axis.axhline(0, color="#9CA3AF", linewidth=0.8)
        axis.axvline(0, color="#9CA3AF", linewidth=0.8)
        axis.grid(True, color="#E5E7EB", linewidth=0.7, zorder=0)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Mean-score gain vs no-deceive")
        axis.set_ylabel("CVaR-10% change vs no-deceive")
    fig.suptitle("E5 empirical gain-tail-safety frontier", fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
