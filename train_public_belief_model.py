"""Train the deployable MC + public-action B_phi model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from experiment_trace import write_json
from public_belief_model import (
    PUBLIC_BELIEF_FEATURE_NAMES,
    PUBLIC_BELIEF_FEATURE_VERSION,
    features_from_h1_row,
    sha256_file,
)


def read_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_estimator(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=25,
        max_depth=10,
        min_samples_leaf=15,
        max_features=0.8,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-seed-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--oracle-samples", type=int, default=60)
    parser.add_argument("--oracle-beta", type=float, default=2.0)
    parser.add_argument("--danger-threshold", type=int, default=0)
    args = parser.parse_args()

    rows = [row for path in args.input for row in read_rows(path)]
    seeds = sorted({int(row["seed"]) for row in rows})
    validation_count = max(1, int(round(len(seeds) * args.validation_seed_fraction)))
    train_seeds = set(seeds[:-validation_count])
    validation_seeds = set(seeds[-validation_count:])
    x_train = [features_from_h1_row(row) for row in rows if int(row["seed"]) in train_seeds]
    y_train = [int(bool(row["b_j_oracle"]["true_tenpai"])) for row in rows if int(row["seed"]) in train_seeds]
    x_validation = [features_from_h1_row(row) for row in rows if int(row["seed"]) in validation_seeds]
    y_validation = [int(bool(row["b_j_oracle"]["true_tenpai"])) for row in rows if int(row["seed"]) in validation_seeds]

    validation_model = make_estimator(args.seed)
    validation_model.fit(x_train, y_train)
    validation_scores = validation_model.predict_proba(x_validation)[:, 1]
    validation = {
        "samples": len(y_validation),
        "positive_rate": sum(y_validation) / len(y_validation),
        "auc": roc_auc_score(y_validation, validation_scores),
        "brier": brier_score_loss(y_validation, validation_scores),
    }

    x_all = [features_from_h1_row(row) for row in rows]
    y_all = [int(bool(row["b_j_oracle"]["true_tenpai"])) for row in rows]
    final_model = make_estimator(args.seed)
    final_model.fit(x_all, y_all)
    metadata = {
        "feature_version": PUBLIC_BELIEF_FEATURE_VERSION,
        "feature_names": list(PUBLIC_BELIEF_FEATURE_NAMES),
        "training_rows": len(rows),
        "training_seed_count": len(seeds),
        "training_seed_first": seeds[0],
        "training_seed_last": seeds[-1],
        "validation": validation,
        "estimator_config": {
            "oracle_samples": args.oracle_samples,
            "oracle_beta": args.oracle_beta,
            "danger_threshold": args.danger_threshold,
            "b_phi_source": "public_model",
        },
        "inputs": [str(path) for path in args.input],
        "input_sha256": {str(path): sha256_file(path) for path in args.input},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "public_belief_model.joblib"
    joblib.dump({"estimator": final_model, "metadata": metadata}, model_path)
    summary = {**metadata, "model_path": str(model_path), "model_sha256": sha256_file(model_path)}
    write_json(args.output_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
