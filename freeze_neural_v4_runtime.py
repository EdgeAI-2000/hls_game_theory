"""Freeze V4 runtime progress guards into existing checkpoints and refresh hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiment_trace import write_json
from run_neural_opponent_v4_training import PROFILES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.training_dir / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("failures"):
        raise RuntimeError("Refusing to freeze a failed training manifest")
    for item in manifest["profiles"]:
        profile = str(item["profile"])
        config = PROFILES[profile]
        checkpoint_path = Path(item["checkpoint"])
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        checkpoint.update(
            {
                "runtime_force_gang": True,
                "runtime_max_shanten_regret": config[
                    "runtime_max_shanten_regret"
                ],
                "runtime_max_effective_copy_loss": config[
                    "runtime_max_effective_copy_loss"
                ],
                "runtime_progress_guard": "shanten_then_effective_copies",
            }
        )
        torch.save(checkpoint, checkpoint_path)
        item["checkpoint_sha256"] = sha256(checkpoint_path)
        item["runtime_config"] = {
            key: checkpoint[key]
            for key in (
                "runtime_force_gang",
                "runtime_max_shanten_regret",
                "runtime_max_effective_copy_loss",
                "runtime_progress_guard",
            )
        }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
