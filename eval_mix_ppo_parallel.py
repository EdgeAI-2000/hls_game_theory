"""Bounded per-seed process evaluation; training never shares output folders."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["train", "holdout"], default="train")
    p.add_argument("--outer-seeds", type=int, nargs="+", required=True)
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    args = p.parse_args()
    if args.workers < 1 or len(set(args.outer_seeds)) != len(args.outer_seeds):
        p.error("Require positive workers and unique seeds")
    args.output.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()

    def worker(item):
        index, seed = item
        target = args.output / f"seed_{seed}"
        with (args.output / f"seed_{seed}.log").open("w") as log:
            subprocess.run([sys.executable, str(Path(__file__).with_name("eval_mix_ppo_adapted.py")),
                "--checkpoint", str(args.checkpoint.resolve()), "--output", str(target.resolve()),
                "--split", args.split, "--outer-seeds", str(seed), "--games", str(args.games),
                "--device", args.devices[index % len(args.devices)]], check=True, stdout=log, stderr=subprocess.STDOUT)
        report = json.loads((target / "summary.json").read_text())
        if report["checkpoint_sha256"] != digest:
            raise ValueError("Checkpoint changed during evaluation")
        return report

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            reports = list(pool.map(worker, enumerate(args.outer_seeds)))
        result = dict(state="complete", workers=args.workers, split=args.split,
                      checkpoint_sha256=digest, by_seed=[r for report in reports for r in report["by_seed"]])
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    except Exception as error:
        (args.output / "FAILED.json").write_text(json.dumps({"error": repr(error)}) + "\n")
        raise


if __name__ == "__main__":
    main()
