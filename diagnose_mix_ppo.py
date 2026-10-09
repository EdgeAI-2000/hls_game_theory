"""Train-split behavior probes; forced hu is diagnostic, not a formal policy."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import torch
from mix_ppo_baseline import SichuanResNet101, ACTIONS
from mix_ppo_rollout import episode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--mode", choices=["argmax", "force_hu", "sample"], default="argmax")
    args = p.parse_args()
    if args.output.exists():
        p.error("Output exists")
    torch.set_num_threads(2)
    torch.manual_seed(2026091702)
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    context_inputs = saved["config"].get("context_inputs", False)
    prospective_inputs = saved["config"].get("prospective_inputs", False)
    search_inputs = saved["config"].get("search_inputs", False)
    privileged_inputs = saved["config"].get("privileged_until_steps", 0) > 0
    model = SichuanResNet101(context_inputs=context_inputs,
                             prospective_inputs=prospective_inputs, search_inputs=search_inputs,
                             privileged_inputs=privileged_inputs).to(args.device).eval()
    model.load_state_dict(saved["model"])
    count, rows, hu_probs = Counter(), [], []
    h = ACTIONS.index("h")
    args.output.mkdir(parents=True)
    with (args.output / "games.jsonl").open("w") as out:
        for i in range(args.games):
            stream = episode(2026091702000000 + i, split="train", context_inputs=context_inputs,
                             prospective_inputs=prospective_inputs, search_inputs=search_inputs,
                             privileged_inputs=privileged_inputs, privileged_scale=0.)
            try:
                obs, mask, _ = next(stream)
                while True:
                    with torch.no_grad():
                        dist, _ = model(obs[None].to(args.device), mask[None].to(args.device))
                    chosen = int(dist.sample()) if args.mode == "sample" else int(dist.probs.argmax())
                    if args.mode == "force_hu" and mask[h]:
                        chosen = h
                    phase = "turn" if mask[:27].any() else "response"
                    count[phase + "_decisions"] += 1
                    count["action_" + ("discard" if chosen < 27 else ACTIONS[chosen])] += 1
                    if mask[h]:
                        count[phase + "_hu_available"] += 1
                        count[phase + "_hu_taken"] += int(chosen == h)
                        hu_probs.append(float(dist.probs[0, h]))
                    obs, mask, _ = stream.send(chosen)
            except StopIteration as end:
                rows.append(end.value)
                out.write(json.dumps(end.value) + "\n")
                out.flush()
    result = {"split": "train", "mode": args.mode, "games": len(rows),
              "privileged_scale": 0.,
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "mean_score": sum(r["net"] for r in rows)/len(rows),
              "hu_rate": sum(r["hu"] for r in rows)/len(rows),
              "ron_rate": sum(r["ron"] for r in rows)/len(rows),
              "dr": sum(r["dealt_in"] for r in rows)/len(rows),
              "mean_legal_hu_probability": sum(hu_probs)/len(hu_probs) if hu_probs else None,
              "counts": dict(count)}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
