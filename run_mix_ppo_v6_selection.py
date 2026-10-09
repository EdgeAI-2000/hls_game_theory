"""Development selection, bounded continuation and auditable stopping."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
OLD = ROOT / "Mix_PPO_Sichuan_20260919_V6_MultiSeed"
RUN = ROOT / "Mix_PPO_Sichuan_20260923_V6_Selection"
SEEDS = (2026091911, 2026091912, 2026091913)
SELECT = tuple(range(2026092601, 2026092609))
MONITOR = tuple(range(2026092701, 2026092709))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def choose(rows):
    return max(rows, key=lambda row: (row["mean_net"], -row["train_seed"]))


def stopping(reference, stale, score):
    return (score, 0) if score >= reference + 1. else (reference, stale + 1)


def audit_outlier():
    import torch
    from unittest.mock import patch
    import mix_ppo_rollout
    from mix_ppo_selfplay import opponent_action
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint = OLD / "seed_2026091911/update_50.pt"
    captured = []
    original = mix_ppo_rollout.init_game
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result[0])
        return result
    with patch.object(mix_ppo_rollout, "init_game", capture):
        stream = mix_ppo_rollout.episode(2026092405000024, split="train",
            context_inputs=True, search_inputs=True, privileged_inputs=True)
        try:
            obs, mask, _ = next(stream)
            from mix_ppo_baseline import ACTIONS
            while True:
                legal = [a for a, valid in zip(ACTIONS, mask) if valid]
                action = opponent_action(str(checkpoint), captured[0], 0, legal,
                                         response=not bool(mask[:27].any()))
                obs, mask, _ = stream.send(ACTIONS.index(action))
        except StopIteration as end:
            result = end.value
    game = captured[0]
    report = dict(seed=2026092405000024, expected_net=7680, actual=result,
                  ledger=game.settlement_ledger, base_score=game.base_score,
                  max_fan=game.max_fan, checkpoint_sha256=digest(checkpoint))
    save(RUN / "outlier_audit.json", report)
    if result["net"] != 7680 or result["hu"]:
        raise ValueError("Outlier replay mismatch; manual inspection required")


def summarize(target, seeds, games, checkpoint):
    report = json.loads((target/"summary.json").read_text())
    rows = report["by_seed"]
    if report["checkpoint_sha256"] != digest(checkpoint) or {r["outer_seed"] for r in rows} != set(seeds):
        raise ValueError("Evaluation identity mismatch")
    if any(r["games"] != games or r["max_steps_or_unsettled"] for r in rows):
        raise ValueError("Incomplete evaluation")
    records = [json.loads(line) for f in target.glob("seed_*/games.jsonl") for line in f.read_text().splitlines()]
    if len(records) != games * len(seeds) or len({r["seed"] for r in records}) != len(records):
        raise ValueError("Duplicate/missing game seeds")
    values = np.array(sorted(r["net"] for r in records), dtype=float)
    means = np.array([r["avg_net"] for r in rows])
    half = float(t.ppf(.975, len(means)-1)*means.std(ddof=1)/len(means)**.5)
    cut = len(values)//10
    return dict(games=len(values), mean_net=float(means.mean()),
                seed_cluster_ci95=[float(means.mean()-half), float(means.mean()+half)],
                trimmed_mean=float(values[cut:-cut].mean()), max_net=float(values[-1]),
                mean_without_max=float(values[:-1].mean()),
                hu_rate=sum(r["hu"] for r in records)/len(records),
                dr=sum(r["dealt_in"] for r in records)/len(records),
                cvar10=float(values[:cut].mean()),
                missed_hu=sum(r["missed_hu"] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if RUN.exists() and not args.resume:
        raise FileExistsError(RUN)
    RUN.mkdir(exist_ok=True)
    def status(state, **kw):
        save(RUN/"status.json", dict(state=state, updated_at=time.time(), **kw))
    def execute(command, logfile):
        with logfile.open("a") as log:
            subprocess.run([sys.executable, *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    def evaluate(checkpoint, name, seeds, games):
        target = RUN/name
        if not (target/"summary.json").exists():
            if target.exists():
                target.rename(RUN/f"{name}_incomplete_{time.time_ns()}")
            execute(["eval_mix_ppo_parallel.py", "--checkpoint", str(checkpoint),
                "--output", str(target), "--split", "train", "--outer-seeds", *map(str,seeds),
                "--games", str(games), "--workers", "8", "--devices", "cuda:0"], RUN/"evaluation.log")
        return summarize(target, seeds, games, checkpoint)
    try:
        if not args.resume:
            cfg = json.loads((OLD/"seed_2026091911/train/config.json").read_text())
            files = set(cfg["source_sha256"]) | {Path(__file__).name, "eval_mix_ppo_parallel.py", "eval_mix_ppo_adapted.py"}
            snapshot = RUN/"source_snapshot"
            snapshot.mkdir()
            for f in files:
                shutil.copy2(ROOT/f, snapshot/f)
            save(RUN/"protocol.json", dict(candidates=SEEDS, selection_seeds=SELECT,
                selection_games_each=2000, selection_rule="maximum raw mean net; exact tie lowest train seed",
                monitor_seeds=MONITOR, monitor_games_each=1000,
                stages=[100,150,200,250,300], plateau_delta=1., plateau_patience=3,
                freeze_rule="highest monitoring mean among evaluated checkpoints, tie earlier",
                plateau_reference="reset only on >=1 point improvement over reference best",
                workers=8, device="cuda:0", holdout=False, convergence_claim=False,
                sources={f:digest(ROOT/f) for f in files},
                candidate_hashes={str(s):digest(OLD/f"seed_{s}/update_50.pt") for s in SEEDS}))
        protocol = json.loads((RUN/"protocol.json").read_text())
        for f,h in protocol["sources"].items():
            if digest(ROOT/f) != h:
                raise ValueError(f"Source changed: {f}")
        for s,h in protocol["candidate_hashes"].items():
            if digest(OLD/f"seed_{s}/update_50.pt") != h:
                raise ValueError("Candidate changed")
        if not (RUN/"outlier_audit.json").exists():
            status("outlier_replay")
            audit_outlier()
        audit = json.loads((RUN/"outlier_audit.json").read_text())
        if audit["actual"]["net"] != 7680:
            raise ValueError("Outlier audit not passed")
        results = []
        for seed in SEEDS:
            status("selection_evaluation", train_seed=seed, games=2000, workers=8)
            results.append(dict(train_seed=seed, **evaluate(OLD/f"seed_{seed}/update_50.pt",f"selection_{seed}",SELECT,250)))
            save(RUN/"selection_results.json", results)
        winner = choose(results)
        seed = winner["train_seed"]
        save(RUN/"selection_decision.json", dict(selected=winner,
            caveat="Development selection, not proof of significant superiority; single selected training seed"))
        cfg = json.loads((OLD/f"seed_{seed}/train/config.json").read_text())
        train = RUN/"train"
        train.mkdir(exist_ok=True)
        if not (train/"latest.pt").exists():
            shutil.copy2(OLD/f"seed_{seed}/update_50.pt",train/"latest.pt")
            save(train/"config.json",{**cfg,"output":str(train)})
        status("monitoring_evaluation", update=50, train_seed=seed)
        initial = evaluate(OLD/f"seed_{seed}/update_50.pt","monitor_50",MONITOR,125)
        history = [dict(update=50,**initial)]
        reference, stale = initial["mean_net"], 0
        for stage in (100,150,200,250,300):
            checkpoint = RUN/f"update_{stage}.pt"
            if not checkpoint.exists():
                status("training", train_seed=seed, target_update=stage, max_update=300, workers=8)
                command = ["train_mix_ppo_adapted.py", "--output",str(train),"--resume","--updates",str(stage)]
                for key in ("context_inputs","prospective_inputs","search_inputs","goal_selfplay"):
                    if cfg[key]:
                        command.append("--"+key.replace("_","-"))
                for key in ("reward_mode","workers","promotion_interval","promotion_seed","privileged_until_steps",
                            "device","envs","rollout_steps","minibatch","epochs","seed","opponents","manifest",
                            "max_steps","reward_scale","gamma","gae_lambda","lr","clip","vf_coef",
                            "entropy_coef","kl_coef","kl_target","max_grad_norm"):
                    command.extend(["--"+key.replace("_","-"),str(cfg[key])])
                execute(command,RUN/"training.log")
                tmp=checkpoint.with_suffix(".tmp")
                shutil.copy2(train/"latest.pt",tmp)
                tmp.replace(checkpoint)
            status("monitoring_evaluation",update=stage,train_seed=seed,games=1000)
            result = evaluate(checkpoint,f"monitor_{stage}",MONITOR,125)
            history.append(dict(update=stage,**result))
            reference,stale = stopping(reference,stale,result["mean_net"])
            save(RUN/"learning_curve.json",history)
            if stale>=3:
                break
        best=max(history,key=lambda r:(r["mean_net"],-r["update"]))
        source=OLD/f"seed_{seed}/update_50.pt" if best["update"]==50 else RUN/f"update_{best['update']}.pt"
        shutil.copy2(source,RUN/"frozen.pt")
        save(RUN/"freeze.json",dict(train_seed=seed,selected=best,checkpoint_sha256=digest(RUN/"frozen.pt"),
             stopping_reason="development_plateau" if stale>=3 else "budget_exhausted",
             converged=False,holdout_run=False))
        status("development_complete",train_seed=seed,selected_update=best["update"],holdout=False)
    except Exception as error:
        status("failed",error=repr(error))
        raise


if __name__=="__main__":
    main()
