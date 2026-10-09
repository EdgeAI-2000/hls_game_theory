"""Synchronous CPU environment workers with centralized, frozen-policy inference."""
import multiprocessing as mp
import os
import traceback
import torch


def _worker(pipe):
    from mix_ppo_rollout import episode
    torch.set_num_threads(1)
    stream = None
    while True:
        try:
            command, data = pipe.recv()
            if command == "close":
                return
            try:
                if command == "reset":
                    stream = episode(**data)
                    observation = next(stream)
                elif command == "step":
                    observation = stream.send(data)
                else:
                    raise ValueError(f"Unknown worker command: {command}")
                obs, mask, balance = observation
                pipe.send(("observation", (obs.numpy(), mask.numpy(), balance)))
            except StopIteration as end:
                pipe.send(("done", {**end.value, "worker_pid": os.getpid()}))
        except EOFError:
            return
        except Exception:
            pipe.send(("error", traceback.format_exc()))
            return


class EnvironmentPool:
    def __init__(self, workers=8, timeout=180):
        if workers < 1:
            raise ValueError("workers must be positive")
        self.timeout = timeout
        self.pipes, self.processes = [], []
        ctx = mp.get_context("spawn")
        for _ in range(workers):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=_worker, args=(child,))
            process.start()
            child.close()
            self.pipes.append(parent)
            self.processes.append(process)

    def exchange(self, requests):
        # Send to all workers before receiving, while consuming results in a
        # stable slot order so completion timing cannot change policy sampling.
        for slot, request in requests.items():
            self.pipes[slot].send(request)
        results = {}
        for slot in sorted(requests):
            if not self.pipes[slot].poll(self.timeout):
                raise TimeoutError(f"Environment worker {slot} timed out")
            kind, payload = self.pipes[slot].recv()
            if kind == "error":
                raise RuntimeError(f"Environment worker {slot}:\n{payload}")
            if kind == "observation":
                obs, mask, balance = payload
                payload = (torch.from_numpy(obs), torch.from_numpy(mask), balance)
            results[slot] = kind, payload
        return results

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for pipe in self.pipes:
            pipe.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def collect_parallel(model, args, game_index, pool):
    from mix_ppo_baseline import episode_gae
    samples, games = [], []
    while len(samples) < args.rollout_steps:
        requests, tracks, pending = {}, {}, {}
        for slot in range(args.envs):
            requests[slot] = ("reset", dict(seed=args.seed * 1000000 + game_index,
                opponent_style=args.opponents, manifest=args.manifest, split="train",
                max_steps=args.max_steps, reward_scale=args.reward_scale,
                context_inputs=args.context_inputs, prospective_inputs=args.prospective_inputs,
                search_inputs=getattr(args, "search_inputs", False),
                reward_mode=getattr(args, "reward_mode", "net"),
                opponent_checkpoint=getattr(args, "opponent_checkpoint", None),
                privileged_inputs=getattr(args, "privileged_until_steps", 0) > 0,
                privileged_scale=getattr(args, "privileged_scale", 0.)))
            game_index += 1
            tracks[slot] = []
        results = pool.exchange(requests)
        while results:
            for slot, (kind, payload) in results.items():
                if kind == "observation":
                    pending[slot] = payload
                    continue
                result = payload
                trajectory = tracks[slot]
                if trajectory:
                    balances = [r["balance"] for r in trajectory] + [result["final_balance"]]
                    rewards = [(b-a)/args.reward_scale for a, b in zip(balances, balances[1:])]
                    if abs(sum(rewards)*args.reward_scale-result["net"]) > 1e-5:
                        raise AssertionError("Incomplete net settlement")
                    if getattr(args, "reward_mode", "net") != "net":
                        rewards = result["learning_rewards"]
                    advantages, returns = episode_gae(rewards, [r["value"] for r in trajectory],
                                                       args.gamma, args.gae_lambda)
                    for row, advantage, target in zip(trajectory, advantages, returns):
                        row["advantage"], row["return"] = advantage, target
                    samples.extend(trajectory)
                games.append(result)
                pending.pop(slot, None)
            if not pending:
                break
            slots = sorted(pending)
            with torch.no_grad():
                dist, values = model(torch.stack([pending[s][0] for s in slots]).to(args.device),
                                     torch.stack([pending[s][1] for s in slots]).to(args.device))
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
            requests = {}
            for i, slot in enumerate(slots):
                obs, mask, balance = pending[slot]
                tracks[slot].append(dict(obs=obs, mask=mask, balance=balance,
                    action=int(actions[i]), old_logits=dist.logits[i].cpu(),
                    old_log_prob=float(log_probs[i]), value=float(values[i])))
                requests[slot] = ("step", int(actions[i]))
            results = pool.exchange(requests)
    return samples, games, game_index
