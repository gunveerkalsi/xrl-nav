"""Training entry point for PPO / DQN agents with MLP or attention policies."""
from __future__ import annotations

import csv
import json
import os
import time

import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from ..envs.nav_env import NavEnv
from .policies import RayAttentionExtractor


class NavLogCallback(BaseCallback):
    """Logs per-episode outcome to CSV and runs periodic deterministic evaluation (train + held-out)."""

    def __init__(self, out_dir: str, env_kwargs: dict, eval_every: int = 20000, eval_episodes: int = 20):
        super().__init__()
        self.out_dir, self.env_kwargs = out_dir, env_kwargs
        self.eval_every, self.eval_episodes = eval_every, eval_episodes
        self._next_eval = eval_every
        self.ep_file = open(os.path.join(out_dir, "episodes.csv"), "w", newline="")
        self.ep_writer = csv.writer(self.ep_file)
        self.ep_writer.writerow(["timestep", "reward", "length", "success", "collision", "wall_time"])
        self.ev_file = open(os.path.join(out_dir, "evaluations.csv"), "w", newline="")
        self.ev_writer = csv.writer(self.ev_file)
        self.ev_writer.writerow(["timestep", "split", "success_rate", "collision_rate", "mean_reward"])
        self.t0 = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.ep_writer.writerow([self.num_timesteps, ep["r"], ep["l"], int(info.get("success", False)),
                                         int(info.get("collision", False)), round(time.time() - self.t0, 1)])
        if self.num_timesteps >= self._next_eval:
            self._next_eval += self.eval_every
            self._evaluate()
        return True

    def _evaluate(self):
        if getattr(self, "_last_eval_t", -1) == self.num_timesteps:
            return
        self._last_eval_t = self.num_timesteps
        for split in ("train", "heldout"):
            env = NavEnv(**{**self.env_kwargs, "layout_mode": split})
            succ = coll = 0
            rews = []
            for i in range(self.eval_episodes):
                obs, _ = env.reset(seed=10_000 + i)
                done, total = False, 0.0
                while not done:
                    a, _ = self.model.predict(obs, deterministic=True)
                    obs, r, term, trunc, info = env.step(a)
                    total += r
                    done = term or trunc
                succ += info["success"]
                coll += info["collision"]
                rews.append(total)
            env.close()
            sr, cr = succ / self.eval_episodes, coll / self.eval_episodes
            self.ev_writer.writerow([self.num_timesteps, split, sr, cr, float(np.mean(rews))])
            self.logger.record(f"eval_{split}/success_rate", sr)
            self.logger.record(f"eval_{split}/collision_rate", cr)
            if self.verbose:
                print(f"[eval {self.num_timesteps}] {split}: success={sr:.2f} collision={cr:.2f}")
        self.ev_file.flush()
        self.ep_file.flush()

    def _on_training_end(self):
        self._evaluate()
        self.ep_file.close()
        self.ev_file.close()


def train(algo: str = "ppo", policy: str = "mlp", continuous: bool = False, timesteps: int = 300_000,
          seed: int = 0, n_envs: int = 4, out_dir: str = "runs/ppo_mlp", env_kwargs: dict | None = None,
          hparams: dict | None = None, eval_every: int = 25_000, subproc: bool = False, verbose: int = 1):
    algo = algo.lower()
    if algo == "dqn" and continuous:
        raise ValueError("DQN requires the discrete action space")
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env_kwargs = {**(env_kwargs or {}), "continuous": continuous, "layout_mode": "train"}
    n_envs = 1 if algo == "dqn" else n_envs
    vec_cls = SubprocVecEnv if subproc and n_envs > 1 else DummyVecEnv
    venv = make_vec_env(NavEnv, n_envs=n_envs, seed=seed, env_kwargs=env_kwargs, vec_env_cls=vec_cls,
                        monitor_kwargs={"info_keywords": ("success", "collision")})

    policy_kwargs: dict = {}
    if policy == "attention":
        policy_kwargs["features_extractor_class"] = RayAttentionExtractor
        policy_kwargs["features_extractor_kwargs"] = {"n_rays": env_kwargs.get("n_rays", 24)}
    elif policy != "mlp":
        raise ValueError(policy)

    if algo == "ppo":
        policy_kwargs["net_arch"] = dict(pi=[128, 128], vf=[128, 128])
        hp = dict(n_steps=1024, batch_size=256, n_epochs=10, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                  clip_range=0.2, ent_coef=0.01)
        hp.update(hparams or {})
        model = PPO("MlpPolicy", venv, policy_kwargs=policy_kwargs, seed=seed, verbose=0, device="cpu", **hp)
    elif algo == "dqn":
        policy_kwargs["net_arch"] = [128, 128]
        hp = dict(learning_rate=5e-4, buffer_size=100_000, learning_starts=5_000, batch_size=128, gamma=0.99,
                  train_freq=4, target_update_interval=2_000, exploration_fraction=0.3,
                  exploration_final_eps=0.05)
        hp.update(hparams or {})
        model = DQN("MlpPolicy", venv, policy_kwargs=policy_kwargs, seed=seed, verbose=0, device="cpu", **hp)
    else:
        raise ValueError(algo)

    fmt = ["csv"]
    try:
        import tensorboard  # noqa: F401
        fmt.append("tensorboard")
    except ImportError:
        pass
    model.set_logger(configure(os.path.join(out_dir, "sb3_logs"), fmt))
    cb = NavLogCallback(out_dir, {k: v for k, v in env_kwargs.items() if k != "layout_mode"}, eval_every)
    cb.verbose = verbose
    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=cb)
    model.save(os.path.join(out_dir, "model.zip"))
    meta = {"algo": algo, "policy": policy, "continuous": continuous, "timesteps": timesteps, "seed": seed,
            "n_envs": n_envs, "env_kwargs": env_kwargs, "hparams": hp, "train_seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(os.path.join(out_dir, "model.meta.json"), "w"), indent=2)
    venv.close()
    return model, meta
