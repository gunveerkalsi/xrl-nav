"""Policy architectures and a uniform differentiable wrapper around trained SB3 models.

``RayAttentionExtractor`` is an *inherently interpretable* feature extractor: every
lidar beam becomes a token, and a query built from the goal/motion state attends
over the beams. The attention weights say which beams the agent "looked at" when
forming its decision, and are compared against post-hoc saliency in the analysis.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class RayAttentionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, n_rays: int = 24, d_model: int = 32, n_heads: int = 2,
                 features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        self.n_rays = n_rays
        n_state = observation_space.shape[0] - n_rays
        angles = torch.arange(n_rays, dtype=torch.float32) * 2 * np.pi / n_rays
        self.register_buffer("ang", torch.stack([torch.sin(angles), torch.cos(angles)], 1))  # (n_rays, 2)
        self.ray_embed = nn.Sequential(nn.Linear(3, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.state_embed = nn.Sequential(nn.Linear(n_state, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * d_model + n_state, features_dim), nn.ReLU())
        self.last_attention: torch.Tensor | None = None  # (B, n_rays), averaged over heads
        self.last_attention_heads: torch.Tensor | None = None  # (B, H, n_rays)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        rays, state = obs[:, : self.n_rays], obs[:, self.n_rays:]
        tokens = torch.cat([rays.unsqueeze(-1), self.ang.expand(obs.shape[0], -1, -1)], -1)
        k = self.ray_embed(tokens)
        q = self.state_embed(state).unsqueeze(1)
        out, w = self.attn(q, k, k, need_weights=True, average_attn_weights=False)
        self.last_attention_heads = w.squeeze(2).detach()
        self.last_attention = self.last_attention_heads.mean(1)
        return self.head(torch.cat([out.squeeze(1), q.squeeze(1), state], -1))


class PolicyModel(nn.Module):
    """Uniform view of a trained SB3 model as ``obs -> scores``.

    * PPO, discrete   : scores = action logits, probs = softmax(logits)
    * PPO, continuous : scores = Gaussian action mean (v, w); probs undefined
    * DQN             : scores = Q-values, probs = softmax(Q / temperature)
    """

    def __init__(self, sb3_model, algo: str, continuous: bool = False, temperature: float = 1.0):
        super().__init__()
        self.model, self.algo, self.continuous, self.temperature = sb3_model, algo.lower(), continuous, temperature
        self.policy = sb3_model.policy
        self.policy.set_training_mode(False)

    @property
    def extractor(self):
        if self.algo == "dqn":
            return self.policy.q_net.features_extractor
        return self.policy.pi_features_extractor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.algo == "dqn":
            return self.policy.q_net(x)
        feats = self.policy.extract_features(x, self.policy.pi_features_extractor)
        latent = self.policy.mlp_extractor.forward_actor(feats)
        return self.policy.action_net(latent)

    def probs(self, x: torch.Tensor) -> torch.Tensor:
        s = self(x)
        if self.continuous:
            return s
        t = self.temperature if self.algo == "dqn" else 1.0
        return torch.softmax(s / t, -1)

    def _t(self, obs):
        return torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(-1, self.n_inputs)

    @property
    def n_inputs(self) -> int:
        return self.model.observation_space.shape[0]

    @torch.no_grad()
    def act(self, obs, deterministic: bool = True):
        a, _ = self.model.predict(np.asarray(obs, dtype=np.float32), deterministic=deterministic)
        return a

    @torch.no_grad()
    def scores_np(self, obs) -> np.ndarray:
        return self(self._t(obs)).numpy()

    @torch.no_grad()
    def probs_np(self, obs) -> np.ndarray:
        return self.probs(self._t(obs)).numpy()

    @torch.no_grad()
    def attention_np(self, obs):
        ext = self.extractor
        if not isinstance(ext, RayAttentionExtractor):
            return None
        self(self._t(obs))
        return ext.last_attention.numpy()

    def has_attention(self) -> bool:
        return isinstance(self.extractor, RayAttentionExtractor)


def load_policy(path: str, algo: str | None = None, continuous: bool | None = None) -> PolicyModel:
    """Load a saved SB3 zip. ``algo``/``continuous`` are read from the sidecar meta json if present."""
    import json
    import os

    from stable_baselines3 import DQN, PPO

    meta_path = os.path.splitext(path)[0] + ".meta.json"
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    algo = (algo or meta.get("algo", "ppo")).lower()
    continuous = meta.get("continuous", False) if continuous is None else continuous
    cls = DQN if algo == "dqn" else PPO
    model = cls.load(path, device="cpu", custom_objects={"RayAttentionExtractor": RayAttentionExtractor})
    pm = PolicyModel(model, algo, continuous)
    pm.meta = meta
    return pm
