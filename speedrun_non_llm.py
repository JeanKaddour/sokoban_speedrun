"""Sokoban Speedrun — non-LLM track.

A from-scratch deep-RL agent that learns to solve Sokoban, racing the SAME metric as the LLM
track in ``speedrun.py``: wall-clock-to-target (and FLOPs-to-target), where the target is a
held-out Sokoban solve-rate whose lower 95% CI clears a threshold.

Quickstart (local RTX-3090, dedicated cu126 venv; see README.md):

    # one-time: build the boxoban env extension (float32) into the fork
    cd third_party/pufferlib && PATH=../../.venv_puffer/bin:$PATH bash build.sh boxoban --float

    # train to target, then eval the final checkpoint (record artifacts -> outputs/<run>/)
    .venv_puffer/bin/python speedrun_non_llm.py --run my-run

Boxoban obs is a 4x10x10 byte grid (channels: agent, walls, boxes, targets); actions are
{noop,down,up,left,right}. Solved == every box on a target. difficulty 0=basic..4=unfiltered.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

SOURCE_PATH = Path(__file__).resolve()
REPO_DIR = SOURCE_PATH.parent
# The vendored PufferLib fork (the environment dependency). Overridable for Modal, where the fork is
# copied to a fixed image path rather than living under the repo tree.
PUFFERLIB_DIR = Path(os.environ.get("PUFFERLIB_DIR") or (REPO_DIR / "third_party" / "pufferlib"))
if str(PUFFERLIB_DIR) not in sys.path:
    sys.path.insert(0, str(PUFFERLIB_DIR))

import contextlib  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

torch.set_float32_matmul_precision("high")  # TF32 on the fp32 matmuls (matches PufferLib's torch path)


def _amp(enabled: bool, device: str):
    """bf16 autocast for the policy forward/loss (matmuls -> tensor cores; reductions/exp stay fp32).
    PufferLib's fused _C trainer runs the whole PPO in bf16, so the recipe is bf16-robust; this is the
    torch-path equivalent. No GradScaler: bf16 has fp32 range. Off => bit-identical to PufferLib (fp32)."""
    if enabled and str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


ENV_NAME = "boxoban"
DIFFICULTY_NAMES = {0: "basic", 1: "easy", 2: "medium", 3: "hard", 4: "unfiltered"}
OBS_CHANNELS, GRID = 4, 10
NUM_ACTIONS = 5


# ============================ RUN RECIPE (single source of truth) ============================
# PufferLib's tuned boxoban PPO hyperparameters (config/boxoban.ini) + speedrun knobs. Every value
# is CLI-overridable (argparse defaults come from here). This is the hackable core — edit freely.
RECIPE = {
    # --- environment (the one piece we don't own) ---
    # Default = UNFILTERED: the official DeepMind Boxoban set with the canonical 1000-level test split
    # (non-leaky). It's the *easiest* official set (medium/hard are filtered to be harder); from-scratch
    # RL reaches a meaningful solve-rate here with the conv+recurrent arch, whereas medium/hard stay ~0
    # full-solve even for PufferLib's own recipe. basic/easy are PufferLib's procedural tiers (leaky).
    "difficulty": 4,            # 0=basic 1=easy 2=medium 3=hard 4=unfiltered
    "num_agents": 32768,        # matches PufferLib's tuned total_agents (use 8192 on a 24GB 3090)
    "max_episode_steps": 150,   # matches PufferLib's tuned boxoban.ini (was 120 — gave fewer solve steps)
    "holdout_frac": 0.1,        # fraction of the level pool held out (disjoint) for the eval gate
    # --- PPO rollout / optimization (PuffeRL) ---
    "total_timesteps": 200_000_000,
    "rollout_horizon": 64,
    "minibatch_size": 32768,    # segment minibatch: minibatch_segments = minibatch_size / horizon
    "replay_ratio": 1.6234,     # num_minibatches = replay_ratio * batch_size / minibatch_size
    "gamma": 0.989717,
    "gae_lambda": 0.759273,
    "vtrace_rho_clip": 3.13347,
    "vtrace_c_clip": 2.75328,
    "prio_alpha": 0.453827,     # prioritized-replay exponent
    "prio_beta0": 0.765589,     # prioritized-replay importance-sampling bias correction
    "clip_coef": 0.01,
    "vf_coef": 5.0,
    "vf_clip_coef": 5.0,
    "ent_coef": 0.000188411,
    "max_grad_norm": 1.20325,
    "learning_rate": 0.00134234,
    "anneal_lr": True,
    "min_lr_ratio": 0.37872,
    "beta1": 0.995526,          # Muon momentum
    "weight_decay": 0.0,
    "muon_eps": 1e-14,
    # --- model ---
    # cnn-mingru (spatial conv encoder + recurrent core) is the winning arch: it cracks the official
    # 4-box sets where the feedforward cnn and PufferLib's linear-encoder recurrent recipe both stall.
    "arch": "cnn-mingru",       # cnn | mingru|lstm|gru|mlp | cnn-{mingru,lstm,gru}
    "hidden_size": 256,
    "num_layers": 3,            # recurrent (planning) depth — deeper generalizes better on official sets
    "bf16": True,               # bf16 autocast for the policy forward/loss (big speedup on H100/5090
                                # tensor cores). --bf16 0 => fp32 (bit-identical to PufferLib's PuffeRL).
    # --- bookkeeping ---
    "seed": 42,
    "wandb": False,             # --wandb 1 to enable; logs train/loss/opt/perf metrics per iter
    "wandb_project": "sokoban-speedrun-non-llm",
    # --- held-out eval / gate ---
    "eval_episodes": 16384,     # total held-out episodes; grouped per-level (~16/level for the 1000-level test)
    "eval_seed": 12345,
    "eval_greedy": True,
    "eval_split": "test",       # official DeepMind held-out split (unfiltered/test = canonical 1000;
                                # medium/valid for difficulty 2); falls back to index-partition if absent
    "target": 0.70,             # leaderboard gate: lower 95% CI on per-level held-out solve-rate > this
}


# ============================ eval aggregates (verbatim from eval_speedrun.py) ===============
def _pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    if k > n:
        raise ValueError(f"pass@k requires k<=n, got k={k}, n={n}")
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1.0 - phat) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def _bootstrap_ci(values: list[float], *, n_boot: int = 10000, seed: int = 0,
                  alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of per-level solve fractions (verbatim from eval_speedrun)."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = arr[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1.0 - alpha / 2)))


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=str(REPO_DIR), timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


class DummyWandb:
    """No-op wandb stand-in (mirrors speedrun.py) so the train loop is wandb-agnostic."""

    def log(self, *a, **k): pass
    def finish(self, *a, **k): pass


# ============================ official DeepMind held-out levels ==============================
# The boxoban env's bin format is [agent(100), walls(100), boxes(100), targets(100), meta(5)] per
# 10x10 puzzle (= PUZZLE_SIZE 405). This encoder is byte-identical to the env's own C parser (verified
# against the generated medium bin), so we can build a held-out pool from the OFFICIAL DeepMind
# train/valid/test splits and point the eval env at it via the BOXOBAN_MAP_BIN env var (the env loads
# it directly when difficulty_id == -1). Clean disjoint held-out, no engine changes.
_BOX_CHARS = {"agent": ("@", "+"), "wall": ("#",), "box": ("$", "*"), "target": (".", "*", "+")}


def _encode_boxoban_puzzle(rows: list[str]) -> bytes:
    agent = bytearray(100); walls = bytearray(100); boxes = bytearray(100); targets = bytearray(100)
    ax = ay = -1; nb = nt = ot = 0
    for r in range(10):
        for c in range(10):
            ch = rows[r][c]
            is_a = ch in _BOX_CHARS["agent"]; is_w = ch in _BOX_CHARS["wall"]
            is_b = ch in _BOX_CHARS["box"]; is_t = ch in _BOX_CHARS["target"]
            i = r * 10 + c
            agent[i] = is_a; walls[i] = is_w; boxes[i] = is_b; targets[i] = is_t
            if is_a: ax, ay = c, r
            nb += is_b; nt += is_t; ot += (is_b and is_t)
    meta = bytes([ax & 0xFF, ay & 0xFF, nb, nt, ot])
    return bytes(agent) + bytes(walls) + bytes(boxes) + bytes(targets) + meta


def build_boxoban_bin(level_dir: Path, out_path: Path) -> int:
    import glob as _glob
    n = 0
    with open(out_path, "wb") as out:
        for path in sorted(_glob.glob(str(level_dir / "*.txt"))):
            rows: list[str] = []
            with open(path) as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if line.startswith(";"):
                        if rows:
                            out.write(_encode_boxoban_puzzle(rows)); rows = []; n += 1
                        continue
                    if not line.strip():
                        continue
                    if len(rows) < 10:
                        rows.append(line[:10])
                    if len(rows) == 10:
                        out.write(_encode_boxoban_puzzle(rows)); rows = []; n += 1
    return n


def ensure_holdout_bin(difficulty_name: str | None, split: str) -> Path | None:
    """Encode the official <difficulty>/<split> levels into a held-out bin (cached). Returns the
    absolute bin path, or None if those official levels aren't present (→ index-partition fallback)."""
    if not difficulty_name:
        return None
    levels = PUFFERLIB_DIR / "resources" / "boxoban" / "levels" / difficulty_name / split
    if not levels.is_dir() or not any(levels.glob("*.txt")):
        return None
    out = PUFFERLIB_DIR / "resources" / "boxoban" / f"boxoban_maps_{difficulty_name}_{split}.bin"
    if not out.exists():
        print(f"[eval] encoding official held-out {difficulty_name}/{split} -> {out.name}")
        build_boxoban_bin(levels, out)
    return out


# ============================ action sampling (ported from pufferlib.torch_pufferl) ==========
# Faithful copies so logprob/entropy match PuffeRL bit-for-bit (tests/test_ppo_equivalence.py).
from torch.distributions.utils import logits_to_probs  # noqa: E402


def _log_prob(logits, value):
    value = value.long().unsqueeze(-1)
    value, log_pmf = torch.broadcast_tensors(value, logits)
    value = value[..., :1]
    return log_pmf.gather(-1, value).squeeze(-1)


def _entropy(logits):
    min_real = torch.finfo(logits.dtype).min
    logits = torch.clamp(logits, min=min_real)
    p_log_p = logits * logits_to_probs(logits)
    return -p_log_p.sum(-1)


def sample_logits(logits, action=None):
    """Discrete (single-head) action sampling, matching PuffeRL. logits: (N, num_actions)."""
    logits = logits.unsqueeze(0)
    normalized_logits = logits - logits.logsumexp(dim=-1, keepdim=True)
    probs = logits_to_probs(logits)
    if action is None:
        probs = torch.nan_to_num(probs, 1e-8, 1e-8, 1e-8)
        action = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1, replacement=True).int()
        action = action.reshape(probs.shape[:-1])
    else:
        batch = logits[0].shape[0]
        action = action.view(batch, -1).T
    logprob = _log_prob(normalized_logits, action)
    logits_entropy = _entropy(normalized_logits).sum(0)
    return action.T, logprob.squeeze(0), logits_entropy.squeeze(0)


# ============================ Muon optimizer (ported from pufferlib.muon) ====================
_NS_COEFS = [(4.0848, -6.8946, 2.9270), (3.9505, -6.3029, 2.6377), (3.7418, -5.5913, 2.3037),
             (2.8769, -3.1427, 1.2046), (2.8366, -3.0525, 1.2012)]


def _zeropower_via_newtonschulz5(G, eps=1e-7):
    x = G.clone()
    if G.size(-2) > G.size(-1):
        x = x.mT
    x = x / torch.clamp(G.norm(dim=(-2, -1)), min=eps)
    for a, b, c in _NS_COEFS:
        s = x @ x.mT
        y = c * s
        y.diagonal(dim1=-2, dim2=-1).add_(b)
        y = y @ s
        y.diagonal(dim1=-2, dim2=-1).add_(a)
        x = y @ x
    if G.size(-2) > G.size(-1):
        x = x.mT
    return x.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon (Newton-Schulz orthogonalized momentum). Ported from pufferlib.muon.Muon; the only
    change is dropping the torch>=2.9 ``_to_scalar`` import (we float() the lr instead)."""

    def __init__(self, params, lr=0.0025, weight_decay=0.0, momentum=0.9, eps=1e-8):
        super().__init__(params, {"lr": lr, "weight_decay": weight_decay, "momentum": momentum,
                                  "eps": eps})

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = float(group["lr"]); wd = group["weight_decay"]; momentum = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                grad = p.grad
                buf.mul_(momentum)
                buf.add_(grad)
                grad = grad + buf * momentum
                if grad.ndim >= 2:
                    g2 = grad.view(grad.shape[0], -1)
                    g2 = _zeropower_via_newtonschulz5(g2)
                    g2 *= max(1, g2.size(-2) / g2.size(-1)) ** 0.5
                    grad = g2.view(p.shape)
                p.mul_(1 - lr * wd)
                p.sub_(lr * grad.view(p.shape))
        return loss


# ============================ VTrace+GAE advantage (ported from src/ kernels) ================
def puff_advantage(values, rewards, terminals, importance, gamma, gae_lambda, rho_clip, c_clip):
    """VTrace-corrected GAE advantage. Faithful torch port of the kernel that PuffeRL actually
    runs: `puff_advantage_row_vec` in src/pufferlib.cu (the vectorized path, selected whenever
    horizon % (16/precision_bytes) == 0, which holds for horizon 64 / float32). Note rho_t scales
    the WHOLE TD residual here — the scalar/CPU variant `rho_t*r + ...` differs and is unused.
    All tensors are (rows, horizon); advantages[:, -1] stays 0."""
    rows, horizon = values.shape
    adv = torch.zeros_like(values)
    lastlam = torch.zeros(rows, device=values.device, dtype=values.dtype)
    for t in range(horizon - 2, -1, -1):
        nextnonterminal = 1.0 - terminals[:, t + 1]
        imp = importance[:, t]
        rho_t = torch.clamp(imp, max=rho_clip)
        c_t = torch.clamp(imp, max=c_clip)
        delta = rho_t * (rewards[:, t + 1] + gamma * values[:, t + 1] * nextnonterminal - values[:, t])
        lastlam = delta + gamma * gae_lambda * c_t * lastlam * nextnonterminal
        adv[:, t] = lastlam
    return adv


# ============================ policy network (hackable) =====================================
class ActorCritic(nn.Module):
    """Small conv actor-critic over the 4x10x10 Sokoban board. forward()/forward_eval() return the
    same shapes as pufferlib.models.Policy so this plugs into the ported PuffeRL train step."""

    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(OBS_CHANNELS, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * GRID * GRID, hidden_size), nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_size, NUM_ACTIONS)
        self.critic = nn.Linear(hidden_size, 1)
        self.apply(self._init)
        nn.init.orthogonal_(self.actor.weight, 0.01); nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, 1.0); nn.init.zeros_(self.critic.bias)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, math.sqrt(2)); nn.init.zeros_(m.bias)

    def _encode(self, obs_flat: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs_flat.view(-1, OBS_CHANNELS, GRID, GRID).float())

    def initial_state(self, batch_size: int, device):
        return ()  # feedforward: fully-observable board, no recurrence

    def forward(self, x: torch.Tensor):
        """Training call: x (B, T, obs_size) -> logits (B*T, num_actions), values (B, T)."""
        B, T = x.shape[:2]
        h = self._encode(x.reshape(B * T, -1))
        return self.actor(h), self.critic(h).reshape(B, T)

    def forward_eval(self, x: torch.Tensor, state):
        """Rollout call: x (A, obs_size) -> logits (A, num_actions), values (A,), state."""
        h = self._encode(x)
        return self.actor(h), self.critic(h).squeeze(-1), state


class ConvEncoder(nn.Module):
    """Spatial conv encoder for the 4x10x10 board, drop-in for pufferlib.models.Policy (whose forward
    hands the encoder a flattened (N, obs_size) batch). Keeps the board's spatial structure that the
    linear DefaultEncoder throws away — the key lever for Sokoban perception."""

    def __init__(self, obs_size: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(OBS_CHANNELS, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * GRID * GRID, hidden_size), nn.ReLU(),
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, math.sqrt(2)); nn.init.zeros_(m.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.view(observations.shape[0], OBS_CHANNELS, GRID, GRID).float()
        return self.net(x)


def build_policy(cfg: argparse.Namespace, env: "BoxobanVecEnv", device: str) -> nn.Module:
    """Policy factory. Arch options:
      cnn                  - in-file feedforward conv actor-critic (no recurrence)
      {mingru,lstm,gru,mlp} - PufferLib's linear DefaultEncoder + that core (their tuned recipe)
      cnn-{mingru,lstm,gru} - spatial conv encoder + that recurrent core (conv perception + planning)
    All expose forward()/forward_eval()/initial_state() the matched train step needs."""
    if cfg.arch == "cnn":
        return ActorCritic(cfg.hidden_size).to(device)
    import pufferlib.models as pm
    nets = {"mingru": pm.MinGRU, "lstm": pm.LSTM, "gru": pm.GRU, "mlp": pm.MLP}
    enc_name, net_name = cfg.arch.split("-", 1) if "-" in cfg.arch else ("linear", cfg.arch)
    if net_name not in nets or enc_name not in ("linear", "cnn"):
        raise SystemExit(f"unknown --arch {cfg.arch!r} (cnn | {'|'.join(nets)} | cnn-{{mingru,lstm,gru}})")
    encoder = (ConvEncoder(env.obs_size, cfg.hidden_size) if enc_name == "cnn"
               else pm.DefaultEncoder(env.obs_size, cfg.hidden_size))
    decoder = pm.DefaultDecoder(env.act_sizes, cfg.hidden_size)
    network = nets[net_name](cfg.hidden_size, num_layers=int(cfg.num_layers))
    return pm.Policy(encoder, decoder, network).to(device)


# ============================ PufferLib boxoban vec-env (the dependency) =====================
_TYPESTR = {torch.uint8: "|u1", torch.float32: "<f4"}


class _CudaPtr:
    def __init__(self, ptr: int, shape: tuple[int, ...], dtype: torch.dtype):
        self.__cuda_array_interface__ = {"data": (ptr, False), "shape": shape,
                                         "typestr": _TYPESTR[dtype], "version": 2}


class BoxobanVecEnv:
    """Thin handle over PufferLib's native boxoban vec-env. Only the environment is borrowed; no
    PufferLib training code runs. obs/reward/terminal buffers are read zero-copy from GPU pointers."""

    def __init__(self, *, difficulty: int, num_agents: int, max_steps: int, seed: int,
                 eval_mode: bool = False, holdout_frac: float = 0.0):
        from pufferlib import pufferl, _C
        assert getattr(_C, "env_name", None) == ENV_NAME, (
            f"_C built for {getattr(_C, 'env_name', None)!r}, not {ENV_NAME!r}. "
            f"Run: cd {PUFFERLIB_DIR} && bash build.sh {ENV_NAME} --float")
        saved = sys.argv
        sys.argv = [saved[0]]
        try:
            args = pufferl.load_config(ENV_NAME)
        finally:
            sys.argv = saved
        args["vec"]["num_buffers"] = 1
        args["vec"]["total_agents"] = int(num_agents)
        args["env"]["difficulty"] = int(difficulty)
        args["env"]["max_steps"] = int(max_steps)
        # Held-out split (forked boxoban kwargs): train env samples the first (1-holdout_frac) of the
        # level pool, the eval env samples the disjoint held-out tail. Same holdout_frac on both sides.
        args["env"]["eval"] = 1 if eval_mode else 0
        args["env"]["holdout_frac"] = float(holdout_frac)
        args["seed"] = int(seed)
        args["train"]["seed"] = int(seed)
        self.args = args
        self.gpu = bool(_C.gpu)
        self.device = "cuda" if self.gpu else "cpu"
        self._vec = _C.create_vec(args, _C.gpu)
        self.num_agents = int(self._vec.total_agents)
        self.obs_size = int(self._vec.obs_size)
        self.act_sizes = list(self._vec.act_sizes)
        self._vec.reset()

    def _view(self, ptr, shape, dtype):
        if self.gpu:
            return torch.as_tensor(_CudaPtr(ptr, shape, dtype))
        n = int(np.prod(shape))
        ctype = {torch.uint8: ctypes.c_uint8, torch.float32: ctypes.c_float}[dtype]
        return torch.frombuffer((ctype * n).from_address(ptr), dtype=dtype).reshape(shape)

    def obs(self) -> torch.Tensor:
        return self._view(self._vec.gpu_obs_ptr if self.gpu else self._vec.obs_ptr,
                          (self.num_agents, self.obs_size), torch.uint8)

    def step(self, actions: torch.Tensor):
        a = actions.reshape(self.num_agents, 1).to(torch.float32).contiguous()
        if self.gpu:
            a = a.cuda()
            self._vec.gpu_step(a.data_ptr()); torch.cuda.synchronize()
        else:
            self._vec.cpu_step(a.data_ptr())
        rew = self._view(self._vec.gpu_rewards_ptr if self.gpu else self._vec.rewards_ptr,
                         (self.num_agents,), torch.float32)
        term = self._view(self._vec.gpu_terminals_ptr if self.gpu else self._vec.terminals_ptr,
                          (self.num_agents,), torch.float32)
        return rew, term

    def log(self) -> dict:
        from pufferlib.pufferl import unroll_nested_dict
        return dict(unroll_nested_dict(self._vec.log()))

    def close(self):
        self._vec.close()


# ============================ rollout (PuffeRL convention) ===================================
def collect_rollout(env: BoxobanVecEnv, policy: ActorCritic, horizon: int, device: str, amp: bool = False):
    """Collect `horizon` steps for all agents, matching PuffeRL.rollouts: stored rewards[t]/terminals[t]
    are those *received arriving at* obs[t] (i.e. for action[t-1]); the advantage uses rewards[t+1]."""
    A, S = env.num_agents, env.obs_size
    obs = torch.empty(horizon, A, S, dtype=torch.uint8, device=device)
    act = torch.empty(horizon, A, dtype=torch.long, device=device)
    logp = torch.empty(horizon, A, device=device)
    val = torch.empty(horizon, A, device=device)
    rew = torch.empty(horizon, A, device=device)
    done = torch.empty(horizon, A, device=device)
    state = policy.initial_state(A, device)
    if amp:  # carry recurrent state in bf16 so it matches the autocast activations (lerp etc.)
        state = tuple(s.to(torch.bfloat16) for s in state)
    r = torch.zeros(A, device=device)
    d = torch.zeros(A, device=device)
    for t in range(horizon):
        o = env.obs().to(device, copy=True)
        with torch.no_grad(), _amp(amp, device):
            logits, value, state = policy.forward_eval(o, state)
            action, logprob, _ = sample_logits(logits)
        obs[t] = o
        act[t] = action.reshape(A)
        logp[t] = logprob
        val[t] = value.reshape(A)
        rew[t] = r
        done[t] = d
        reward, terminal = env.step(action.reshape(A))
        r = reward.to(device); d = terminal.to(device)
    return obs, act, val, logp, rew, done


# ============================ PuffeRL train step (faithful port) =============================
def puff_train(policy: nn.Module, optimizer: torch.optim.Optimizer, obs, act, val, logp, rew, ter,
               cfg, epoch: int, total_epochs: int) -> dict:
    """One PuffeRL training pass over a rollout. Faithful port of
    pufferlib.torch_pufferl.PuffeRL.train(): reward clamp, prioritized segment replay, VTrace+GAE
    advantage recomputed per minibatch with refreshed values, clipped policy+value loss, Muon step,
    cosine LR. Buffers are (horizon, agents[, obs]); updates `policy` in place. Returns mean losses."""
    device = val.device
    horizon, total_agents = val.shape
    batch_size = total_agents * horizon
    minibatch_segments = cfg.minibatch_size // horizon

    b0, a = cfg.prio_beta0, cfg.prio_alpha
    clip_coef, vf_clip = cfg.clip_coef, cfg.vf_clip_coef
    anneal_beta = b0 + (1 - b0) * a * epoch / total_epochs
    ratio_buf = torch.ones(total_agents, horizon, device=device)

    if cfg.anneal_lr and epoch > 0:
        lr_ratio = epoch / total_epochs
        lr_min = cfg.learning_rate * cfg.min_lr_ratio
        lr = lr_min + 0.5 * (cfg.learning_rate - lr_min) * (1 + math.cos(math.pi * lr_ratio))
        optimizer.param_groups[0]["lr"] = lr

    # [horizon, agents] (contiguous writes) -> [agents, horizon] (segment indexing)
    obs_t = obs.transpose(0, 1).contiguous()
    act_t = act.transpose(0, 1).contiguous()
    val_t = val.T.contiguous()
    lp_t = logp.T.contiguous()
    rew_t = rew.T.contiguous().clamp(-1, 1)
    ter_t = ter.T.contiguous()

    num_minibatches = int(cfg.replay_ratio * batch_size / cfg.minibatch_size)
    sums = {k: 0.0 for k in ("policy", "value", "entropy", "approx_kl", "clipfrac", "grad_norm")}
    advantages = torch.zeros_like(val_t)
    for _ in range(num_minibatches):
        advantages = puff_advantage(val_t, rew_t, ter_t, ratio_buf, cfg.gamma, cfg.gae_lambda,
                                    cfg.vtrace_rho_clip, cfg.vtrace_c_clip)
        seg_adv = advantages.abs().sum(axis=1)
        prio_weights = torch.nan_to_num(seg_adv ** a, 0, 0, 0)
        prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)
        idx = torch.multinomial(prio_probs, minibatch_segments, replacement=True)
        mb_prio = (total_agents * prio_probs[idx, None]) ** -anneal_beta

        mb_obs = obs_t[idx]
        mb_actions = act_t[idx]
        mb_logprobs = lp_t[idx]
        mb_values = val_t[idx]
        mb_returns = advantages[idx] + mb_values
        mb_advantages = advantages[idx]

        with _amp(getattr(cfg, "bf16", False), device):   # bf16 autocast: matmuls -> tensor cores
            logits, newvalue = policy(mb_obs)
            _, newlogprob, entropy = sample_logits(logits, action=mb_actions)
            newlogprob = newlogprob.reshape(mb_logprobs.shape)
            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()
            ratio_buf[idx] = ratio.detach().float()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()

            adv = mb_advantages
            adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)
            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(mb_returns.shape)
            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
            v_loss = 0.5 * torch.max((newvalue - mb_returns) ** 2, (v_clipped - mb_returns) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy_loss
        val_t[idx] = newvalue.detach().float()

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
        optimizer.step()

        sums["policy"] += pg_loss.item()
        sums["value"] += v_loss.item()
        sums["entropy"] += entropy_loss.item()
        sums["approx_kl"] += approx_kl.item()
        sums["clipfrac"] += clipfrac.item()
        sums["grad_norm"] += float(grad_norm)
    return {k: v / max(1, num_minibatches) for k, v in sums.items()}


# ============================ record log (mirrors speedrun.py RunLogger) =====================
class RunLogger:
    _DIVIDER = "=" * 100

    def __init__(self, run_dir: Path, args_dict: dict):
        self.path = None
        self._fh = None
        self._t0 = None
        run_dir.mkdir(parents=True, exist_ok=True)
        source = SOURCE_PATH.read_text(encoding="utf-8")
        sha = self._save_source_snapshot(run_dir, source)
        self.path = run_dir / f"log_{uuid.uuid4().hex[:8]}.txt"
        self._fh = self.path.open("w", encoding="utf-8")
        self._fh.write(self._header(args_dict, source, sha)); self._fh.flush()
        print(f"RunLogger: writing record log to {self.path}")

    @staticmethod
    def _save_source_snapshot(run_dir: Path, source: str) -> str:
        sd = run_dir / "source"; sd.mkdir(parents=True, exist_ok=True)
        (sd / "speedrun_non_llm.py").write_text(source, encoding="utf-8")
        sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
        (sd / "manifest.json").write_text(json.dumps(
            {"version": 1, "git_commit": _git_commit(), "source_file": str(SOURCE_PATH),
             "files": {"speedrun_non_llm.py": {"sha256": sha, "bytes": len(source.encode())}}},
            indent=2) + "\n", encoding="utf-8")
        return sha

    @staticmethod
    def _header(args_dict: dict, source: str, sha: str) -> str:
        lines = ["python: " + " ".join(sys.version.split()), f"torch: {torch.__version__}"]
        try:
            import importlib.metadata
            lines.append(f"pufferlib: {importlib.metadata.version('pufferlib')}")
        except Exception:
            lines.append("pufferlib: (vendored fork third_party/pufferlib)")
        lines.append(f"git commit: {_git_commit()}")
        lines.append(f"source snapshot: source/speedrun_non_llm.py sha256:{sha}")
        try:
            smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
            lines.append(smi.stdout if smi.returncode == 0 else f"nvidia-smi failed: {smi.stderr}")
        except Exception as exc:
            lines.append(f"nvidia-smi unavailable: {exc!r}")
        lines.append(f"argv: {sys.argv}")
        lines.append(f"args: {args_dict}")
        d = RunLogger._DIVIDER
        return f"{source}\n{d}\n" + "\n".join(lines) + f"\n{d}\n"

    def start_clock(self, anchor): self._t0 = anchor if self._t0 is None else self._t0
    def record_time(self): return time.monotonic() - self._t0 if self._t0 is not None else 0.0
    def _write(self, line):
        if self._fh is not None: self._fh.write(line + "\n"); self._fh.flush()

    def log_step(self, step, num_steps, *, reward_mean, solved_frac, loss, grad_norm, cum_flops):
        rt = self.record_time()
        self._write(f"step:{step + 1}/{num_steps} record_time:{rt:.1f}s step_avg:{rt / (step + 1):.1f}s "
                    f"reward_mean:{max(0.0, reward_mean):.4f} solved_frac:{solved_frac:.4f} "
                    f"loss:{loss:.4f} grad_norm:{grad_norm:.4f} cum_flops:{cum_flops:.6e}")

    def log_final_checkpoint(self, path, cum_flops):
        self._write(f"final_checkpoint:{path} record_time:{self.record_time():.1f}s cum_flops:{cum_flops:.6e}")

    def close(self):
        if self._fh is not None: self._fh.close(); self._fh = None


# ============================ training ======================================================
def set_seed(seed: int):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def flops_per_env_step(num_params: int, replay_ratio: float) -> float:
    """2*N for the rollout forward + 6*N per gradient pass; each collected step is replayed
    replay_ratio times (analogous to speedrun.py's dense-model estimator)."""
    return num_params * (2.0 + 6.0 * float(replay_ratio))


def train(cfg: argparse.Namespace) -> Path:
    out_dir = (REPO_DIR / cfg.output_dir / cfg.run).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(PUFFERLIB_DIR)  # boxoban C env caches level files under a cwd-relative resources/
    set_seed(cfg.seed)

    env = BoxobanVecEnv(difficulty=cfg.difficulty, num_agents=cfg.num_agents,
                        max_steps=cfg.max_episode_steps, seed=cfg.seed,
                        eval_mode=False, holdout_frac=cfg.holdout_frac)
    device = env.device
    policy = build_policy(cfg, env, device)
    optimizer = Muon(policy.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
                     momentum=cfg.beta1, eps=cfg.muon_eps)
    num_params = sum(p.numel() for p in policy.parameters())
    fpes = flops_per_env_step(num_params, cfg.replay_ratio)

    wandb_run = DummyWandb()
    if cfg.wandb:
        import wandb
        wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.run,
                               config={**vars(cfg), "num_params": num_params})

    logger = RunLogger(out_dir, {**vars(cfg), "env_config": env.args["env"], "num_params": num_params})
    steps_per_iter = cfg.num_agents * cfg.rollout_horizon
    total_iters = max(1, math.ceil(cfg.total_timesteps / steps_per_iter))
    print(f"[train] params={num_params:,} device={device} difficulty={DIFFICULTY_NAMES.get(cfg.difficulty)} "
          f"agents={cfg.num_agents} horizon={cfg.rollout_horizon} iters={total_iters} "
          f"minibatches/iter={int(cfg.replay_ratio * steps_per_iter / cfg.minibatch_size)}")

    logger.start_clock(time.monotonic())
    global_step = 0
    for it in range(total_iters):
        obs, act, val, logp, rew, done = collect_rollout(env, policy, cfg.rollout_horizon, device, amp=cfg.bf16)
        global_step += steps_per_iter
        stats = puff_train(policy, optimizer, obs, act, val, logp, rew, done, cfg, it, total_iters)
        elog = env.log()
        solved = float(elog.get("perf", 0.0))
        cum_flops = fpes * global_step
        logger.log_step(it, total_iters, reward_mean=float(elog.get("targets_hit", 0.0)),
                        solved_frac=solved, loss=float(stats["policy"] + cfg.vf_coef * stats["value"]),
                        grad_norm=float(stats["grad_norm"]), cum_flops=cum_flops)
        sps = global_step / max(1e-6, logger.record_time())
        wandb_run.log({
            "train/solved_frac": solved,
            "train/targets_hit": float(elog.get("targets_hit", 0.0)),
            "train/episode_return": float(elog.get("episode_return", 0.0)),
            "loss/policy": stats["policy"], "loss/value": stats["value"], "loss/entropy": stats["entropy"],
            "loss/approx_kl": stats["approx_kl"], "loss/clipfrac": stats["clipfrac"],
            "opt/grad_norm": stats["grad_norm"], "opt/lr": optimizer.param_groups[0]["lr"],
            "perf/sps": sps, "perf/cum_flops": cum_flops, "perf/record_time_s": logger.record_time(),
            "epoch": it + 1,
        }, step=global_step)
        if it % max(1, cfg.print_every) == 0 or it == total_iters - 1:
            print(f"  iter {it + 1}/{total_iters} gstep={global_step:,} solved={solved:.3f} "
                  f"ent={stats['entropy']:.3f} kl={stats['approx_kl']:.4f} gnorm={stats['grad_norm']:.2f} "
                  f"SPS={sps:,.0f} t={logger.record_time():.0f}s")
        if cfg.checkpoint_every and (it + 1) % cfg.checkpoint_every == 0:
            torch.save(policy.state_dict(), out_dir / f"step_{global_step:012d}.pt")

    final_ckpt = out_dir / "final.pt"
    torch.save(policy.state_dict(), final_ckpt)
    logger.log_final_checkpoint(final_ckpt, fpes * global_step)
    logger.close(); env.close(); wandb_run.finish()
    print(f"[train] done. final checkpoint: {final_ckpt}")
    return final_ckpt


# ============================ held-out evaluation ===========================================
def evaluate(cfg: argparse.Namespace, checkpoint: Path) -> dict:
    """Roll the trained policy over the DISJOINT held-out Boxoban split; gate on solve-rate's lower
    95% CI. The forked boxoban env reserves the last holdout_frac of the level pool for eval (never
    sampled during training). Each episode is one attempt (k=1) -> pass@1 = solve rate."""
    os.chdir(PUFFERLIB_DIR)
    set_seed(cfg.eval_seed)
    diff_name = DIFFICULTY_NAMES.get(cfg.difficulty)
    holdout_bin = ensure_holdout_bin(diff_name, cfg.eval_split)
    if holdout_bin is not None:
        # official DeepMind held-out split: load that pool directly (difficulty=-1 => BOXOBAN_MAP_BIN)
        os.environ["BOXOBAN_MAP_BIN"] = str(holdout_bin)
        eval_difficulty, eval_mode, eval_holdout = -1, False, 0.0
        split_label = f"{diff_name}/{cfg.eval_split}-official"
    else:
        # fallback: disjoint index-partition of the (procedural) difficulty pool
        eval_difficulty, eval_mode, eval_holdout = cfg.difficulty, True, cfg.holdout_frac
        split_label = f"{diff_name}-holdout{cfg.holdout_frac}"
    # Eval is num_agents-INVARIANT: cap the eval pool so each agent completes many episodes (>= ~32).
    # The per-level mean only converges when every held-out level is sampled enough; with the training
    # num_agents (e.g. 32768) vs eval_episodes (16384) each agent finishes <1 episode, levels are
    # starved, and the score collapses (32768 agents -> 0.27 vs 1024 -> 0.69 on the SAME checkpoint).
    # PufferLib sidesteps this entirely by reading the C-aggregated env.log()['perf'] (a true
    # per-episode solve mean) over long rollouts; we cross-check against it below.
    eval_num_agents = min(cfg.num_agents, max(256, cfg.eval_episodes // 32))
    env = BoxobanVecEnv(difficulty=eval_difficulty, num_agents=eval_num_agents,
                        max_steps=cfg.max_episode_steps, seed=cfg.eval_seed,
                        eval_mode=eval_mode, holdout_frac=eval_holdout)
    device = env.device
    policy = build_policy(cfg, env, device)
    policy.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    policy.eval()
    print(f"[eval] loaded {checkpoint}; scoring >= {cfg.eval_episodes} episodes "
          f"(held-out: {split_label}, greedy={cfg.eval_greedy})")

    # Per-LEVEL scoring (unbiased): roll greedily and key each completed episode by its INITIAL board
    # (which uniquely identifies the held-out level). A win gives terminal reward >= +1; max-steps
    # truncation gives <= -0.75 (no deadlock early-stop), so solved = (terminal reward > 0.5).
    # Averaging over DISTINCT levels (each weighted equally) removes the startup-transient and
    # fast-cycling biases that make episode-stream averaging depend on how many episodes are run.
    from collections import defaultdict
    init_board = env.obs().to(device, copy=True).clone()   # each agent's current-episode start board
    state = policy.initial_state(env.num_agents, device)
    if getattr(cfg, "bf16", False):
        state = tuple(s.to(torch.bfloat16) for s in state)
    lvl_solved: dict = defaultdict(int)
    lvl_n: dict = defaultdict(int)
    total_eps = 0
    while total_eps < cfg.eval_episodes:
        with torch.no_grad(), _amp(getattr(cfg, "bf16", False), device):
            logits, _, state = policy.forward_eval(env.obs().to(device, copy=True), state)
            action = logits.argmax(-1) if cfg.eval_greedy else sample_logits(logits)[0].reshape(-1)
        rew, term = env.step(action)
        done = term.to(device) > 0.5
        solved = done & (rew.to(device) > 0.5)
        idx = done.nonzero(as_tuple=True)[0]
        if idx.numel():
            for b, s in zip(init_board[idx].cpu().numpy(), solved[idx].cpu().numpy()):
                h = b.tobytes(); lvl_n[h] += 1; lvl_solved[h] += int(s)
            total_eps += int(idx.numel())
            for s_ in state:                       # reset recurrent state for finished envs
                s_[:, idx, :] = 0.0
            init_board[idx] = env.obs().to(device, copy=True)[idx]   # new episode's start board
    env_perf = float(env.log().get("perf", float("nan")))  # PufferLib's C-aggregated solve mean (num_agents-invariant)
    env.close()

    keys = list(lvl_n.keys())
    per_frac = [lvl_solved[h] / lvl_n[h] for h in keys]
    per_n = [lvl_n[h] for h in keys]
    per_c = [lvl_solved[h] for h in keys]
    n = len(keys)                                  # distinct held-out levels seen
    pass_at_1 = sum(per_frac) / max(1, n)          # mean per-level solve fraction (each level weight 1)
    ci_low, ci_high = _bootstrap_ci(per_frac, seed=cfg.eval_seed)
    se = float(np.std(per_frac) / math.sqrt(max(1, n))) if n else 0.0
    record = {
        "seed": cfg.eval_seed, "run": cfg.run, "step": None, "checkpoint": str(checkpoint),
        "model": f"ppo-{cfg.arch}-h{cfg.hidden_size}-L{cfg.num_layers}", "eval_data": f"boxoban:{split_label}",
        "git_commit": _git_commit(), "n_puzzles": n, "k": int(np.median(per_n)) if n else 1,
        "pass_at_1": pass_at_1, "pass_at_k": {"1": pass_at_1},
        "solve_rate_episode": env_perf,    # PufferLib env.log()['perf'] cross-check (per-episode mean)
        "eval_num_agents": int(eval_num_agents),
        "ci_low": ci_low, "ci_high": ci_high, "se": se,
        "n_extract_fail": 0, "n_answered": int(sum(per_n)), "n_length_trunc": 0,
        "answer_rate": 1.0, "solve_given_answer": pass_at_1, "trunc_frac": 0.0,
        "sampling": {"temperature": None, "top_p": None, "top_k": None, "min_p": 0.0,
                     "max_tokens": cfg.max_episode_steps, "seed": cfg.eval_seed,
                     "greedy": cfg.eval_greedy, "episodes": int(sum(per_n)), "logprobs": 0, "interrupt": None},
        "per_puzzle_solve_frac": per_frac, "per_puzzle_n": per_n,
        "per_puzzle_solved_count": per_c, "per_puzzle_answered_count": per_n,
        "per_puzzle_length_trunc_count": [0] * n,
    }
    out_dir = (REPO_DIR / cfg.output_dir / cfg.run).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_step000000_seed{cfg.eval_seed}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    cleared = "CLEARS" if ci_low > cfg.target else "DOES NOT CLEAR"
    print(f"[eval] per-level pass@1={pass_at_1:.4f}  95% CI [{ci_low:.4f}, {ci_high:.4f}]  "
          f"({n} levels, {int(sum(per_n))} episodes @ {eval_num_agents} agents, ~{sum(per_n)/max(1,n):.0f}/level)  "
          f"vs target {cfg.target}: {cleared}")
    print(f"[eval] PufferLib env.log perf (per-episode solve, cross-check) = {env_perf:.4f}")
    print(f"[eval] wrote {out_path}")
    return record


# ============================ throughput profile ============================================
def profile_run(cfg: argparse.Namespace) -> None:
    """Time per-iteration cost split into env-step / policy-inference (rollout) / train (fwd+bwd+Muon),
    so we can reason about where wall-clock goes and how it scales to other GPUs."""
    os.chdir(PUFFERLIB_DIR)
    set_seed(cfg.seed)
    env = BoxobanVecEnv(difficulty=cfg.difficulty, num_agents=cfg.num_agents,
                        max_steps=cfg.max_episode_steps, seed=cfg.seed)
    device = env.device
    policy = build_policy(cfg, env, device)
    optimizer = Muon(policy.parameters(), lr=cfg.learning_rate, momentum=cfg.beta1, eps=cfg.muon_eps)
    H, A = cfg.rollout_horizon, env.num_agents

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    roll = collect_rollout(env, policy, H, device, amp=cfg.bf16)            # warmup
    puff_train(policy, optimizer, *roll, cfg, 1, 100); sync()
    N = 10
    sync(); t0 = time.perf_counter()
    for _ in range(N * H):
        env.step(torch.randint(0, NUM_ACTIONS, (A,), device=device))
    sync(); t_env = (time.perf_counter() - t0) / N
    sync(); t0 = time.perf_counter()
    for _ in range(N):
        roll = collect_rollout(env, policy, H, device, amp=cfg.bf16)
    sync(); t_roll = (time.perf_counter() - t0) / N
    sync(); t0 = time.perf_counter()
    for _ in range(N):
        puff_train(policy, optimizer, *roll, cfg, 1, 100)
    sync(); t_train = (time.perf_counter() - t0) / N

    steps = A * H
    t_pol = max(0.0, t_roll - t_env)
    t_iter = t_roll + t_train
    nparams = sum(p.numel() for p in policy.parameters())
    print(f"[profile] arch={cfg.arch} h{cfg.hidden_size} L{cfg.num_layers} params={nparams:,} "
          f"agents={A} horizon={H} minibatches/iter={int(cfg.replay_ratio * steps / cfg.minibatch_size)} device={device}")
    print(f"[profile] per-iter ({steps:,} env-steps):")
    print(f"  env step (C/GPU env)       {t_env*1e3:7.1f} ms  ({100*t_env/t_iter:4.1f}%)")
    print(f"  policy inference (rollout) {t_pol*1e3:7.1f} ms  ({100*t_pol/t_iter:4.1f}%)")
    print(f"  train (fwd+bwd+Muon)       {t_train*1e3:7.1f} ms  ({100*t_train/t_iter:4.1f}%)")
    print(f"  TOTAL                      {t_iter*1e3:7.1f} ms  -> {steps/t_iter:,.0f} env-steps/s")
    print(f"[profile] GPU policy work (inference+train) = {100*(t_pol+t_train)/t_iter:.0f}% of wall-clock "
          f"-> faster GPUs scale most of it; env step = {100*t_env/t_iter:.0f}%")
    env.close()


# ============================ CLI ===========================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sokoban Speedrun — non-LLM (PufferLib boxoban + in-file PuffeRL PPO)")
    p.add_argument("--run", type=str, default=f"boxoban-{int(time.time())}")
    p.add_argument("--output-dir", type=str, default="outputs")
    for key, val in RECIPE.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(val, bool):
            p.add_argument(flag, default=val, type=lambda s: str(s).lower() not in ("0", "false", "no"))
        else:
            p.add_argument(flag, default=val, type=type(val))
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--checkpoint-every", type=int, default=0)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--eval-checkpoint", type=str, default=None)
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--profile", action="store_true", help="time env/policy/train per iter and exit")
    return p


def main(argv: list[str] | None = None) -> None:
    cfg = build_parser().parse_args(argv)
    if cfg.profile:
        profile_run(cfg)
        return
    if cfg.eval_only:
        ckpt = Path(cfg.eval_checkpoint).resolve() if cfg.eval_checkpoint else None
        if ckpt is None or not ckpt.exists():
            raise SystemExit(f"--eval-only needs an existing --eval-checkpoint (got {cfg.eval_checkpoint})")
        evaluate(cfg, ckpt)
        return
    final_ckpt = train(cfg)
    if cfg.no_eval:
        return
    # Eval in a FRESH process: the boxoban env caches its level pool in process-global state, so the
    # held-out pool (a different bin via BOXOBAN_MAP_BIN) only loads cleanly in a new process.
    cmd = [sys.executable, str(SOURCE_PATH), "--eval-only", "--eval-checkpoint", str(final_ckpt),
           "--run", cfg.run, "--difficulty", str(cfg.difficulty), "--eval-split", cfg.eval_split,
           "--eval-seed", str(cfg.eval_seed), "--eval-episodes", str(cfg.eval_episodes),
           "--target", str(cfg.target), "--holdout-frac", str(cfg.holdout_frac),
           "--num-agents", str(cfg.num_agents), "--max-episode-steps", str(cfg.max_episode_steps),
           "--arch", cfg.arch, "--hidden-size", str(cfg.hidden_size),
           "--num-layers", str(cfg.num_layers), "--bf16", "1" if cfg.bf16 else "0",
           "--output-dir", cfg.output_dir]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
