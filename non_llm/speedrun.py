"""Sokoban Speedrun — non-LLM track, using Pufferlib's Boxoban environment.

A from-scratch deep-RL agent that learns to solve Sokoban.

Boxoban obs is a 4x10x10 byte grid (channels: agent, walls, boxes, targets); actions are
{noop,down,up,left,right}. Solved == every box on a target. difficulty 0=basic..4=unfiltered.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np

SOURCE_PATH = Path(__file__).resolve()
REPO_DIR = SOURCE_PATH.parent

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_float32_matmul_precision("high")  # TF32 on the fp32 matmuls (matches PufferLib's torch path)
torch.backends.cudnn.benchmark = True  # autotune conv kernels for our fixed shapes (lossless speedup)


def _amp(enabled: bool, device: str):
    """bf16 autocast for the policy forward/loss (matmuls -> tensor cores; reductions/exp stay fp32).
    PufferLib's fused _C trainer runs the whole PPO in bf16, so the recipe is bf16-robust; this is the
    torch-path equivalent. No GradScaler: bf16 has fp32 range. Off => bit-identical to PufferLib (fp32)."""
    if enabled and str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


ENV_NAME = "boxoban"
DEFAULT_RUN_NAME_PREFIX = "boxoban-non-llm"
PUFFERLIB_REPO_URL = "https://github.com/JeanKaddour/PufferLib.git"
PUFFERLIB_COMMIT = "e90b58edef5445d1a3689e2f908f5841ad6a0f23"
LOCAL_PUFFERLIB_DIR = Path(os.environ.get("PUFFERLIB_SRC_DIR", REPO_DIR / ".pufferlib-src")).expanduser()
DIFFICULTY_NAMES = {0: "basic", 1: "easy", 2: "medium", 3: "hard", 4: "unfiltered"}
OBS_CHANNELS, GRID = 4, 10
GRID_CELLS = GRID * GRID
BOXOBAN_META_BYTES = 5
BOXOBAN_PUZZLE_BYTES = OBS_CHANNELS * GRID_CELLS + BOXOBAN_META_BYTES
NUM_ACTIONS = 5


# ============================ RUN RECIPE (single source of truth) ============================
# PufferLib's tuned boxoban PPO hyperparameters (config/boxoban.ini) + speedrun knobs. These are the
# train-time defaults; argparse exposes them for experimentation without changing the eval contract.
RECIPE = {
    # --- environment ---
    # Default = official DeepMind unfiltered levels with the canonical 1000-level test split.
    "difficulty": 4,            # 0=basic 1=easy 2=medium 3=hard 4=unfiltered
    "num_agents": 8192,
    "max_episode_steps": 120,
    "holdout_frac": 0.1,        # procedural/index-partition holdout knob; canonical eval ignores it
    # --- PPO rollout / optimization ---
    "total_timesteps": 498_073_600,  # 950 iters — record #5's matched-anneal horizon (cosine LR anneals over the run)
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
    # sgpm2-mingru = conv-free shift+pooled-global encoder + recurrent core (the record recipe).
    "arch": "sgpm2-mingru",     # the only arch in-tree; other archs live in records/*/source snapshots
    "hidden_size": 256,
    "num_layers": 3,            # recurrent (planning) depth — deeper generalizes better on official sets
    "enc_dim": 64,              # sgpm2 per-cell channel dim (0 = arch default)
    "bf16": False,              # fp32 by default
    "compile": True,            # torch.compile the policy for train-bound fwd/bwd kernel fusion.
    "compile_mode": "default",  # default | reduce-overhead (CUDA graphs) | max-autotune
    # --- bookkeeping ---
    "seed": 42,
    "wandb": False,             # --wandb 1 to enable; logs train/loss/opt/perf metrics per iter
    "wandb_project": "sokoban-speedrun-non-llm",
}


# ============================ eval aggregates (verbatim from eval_speedrun.py) ===============
# Single source of truth for the record aggregates: verify_record.py imports these so an offline
# re-derivation of pass@1 / pass@k / CI can never drift from what evaluate() actually computed.
# This track always uses percentile bootstrap CIs over per-level solve fractions.
def _pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    if k > n:
        raise ValueError(f"pass@k requires k<=n, got k={k}, n={n}")
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


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
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(REPO_DIR),
            timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _file_sha256(path) -> str | None:
    """sha256 of a file (the held-out level bin), so the record pins the exact eval pool and
    verify_record.py can confirm it offline. Mirrors eval_speedrun._file_sha256."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


class DummyWandb:
    """No-op wandb stand-in (mirrors speedrun.py) so the train loop is wandb-agnostic."""

    def log(self, *a, **k):
        pass

    def finish(self, *a, **k):
        pass


def pufferlib_root() -> Path:
    import pufferlib
    return Path(pufferlib.__file__).resolve().parent.parent


def _default_run_name() -> str:
    return f"{DEFAULT_RUN_NAME_PREFIX}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _clear_pufferlib_modules() -> None:
    for name in list(sys.modules):
        if name == "pufferlib" or name.startswith("pufferlib."):
            del sys.modules[name]


def _pufferlib_extension_env_name() -> str | None:
    from pufferlib import _C
    return getattr(_C, "env_name", None)


def _pufferlib_source_ready(path: Path) -> bool:
    return (path / "build.sh").is_file() and (path / "ocean" / ENV_NAME / "binding.c").is_file()


def _pufferlib_checkout_matches(path: Path) -> bool:
    if not _pufferlib_source_ready(path):
        return False
    out = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return out.returncode == 0 and out.stdout.strip() == PUFFERLIB_COMMIT


def _find_cached_pufferlib_checkout() -> Path | None:
    cache_root = Path(os.environ.get("UV_CACHE_DIR", Path.home() / ".cache" / "uv")) / "git-v0" / "checkouts"
    if not cache_root.exists():
        return None
    for candidate in cache_root.glob("*/*"):
        if _pufferlib_checkout_matches(candidate):
            return candidate
    return None


def _patch_pufferlib_float_build(src: Path) -> None:
    path = src / "src" / "kernels.cu"
    needle = """inline void cast_dispatch(precision_t* dst, const float* src, int n, cudaStream_t stream) {
    cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(dst, src, n);
}

"""
    guard = "#ifndef PRECISION_FLOAT\n" + needle + "#endif\n\n"
    text = path.read_text(encoding="utf-8")
    if guard in text:
        return
    if needle not in text:
        raise SystemExit(f"Could not apply PufferLib float-build patch to {path}")
    print(f"[setup] patching PufferLib float build in {path}", flush=True)
    path.write_text(text.replace(needle, guard, 1), encoding="utf-8")


def _ensure_local_pufferlib_source() -> Path:
    src = LOCAL_PUFFERLIB_DIR.resolve()
    if _pufferlib_checkout_matches(src):
        _patch_pufferlib_float_build(src)
        return src
    if src.exists():
        raise SystemExit(
            f"{src} exists but is not the pinned PufferLib checkout {PUFFERLIB_COMMIT}. "
            "Remove it, or set PUFFERLIB_SRC_DIR to a checkout of the pinned commit."
        )
    cached = _find_cached_pufferlib_checkout()
    if cached is not None:
        print(f"[setup] cloning cached PufferLib checkout to {src}", flush=True)
        subprocess.run(["git", "clone", "--no-hardlinks", str(cached), str(src)], check=True)
    else:
        print(f"[setup] cloning PufferLib {PUFFERLIB_COMMIT[:7]} to {src}", flush=True)
        subprocess.run(["git", "clone", PUFFERLIB_REPO_URL, str(src)], check=True)
    subprocess.run(["git", "-C", str(src), "checkout", "--detach", PUFFERLIB_COMMIT], check=True)
    _patch_pufferlib_float_build(src)
    return src


def _install_editable_pufferlib(src: Path) -> None:
    print(f"[setup] installing editable PufferLib from {src}", flush=True)
    uv = shutil.which("uv")
    if uv is not None:
        cmd = [uv, "pip", "install", "--python", sys.executable, "--no-deps", "-e", str(src)]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(src)]
    subprocess.run(cmd, check=True)
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    _clear_pufferlib_modules()


def _ensure_nvidia_lib_symlink(module_name: str, link_name: str, target_name: str) -> None:
    try:
        mod = __import__(module_name, fromlist=["_"])
        lib_dir = Path(next(iter(mod.__path__))) / "lib"
    except Exception:
        return
    link = lib_dir / link_name
    target = lib_dir / target_name
    if target.exists() and not link.exists():
        link.symlink_to(target.name)


def _build_boxoban_extension(src: Path) -> None:
    print(f"[setup] building PufferLib native extension: bash build.sh {ENV_NAME} --float", flush=True)
    _ensure_nvidia_lib_symlink("nvidia.cudnn", "libcudnn.so", "libcudnn.so.9")
    _ensure_nvidia_lib_symlink("nvidia.nccl", "libnccl.so", "libnccl.so.2")
    env = dict(os.environ)
    if shutil.which("ccache", path=env.get("PATH")) is None:
        with tempfile.TemporaryDirectory(prefix="pufferlib-build-") as td:
            shim = Path(td) / "ccache"
            shim.write_text("#!/bin/sh\nexec \"$@\"\n", encoding="utf-8")
            shim.chmod(0o755)
            env["PATH"] = td + os.pathsep + env.get("PATH", "")
            subprocess.run(["bash", "build.sh", ENV_NAME, "--float"], cwd=str(src), env=env, check=True)
    else:
        subprocess.run(["bash", "build.sh", ENV_NAME, "--float"], cwd=str(src), env=env, check=True)
    _clear_pufferlib_modules()


def ensure_boxoban_extension() -> None:
    """Validate, and locally build if needed, the PufferLib Boxoban native env."""
    try:
        env_name = _pufferlib_extension_env_name()
        if env_name == ENV_NAME and _pufferlib_source_ready(pufferlib_root()):
            return
        reason = f"built for {env_name!r}" if env_name != ENV_NAME else "installed wheel lacks source resources"
    except Exception as exc:
        reason = f"`pufferlib._C` unavailable ({exc})"

    print(f"[setup] PufferLib Boxoban extension needs a local build: {reason}", flush=True)
    src = _ensure_local_pufferlib_source()
    _install_editable_pufferlib(src)
    try:
        env_name = _pufferlib_extension_env_name()
    except Exception:
        _build_boxoban_extension(src)
        try:
            env_name = _pufferlib_extension_env_name()
        except Exception as exc:
            raise SystemExit(
                "PufferLib's native extension was built, but `pufferlib._C` still cannot be loaded. "
                "Check the compiler/CUDA library output above."
            ) from exc
    if env_name != ENV_NAME:
        _build_boxoban_extension(src)
        try:
            env_name = _pufferlib_extension_env_name()
        except Exception as exc:
            raise SystemExit(
                "PufferLib's native extension was rebuilt, but `pufferlib._C` still cannot be loaded. "
                "Check the compiler/CUDA library output above."
            ) from exc
    if env_name != ENV_NAME:
        raise SystemExit(f"PufferLib native extension is built for {env_name!r}, expected {ENV_NAME!r}.")


# ============================ official DeepMind held-out levels ==============================
# The boxoban env's bin format is [agent(100), walls(100), boxes(100), targets(100), meta(5)] per
# 10x10 puzzle. BOXOBAN_MAP_BIN lets eval load the official held-out split directly.
_BOX_CHARS = {"agent": ("@", "+"), "wall": ("#",), "box": ("$", "*"), "target": (".", "*", "+")}


def _encode_boxoban_puzzle(rows: list[str]) -> bytes:
    agent = bytearray(GRID_CELLS)
    walls = bytearray(GRID_CELLS)
    boxes = bytearray(GRID_CELLS)
    targets = bytearray(GRID_CELLS)
    ax = ay = -1
    nb = nt = ot = 0

    for r in range(GRID):
        for c in range(GRID):
            ch = rows[r][c]
            is_a = ch in _BOX_CHARS["agent"]
            is_w = ch in _BOX_CHARS["wall"]
            is_b = ch in _BOX_CHARS["box"]
            is_t = ch in _BOX_CHARS["target"]
            i = r * GRID + c
            agent[i] = is_a
            walls[i] = is_w
            boxes[i] = is_b
            targets[i] = is_t
            if is_a:
                ax, ay = c, r
            nb += is_b
            nt += is_t
            ot += is_b and is_t

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
                            out.write(_encode_boxoban_puzzle(rows))
                            rows = []
                            n += 1
                        continue
                    if not line.strip():
                        continue
                    if len(rows) < GRID:
                        rows.append(line[:GRID])
                    if len(rows) == GRID:
                        out.write(_encode_boxoban_puzzle(rows))
                        rows = []
                        n += 1
    return n


def ensure_holdout_bin(difficulty_name: str | None, split: str) -> Path | None:
    """Resolve the held-out level bin for <difficulty>/<split>. Returns the absolute bin path, or None
    if neither a committed bin nor the raw levels are present (→ index-partition fallback).

    Priority: (1) a SINGLE committed `data/boxoban_<diff>_<split>.bin` shipped in the repo — this is
    what makes the repo self-contained (the leaderboard held-out is ~400KB; no 200MB of raw text). If
    absent, (2) encode it from the raw `resources/.../levels/<diff>/<split>` text (local dev / other
    splits the repo doesn't ship)."""
    if not difficulty_name:
        return None
    committed = REPO_DIR / "data" / f"boxoban_{difficulty_name}_{split}.bin"
    if committed.exists():
        return committed
    levels = pufferlib_root() / "resources" / "boxoban" / "levels" / difficulty_name / split
    if not levels.is_dir() or not any(levels.glob("*.txt")):
        return None
    out = REPO_DIR / "resources" / "boxoban" / f"boxoban_maps_{difficulty_name}_{split}.bin"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        print(f"[eval] encoding official held-out {difficulty_name}/{split} -> {out.name}")
        build_boxoban_bin(levels, out)
    return out


# ============================ action sampling ===============================================
from torch.distributions.utils import logits_to_probs  # noqa: E402


def _log_prob(logits: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    value = value.long().unsqueeze(-1)
    value, log_pmf = torch.broadcast_tensors(value, logits)
    value = value[..., :1]
    return log_pmf.gather(-1, value).squeeze(-1)


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    min_real = torch.finfo(logits.dtype).min
    logits = torch.clamp(logits, min=min_real)
    p_log_p = logits * logits_to_probs(logits)
    return -p_log_p.sum(-1)


def sample_logits(logits: torch.Tensor, action: torch.Tensor | None = None):
    """Discrete (single-head) action sampling. logits: (N, num_actions)."""
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


# ============================ Muon optimizer =================================================
_NS_COEFS = [
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
]


def _zeropower_via_newtonschulz5(G: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
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
    """Muon (Newton-Schulz orthogonalized momentum)."""

    def __init__(self, params, lr=0.0025, weight_decay=0.0, momentum=0.9, eps=1e-8):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "eps": eps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = float(group["lr"])
            wd = group["weight_decay"]
            momentum = group["momentum"]
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


# ============================ VTrace+GAE advantage ===========================================
def compute_advantages(values, rewards, terminals, importance, gamma, gae_lambda, rho_clip, c_clip):
    """VTrace-corrected GAE advantage. Matches the vectorized kernel path selected whenever
    horizon % (16/precision_bytes) == 0, which holds for horizon 64 / float32. Note rho_t scales
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
class ShiftGlobalPoolEncoder(nn.Module):
    """sgpm2: conv-free local 3x3 shift-mix + broadcast global context. One shift layer for local
    perception (pad once, gather the 9 neighbor slices, one dense GEMM), a mean+max-pooled
    squeeze-excite-style global vector concatenated back per cell, a second channel mix, then a
    second pooled-global round appended to the flatten readout. All channel-side GEMMs
    (efficient K) plus pooling kernels — the cheapest way to give every cell board-global
    information without token mixing."""

    def __init__(self, obs_size: int, hidden_size: int, dim: int = 48):
        super().__init__()
        self.mix1 = nn.Linear(9 * OBS_CHANNELS, dim)
        self.glob = nn.Linear(2 * dim, dim)
        self.mix2 = nn.Linear(2 * dim, 64)
        self.glob2 = nn.Linear(2 * 64, 64)
        self.proj = nn.Linear(64 * GRID_CELLS + 64, hidden_size)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                nn.init.zeros_(m.bias)

    @staticmethod
    def _neighbors(h: torch.Tensor) -> torch.Tensor:
        """(N, H, W, C) -> (N, H, W, 9C): 3x3 neighborhoods, zero-padded borders."""
        p = F.pad(h, (0, 0, 1, 1, 1, 1))
        return torch.cat([p[:, dr:dr + GRID, dc:dc + GRID]
                          for dr in range(3) for dc in range(3)], dim=-1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        n = observations.shape[0]
        x = observations.view(n, OBS_CHANNELS, GRID, GRID).float().permute(0, 2, 3, 1)   # (N,H,W,C)
        h = torch.relu(self.mix1(self._neighbors(x)))                                    # (N,H,W,d)
        g = torch.cat([h.mean(dim=(1, 2)), h.amax(dim=(1, 2))], dim=-1)
        g = torch.relu(self.glob(g))                                                     # (N,d)
        g = g[:, None, None, :].expand(-1, GRID, GRID, -1)
        h = torch.relu(self.mix2(torch.cat([h, g], dim=-1)))                             # (N,H,W,64)
        g2 = torch.relu(self.glob2(torch.cat([h.mean(dim=(1, 2)), h.amax(dim=(1, 2))], dim=-1)))
        return torch.relu(self.proj(torch.cat([h.reshape(n, -1), g2], dim=-1)))


def build_policy(cfg: argparse.Namespace, env: "BoxobanVecEnv", device: str) -> nn.Module:
    """Policy factory. Single arch: sgpm2-mingru — the conv-free shift+pooled-global encoder plus
    PufferLib's MinGRU recurrent core (the record recipe). Other archs from earlier records and
    sweeps live in each record's source snapshot. Exposes forward()/forward_eval()/initial_state()
    the matched train step needs."""
    if cfg.arch != "sgpm2-mingru":
        raise SystemExit(f"unknown --arch {cfg.arch!r} (only 'sgpm2-mingru' is in-tree; "
                         "other archs live in records/*/source snapshots)")
    import pufferlib.models as pm
    enc_kwargs = {"dim": int(cfg.enc_dim)} if getattr(cfg, "enc_dim", 0) else {}
    encoder = ShiftGlobalPoolEncoder(env.obs_size, cfg.hidden_size, **enc_kwargs)
    decoder = pm.DefaultDecoder(env.act_sizes, cfg.hidden_size)
    network = pm.MinGRU(cfg.hidden_size, num_layers=int(cfg.num_layers))
    policy = pm.Policy(encoder, decoder, network).to(device)
    if not getattr(cfg, "compile", False):
        return policy
    mode = getattr(cfg, "compile_mode", "default")
    return torch.compile(policy, mode=None if mode in ("", "default") else mode)


def _policy_module(policy: nn.Module) -> nn.Module:
    return getattr(policy, "_orig_mod", policy)


# ============================ PufferLib boxoban vec-env (the dependency) =====================
_TYPESTR = {torch.uint8: "|u1", torch.float32: "<f4"}


class _CudaPtr:
    def __init__(self, ptr: int, shape: tuple[int, ...], dtype: torch.dtype):
        self.__cuda_array_interface__ = {
            "data": (ptr, False),
            "shape": shape,
            "typestr": _TYPESTR[dtype],
            "version": 2,
        }


class BoxobanVecEnv:
    """Thin handle over PufferLib's native boxoban vec-env. Only the environment is borrowed; no
    PufferLib training code runs. obs/reward/terminal buffers are read zero-copy from GPU pointers."""

    def __init__(self, *, difficulty: int, num_agents: int, max_steps: int, seed: int,
                 eval_mode: bool = False, holdout_frac: float = 0.0):
        ensure_boxoban_extension()
        from pufferlib import pufferl, _C
        assert getattr(_C, "env_name", None) == ENV_NAME, (
            f"_C built for {getattr(_C, 'env_name', None)!r}, not {ENV_NAME!r}. "
            "Run `uv run python speedrun.py` from non_llm/ to rebuild the local extension.")
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
        # Forked boxoban kwargs for disjoint index-partition holdouts.
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
            self._vec.gpu_step(a.data_ptr())
            torch.cuda.synchronize()
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


# ============================ rollout buffers ===============================================
def collect_rollout(env: BoxobanVecEnv, policy: nn.Module, horizon: int, device: str, amp: bool = False):
    """Collect `horizon` steps for all agents. Stored rewards[t]/terminals[t]
    are those *received arriving at* obs[t] (i.e. for action[t-1]); the advantage uses rewards[t+1]."""
    num_agents = env.num_agents
    obs_size = env.obs_size
    obs = torch.empty(horizon, num_agents, obs_size, dtype=torch.uint8, device=device)
    act = torch.empty(horizon, num_agents, dtype=torch.long, device=device)
    logp = torch.empty(horizon, num_agents, device=device)
    val = torch.empty(horizon, num_agents, device=device)
    rew = torch.empty(horizon, num_agents, device=device)
    done = torch.empty(horizon, num_agents, device=device)
    state = policy.initial_state(num_agents, device)
    if amp:  # carry recurrent state in bf16 so it matches the autocast activations (lerp etc.)
        state = tuple(s.to(torch.bfloat16) for s in state)
    r = torch.zeros(num_agents, device=device)
    d = torch.zeros(num_agents, device=device)
    for t in range(horizon):
        o = env.obs().to(device, copy=True)
        with torch.no_grad(), _amp(amp, device):
            logits, value, state = policy.forward_eval(o, state)
            action, logprob, _ = sample_logits(logits)
        obs[t] = o
        act[t] = action.reshape(num_agents)
        logp[t] = logprob
        val[t] = value.reshape(num_agents)
        rew[t] = r
        done[t] = d
        reward, terminal = env.step(action.reshape(num_agents))
        r = reward.to(device)
        d = terminal.to(device)
    return obs, act, val, logp, rew, done


# ============================ PPO train step =================================================
def train_step(policy: nn.Module, optimizer: torch.optim.Optimizer, obs, act, val, logp, rew, ter,
               cfg, epoch: int, total_epochs: int) -> dict:
    """One PPO training pass over a rollout: reward clamp, prioritized segment replay, VTrace+GAE
    advantage recomputed per minibatch with refreshed values, clipped policy+value loss, Muon step,
    cosine LR. Buffers are (horizon, agents[, obs]); updates `policy` in place. Returns mean losses."""
    device = val.device
    horizon, total_agents = val.shape
    batch_size = total_agents * horizon
    minibatch_segments = cfg.minibatch_size // horizon

    prio_beta0 = cfg.prio_beta0
    prio_alpha = cfg.prio_alpha
    clip_coef, vf_clip = cfg.clip_coef, cfg.vf_clip_coef
    # Prioritized-replay IS-bias anneal. The `* prio_alpha` factor is intentional: it matches
    # Pufferlib's trainer code, NOT the textbook `beta0 + (1-beta0)*epoch/total`.
    anneal_beta = prio_beta0 + (1 - prio_beta0) * prio_alpha * epoch / total_epochs
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
    logp_t = logp.T.contiguous()
    rew_t = rew.T.contiguous().clamp(-1, 1)
    done_t = ter.T.contiguous()

    num_minibatches = int(cfg.replay_ratio * batch_size / cfg.minibatch_size)
    sums = {k: 0.0 for k in ("policy", "value", "entropy", "approx_kl", "clipfrac", "grad_norm")}
    advantages = torch.zeros_like(val_t)
    for _ in range(num_minibatches):
        advantages = compute_advantages(val_t, rew_t, done_t, ratio_buf, cfg.gamma, cfg.gae_lambda,
                                        cfg.vtrace_rho_clip, cfg.vtrace_c_clip)
        seg_adv = advantages.abs().sum(axis=1)
        prio_weights = torch.nan_to_num(seg_adv ** prio_alpha, 0, 0, 0)
        prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)
        idx = torch.multinomial(prio_probs, minibatch_segments, replacement=True)
        mb_is_weight = (total_agents * prio_probs[idx, None]) ** -anneal_beta

        mb_obs = obs_t[idx]
        mb_actions = act_t[idx]
        mb_logprobs = logp_t[idx]
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
            adv = mb_is_weight * (adv - adv.mean()) / (adv.std() + 1e-8)
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
        self._t1 = None
        run_dir.mkdir(parents=True, exist_ok=True)
        source = SOURCE_PATH.read_text(encoding="utf-8")
        sha = self._save_source_snapshot(run_dir, source)
        self.path = run_dir / f"log_{uuid.uuid4().hex[:8]}.txt"
        self._fh = self.path.open("w", encoding="utf-8")
        self._fh.write(self._header(args_dict, source, sha))
        self._fh.flush()
        print(f"RunLogger: writing record log to {self.path}")

    @staticmethod
    def _save_source_snapshot(run_dir: Path, source: str) -> str:
        sd = run_dir / "source"
        sd.mkdir(parents=True, exist_ok=True)
        name = SOURCE_PATH.name  # track the recipe filename so the snapshot can't drift on rename
        (sd / name).write_text(source, encoding="utf-8")
        sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
        (sd / "manifest.json").write_text(json.dumps(
            {"version": 1, "git_commit": _git_commit(), "source_file": str(SOURCE_PATH),
             "files": {name: {"sha256": sha, "bytes": len(source.encode())}}},
            indent=2) + "\n", encoding="utf-8")
        return sha

    @staticmethod
    def _header(args_dict: dict, source: str, sha: str) -> str:
        lines = ["python: " + " ".join(sys.version.split()), f"torch: {torch.__version__}"]
        try:
            import importlib.metadata
            lines.append(f"pufferlib: {importlib.metadata.version('pufferlib')}")
        except Exception:
            lines.append("pufferlib: unknown")
        lines.append(f"git commit: {_git_commit()}")
        lines.append(f"source snapshot: source/{SOURCE_PATH.name} sha256:{sha}")
        try:
            smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
            lines.append(smi.stdout if smi.returncode == 0 else f"nvidia-smi failed: {smi.stderr}")
        except Exception as exc:
            lines.append(f"nvidia-smi unavailable: {exc!r}")
        lines.append(f"argv: {sys.argv}")
        lines.append(f"args: {args_dict}")
        d = RunLogger._DIVIDER
        return f"{source}\n{d}\n" + "\n".join(lines) + f"\n{d}\n"

    def start_clock(self, anchor):
        self._t0 = anchor if self._t0 is None else self._t0

    def stop_clock(self, anchor):
        self._t1 = anchor if self._t1 is None else self._t1

    def record_time(self):
        if self._t0 is None:
            return 0.0
        return (self._t1 if self._t1 is not None else time.monotonic()) - self._t0

    def _write(self, line):
        if self._fh is not None:
            self._fh.write(line + "\n")
            self._fh.flush()

    def log_step(self, step, num_steps, *, reward_mean, solved_frac, loss, grad_norm):
        rt = self.record_time()
        self._write(f"step:{step + 1}/{num_steps} record_time:{rt:.1f}s step_avg:{rt / (step + 1):.1f}s "
                    f"reward_mean:{max(0.0, reward_mean):.4f} solved_frac:{solved_frac:.4f} "
                    f"loss:{loss:.4f} grad_norm:{grad_norm:.4f}")

    def log_final_checkpoint(self, path):
        self._write(f"final_checkpoint:{path} record_time:{self.record_time():.1f}s")

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# ============================ training ======================================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng_state() -> dict:
    np_state = np.random.get_state()
    state = {
        "torch": torch.get_rng_state().clone(),
        "numpy": (np_state[0], np_state[1].copy(), *np_state[2:]),
    }
    if torch.cuda.is_available():
        state["cuda"] = [s.clone() for s in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def train(cfg: argparse.Namespace) -> Path:
    out_dir = (REPO_DIR / cfg.output_dir / cfg.run).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_boxoban_extension()
    os.chdir(pufferlib_root())  # boxoban C env caches level files under a cwd-relative resources/
    set_seed(cfg.seed)

    def make_train_env() -> BoxobanVecEnv:
        return BoxobanVecEnv(difficulty=cfg.difficulty, num_agents=cfg.num_agents,
                             max_steps=cfg.max_episode_steps, seed=cfg.seed,
                             eval_mode=False, holdout_frac=cfg.holdout_frac)

    env = make_train_env()
    device = env.device
    policy = build_policy(cfg, env, device)
    optimizer = Muon(policy.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
                     momentum=cfg.beta1, eps=cfg.muon_eps)
    num_params = sum(p.numel() for p in policy.parameters())

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

    if cfg.compile:
        import copy
        # Warm up compile/autotune outside the timed run, then restore every mutable training input.
        _orig = _policy_module(policy)
        init_model = copy.deepcopy(_orig.state_dict())
        init_opt = copy.deepcopy(optimizer.state_dict())
        init_rng = _capture_rng_state()
        for _ in range(2):
            warmup_obs, warmup_act, warmup_val, warmup_logp, warmup_rew, warmup_done = collect_rollout(
                env, policy, cfg.rollout_horizon, device, amp=cfg.bf16,
            )
            train_step(
                policy, optimizer,
                warmup_obs, warmup_act, warmup_val, warmup_logp, warmup_rew, warmup_done,
                cfg, 0, total_iters,
            )
        _orig.load_state_dict(init_model)
        optimizer.load_state_dict(init_opt)
        env.close()
        set_seed(cfg.seed)
        env = make_train_env()
        _restore_rng_state(init_rng)
        if env.device != device:
            raise RuntimeError(f"fresh env device changed after compile warmup: {device!r} -> {env.device!r}")

    # Time the full schedule from step 1.
    logger.start_clock(time.monotonic())
    global_step = 0
    for it in range(total_iters):
        obs, act, val, logp, rew, done = collect_rollout(env, policy, cfg.rollout_horizon, device, amp=cfg.bf16)
        global_step += steps_per_iter
        stats = train_step(policy, optimizer, obs, act, val, logp, rew, done, cfg, it, total_iters)
        if it == total_iters - 1:
            logger.stop_clock(time.monotonic())
        elog = env.log()
        solved = float(elog.get("perf", 0.0))
        logger.log_step(it, total_iters, reward_mean=float(elog.get("targets_hit", 0.0)),
                        solved_frac=solved, loss=float(stats["policy"] + cfg.vf_coef * stats["value"]),
                        grad_norm=float(stats["grad_norm"]))
        sps = global_step / max(1e-6, logger.record_time())
        wandb_run.log({
            "train/solved_frac": solved,
            "train/targets_hit": float(elog.get("targets_hit", 0.0)),
            "train/episode_return": float(elog.get("episode_return", 0.0)),
            "loss/policy": stats["policy"], "loss/value": stats["value"], "loss/entropy": stats["entropy"],
            "loss/approx_kl": stats["approx_kl"], "loss/clipfrac": stats["clipfrac"],
            "opt/grad_norm": stats["grad_norm"], "opt/lr": optimizer.param_groups[0]["lr"],
            "perf/sps": sps, "perf/record_time_s": logger.record_time(),
            "epoch": it + 1,
        }, step=global_step)
        if it % max(1, cfg.print_every) == 0 or it == total_iters - 1:
            print(f"  iter {it + 1}/{total_iters} gstep={global_step:,} solved={solved:.3f} "
                  f"ent={stats['entropy']:.3f} kl={stats['approx_kl']:.4f} gnorm={stats['grad_norm']:.2f} "
                  f"SPS={sps:,.0f} t={logger.record_time():.0f}s")
        if cfg.checkpoint_every and (it + 1) % cfg.checkpoint_every == 0:
            torch.save(_policy_module(policy).state_dict(), out_dir / f"step_{global_step:012d}.pt")

    final_ckpt = out_dir / "final.pt"
    torch.save(_policy_module(policy).state_dict(), final_ckpt)
    logger.log_final_checkpoint(final_ckpt)
    logger.close()
    env.close()
    wandb_run.finish()
    print(f"[train] done. final checkpoint: {final_ckpt}")
    return final_ckpt


# ============================ held-out evaluation ===========================================
def evaluate(cfg: argparse.Namespace, checkpoint: Path) -> dict:
    """Run the fixed non-LLM leaderboard eval: greedy policy over unfiltered/test."""
    eval_difficulty = 4
    eval_split = "test"
    eval_seed = 12345
    eval_episodes = 16_384
    eval_num_agents = 512
    eval_max_episode_steps = 120
    target_solve_rate = 0.70

    ensure_boxoban_extension()
    os.chdir(pufferlib_root())
    set_seed(eval_seed)

    eval_name = DIFFICULTY_NAMES[eval_difficulty]
    holdout_bin = ensure_holdout_bin(eval_name, eval_split)
    if holdout_bin is None:
        raise SystemExit(f"canonical eval bin not found for {eval_name}/{eval_split}")

    os.environ["BOXOBAN_MAP_BIN"] = str(holdout_bin)
    split_label = f"{eval_name}/{eval_split}-official"
    env = BoxobanVecEnv(difficulty=-1, num_agents=eval_num_agents,
                        max_steps=eval_max_episode_steps, seed=eval_seed,
                        eval_mode=False, holdout_frac=0.0)
    device = env.device
    policy_args = vars(cfg).copy()
    policy_args["compile"] = True
    policy_cfg = argparse.Namespace(**policy_args)
    policy = build_policy(policy_cfg, env, device)
    _policy_module(policy).load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    policy.eval()
    print(f"[eval] loaded {checkpoint}; scoring >= {eval_episodes} episodes "
          f"(held-out: {split_label}, greedy=True)")

    # Key completed episodes by their initial board and average equally over distinct levels.
    from collections import defaultdict
    init_board = env.obs().to(device, copy=True).clone()   # each agent's current-episode start board
    state = policy.initial_state(env.num_agents, device)
    level_solved: dict = defaultdict(int)
    level_counts: dict = defaultdict(int)
    total_eps = 0
    # Record eval is greedy over the canonical held-out bin; verifier rejects incomplete coverage.
    eps_since_new = 0
    while total_eps < eval_episodes:
        with torch.no_grad():
            logits, _, state = policy.forward_eval(env.obs().to(device, copy=True), state)
            action = logits.argmax(-1)
        rew, term = env.step(action)
        done = term.to(device) > 0.5
        solved = done & (rew.to(device) > 0.5)
        idx = done.nonzero(as_tuple=True)[0]
        if idx.numel():
            before = len(level_counts)
            for b, s in zip(init_board[idx].cpu().numpy(), solved[idx].cpu().numpy()):
                h = b.tobytes()
                level_counts[h] += 1
                level_solved[h] += int(s)
            ndone = int(idx.numel())
            total_eps += ndone
            eps_since_new = 0 if len(level_counts) > before else eps_since_new + ndone
            for s_ in state:                       # reset recurrent state for finished envs
                s_[:, idx, :] = 0.0
            init_board[idx] = env.obs().to(device, copy=True)[idx]   # new episode's start board
            if (
                len(level_counts)
                and eps_since_new >= max(4096, 4 * len(level_counts))
            ):
                break
    env_perf = float(env.log().get("perf", float("nan")))  # PufferLib's C-aggregated solve mean (num_agents-invariant)
    env.close()

    keys = list(level_counts.keys())
    per_frac = [level_solved[h] / level_counts[h] for h in keys]
    per_n = [level_counts[h] for h in keys]
    per_c = [level_solved[h] for h in keys]
    per_level_sha = [hashlib.sha256(h).hexdigest() for h in keys]   # board identity of each held-out level
    n = len(keys)                                  # distinct held-out levels seen
    pass_at_1 = sum(per_frac) / max(1, n)          # mean per-level solve fraction (each level weight 1)
    ci_low, ci_high = _bootstrap_ci(per_frac, seed=eval_seed)
    se = float(np.std(per_frac) / math.sqrt(max(1, n))) if n else 0.0
    # Pin the exact eval pool for offline verification.
    holdout_n_levels = (holdout_bin.stat().st_size // BOXOBAN_PUZZLE_BYTES) if holdout_bin else None
    if holdout_n_levels and n < holdout_n_levels:
        print(f"[eval] WARNING: scored {n}/{holdout_n_levels} held-out levels — pool not fully covered")
    record = {
        "seed": eval_seed, "run": cfg.run, "step": None, "checkpoint": str(checkpoint),
        "model": f"ppo-{cfg.arch}-h{cfg.hidden_size}-L{cfg.num_layers}", "eval_data": f"boxoban:{split_label}",
        "holdout_bin": holdout_bin.name if holdout_bin else None,
        "holdout_bin_sha256": _file_sha256(holdout_bin) if holdout_bin else None,
        "holdout_n_levels": holdout_n_levels,
        "git_commit": _git_commit(), "n_puzzles": n, "k": int(np.median(per_n)) if n else 1,
        "pass_at_1": pass_at_1, "pass_at_k": {"1": pass_at_1},
        "solve_rate_episode": env_perf,    # PufferLib env.log()['perf'] cross-check (per-episode mean)
        "eval_num_agents": int(eval_num_agents),
        "ci_low": ci_low, "ci_high": ci_high, "se": se,
        "n_extract_fail": 0, "n_answered": int(sum(per_n)), "n_length_trunc": 0,
        "answer_rate": 1.0, "solve_given_answer": pass_at_1, "trunc_frac": 0.0,
        "sampling": {"temperature": None, "top_p": None, "top_k": None, "min_p": 0.0,
                     "max_tokens": eval_max_episode_steps, "seed": eval_seed,
                     "greedy": True, "episodes": int(sum(per_n)), "logprobs": 0, "interrupt": None},
        "per_puzzle_solve_frac": per_frac, "per_puzzle_n": per_n,
        "per_puzzle_solved_count": per_c, "per_puzzle_answered_count": per_n,
        "per_puzzle_length_trunc_count": [0] * n,
        "per_puzzle_level_sha": per_level_sha,
    }
    out_dir = (REPO_DIR / cfg.output_dir / cfg.run).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_step000000_seed{eval_seed}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    cleared = "CLEARS" if ci_low > target_solve_rate else "DOES NOT CLEAR"
    print(f"[eval] per-level pass@1={pass_at_1:.4f}  95% CI [{ci_low:.4f}, {ci_high:.4f}]  "
          f"({n} levels, {int(sum(per_n))} episodes @ {eval_num_agents} agents, ~{sum(per_n)/max(1,n):.0f}/level)  "
          f"vs target {target_solve_rate}: {cleared}")
    print(f"[eval] PufferLib env.log perf (per-episode solve, cross-check) = {env_perf:.4f}")
    print(f"[eval] wrote {out_path}")
    return record


# ============================ CLI ===========================================================
def _parse_bool(value: str) -> bool:
    return str(value).lower() not in ("0", "false", "no")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sokoban Speedrun — non-LLM (PufferLib boxoban + in-file PPO)")
    p.add_argument("--run", type=str, default=_default_run_name())
    p.add_argument("--output-dir", type=str, default="outputs")
    for key, val in RECIPE.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(val, bool):
            p.add_argument(flag, default=val, type=_parse_bool)
        else:
            p.add_argument(flag, default=val, type=type(val))
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--checkpoint-every", type=int, default=0)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--eval-checkpoint", type=str, default=None)
    p.add_argument("--no-eval", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    cfg = build_parser().parse_args(argv)
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
           "--run", cfg.run, "--arch", cfg.arch, "--hidden-size", str(cfg.hidden_size),
           "--num-layers", str(cfg.num_layers), "--output-dir", cfg.output_dir,
           "--enc-dim", str(cfg.enc_dim)]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
