"""Modal entrypoint for the non-LLM Sokoban speedrun track (PufferLib boxoban + in-file PuffeRL PPO).

Sibling of `modal_app.py` (which drives the LLM `speedrun.py`) and `modal_verl_app.py`. The non-LLM
track is a different stack, so it gets its own image: a CUDA-devel base with clang + a cu126 torch,
into which the `boxoban` env extension is built (`--float`) at image
build time. Records run on a single H100 (Boxoban PPO is GPU-light; a full run is minutes).

Usage (mirrors modal_app.py's env-var local entrypoint):

    # train to target + final held-out eval, artifacts -> volume /vol/outputs/<run>/
    modal run --detach modal_app_non_llm.py

    # eval an existing checkpoint only
    EVAL_CHECKPOINT=/vol/outputs/<run>/final.pt modal run modal_app_non_llm.py
"""

from __future__ import annotations

import os
import subprocess
import sys

import modal

APP_NAME = "sokoban-non-llm"
VOLUME_NAME = "nanochat-rl-hf"          # share the LLM track's volume so records sit side by side
VOLUME_MOUNT_PATH = "/vol"
GPU_TYPE = os.environ.get("GPU_TYPE", "H100")
NUM_GPUS = int(os.environ.get("NUM_GPUS", "1"))
# H100 = sm_90. The image builds on a CPU builder (no GPU), so build.sh's default `-arch=native`
# can't probe — pin the arch explicitly (nvcc wants the full `sm_90`, not bare `90`).
NVCC_ARCH = os.environ.get("NVCC_ARCH", "sm_90")
PUFFERLIB_IMG_DIR = "/root/pufferlib"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

_FORWARD_KEYS = ("WANDB_API_KEY", "WANDB_ENTITY")
_forward = {k: os.environ[k] for k in _FORWARD_KEYS if os.environ.get(k)}
runtime_secrets = [modal.Secret.from_dict(_forward)] if _forward else []

# Build the native boxoban extension into the fork at image-build time: create the libcudnn.so /
# libnccl.so dev symlinks the pip nvidia wheels omit, then `build.sh boxoban --float`.
_PYLIB = "import nvidia.{m}, os; print(os.path.join(nvidia.{m}.__path__[0], 'lib'))"
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.2-devel-ubuntu22.04", add_python="3.12")
    .apt_install("clang", "libomp-dev", "git", "build-essential", "curl", "ca-certificates", "unzip")
    .pip_install("torch==2.9.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("numpy", "rich", "rich_argparse", "pybind11", "scikit-learn", "wandb")
    .env({"NVCC_ARCH": NVCC_ARCH, "PUFFERLIB_DIR": PUFFERLIB_IMG_DIR, "OMP_NUM_THREADS": "1"})
    # Upload only the source needed to BUILD the env: skip .git, the ~200MB resources/ map cache
    # (regenerated/downloaded at runtime), build artifacts, the raylib download (re-fetched by
    # build.sh), and any local .so (rebuilt for the H100's sm_90 in run_commands below).
    .add_local_dir("third_party/pufferlib", PUFFERLIB_IMG_DIR, copy=True,
                   ignore=[".git", "resources", "build", "raylib-*", "*.so", "__pycache__", "*.pyc"])
    .run_commands(
        f"ln -sf libcudnn.so.9 $(python -c \"{_PYLIB.format(m='cudnn')}\")/libcudnn.so",
        f"ln -sf libnccl.so.2 $(python -c \"{_PYLIB.format(m='nccl')}\")/libnccl.so",
        f"cd {PUFFERLIB_IMG_DIR} && bash build.sh boxoban --float",
    )
    # Bake the small official held-out splits (valid/test ~18MB) so the eval gates on the real
    # DeepMind set; the big train splits are excluded (the env downloads them at runtime), as are the
    # procedural basic/easy tiers.
    .add_local_dir("third_party/pufferlib/resources/boxoban/levels",
                   f"{PUFFERLIB_IMG_DIR}/resources/boxoban/levels", copy=True,
                   ignore=["train", "basic", "easy"])
    .add_local_python_source("speedrun_non_llm")
)


def _runtime_env() -> dict[str, str]:
    """LD_LIBRARY_PATH must include the pip nvidia wheel lib dirs (cudnn/nccl) so the built _C.so and
    torch resolve their CUDA libs in the launched process."""
    import glob
    import sysconfig

    env = dict(os.environ)
    libs = glob.glob(os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "*", "lib"))
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(libs + ([existing] if existing else []))
    env["PUFFERLIB_DIR"] = PUFFERLIB_IMG_DIR
    return env


def _run(extra_args: list[str]) -> None:
    cmd = [sys.executable, "-m", "speedrun_non_llm", "--output-dir", f"{VOLUME_MOUNT_PATH}/outputs",
           *extra_args]
    print(f"Launching: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=_runtime_env())
    volume.commit()


@app.function(image=image, gpu=f"{GPU_TYPE}:{NUM_GPUS}", timeout=24 * 60 * 60,
              volumes={VOLUME_MOUNT_PATH: volume}, secrets=runtime_secrets)
def train(run_name: str | None, difficulty: int | None, total_timesteps: int | None, target: float | None,
          holdout_frac: float | None, extra_args: list[str] | None = None) -> None:
    args = []
    if run_name is not None:
        args += ["--run", run_name]
    if difficulty is not None:
        args += ["--difficulty", str(difficulty)]
    if total_timesteps is not None:
        args += ["--total-timesteps", str(total_timesteps)]
    if target is not None:
        args += ["--target", str(target)]
    if holdout_frac is not None:
        args += ["--holdout-frac", str(holdout_frac)]
    args += [*(extra_args or [])]
    _run(args)


@app.function(image=image, gpu=f"{GPU_TYPE}:{NUM_GPUS}", timeout=6 * 60 * 60,
              volumes={VOLUME_MOUNT_PATH: volume}, secrets=runtime_secrets)
def evaluate(checkpoint: str, run_name: str | None, difficulty: int | None, eval_episodes: int | None,
             target: float | None, holdout_frac: float | None) -> None:
    args = ["--eval-only", "--eval-checkpoint", checkpoint]
    if run_name is not None:
        args += ["--run", run_name]
    if difficulty is not None:
        args += ["--difficulty", str(difficulty)]
    if eval_episodes is not None:
        args += ["--eval-episodes", str(eval_episodes)]
    if target is not None:
        args += ["--target", str(target)]
    if holdout_frac is not None:
        args += ["--holdout-frac", str(holdout_frac)]
    _run(args)


@app.function(image=image, gpu=f"{GPU_TYPE}:{NUM_GPUS}", timeout=30 * 60, volumes={VOLUME_MOUNT_PATH: volume})
def profile(extra_args: list[str]) -> None:
    for bf in ("0", "1"):                                  # fp32(TF32) vs bf16, same container/GPU
        print(f"\n================ PROFILE bf16={bf} ================", flush=True)
        _run(["--profile", "--bf16", bf, *extra_args])


@app.function(image=image, gpu=f"{GPU_TYPE}:{NUM_GPUS}", timeout=30 * 60, volumes={VOLUME_MOUNT_PATH: volume})
def smoke(run_name: str | None, total_timesteps: int, extra_args: list[str]) -> None:
    # short train (no eval) to measure real-loop SPS on the GPU
    args = ["--no-eval", "--total-timesteps", str(total_timesteps), *extra_args]
    if run_name is not None:
        args = ["--run", run_name, *args]
    _run(args)


@app.local_entrypoint()
def main() -> None:
    run_name = os.environ.get("RUN_NAME")
    difficulty = int(os.environ["DIFFICULTY"]) if os.environ.get("DIFFICULTY") else None
    holdout_frac = float(os.environ["HOLDOUT_FRAC"]) if os.environ.get("HOLDOUT_FRAC") else None
    target = float(os.environ["TARGET"]) if os.environ.get("TARGET") else None
    extra = os.environ.get("EXTRA_ARGS", "").split() or None

    # PROFILE/SMOKE: measure H100 speed. Default to the winning config on difficulty 0 (basic = procedural,
    # no big download), 8192 agents — directly comparable to the 3090's 172k env-steps/s.
    cfg_args = ["--difficulty", os.environ.get("DIFFICULTY", "0"),
                "--arch", os.environ.get("ARCH", "cnn-mingru"),
                "--num-layers", os.environ.get("NUM_LAYERS", "3"),
                "--hidden-size", os.environ.get("HIDDEN_SIZE", "256"),
                "--num-agents", os.environ.get("NUM_AGENTS", "8192")]
    if os.environ.get("PROFILE"):
        profile.remote(cfg_args)
        return
    if os.environ.get("SMOKE"):
        smoke.remote(run_name, int(os.environ.get("SMOKE_STEPS", "60000000")), cfg_args)
        return

    eval_ckpt = os.environ.get("EVAL_CHECKPOINT")
    if eval_ckpt:
        evaluate.remote(eval_ckpt, run_name, difficulty,
                        int(os.environ["EVAL_EPISODES"]) if os.environ.get("EVAL_EPISODES") else None,
                        target, holdout_frac)
        return

    total = int(os.environ["TOTAL_TIMESTEPS"]) if os.environ.get("TOTAL_TIMESTEPS") else None
    train.remote(run_name, difficulty, total, target, holdout_frac, extra)
