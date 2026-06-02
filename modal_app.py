from __future__ import annotations

import os
import subprocess
import sys

import modal


APP_NAME = "nanochat-rl-hf"
VOLUME_NAME = "nanochat-rl-hf"
VOLUME_MOUNT_PATH = "/vol"
DEFAULT_OUTPUT_DIR = f"{VOLUME_MOUNT_PATH}/outputs"
DEFAULT_TRAIN_DATA = f"{VOLUME_MOUNT_PATH}/datasets/sokoban_train.jsonl"
DEFAULT_EVAL_DATA = f"{VOLUME_MOUNT_PATH}/datasets/sokoban_eval.jsonl"

# GPU fleet is configurable per launch via env vars, so we never edit this file
# to change the run size, e.g.:
#   PIPELINE_TOTAL=8 PIPELINE_NTRAINERS=4 GPU_TYPE=H100 modal run --detach modal_app.py -- --spawn --ddp ...
GPU_TYPE = os.environ.get("GPU_TYPE", "H100")
# Async/pipeline mode runs as ONE process: trainer on GPU 0 + vLLM generators on GPUs 1..K.
# Total GPUs must be >= --vllm-dp + 1. Default 2 (1 trainer + 1 vLLM) keeps smoke runs cheap.
PIPELINE_NGPU = int(os.environ.get("PIPELINE_NGPU", "2"))
# Multi-trainer pipeline (train_pipeline_ddp): T data-parallel trainer ranks launched
# under torchrun on GPUs 0..T-1, plus M = PIPELINE_TOTAL - PIPELINE_NTRAINERS vLLM
# generators on GPUs T..T+M-1. Default 4 trainers + 4 generators on an 8-GPU node.
PIPELINE_NTRAINERS = int(os.environ.get("PIPELINE_NTRAINERS", "4"))
PIPELINE_TOTAL = int(os.environ.get("PIPELINE_TOTAL", "8"))


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
runtime_secret_keys = [
    key for key in ("WANDB_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
    if os.environ.get(key)
]
runtime_secrets = (
    [modal.Secret.from_local_environ(runtime_secret_keys)]
    if runtime_secret_keys
    else []
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(frozen=True)
    # .env AFTER the expensive .uv_sync (so changing these run-shape vars doesn't bust the
    # reinstall cache) but BEFORE .add_local_* (Modal requires add_local_* to be the LAST
    # build steps). PIPELINE_* are read INSIDE the container to size torchrun --nproc_per_node
    # and --vllm-dp, so they must match the locally-resolved gpu=H100:{PIPELINE_TOTAL}
    # decorations; without baking them in the container falls back to defaults and
    # mis-allocates GPUs.
    .env({
        "OMP_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PIPELINE_NGPU": str(PIPELINE_NGPU),
        "PIPELINE_NTRAINERS": str(PIPELINE_NTRAINERS),
        "PIPELINE_TOTAL": str(PIPELINE_TOTAL),
    })
    .add_local_python_source("run_rl")
)


def _has_option(args: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in args)


def _training_args(args: list[str]) -> list[str]:
    train_args = list(args)
    if not _has_option(train_args, "--device"):
        train_args.extend(["--device", "cuda"])
    if not _has_option(train_args, "--output-dir"):
        train_args.extend(["--output-dir", DEFAULT_OUTPUT_DIR])
    if not _has_option(train_args, "--train-data"):
        train_args.extend(["--train-data", DEFAULT_TRAIN_DATA])
    if not _has_option(train_args, "--eval-data"):
        train_args.extend(["--eval-data", DEFAULT_EVAL_DATA])
    return train_args


def _pipeline_ddp_command(args: list[str], nproc: int) -> list[str]:
    """Build the torchrun command for the multi-trainer pipeline launch.

    Launches `nproc` data-parallel trainer ranks via torch.distributed.run (so
    WORLD_SIZE == nproc), each running `run_rl` in --pipeline mode. Rank 0 spawns
    the single vLLM child on the remaining GPUs. Kept pure (no side effects) so
    the command shape is unit-testable without a GPU container.
    """
    pipeline_args = list(args)
    if not _has_option(pipeline_args, "--pipeline"):
        pipeline_args = ["--pipeline", *pipeline_args]
    if not _has_option(pipeline_args, "--num-trainer-gpus"):
        pipeline_args.extend(["--num-trainer-gpus", str(nproc)])
    if not _has_option(pipeline_args, "--vllm-dp"):
        pipeline_args.extend(["--vllm-dp", str(PIPELINE_TOTAL - nproc)])
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        "-m",
        "run_rl",
        *_training_args(pipeline_args),
    ]


def _pipeline_command(args: list[str], pipeline_ngpu: int = PIPELINE_NGPU) -> list[str]:
    """Build the single-trainer pipeline command for a Modal allocation.

    `pipeline_ngpu` is the total Modal GPU count: one trainer GPU plus the vLLM
    data-parallel workers. If the caller did not pass --vllm-dp explicitly, make
    run_rl match the allocation instead of falling back to its larger parser default.
    """
    if pipeline_ngpu < 2:
        raise ValueError("single-pipeline launch needs at least 2 GPUs (1 trainer + >=1 vLLM)")
    pipeline_args = list(args)
    if not _has_option(pipeline_args, "--pipeline"):
        pipeline_args = ["--pipeline", *pipeline_args]
    if not _has_option(pipeline_args, "--vllm-dp"):
        pipeline_args.extend(["--vllm-dp", str(pipeline_ngpu - 1)])
    return [sys.executable, "-m", "run_rl", *_training_args(pipeline_args)]


def _nvidia_ld_library_path() -> str:
    """vLLM 0.22 links cu13 runtime libs that ship as pip wheels but aren't on the
    default loader path. Prepend the installed nvidia/*/lib dirs so `import vllm`
    works in the launched process (and its spawned vLLM workers)."""
    import glob
    import sysconfig

    libs = glob.glob(os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "*", "lib"))
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    return ":".join(libs + ([existing] if existing else []))


def _run_pipeline_subprocess(command: list[str], label: str) -> None:
    """Launch a pipeline subprocess with the cu13 loader path + allocator-fragmentation
    env, committing the volume every 60s (and once on exit) so rollouts.jsonl / metrics
    are downloadable mid-run and survive a crash, not just at the end."""
    import threading

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = _nvidia_ld_library_path()
    # Reduce allocator fragmentation on the memory-tight fp32 trainer GPU(s);
    # setdefault so it reaches every torchrun-spawned rank.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"Launching {label}:", " ".join(command), flush=True)
    print("LD_LIBRARY_PATH prefix:", env["LD_LIBRARY_PATH"][:160], flush=True)

    stop_commit = threading.Event()

    def _periodic_commit():
        while not stop_commit.wait(60):
            try:
                volume.commit()
            except Exception:
                pass

    committer = threading.Thread(target=_periodic_commit, daemon=True)
    committer.start()
    try:
        subprocess.run(command, check=True, env=env)
    finally:
        stop_commit.set()
        volume.commit()


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{PIPELINE_NGPU}",
    timeout=24 * 60 * 60,
    volumes={VOLUME_MOUNT_PATH: volume},
    secrets=runtime_secrets,
)
def train_pipeline(*arglist: str) -> None:
    """Async/pipeline RL: trainer on GPU 0; run_pipeline spawns a vLLM child process
    that runs the generators on GPUs 1..K. No torchrun (single trainer rank)."""
    args = list(arglist)
    if args and args[0] == "--":
        args = args[1:]
    _run_pipeline_subprocess(_pipeline_command(args), "pipeline")


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{PIPELINE_TOTAL}",
    timeout=24 * 60 * 60,
    volumes={VOLUME_MOUNT_PATH: volume},
    secrets=runtime_secrets,
)
def train_pipeline_ddp(*arglist: str) -> None:
    """Multi-trainer async/pipeline RL: PIPELINE_NTRAINERS data-parallel trainer
    ranks under torchrun on GPUs 0..T-1, plus the vLLM child on GPUs T..T+M-1
    (M = PIPELINE_TOTAL - PIPELINE_NTRAINERS). Rank 0 owns the vLLM child + mp
    queues; ranks 1..T-1 are pure trainers."""
    args = list(arglist)
    if args and args[0] == "--":
        args = args[1:]
    _run_pipeline_subprocess(_pipeline_ddp_command(args, PIPELINE_NTRAINERS), "pipeline (DDP)")


@app.local_entrypoint()
def main(*arglist: str) -> None:
    args = list(arglist)
    if args and args[0] == "--":
        args = args[1:]
    should_spawn = False
    if "--spawn" in args:
        args.remove("--spawn")
        should_spawn = True
    ddp_pipeline = False
    if "--ddp" in args:
        args.remove("--ddp")  # modal-routing flag only; not passed to run_rl
        ddp_pipeline = True
    # Async/pipeline is the only training mode: --ddp routes to the multi-trainer DDP
    # pipeline (torchrun on T trainer GPUs), otherwise the single-trainer pipeline.
    train_fn = train_pipeline_ddp if ddp_pipeline else train_pipeline
    if should_spawn:
        call = train_fn.spawn(*args)
        print(f"Spawned Modal function call: {call.object_id}", flush=True)
        try:
            print(f"Function call dashboard: {call.get_dashboard_url()}", flush=True)
        except Exception:
            pass
        return
    train_fn.remote(*args)
