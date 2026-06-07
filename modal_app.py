from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import threading
from datetime import datetime

import modal


APP_NAME = "nanochat-rl-hf"
VOLUME_NAME = "nanochat-rl-hf"
VOLUME_MOUNT_PATH = "/vol"
PERSISTENT_CACHE_DIR = f"{VOLUME_MOUNT_PATH}/.cache"
PERSISTENT_HF_HOME = f"{PERSISTENT_CACHE_DIR}/huggingface"
PERSISTENT_VLLM_CACHE = f"{PERSISTENT_CACHE_DIR}/vllm"

# speedrun.py fills one node: it runs the trainer on GPU 0 and spawns vLLM generators on the
# rest (vllm_dp = NODE_GPUS - world_size), so the allocation must match speedrun.NODE_GPUS (8).
GPU_TYPE = os.environ.get("GPU_TYPE", "H100")
NUM_GPUS = int(os.environ.get("NUM_GPUS", "8"))

# One-hour target recipe. Keep the ScaleRL-ish optimizer hparams, but use a benchmark-defined
# non-trivial 1-box train/eval split and avoid debug/measurement I/O that slows the sprint.
RECIPE = [
    "--dtype", "float32",            # MANDATORY: speedrun guards bf16 (untying lm_head desyncs vLLM on Qwen3's tied embeds)
    "--train-data", "datasets/sokoban_70_easy_train.jsonl",
    "--eval-data", "datasets/sokoban_70_easy_eval.jsonl",
    "--examples-per-step", "16",     # batch 128: faster feedback than batch 256; previous stable run's batch size.
    "--num-samples", "8",            # k=8: halves advantage noise vs run1's k=4; batch=128
    "--temperature", "0.8",          # lower-entropy rollouts improved solve|answer EWMA in the baseline runs
    "--top-p", "0.95",               # less aggressive than the old 0.7 truncation, but avoids fully open-tail samples
    "--max-new-tokens", "5632",      # fixed think budget before answer forcing; 5632+512 matches the 6144 eval budget
    "--max-model-len", "7168",       # covers max train prompt(~590)+5632+marker+512 answer continuation
    # fp32 head + params/AdamW/grad ~64GB is fixed; the fp32 logits chunk (chunk x vocab x batch) is the
    # batch-scaled cost. device-batch-size 4 + logits-chunk 2048 OOM'd; batch 2 + chunk 2048 fits in probes.
    "--device-batch-size", "2",      # fp32 trainer fits at batch 2 (OOM'd at 4)
    "--logits-chunk-size", "2048",
    "--gradient-checkpointing",
    "--interruption",                # ScaleRL A.10: force an answer instead of hard-truncating to 0 reward (keeps TRAIN trunc low)
    "--interrupt-answer-tokens", "512",
    "--learning-rate", "8e-7",       # fixed LR, close to ScaleRL 8B->4B adaptation (~7e-7)
    "--init-lr-frac", "0.05",        # warmup starts at 5% of peak
    "--warmup-steps", "4",           # peak by step 3, then constant LR
    "--lr-schedule", "constant",
    "--grad-clip", "0.5",            # tighter than default 1.0 (run1 showed 1.0 is no backstop); extra damper on the unguarded gnorm
    "--cispo-eps", "4.0",            # ScaleRL-faithful CISPO upper IS cap; tighten only if clip/gnorm rises
    "--max-staleness", "4",          # less off-policy than ScaleRL's 8 (run2-proven stability lever)
    # Measured sweet spot (non-monotonic): inflight 96 peaks generator throughput (~22k tok/s, collect ~33s).
    # 32 starves the backlog (collect ~96s); 160 over-saturates + balloons the weight-sync pause.
    "--inflight-requests", "96",
    "--vllm-gpu-mem-util", "0.9",
    "--eval-top-p", "0.95",
    "--save-every", "0",
    "--save-final",
    "--no-save-rollouts",
    "--wandb-rollout-samples", "0",
]


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _dotenv_values(path: str) -> dict[str, str]:
    """Minimal .env parser (`KEY=VALUE` / `export KEY=VALUE`), read locally at launch time."""
    values: dict[str, str] = {}
    if not os.path.exists(path):
        return values
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, val = line.split("=", 1)
            values[key.strip()] = val.strip().strip('"').strip("'")
    return values


# Forward ONLY what the training container needs — HF for fast model downloads, W&B for logging.
# Read the local environment first, then fall back to the gitignored .env. (Deliberately excludes
# the MODAL_* tokens in .env: the container doesn't need them.)
_FORWARD_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY", "WANDB_ENTITY")
_dotenv = _dotenv_values(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
_forward = {k: (os.environ.get(k) or _dotenv.get(k)) for k in _FORWARD_KEYS}
_forward = {k: v for k, v in _forward.items() if v}
runtime_secrets = [modal.Secret.from_dict(_forward)] if _forward else []

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(frozen=True)
    .env({
        "OMP_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HOME": PERSISTENT_HF_HOME,
        "HF_HUB_CACHE": f"{PERSISTENT_HF_HOME}/hub",
        "TRANSFORMERS_CACHE": f"{PERSISTENT_HF_HOME}/hub",
        "HF_DATASETS_CACHE": f"{PERSISTENT_HF_HOME}/datasets",
        "VLLM_CACHE_ROOT": PERSISTENT_VLLM_CACHE,
        "XDG_CACHE_HOME": PERSISTENT_CACHE_DIR,
    })
    .add_local_python_source("speedrun")
)


def _nvidia_ld_library_path() -> str:
    """vLLM 0.22 links cu13 runtime libs that ship as pip wheels but aren't on the default
    loader path. Prepend the installed nvidia/*/lib dirs so `import vllm` works in the launched
    process (and its spawned vLLM workers)."""
    import glob
    import sysconfig

    libs = glob.glob(os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "*", "lib"))
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    return ":".join(libs + ([existing] if existing else []))


def _persistent_cache_env() -> dict[str, str]:
    """Cache heavyweight downloads on the mounted Modal volume across ephemeral containers."""
    return {
        "HF_HOME": PERSISTENT_HF_HOME,
        "HF_HUB_CACHE": f"{PERSISTENT_HF_HOME}/hub",
        "TRANSFORMERS_CACHE": f"{PERSISTENT_HF_HOME}/hub",
        "HF_DATASETS_CACHE": f"{PERSISTENT_HF_HOME}/datasets",
        "VLLM_CACHE_ROOT": PERSISTENT_VLLM_CACHE,
        "XDG_CACHE_HOME": PERSISTENT_CACHE_DIR,
    }


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{NUM_GPUS}",
    timeout=24 * 60 * 60,
    volumes={VOLUME_MOUNT_PATH: volume},
    secrets=runtime_secrets,
)
def train(
    run_name: str,
    max_steps: int,
    num_trainers: int,
    extra_args: list[str] | None = None,
    final_eval_k: int = 0,
    final_eval_limit: int = 0,
    final_eval_seeds: str = "12345",
) -> None:
    """Launch the speedrun sprint recipe from the volume (relative data/output paths resolve under
    /vol). `run_name` (non-"dummy") turns on W&B logging; `max_steps` bounds the run. `num_trainers`
    is the data-parallel trainer count: 1 => single process (1 trainer + 7 vLLM); N>1 => torchrun
    with N trainer ranks (+ NODE_GPUS-N vLLM generators). speedrun derives the vLLM split from
    WORLD_SIZE, so no GPU-split args are passed."""
    env = dict(os.environ)
    # speedrun maps each trainer rank to a physical GPU itself, so CUDA_VISIBLE_DEVICES must be unset.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(_persistent_cache_env())
    env["LD_LIBRARY_PATH"] = _nvidia_ld_library_path()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    speedrun_args = ["--run", run_name, "--max-steps", str(max_steps), *RECIPE, *(extra_args or [])]
    if num_trainers > 1:
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   f"--nproc_per_node={num_trainers}", "-m", "speedrun", *speedrun_args]
    else:
        command = [sys.executable, "-m", "speedrun", *speedrun_args]
    print(f"Launching ({num_trainers} trainer(s)): {' '.join(command)} (cwd={VOLUME_MOUNT_PATH})", flush=True)

    # Commit the volume every 60s (and once on exit) so checkpoints/rollouts are downloadable
    # mid-run and survive a crash, not just at the end.
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
        subprocess.run(command, check=True, env=env, cwd=VOLUME_MOUNT_PATH)
        if final_eval_k > 0:
            checkpoint = f"outputs/{run_name}/step_{max_steps - 1:06d}"
            eval_run_name = f"{run_name}-final-eval"
            for eval_command in eval_commands(
                checkpoint,
                eval_run_name,
                final_eval_k,
                final_eval_limit,
                _parse_seeds(final_eval_seeds),
            ):
                print(f"Final checkpoint eval: {' '.join(eval_command)} (cwd={VOLUME_MOUNT_PATH})", flush=True)
                subprocess.run(eval_command, check=True, env=env, cwd=VOLUME_MOUNT_PATH)
                volume.commit()
    finally:
        stop_commit.set()
        volume.commit()


def _parse_seeds(spec: str) -> list[int]:
    """Comma-separated eval-seed list, e.g. "1,2,3,4,5" (single-checkpoint mode only; the record
    protocol replicates over training runs via comma-separated EVAL_CHECKPOINT, with one pinned
    eval seed). Duplicates are rejected: rerunning one seed would just overwrite its JSON."""
    tokens = [tok for tok in spec.replace(" ", "").split(",") if tok]
    if not tokens:
        raise ValueError(f"EVAL_SEEDS must be a comma-separated list of ints, got {spec!r}")
    try:
        seeds = [int(tok) for tok in tokens]
    except ValueError:
        raise ValueError(f"EVAL_SEEDS must be a comma-separated list of ints, got {spec!r}") from None
    if len(set(seeds)) < len(seeds):
        raise ValueError(f"duplicate seeds in EVAL_SEEDS: {seeds}")
    return seeds


def _recipe_arg_value(flag: str, default: str | None = None) -> str | None:
    try:
        return RECIPE[RECIPE.index(flag) + 1]
    except (ValueError, IndexError):
        return default


def eval_commands(checkpoint: str, run_name: str, k: int, eval_limit: int, seeds: list[int],
                  target: float = 0.70) -> list[list[str]]:
    """speedrun --eval-only command(s). `checkpoint` may be a comma-separated list of final
    checkpoints (one per training seed): that emits ONE record-eval command that evaluates all
    of them under the pinned protocol at a single eval seed and ends with the built-in
    significance verdict (mean pass@1 > `target` at p<0.01). A single checkpoint keeps the
    legacy behavior: one command per eval seed, each with an EXPLICIT --eval-seed and a
    seed-distinct --eval-output. (Without these, every run silently sampled at speedrun.py's
    default seed 12345 and successive runs overwrote one seed-agnostic JSON — a 5-seed record
    claim would have had ~zero seed-to-seed variance and a meaningless p-value.)"""
    if len(set(seeds)) < len(seeds):
        raise ValueError(f"duplicate seeds: {seeds}")
    checkpoints = [c for c in checkpoint.replace(" ", "").split(",") if c]
    if not checkpoints:
        raise ValueError(f"EVAL_CHECKPOINT must name at least one checkpoint, got {checkpoint!r}")
    eval_data = _recipe_arg_value("--eval-data", "datasets/sokoban_eval.jsonl")
    eval_max_tokens = _recipe_arg_value("--inloop-eval-max-tokens", "6144")
    # Standalone eval has its own engine; keep enough room for prompt + the full 6144-token
    # measurement budget. Eval deliberately does not use interruption-based answer forcing.
    eval_max_model_len = "8192"
    eval_top_p = _recipe_arg_value("--eval-top-p", _recipe_arg_value("--top-p", "0.95")) or "0.95"
    common = [sys.executable, "-m", "speedrun", "--eval-only",
              "--run", run_name,
              "--eval-data", eval_data,
              "--eval-k", str(k), "--eval-max-tokens", eval_max_tokens,
              "--eval-max-model-len", eval_max_model_len, "--eval-vllm-dp", str(NUM_GPUS),
              "--eval-top-p", eval_top_p]
    if eval_limit > 0:
        common += ["--eval-limit", str(eval_limit)]
    if len(checkpoints) > 1:
        # Record eval: all checkpoints in one invocation (speedrun auto-names one JSON per
        # checkpoint and prints the PASS/FAIL verdict; exit code 0/1 mirrors it).
        return [common + ["--eval-checkpoint", *checkpoints,
                          "--eval-seed", str(seeds[0]), "--eval-target", str(target)]]
    # Mirror speedrun.py's own suffix derivation (step from the checkpoint path, else "latest")
    # and its run-name sanitization, so the per-seed files sit next to the default-path ones.
    m = re.search(r"step_?(\d+)", checkpoints[0])
    suffix = f"step{int(m.group(1)):06d}" if m else "latest"
    safe_run = re.sub(r"[^A-Za-z0-9._-]+", "_", run_name)
    return [common + ["--eval-checkpoint", checkpoints[0],
                      "--eval-seed", str(seed),
                      "--eval-output", f"outputs/{safe_run}/eval_{suffix}_seed{seed}.json"]
            for seed in seeds]


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{NUM_GPUS}",
    timeout=6 * 60 * 60,
    volumes={VOLUME_MOUNT_PATH: volume},
    secrets=runtime_secrets,
)
def evaluate(checkpoint: str, run_name: str, k: int, eval_limit: int, seeds: str = "12345",
             target: float = 0.70) -> None:
    """Authoritative held-out eval (speedrun.py --eval-only): own vLLM engine over all GPUs at the
    full 6144-token leaderboard budget. `checkpoint` is a /vol path or an HF id (e.g. Qwen/Qwen3-4B
    for the base) — or a COMMA-SEPARATED list of final checkpoints (one per training seed), which
    runs the whole record eval end-to-end in one call: every checkpoint evaluated under the pinned
    protocol, one JSON each, then the significance verdict (mean pass@1 > `target`, p<0.01).
    For a single checkpoint, `seeds` (comma-separated) runs one eval/JSON per eval seed
    sequentially. `eval_limit`>0 evals only the first N puzzles (cheap dev runs); 0 = full set.
    Writes outputs/<run>/eval_*.json with pass@1/pass@k + bootstrap CI per run."""
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(_persistent_cache_env())
    env["LD_LIBRARY_PATH"] = _nvidia_ld_library_path()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    record_mode = "," in checkpoint
    for command in eval_commands(checkpoint, run_name, k, eval_limit, _parse_seeds(seeds), target):
        print(f"Eval: {' '.join(command)} (cwd={VOLUME_MOUNT_PATH})", flush=True)
        try:
            if record_mode:
                # speedrun exits 0/1 on the PASS/FAIL verdict; don't crash the container on FAIL.
                rc = subprocess.run(command, check=False, env=env, cwd=VOLUME_MOUNT_PATH).returncode
                print(f"Record eval finished with exit code {rc} "
                      f"({'PASS' if rc == 0 else 'FAIL or error — see verdict above'})", flush=True)
            else:
                subprocess.run(command, check=True, env=env, cwd=VOLUME_MOUNT_PATH)
        finally:
            volume.commit()  # commit per eval so partial results survive a crash


@app.local_entrypoint()
def main() -> None:
    # Eval mode: EVAL_CHECKPOINT=<vol path or HF id> modal run --detach modal_app.py
    # Record mode: EVAL_CHECKPOINT="ckptA,ckptB,..." (one final checkpoint per training seed) runs
    # the whole record eval + significance verdict in one call; EVAL_TARGET sets the bar (0.70).
    eval_ckpt = os.environ.get("EVAL_CHECKPOINT")
    if eval_ckpt:
        k = int(os.environ.get("EVAL_K", "8"))
        eval_limit = int(os.environ.get("EVAL_LIMIT", "0"))  # 0 = full eval set; >0 = first N (cheap dev)
        seeds = ",".join(str(s) for s in _parse_seeds(os.environ.get("EVAL_SEEDS", "12345")))
        target = float(os.environ.get("EVAL_TARGET", "0.70"))
        run_name = os.environ.get("RUN_NAME") or f"sokoban-eval-{datetime.now():%Y%m%d-%H%M%S}"
        print(f"Running eval: ckpt={eval_ckpt}, run={run_name}, k={k}, "
              f"limit={eval_limit}, seeds=[{seeds}], target={target}", flush=True)
        evaluate.remote(eval_ckpt, run_name, k, eval_limit, seeds, target)
        return
    # NTRAINERS=1 => single trainer + 7 vLLM; NTRAINERS=2 => 2 trainers + 6 vLLM (torchrun), etc.
    # Default = short throughput probe. Full sprint e.g.: MAX_STEPS=22 modal run --detach modal_app.py
    max_steps = int(os.environ.get("MAX_STEPS", "4"))
    num_trainers = int(os.environ.get("NTRAINERS", "1"))
    extra_args = shlex.split(os.environ.get("EXTRA_ARGS", ""))
    final_eval_k = int(os.environ.get("FINAL_EVAL_K", "0"))
    final_eval_limit = int(os.environ.get("FINAL_EVAL_LIMIT", "0"))
    final_eval_seeds = ",".join(str(s) for s in _parse_seeds(os.environ.get("FINAL_EVAL_SEEDS", "12345")))
    kind = "probe" if max_steps <= 6 else "sprint"
    run_name = os.environ.get("RUN_NAME") or f"sokoban-{kind}-nt{num_trainers}-{datetime.now():%Y%m%d-%H%M%S}"
    call = train.spawn(run_name, max_steps, num_trainers, extra_args,
                       final_eval_k, final_eval_limit, final_eval_seeds)
    print(f"Spawned Modal function call: {call.object_id} "
          f"(run={run_name}, max_steps={max_steps}, trainers={num_trainers}, "
          f"extra_args={extra_args}, final_eval_k={final_eval_k}, "
          f"final_eval_limit={final_eval_limit}, final_eval_seeds=[{final_eval_seeds}])", flush=True)
    try:
        print(f"Function call dashboard: {call.get_dashboard_url()}", flush=True)
    except Exception:
        pass
