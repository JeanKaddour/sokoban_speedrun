from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import threading
from contextlib import contextmanager
from datetime import datetime

import modal


APP_NAME = "nanochat-rl-hf"
VOLUME_NAME = "nanochat-rl-hf"
VOLUME_MOUNT_PATH = "/vol"
PERSISTENT_CACHE_DIR = f"{VOLUME_MOUNT_PATH}/.cache"
PERSISTENT_HF_HOME = f"{PERSISTENT_CACHE_DIR}/huggingface"
PERSISTENT_VLLM_CACHE = f"{PERSISTENT_CACHE_DIR}/vllm"
# Container-LOCAL on purpose: inductor writes thousands of small artifact files during codegen,
# and on the FUSE-mounted volume that made a 3-rank --compile cold start take >25 min vs the
# expected 2-5 min on local NVMe. Cold compile per run is the cheaper trade;
# revisit cross-run warm starts with a tar-to-volume scheme if --compile sticks.
LOCAL_INDUCTOR_CACHE = "/tmp/torchinductor"
# Container-LOCAL for the same reason: triton JITs hundreds of small files during vLLM's cold
# torch.compile. The durable cross-run artifact is vLLM's PACKED compile cache under
# VLLM_CACHE_ROOT, which lives on the volume — a warm start loads that and skips inductor/triton
# entirely, so persisting their scratch caches would only buy FUSE risk.
LOCAL_TRITON_CACHE = "/tmp/triton"

# rank-0 runs the decode/trim/score pipeline (thread pool) plus 6-7 vLLM engine processes share
# the container; without an explicit reservation Modal's default CPU allocation can starve them.
CPU_REQUEST = float(os.environ.get("CPU_REQUEST", "16"))

# speedrun.py fills one node: it runs trainer ranks on the first GPUs and spawns vLLM generators
# on the rest (vllm_dp = NODE_GPUS - world_size), so the allocation must match speedrun.NODE_GPUS.
# NUM_GPUS is resolved from the launch shell, then baked into the image env as NODE_GPUS (see
# `image` below): the container re-imports this module with no launch env, so without the bake the
# remote eval_commands/--eval-vllm-dp and speedrun's trainer/generator split would silently
# mismatch the box. Default 8 = the 4B node: NTRAINERS defaults to 3 (3 trainers + 5 vLLM
# generators) — the confirmed sweet spot (generation keeps the async buffer full at gen-wait ~3s
# while the trainer is compute-bound at ~35s/step; nt4 drops to 4 generators and goes gen-bound).
# A smaller NUM_GPUS (e.g. 4) just changes the split and defaults NTRAINERS to 1.
def _strict_h100_type(raw: str) -> str:
    # Modal's plain H100 request can auto-upgrade to H200; H100! opts out for benchmark runs.
    gpu_type = raw.strip()
    if gpu_type == "H100":
        return "H100!"
    if gpu_type != "H100!":
        raise ValueError(
            "LLM record runs must use Modal gpu='H100!' so they cannot be auto-upgraded. "
            f"Got GPU_TYPE={raw!r}."
        )
    return gpu_type


GPU_TYPE = _strict_h100_type(os.environ.get("GPU_TYPE", "H100!"))
NUM_GPUS = int(os.environ.get("NUM_GPUS", os.environ.get("NODE_GPUS", "8")))
GPU_CONFIG = f"{GPU_TYPE}:{NUM_GPUS}"

# One-hour target recipe. Keep the ScaleRL-ish optimizer hparams, but use a benchmark-defined
# non-trivial 1-box train/eval split and avoid debug/measurement I/O that slows the sprint.
# The run recipe now lives in speedrun.py as the top-level RECIPE constant, which speedrun.main()
# prepends to the CLI args. The launcher below only passes per-run args (--run / EXTRA_ARGS, plus
# an optional MAX_STEPS override). Held-out eval defaults live in eval_speedrun.py.


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
        # Launch-time GPU count, baked in so the container agrees with the gpu= allocation:
        # speedrun reads NODE_GPUS for its trainer/generator split, and this module's remote
        # re-import resolves NUM_GPUS from it (the launch shell's NUM_GPUS never reaches Modal).
        "NODE_GPUS": str(NUM_GPUS),
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
        "HF_XET_CACHE": f"{PERSISTENT_HF_HOME}/xet",
        "TORCH_HOME": f"{PERSISTENT_CACHE_DIR}/torch",
        "VLLM_CACHE_ROOT": PERSISTENT_VLLM_CACHE,
        # flashinfer JIT-compiled ops persist under <base>/.cache/flashinfer/<version>/<arch>/
        # (cubins ship in the flashinfer-cubin wheel; this only holds the JIT'd ops). Default base
        # is $HOME, i.e. container-local — without this, every container re-JITs them.
        "FLASHINFER_WORKSPACE_BASE": PERSISTENT_CACHE_DIR,
        # Local NVMe, NOT the volume — see LOCAL_INDUCTOR_CACHE / LOCAL_TRITON_CACHE for the
        # measured FUSE pathology; vLLM's packed compile cache (on the volume) is the durable copy.
        "TORCHINDUCTOR_CACHE_DIR": LOCAL_INDUCTOR_CACHE,
        "TRITON_CACHE_DIR": LOCAL_TRITON_CACHE,
        "XDG_CACHE_HOME": PERSISTENT_CACHE_DIR,
    }


def _ensure_persistent_cache_dirs() -> None:
    """Create Modal-volume cache dirs before libraries try to populate them."""
    for path in {
        PERSISTENT_CACHE_DIR,
        PERSISTENT_HF_HOME,
        f"{PERSISTENT_HF_HOME}/hub",
        f"{PERSISTENT_HF_HOME}/datasets",
        f"{PERSISTENT_HF_HOME}/xet",
        f"{PERSISTENT_CACHE_DIR}/torch",
        PERSISTENT_VLLM_CACHE,
        LOCAL_INDUCTOR_CACHE,
        LOCAL_TRITON_CACHE,
    }:
        os.makedirs(path, exist_ok=True)


def _last_arg_value(args: list[str], flag: str, default: str | None = None) -> str | None:
    value = default
    for i, arg in enumerate(args[:-1]):
        if arg == flag:
            value = args[i + 1]
    return value


def _last_bool_flag(args: list[str], enabled_flag: str, disabled_flag: str, default: bool) -> bool:
    value = default
    for arg in args:
        if arg == enabled_flag:
            value = True
        elif arg == disabled_flag:
            value = False
    return value


def _safe_run_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return safe or "run"


def _latest_checkpoint_for_run(run_name: str) -> str:
    run_dir = os.path.join(VOLUME_MOUNT_PATH, "outputs", _safe_run_name(run_name))
    try:
        names = os.listdir(run_dir)
    except FileNotFoundError:
        raise FileNotFoundError(f"no output directory found for run {run_name!r}: {run_dir}") from None

    candidates: list[tuple[int, str]] = []
    for name in names:
        match = re.fullmatch(r"step_(\d+)", name)
        if match and os.path.isdir(os.path.join(run_dir, name)):
            candidates.append((int(match.group(1)), name))
    if not candidates:
        raise FileNotFoundError(f"no step_* checkpoints found for run {run_name!r} in {run_dir}")
    _, checkpoint_name = max(candidates)
    return f"outputs/{_safe_run_name(run_name)}/{checkpoint_name}"


def _latest_checkpoint() -> str:
    outputs_dir = os.path.join(VOLUME_MOUNT_PATH, "outputs")
    try:
        run_names = os.listdir(outputs_dir)
    except FileNotFoundError:
        raise FileNotFoundError(f"no outputs directory found: {outputs_dir}") from None

    candidates: list[tuple[float, int, str, str]] = []
    for run_name in run_names:
        run_dir = os.path.join(outputs_dir, run_name)
        if not os.path.isdir(run_dir):
            continue
        for checkpoint_name in os.listdir(run_dir):
            match = re.fullmatch(r"step_(\d+)", checkpoint_name)
            checkpoint_dir = os.path.join(run_dir, checkpoint_name)
            if match and os.path.isdir(checkpoint_dir):
                candidates.append((
                    os.path.getmtime(checkpoint_dir),
                    int(match.group(1)),
                    run_name,
                    checkpoint_name,
                ))
    if not candidates:
        raise FileNotFoundError(f"no step_* checkpoints found under {outputs_dir}")
    _, _, run_name, checkpoint_name = max(candidates)
    return f"outputs/{run_name}/{checkpoint_name}"


def _needs_periodic_volume_commit(speedrun_args: list[str]) -> bool:
    """True when the run streams mid-run artifacts and wants the tight 60s commit cadence."""
    # The rollout STREAM is on by default (speedrun.py: --save-rollouts + --rollout-flush-every 10;
    # batched ~6MB gzip appends, not the old per-group small writes): without periodic commits a
    # hard container death would discard everything since the last commit, defeating the stream's
    # keep-logs-on-premature-stop purpose. Defaults here must mirror speedrun.py's argparse defaults.
    save_every = int(_last_arg_value(speedrun_args, "--save-every", "0") or "0")
    save_rollouts = _last_bool_flag(speedrun_args, "--save-rollouts", "--no-save-rollouts", True)
    flush_every = int(_last_arg_value(speedrun_args, "--rollout-flush-every", "10") or "0")
    return save_every > 0 or (save_rollouts and flush_every > 0)


@contextmanager
def _periodic_volume_commits(interval_s: float):
    """Commit the volume every `interval_s` while the body runs, plus once on exit. Cache writes
    (HF model download, vLLM packed compile cache, flashinfer JIT ops) land on the volume as plain
    file writes and are LOST on a hard container death unless committed — a crash on the first run
    with a new model would re-pay the whole download+compile tax on the next launch. Commits are
    cheap; the interval only bounds contention with training/generation I/O on the FUSE mount."""
    stop = threading.Event()

    def _loop():
        while not stop.wait(interval_s):
            try:
                volume.commit()
            except Exception:
                pass

    committer = threading.Thread(target=_loop, daemon=True)
    committer.start()
    try:
        yield
    finally:
        stop.set()
        committer.join(timeout=5)
        volume.commit()


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    cpu=CPU_REQUEST,
    timeout=24 * 60 * 60,
    volumes={VOLUME_MOUNT_PATH: volume},
    secrets=runtime_secrets,
)
def train(
    run_name: str,
    max_steps: int | None,
    num_trainers: int,
    extra_args: list[str] | None = None,
    final_eval_k: int = 0,
    final_eval_limit: int = 0,
    final_eval_seeds: str = "12345",
) -> None:
    """Launch the speedrun sprint recipe from the volume (relative data/output paths resolve under
    /vol). `run_name` labels the artifacts; `max_steps` optionally overrides the recipe.
    `num_trainers` is the data-parallel trainer count: 1 => single process (1 trainer +
    NUM_GPUS-1 vLLM); N>1 => torchrun with N trainer ranks (+ NUM_GPUS-N vLLM generators).
    speedrun derives the vLLM split from WORLD_SIZE and the baked-in NODE_GPUS, so no GPU-split
    args are passed."""
    env = dict(os.environ)
    # speedrun maps each trainer rank to a physical GPU itself, so CUDA_VISIBLE_DEVICES must be unset.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    _ensure_persistent_cache_dirs()
    env.update(_persistent_cache_env())
    print(
        f"Persistent caches: HF_HOME={env['HF_HOME']} "
        f"HF_HUB_CACHE={env['HF_HUB_CACHE']} VLLM_CACHE_ROOT={env['VLLM_CACHE_ROOT']}",
        flush=True,
    )
    env["LD_LIBRARY_PATH"] = _nvidia_ld_library_path()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # wandb's run dir must NOT live on the FUSE volume (cwd): wandb-core's many small appends to
    # the .wandb transaction log wedge on FUSE and the run uploads nothing (a 7-byte .wandb even
    # after tens of minutes). Container-local disk keeps the sender fed; the server +
    # RunLogger hold the durable record, so losing the local dir with the container is fine.
    env.setdefault("WANDB_DIR", "/tmp/wandb")
    os.makedirs("/tmp/wandb", exist_ok=True)

    speedrun_args = ["--run", run_name]
    if max_steps is not None:
        speedrun_args += ["--max-steps", str(max_steps)]
    speedrun_args += [*(extra_args or [])]
    if num_trainers > 1:
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   f"--nproc_per_node={num_trainers}", "-m", "speedrun", *speedrun_args]
    else:
        command = [sys.executable, "-m", "speedrun", *speedrun_args]
    print(f"Launching ({num_trainers} trainer(s)): {' '.join(command)} (cwd={VOLUME_MOUNT_PATH})", flush=True)

    # Tight 60s commits when checkpoints/rollouts stream mid-run (a hard container death should
    # keep the logs); otherwise a relaxed 300s cadence that exists purely to checkpoint cache
    # writes (model download, vLLM compile cache) without contending with training I/O.
    commit_interval = 60 if _needs_periodic_volume_commit(speedrun_args) else 300
    with _periodic_volume_commits(commit_interval):
        subprocess.run(command, check=True, env=env, cwd=VOLUME_MOUNT_PATH)
        if final_eval_k > 0:
            checkpoint = _latest_checkpoint_for_run(run_name)
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


def _parse_seeds(spec: str) -> list[int]:
    """Comma-separated eval-seed list, e.g. "1,2,3,4,5".

    Single-checkpoint mode runs one JSON per seed. Comma-separated checkpoint mode uses the
    first seed and evaluates each checkpoint once under the shared protocol.
    """
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


def eval_commands(checkpoint: str, run_name: str, k: int, eval_limit: int, seeds: list[int],
                  eval_max_tokens: int = 12288, eval_max_model_len: int = 16384,
                  interruption: bool = True, interrupt_answer_tokens: int = 512,
                  eval_gpu_mem_util: float = 0.9, eval_vllm_max_num_batched_tokens: int = 40960,
                  eval_vllm_max_num_seqs: int = 32, eval_concurrency: int = 32,
                  eval_data: str | None = None) -> list[list[str]]:
    """eval_speedrun command(s). `checkpoint` may be a comma-separated list of final
    checkpoints (one per training seed): that emits one command evaluating them sequentially
    under the same protocol at a single eval seed, with one JSON per checkpoint. A single
    checkpoint keeps the per-seed behavior: one command per eval seed, each with an explicit
    --eval-seed and a seed-distinct --eval-output."""
    if len(set(seeds)) < len(seeds):
        raise ValueError(f"duplicate seeds: {seeds}")
    checkpoints = [c for c in checkpoint.replace(" ", "").split(",") if c]
    if not checkpoints:
        raise ValueError(f"EVAL_CHECKPOINT must name at least one checkpoint, got {checkpoint!r}")
    # Defaults are the leaderboard protocol: a generous 12288-token budget + interruption
    # answer-forcing, which eliminates the budget-truncation confound and measures solving.
    # A strict 6144/no-interruption protocol remains available via EVAL_MAX_TOKENS /
    # EVAL_MAX_MODEL_LEN / EVAL_INTERRUPTION for diagnostics; --eval-max-model-len must cover
    # prompt + max_tokens + the forced-answer continuation.
    common = [sys.executable, "-m", "eval_speedrun",
              "--run", run_name,
              "--eval-k", str(k), "--eval-max-tokens", str(eval_max_tokens),
              "--eval-max-model-len", str(eval_max_model_len), "--eval-vllm-dp", str(NUM_GPUS),
              "--eval-gpu-mem-util", str(eval_gpu_mem_util),
              "--eval-vllm-max-num-batched-tokens", str(eval_vllm_max_num_batched_tokens),
              "--eval-vllm-max-num-seqs", str(eval_vllm_max_num_seqs),
              "--eval-concurrency", str(eval_concurrency)]
    if eval_data:
        # Leave unset for the default committed eval set. Set only to evaluate on a different set
        # (e.g. a harder transfer slice for ablations).
        common += ["--eval-data", eval_data]
    if interruption:
        common += ["--eval-interruption", "--eval-interrupt-answer-tokens", str(interrupt_answer_tokens)]
    else:
        common += ["--no-eval-interruption"]
    if eval_limit > 0:
        common += ["--eval-limit", str(eval_limit)]
    if len(checkpoints) > 1:
        # Record-style eval: all checkpoints in one invocation. eval_speedrun auto-names one JSON
        # per checkpoint and does not emit an in-command verdict.
        return [common + ["--eval-checkpoint", *checkpoints, "--eval-seed", str(seeds[0])]]
    # Mirror eval_speedrun.py's own suffix derivation (step from the checkpoint path, else "latest")
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
    gpu=GPU_CONFIG,
    cpu=CPU_REQUEST,
    timeout=6 * 60 * 60,
    volumes={VOLUME_MOUNT_PATH: volume},
    secrets=runtime_secrets,
)
def evaluate(checkpoint: str, run_name: str, k: int, eval_limit: int, seeds: str = "12345",
             eval_max_tokens: int = 12288, eval_max_model_len: int = 16384,
             interruption: bool = True, eval_gpu_mem_util: float = 0.9,
             eval_vllm_max_num_batched_tokens: int = 40960,
             eval_vllm_max_num_seqs: int = 32, eval_concurrency: int = 32,
             eval_data: str | None = None) -> None:
    """Authoritative held-out eval (eval_speedrun.py): own vLLM engine over all GPUs at the
    full 12288-token leaderboard budget. `checkpoint` is "latest", a /vol path, an HF id (e.g. the
    base model), or a comma-separated list of final checkpoints (one per training seed), which runs
    sequential independent evals in one call: every checkpoint evaluated under the same protocol,
    one JSON each. For a single checkpoint, `seeds` (comma-separated) runs one eval/JSON per eval
    seed sequentially. `eval_limit`>0 evals only the first N puzzles (cheap dev runs); 0 = full set.
    Writes outputs/<run>/eval_*.json with pass@1/pass@k + bootstrap CI per run."""
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    _ensure_persistent_cache_dirs()
    env.update(_persistent_cache_env())
    print(
        f"Persistent caches: HF_HOME={env['HF_HOME']} "
        f"HF_HUB_CACHE={env['HF_HUB_CACHE']} VLLM_CACHE_ROOT={env['VLLM_CACHE_ROOT']}",
        flush=True,
    )
    env["LD_LIBRARY_PATH"] = _nvidia_ld_library_path()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if checkpoint.lower() == "latest":
        checkpoint = _latest_checkpoint()
        print(f"Resolved latest checkpoint to {checkpoint}", flush=True)
    # 300s cadence so a mid-eval crash keeps a fresh model download / vLLM compile cache (the
    # per-command commit below only fires AFTER a full eval, ~30-60 min after those cache writes).
    with _periodic_volume_commits(300):
        for command in eval_commands(checkpoint, run_name, k, eval_limit, _parse_seeds(seeds),
                                     eval_max_tokens, eval_max_model_len, interruption,
                                     eval_gpu_mem_util=eval_gpu_mem_util,
                                     eval_vllm_max_num_batched_tokens=eval_vllm_max_num_batched_tokens,
                                     eval_vllm_max_num_seqs=eval_vllm_max_num_seqs,
                                     eval_concurrency=eval_concurrency,
                                     eval_data=eval_data):
            print(f"Eval: {' '.join(command)} (cwd={VOLUME_MOUNT_PATH})", flush=True)
            try:
                subprocess.run(command, check=True, env=env, cwd=VOLUME_MOUNT_PATH)
            finally:
                volume.commit()  # commit per eval so partial results survive a crash


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    cpu=CPU_REQUEST,
    timeout=6 * 60 * 60,
    volumes={VOLUME_MOUNT_PATH: volume},
    secrets=runtime_secrets,
)
def probe_datasets(eval_datasets: list[str], run_name: str, k: int, eval_limit: int,
                   max_tokens: int, max_model_len: int, interruption: bool,
                   model: str = "Qwen/Qwen3-4B-Instruct-2507") -> None:
    """Base-model pass@k probe (P0) across a difficulty ladder. Runs eval_speedrun against
    the BASE model on each eval dataset (relative /vol path) and writes one JSON per set to
    outputs/<run>/probe_<stem>.json. Used to locate the productive (0,1) reward-variance band from
    the per_puzzle_solved_count distribution BEFORE committing a training run to a dataset. Mirrors
    the TRAINING rollout regime (5632 think budget + interruption answer-forcing, temp 0.8) so the
    saturation/all-fail readout predicts the live group_pass@k, not the leaderboard budget."""
    import os.path as osp
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    _ensure_persistent_cache_dirs()
    env.update(_persistent_cache_env())
    print(
        f"Persistent caches: HF_HOME={env['HF_HOME']} HF_HUB_CACHE={env['HF_HUB_CACHE']} "
        f"VLLM_CACHE_ROOT={env['VLLM_CACHE_ROOT']}",
        flush=True,
    )
    env["LD_LIBRARY_PATH"] = _nvidia_ld_library_path()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with _periodic_volume_commits(300):  # keep cache writes through a mid-probe crash
        for ds in eval_datasets:
            stem = osp.splitext(osp.basename(ds))[0]
            out = f"outputs/{run_name}/probe_{stem}.json"
            command = [sys.executable, "-m", "eval_speedrun",
                       "--run", run_name,
                       "--eval-checkpoint", model,
                       "--eval-data", ds,
                       "--eval-k", str(k),
                       "--eval-temperature", "0.8", "--eval-top-p", "0.95", "--eval-seed", "12345",
                       "--eval-max-tokens", str(max_tokens),
                       "--eval-max-model-len", str(max_model_len),
                       "--eval-vllm-dp", str(NUM_GPUS),
                       "--eval-output", out]
            if eval_limit > 0:
                command += ["--eval-limit", str(eval_limit)]
            if interruption:
                command += ["--eval-interruption", "--eval-interrupt-answer-tokens", "512"]
            else:
                command += ["--no-eval-interruption"]
            print(f"Probe: {' '.join(command)} (cwd={VOLUME_MOUNT_PATH})", flush=True)
            try:
                subprocess.run(command, check=True, env=env, cwd=VOLUME_MOUNT_PATH)
            finally:
                volume.commit()  # commit per dataset so partial results survive a crash


@app.local_entrypoint()
def main() -> None:
    # Probe mode: PROBE_DATASETS="datasets/a_eval.jsonl,datasets/b_eval.jsonl" uv run modal run modal_app.py
    #   (blocking .remote(); no --detach needed). Locates the (0,1) reward-variance band on the base
    #   model. RUN_NAME pins the output dir; pull with `modal volume get nanochat-rl-hf /outputs/<run>`.
    probe_spec = os.environ.get("PROBE_DATASETS")
    if probe_spec:
        eval_datasets = [d.strip() for d in probe_spec.split(",") if d.strip()]
        k = int(os.environ.get("EVAL_K", "8"))
        eval_limit = int(os.environ.get("EVAL_LIMIT", "100"))
        max_tokens = int(os.environ.get("PROBE_MAX_TOKENS", "5632"))
        max_model_len = int(os.environ.get("PROBE_MAX_MODEL_LEN", "7168"))
        interruption = os.environ.get("PROBE_INTERRUPTION", "1") not in ("0", "false", "False", "")
        model = os.environ.get("PROBE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
        run_name = os.environ.get("RUN_NAME") or f"sokoban-probe-{datetime.now():%Y%m%d-%H%M%S}"
        print(f"Running base-model probe: sets={eval_datasets} run={run_name} k={k} "
              f"limit={eval_limit} max_tokens={max_tokens} interruption={interruption}", flush=True)
        probe_datasets.remote(eval_datasets, run_name, k, eval_limit, max_tokens, max_model_len,
                              interruption, model)
        print(f"Probe complete. Pull results: "
              f"modal volume get nanochat-rl-hf /outputs/{run_name} ./reports/probe_modal --force", flush=True)
        return
    # Eval mode: EVAL_CHECKPOINT=latest (or <vol path or HF id>) uv run modal run --detach modal_app.py
    # Record-style mode: EVAL_CHECKPOINT="ckptA,ckptB,..." (one final checkpoint per training seed)
    # evaluates each checkpoint sequentially and writes one independent eval JSON per checkpoint.
    eval_ckpt = os.environ.get("EVAL_CHECKPOINT")
    if eval_ckpt:
        # Leaderboard eval protocol: k=8 on the fixed eval set, 12,288-token budget +
        # interruption, seed 12345. EVAL_K / EVAL_DATA / EVAL_LIMIT override for ablations.
        k = int(os.environ.get("EVAL_K", "8"))
        eval_data = os.environ.get("EVAL_DATA")  # None => eval_speedrun's default committed eval set
        eval_limit = int(os.environ.get("EVAL_LIMIT", "0"))  # 0 = full eval set; >0 = first N (cheap dev)
        seeds = ",".join(str(s) for s in _parse_seeds(os.environ.get("EVAL_SEEDS", "12345")))
        eval_max_tokens = int(os.environ.get("EVAL_MAX_TOKENS", "12288"))
        eval_max_model_len = int(os.environ.get("EVAL_MAX_MODEL_LEN", "16384"))
        eval_interruption = os.environ.get("EVAL_INTERRUPTION", "1") not in ("0", "false", "False", "")
        eval_gpu_mem_util = float(os.environ.get("EVAL_GPU_MEM_UTIL", "0.9"))
        eval_vllm_max_num_batched_tokens = int(os.environ.get("EVAL_VLLM_MAX_NUM_BATCHED_TOKENS", "40960"))
        eval_vllm_max_num_seqs = int(os.environ.get("EVAL_VLLM_MAX_NUM_SEQS", "32"))
        eval_concurrency = int(os.environ.get("EVAL_CONCURRENCY", "32"))
        run_name = os.environ.get("RUN_NAME") or f"sokoban-eval-{datetime.now():%Y%m%d-%H%M%S}"
        print(f"Running eval: ckpt={eval_ckpt}, run={run_name}, k={k}, limit={eval_limit}, "
              f"data={eval_data or 'default'}, seeds=[{seeds}], "
              f"max_tokens={eval_max_tokens}, max_model_len={eval_max_model_len}, "
              f"interruption={eval_interruption}, gpu_mem_util={eval_gpu_mem_util}, "
              f"batched_tokens={eval_vllm_max_num_batched_tokens}, "
              f"max_num_seqs={eval_vllm_max_num_seqs}, concurrency={eval_concurrency}", flush=True)
        evaluate.remote(eval_ckpt, run_name, k, eval_limit, seeds,
                        eval_max_tokens, eval_max_model_len, eval_interruption,
                        eval_gpu_mem_util, eval_vllm_max_num_batched_tokens,
                        eval_vllm_max_num_seqs, eval_concurrency, eval_data)
        return
    # NTRAINERS=1 => single trainer + NUM_GPUS-1 vLLM; NTRAINERS=2 => 2 trainers + the rest (torchrun), etc.
    # Default NTRAINERS=3 on the 8-GPU benchmark node: it is trainer-bound at inflight 96, so 3
    # trainers is the sweet spot (~30% faster per step than 2 — fewer microbatches per rank). On
    # smaller boxes default to 1 trainer (3 trainers + 1 generator would starve generation).
    # Trainer count is a torchrun launch param (not a speedrun CLI arg), so it lives here; the rest
    # of the recipe is speedrun.py RECIPE.
    # Bare launch runs speedrun.py's RECIPE. Set MAX_STEPS only for probes, smoke tests, or longer runs.
    max_steps_env = os.environ.get("MAX_STEPS")
    max_steps = int(max_steps_env) if max_steps_env else None
    num_trainers = int(os.environ.get("NTRAINERS", "3" if NUM_GPUS >= 8 else "1"))
    extra_args = shlex.split(os.environ.get("EXTRA_ARGS", ""))
    # MODEL overrides the RECIPE/argparse base model (default Qwen/Qwen3-4B-Instruct-2507). Prepended
    # so an explicit --model in EXTRA_ARGS still wins (argparse last-wins).
    model = os.environ.get("MODEL")
    if model:
        extra_args = ["--model", model, *extra_args]
    final_eval_k = int(os.environ.get("FINAL_EVAL_K", "0"))
    final_eval_limit = int(os.environ.get("FINAL_EVAL_LIMIT", "0"))
    final_eval_seeds = ",".join(str(s) for s in _parse_seeds(os.environ.get("FINAL_EVAL_SEEDS", "12345")))
    kind = "probe" if max_steps is not None and max_steps <= 6 else "speedrun"
    run_name = os.environ.get("RUN_NAME") or f"sokoban-{kind}-{datetime.now():%Y%m%d-%H%M%S}"
    call = train.spawn(run_name, max_steps, num_trainers, extra_args,
                       final_eval_k, final_eval_limit, final_eval_seeds)
    max_steps_desc = str(max_steps) if max_steps is not None else "RECIPE"
    print(f"Spawned Modal function call: {call.object_id} "
          f"(run={run_name}, max_steps={max_steps_desc}, trainers={num_trainers}, "
          f"extra_args={extra_args}, final_eval_k={final_eval_k}, "
          f"final_eval_limit={final_eval_limit}, final_eval_seeds=[{final_eval_seeds}])", flush=True)
    try:
        print(f"Function call dashboard: {call.get_dashboard_url()}", flush=True)
    except Exception:
        pass
