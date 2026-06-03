from __future__ import annotations

import os
import subprocess
import sys
import threading

import modal


APP_NAME = "nanochat-rl-hf"
VOLUME_NAME = "nanochat-rl-hf"
VOLUME_MOUNT_PATH = "/vol"

# speedrun.py fills one node: it runs the trainer on GPU 0 and spawns vLLM generators on the
# rest (vllm_dp = NODE_GPUS - world_size), so the allocation must match speedrun.NODE_GPUS (8).
GPU_TYPE = os.environ.get("GPU_TYPE", "H100")
NUM_GPUS = int(os.environ.get("NUM_GPUS", "8"))


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
    .env({"OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"})
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


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{NUM_GPUS}",
    timeout=24 * 60 * 60,
    volumes={VOLUME_MOUNT_PATH: volume},
    secrets=runtime_secrets,
)
def train() -> None:
    """Launch the speedrun config as-is: `python -m speedrun` with its own defaults, run from
    the volume so its relative data/output paths resolve under /vol. No arguments are passed."""
    env = dict(os.environ)
    # speedrun maps each trainer rank to a physical GPU itself, so CUDA_VISIBLE_DEVICES must be unset.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["LD_LIBRARY_PATH"] = _nvidia_ld_library_path()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    command = [sys.executable, "-m", "speedrun"]
    print(f"Launching: {' '.join(command)} (cwd={VOLUME_MOUNT_PATH})", flush=True)

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
    finally:
        stop_commit.set()
        volume.commit()


@app.local_entrypoint()
def main() -> None:
    call = train.spawn()
    print(f"Spawned Modal function call: {call.object_id}", flush=True)
    try:
        print(f"Function call dashboard: {call.get_dashboard_url()}", flush=True)
    except Exception:
        pass
