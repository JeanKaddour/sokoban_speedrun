# Sokoban Speedrun

Goal: the fastest recipe to RL [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) from 14% to 50% pass@1 solve-rate on Sokoban puzzles, using a single 8xH100 node.

Sokoban: If you don't know it, the best way to familiarize yourself with it is [to play it here](https://www.jeankaddour.com/sokoban).

Motivation: A lot of LLM RL papers don't reproduce and there are many high-variance moving parts in their pipelines. This is an attempt to standardize things.

# World record history

| # | Record time | Description | Date | Log | pass@1 | Contributors |
| - | - | - | - | - | - | - |
| 1 | 1:45:39 | [ScaleRL](https://arxiv.org/abs/2510.13786)-like: CISPO, LR 8e-7 annealed to 1.2e-7 | 2026-06-11 | [records/2026-06-11_01](records/2026-06-11_01_cispo/) | 0.558 (3 seeds, p=0.006 vs 0.50) | @JeanKaddour |

Each record links its full training log (script source, environment attestation, per-step record clock, final-checkpoint stamp); the eval JSON and verification-run logs live in the same [`records/`](records/) directory.

# Rules

- **TARGET**: pass@1 > **0.50** on the held-out eval set [`datasets/sokoban_eval.jsonl`](datasets/sokoban_eval.jsonl) (384 puzzles). The submission must clear it via lower 95% bootstrap CI.
- **Eval protocol**: sample k=16 completions per puzzle to estimate per-sample pass@1 (pass@16 is reported separately), with a 12,288-token budget, interruption answer-forcing, temperature 0.8 / top-p 0.95, and eval seed 12345.
- **Record time** = the run log's record clock: it starts at training step 1 and ends when the final checkpoint finishes writing. Startup (model load, engine build) is untimed.
- **Fixed**: the base model (Qwen3-4B), the training set [`datasets/sokoban_train.jsonl`](datasets/sokoban_train.jsonl) (10,000 puzzles, consumed in file order), the eval set.
- **Changeable**: RL algorithm, loss function, schedules, training/inference engine, parallelism, dataset-agnostic auxiliary rewards (e.g. entropy or uncertainty proxies), and the prompt, as long as the changes do not add Sokoban-specific solving knowledge (see FAQ).
- **Hardware**: one 8xH100 node.

## Submitting a record

1. Train with your recipe, then run the eval on the final checkpoint. All artifacts are written automatically: run log, `final_rollouts.jsonl.gz`, eval JSON + eval rollouts.
2. Run `python make_record_report.py records/<your-dir>` to generate the standard plots and the per-record README (artifact-only; add `--wandb-runs` for richer panels), write its `Idea` section, then open a PR adding the directory plus a row in the table.
3. The PR must pass `python verify_record.py records/<your-dir>` (offline re-scoring of every eval completion + health checks); records are then verified by an independent rerun before merging.

# How it works

`speedrun.py` is a self-contained async RL pipeline in one file. It is designed to run on one 8×H100 node, using `torch` and `vllm`.

Two roles share the node:

- **Generator**: one vLLM process (data-parallel over its GPUs) samples completions for each puzzle.
- **Trainers**: data-parallel ranks score the rollouts (solutions verified by ReasoningGym), update the policy, and broadcast fresh weights to the generator over NCCL.

The two run concurrently, so the generator never idles waiting for a step. The benchmark recipe lives at the top of `speedrun.py` as the `RECIPE` constant — any CLI flag overrides it.

## How to run

On an 8xH100-GPU node (3 trainer ranks + 5 vLLM generators):

```bash
torchrun --standalone --nproc_per_node=3 -m speedrun --run myrun --max-steps 100
```

## Modal

If you'd rather not manage a node yourself, `modal_app.py` rents an 8×H100 box on [Modal](https://modal.com) and launches the recipe:

```bash
# one-time: push the fixed datasets to the Modal volume
modal volume put nanochat-rl-hf datasets/sokoban_train.jsonl /datasets/sokoban_train.jsonl
modal volume put nanochat-rl-hf datasets/sokoban_eval.jsonl  /datasets/sokoban_eval.jsonl

# train (use --detach so the run survives the client disconnecting)
MAX_STEPS=100 RUN_NAME=myrun modal run --detach modal_app.py

# evaluate a checkpoint under the leaderboard protocol
EVAL_CHECKPOINT=/vol/outputs/myrun/step_000099 modal run modal_app.py
```

The function runs `python -m speedrun` from the `nanochat-rl-hf` volume (mounted at `/vol`), so its relative paths resolve there: datasets at `/vol/datasets/`, with checkpoints and logs written to `/vol/outputs/`.

## Evals

```bash
EVAL_CHECKPOINT=/vol/outputs/<run>/step_NNNNNN modal run modal_app.py
```

# FAQ

## Why Sokoban?

* Small contamination risk: We generate fresh puzzles, so unlike eg. GSM8k there is less risk the base model has memorized them.
* Simple enough to yield RL gains in `O(10)` GPU hours.
* Hard enough for gains to be meaningful: Sokoban is PSPACE-complete and can't be brute-forced; it requires genuine reasoning capabilities.
* Naturally disincentivizes entropy collapse, as puzzles naturally permit multiple solutions.

## Why Qwen3-4B?

- Its base pass@1 is 14.0%.
- It got released around the same time as [ReasoningGym](https://arxiv.org/abs/2505.24760) (which includes Sokoban), so it is unlikely to have been trained on it.
- In my (Jean's) experience, 4B seems just large enough for the model to elicit meaningful CoT traces, which are necessary for RL.  


## Why is the training set ordered?

- The ordering follows a curriculum: difficulty ramps continuously from short 2-box puzzles to longer-solution and 2–3-box ones, paced so a ~100-step run stays in the high-signal regime. 
- The exact schedule lives in [`datasets/generate_sokoban_datasets.py`](datasets/generate_sokoban_datasets.py).
- Every recipe should train on the same puzzle sequence, so records differ by algorithm and systems, not data shuffling.
- Dynamic filtering is part of the algorithm and remains fair game, e.g. the zero-variance filter in DAPO.  

## Why handroll an async RL stack instead of using verl/slime/TRL/etc.?

- Since wall-clock time to a target solve-rate is the metric, the RL stack itself is part of the optimization target.
- General-purpose RL frameworks optimize for broad coverage; here we want a narrow, inspectable fast path.
- This follows the spirit of [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt): strip the stack down until bottlenecks are visible, then make the fastest version easy to reproduce and improve.

## What prompt changes are allowed?

- The point of the benchmark is generalizable reasoning, not overfitting to Sokoban.
- Changes are allowed as long as they remain **domain-agnostic**: generic reasoning scaffolds ("plan, then verify each step"), output-format or termination guidance, self-check instructions. 
- In other words, they are allowed if they would be equally sensible for a different puzzle domain. 
- What's not allowed are Sokoban strategy hints, heuristics, or deadlock rules ("don't push a box into a goal-less corner"), and no worked examples (zero-shot). 

# Shoutouts

* [nanochat](https://github.com/karpathy/nanochat): We forked [chat_rl.py](https://github.com/karpathy/nanochat/blob/master/scripts/chat_rl.py)
* [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt): The OG of LLM speedruns, and the template for these rules
* [nanoRL](https://joshuaharrissite.substack.com/p/nanorl): Fundoku RL LLM speedrun
* [ScaleRL](https://arxiv.org/abs/2510.13786): Our initial recipe took a lot of inspiration from theirs
* [ReasoningGym](https://github.com/open-thought/reasoning-gym) provided the Sokoban implementation
