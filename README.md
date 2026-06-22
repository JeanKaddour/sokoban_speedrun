# Sokoban Speedrun

Fastest recipes to RL models to solve Sokoban to a held-out target on one node:

- **[LLM Track](#llm-track)**: RL-fine-tune [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) from 57% to **>80% held-out pass@1** on one **8xH100**.
- **[Non-LLM Track](#non-llm-track)**: train a **from-scratch** agent on a **single H100**.

[Play Sokoban](https://www.jeankaddour.com/sokoban) if the task is unfamiliar.

## LLM Track

### World record history


| #   | Record time (h:mm:ss) | FLOPs        | Description                        | Date       | Log                                                          | held-out pass@1         | Contributors |
| --- | --------------------- | ------------ | ---------------------------------- | ---------- | ------------------------------------------------------------ | ----------------------- | ------------ |
| 1   | 1:27:31               | 1.250837e+18 | GRPO, LR 1.6e-6 annealed, 75 steps | 2026-06-17 | [records/llm/2026-06-17_01](records/llm/2026-06-17_01_grpo/) | 0.891 (CI [0.86, 0.92]) | @JeanKaddour |


### Rules

Fastest wall-clock run wins: one run on one 8xH100 node, from training step 1 through the final checkpoint, which must clear the target.

- **Target:** lower 95% bootstrap CI > 0.80 on [datasets/sokoban_eval.jsonl](datasets/sokoban_eval.jsonl).
- **Eval:** 8 completions/puzzle, 12,288 tokens, temperature 0.8, top-p 0.95, seed 12345.
- **Fixed:** model, [train set](datasets/sokoban_train.jsonl), eval set, reward function, hardware.
- **Open:** RL algorithm, loss, schedules, engine, parallelism, domain-agnostic rewards, prompt.
- **Not allowed:** Sokoban-specific hints, heuristics, or few-shot examples.
- **Verification:** maintainers rerun at a second seed; both runs must clear the target.

### Running

```bash
NODE_GPUS=8 torchrun --standalone --nproc_per_node=3 -m speedrun
python -m eval_speedrun --eval-checkpoint outputs/<run>/step_000075
# or on Modal (modal_app.py rents an 8xH100)
modal volume put nanochat-rl-hf datasets/sokoban_train.jsonl /datasets/sokoban_train.jsonl
modal run --detach modal_app.py
EVAL_CHECKPOINT=latest modal run modal_app.py
```

## Non-LLM Track

### World record history


| #   | Record time (mm:ss) | FLOPs        | Description     | Date       | Log                                                                     | held-out pass@1         | Contributors |
| --- | ------------------- | ------------ | --------------- | ---------- | ----------------------------------------------------------------------- | ----------------------- | ------------ |
| 1   | 19:36               | 1.550943e+16 | cnn-mingru h256 | 2026-06-21 | [records/non_llm/2026-06-21_01](records/non_llm/2026-06-21_01_non_llm/) | 0.728 (CI [0.70, 0.75]) | @JeanKaddour |


### Rules

Fastest wall-clock run wins: one run on one node, from training step 1 through the first checkpoint whose held-out CI clears the target.

- **Target:** lower 95% CI on held-out Boxoban solve-rate > **0.70**.
- **Eval:** official [DeepMind Boxoban](https://github.com/google-deepmind/boxoban-levels) held-out splits (per-level scoring); default `unfiltered/test`.
- **Open:** policy architecture, RL algorithm, optimizer, schedules, implementation.
- **Verification:** maintainers rerun at a second seed; both runs must clear the target.

### Running

```bash
uv sync
uv run python speedrun_non_llm.py                         # local
uv run modal run --detach modal_app_non_llm.py            # Modal 1xH100
```

## Submitting a record

1. Train, then eval the final checkpoint — logs, source snapshots, and the eval JSON are written automatically.
2. Run `python make_record_report.py records/<track>/<your-dir> --track <llm|non-llm>` and fill in the `Idea` section.
3. Open a PR adding the record dir + a row in the matching track's world record history. CI runs `python verify_record.py records/<track>/<your-dir>`.

## Credits

[@joshua-a-harris](https://github.com/joshua-a-harris)'s [nanoRL speedrun](https://joshuaharrissite.substack.com/p/nanorl), [nanochat](https://github.com/karpathy/nanochat), [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), [ScaleRL](https://arxiv.org/abs/2510.13786), [ReasoningGym for the LLM-track Sokoban env](https://github.com/open-thought/reasoning-gym), [DeepMind for Boxoban](https://github.com/google-deepmind/boxoban-levels) and [PufferLib for the efficient `boxoban` implementation](https://github.com/PufferAI/PufferLib).