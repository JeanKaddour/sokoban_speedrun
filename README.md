# Sokoban Speedrun

Fastest recipes to RL models to solve Sokoban to a held-out target on one node:

- **[LLM Track](#llm-track)**: RL-fine-tune [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) from 57% to **>80% held-out pass@1** on one **8xH100**.
- **[Non-LLM Track](#non-llm-track)**: train a **from-scratch** agent on a **single H100**.

[Play Sokoban](https://www.jeankaddour.com/sokoban) if the task is unfamiliar.


## LLM Track

### World record history

![LLM track world records — held-out pass@1 vs wall-clock time to target](assets/llm_records.png)

| #   | Record time (mm:ss) | Description                        | Date       | Log                                                          | held-out pass@1         | Contributors |
| --- | ------------------- | ---------------------------------- | ---------- | ------------------------------------------------------------ | ----------------------- | ------------ |
| 1   | 48:53               | GRPO, LR 1.6e-6 annealed, 75 steps | 2026-06-29 | [llm/records/2026-06-29_01](llm/records/2026-06-29_01_grpo/) | 0.869 (CI [0.83, 0.90]) | @JeanKaddour |
| 2   | 36:53               | GRPO, LR 1.6e-6 annealed, 60 steps | 2026-06-29 | [llm/records/2026-06-29_02](llm/records/2026-06-29_02_grpo_60step/) | 0.843 (CI [0.81, 0.88]) | @dexhunter   |


### Rules

Fastest wall-clock run wins: one run on one 8xH100 node, from training step 1 through the final checkpoint, which must clear the target.

- **Target:** lower 95% bootstrap CI > 0.80 on [llm/datasets/sokoban_eval.jsonl](llm/datasets/sokoban_eval.jsonl).
- **Eval:** 8 completions/puzzle, 12,288 tokens, temperature 0.8, top-p 0.95, seed 12345.
- **Fixed:** model, [train set](llm/datasets/sokoban_train.jsonl), eval set, reward function, hardware.
- **Open:** RL algorithm, loss, schedules, engine, parallelism, domain-agnostic rewards, prompt.
- **Not allowed:** Sokoban-specific hints, heuristics, or few-shot examples.
- **Verification:** Rerun with a second seed; both runs must clear the target. The held-out pass@1 column reports the **worst** of the two seeds (the binding one), so it can't be seed-shopped; records are ranked by wall-clock, not pass@1.

### Running

```bash
cd llm
uv sync
NODE_GPUS=8 uv run torchrun --standalone --nproc_per_node=3 -m speedrun
uv run python -m eval_speedrun --eval-checkpoint outputs/<run>/step_000075
```

## Non-LLM Track

### World record history

![Non-LLM track world records — held-out solve-rate vs wall-clock time to target](assets/non_llm_records.png)

| #   | Record time (mm:ss) | Description     | Date       | Log                                                                     | held-out pass@1         | Contributors |
| --- | ------------------- | --------------- | ---------- | ----------------------------------------------------------------------- | ----------------------- | ------------ |
| 1   | 22:24               | cnn-mingru h256 | 2026-06-21 | [non_llm/records/2026-06-21_01](non_llm/records/2026-06-21_01_non_llm/) | 0.744 (CI [0.72, 0.77]) | @JeanKaddour |
| 2 | 21:00 | cnn-mingru h256 — earliest clearing checkpoint (iter 1200, same recipe as #1) | 2026-06-29 | [non_llm/records/2026-06-29_01](non_llm/records/2026-06-29_01_non_llm/) | 0.735 (CI [0.71, 0.76]) | @JeanKaddour |


### Rules

Fastest wall-clock run wins: one run on a single **1×H100** node, from training step 1 through the first checkpoint whose held-out CI clears the target.

- **Target:** lower 95% CI on held-out Boxoban solve-rate > **0.70**.
- **Eval:** official [DeepMind Boxoban](https://github.com/google-deepmind/boxoban-levels) held-out splits (per-level greedy scoring); default `unfiltered/test`.
- **Disjointness:** training draws only from the official `unfiltered/train` split; eval uses the disjoint `unfiltered/test`.
- **Open:** policy architecture, RL algorithm, optimizer, schedules, implementation.
- **Verification:** Rerun with a second seed; both runs must clear the target. The held-out pass@1 column reports the **worst** of the two seeds (the binding one), so it can't be seed-shopped; records are ranked by wall-clock, not pass@1.

### Running

```bash
cd non_llm
uv sync
uv run python speedrun.py

# Modal rents the 1×H100 for you (handy for this track):
uv run modal run --detach modal_app_non_llm.py
```

## Submitting a record

Each track's `assemble_record.sh` ([`llm/`](llm/assemble_record.sh), [`non_llm/`](non_llm/assemble_record.sh)) turns a finished run into a record dir: it collects the log, eval JSON, and source snapshot, builds the report, pins the top-level `speedrun.py`, runs `verify_record.py`, and adds the record's row + redraws the leaderboard figure. It reads a local `outputs/<RUN>/` by default; pass `SOURCE=modal` to pull off the volume.

1. **Train + eval** with your track's [Running](#running) commands.

2. **Assemble** the record:

   ```bash
   cd llm        # or: cd non_llm
   RUN=<RUN> DEST=records/<date>_01_<name> ./assemble_record.sh
   ```

   The record's `README.md` is scaffolded with a placeholder **`## Idea`** section. Review the pinned `speedrun.py` diff, then fill in by hand: the record's `## Idea`, and the new leaderboard row's **Description** + **Contributors** in this top-level `README.md`.

3. **Open a PR** with the record dir + new row.

*Optional:* verify it yourself with a second seed (otherwise the maintainers do; either way both seeds must clear the target), assembled into the record's `verification/` subdir:

```bash
RUN=<VRUN> VERIFY_OF=records/<date>_01_<name> ./assemble_record.sh
```

The top-level `speedrun.py` files always hold the current record's recipe.

## Credits

[@joshua-a-harris](https://github.com/joshua-a-harris)'s [nanoRL speedrun](https://joshuaharrissite.substack.com/p/nanorl), [nanochat](https://github.com/karpathy/nanochat), [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), [ScaleRL](https://arxiv.org/abs/2510.13786), [ReasoningGym for the LLM-track Sokoban env](https://github.com/open-thought/reasoning-gym), [DeepMind for Boxoban](https://github.com/google-deepmind/boxoban-levels) and [PufferLib for the efficient `boxoban` implementation](https://github.com/PufferAI/PufferLib).
