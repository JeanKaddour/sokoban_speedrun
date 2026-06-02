# Sokoban Speedrun

Goal: the fastest recipe to RL [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) up to a `{TARGET}` solve-rate on held-out Sokoban puzzles, using a single 8xH100 node.

Sokoban: If you don't know it, the best way to familiarize yourself with it is [to play it here](https://www.jeankaddour.com/boxy). 

Motivation: A lot of LLM RL papers don't reproduce and there are many high-variance moving parts in their pipelines. This is an attempt to standardize things.

## Rules

The primary metric is the pass@1 solve-rate, averaged over `{N_SEEDS}` seeds, but contributors are encouraged to also report pass@k for {4,8,16}. 

A new record must reach `{TARGET}` with statistical significance `p<0.01` over the prior record.

What is fixed: 
- Base model: Qwen3-4B
- Training dataset: A fixed set of `{N_TRAIN}` puzzles, frozen and published [here](https://github.com/JeanKaddour/sokoban_speedrun/blob/main/datasets/sokoban_train.jsonl). Regenerable using [ReasoningGym](https://github.com/JeanKaddour/sokoban_speedrun/blob/main/generate_sokoban_datasets.py).
- Eval: A fixed, held-out set of `{N_EVAL}` puzzles, disjoint from the training set. 
- Rollout budget: 6144 tokens per puzzle, single-turn, thinking included.

What can be changed: Pretty much anything else.
- RL algorithm, loss function, etc.
- Dataset-agnostic auxiliary rewards (eg. entropy or uncertainty proxies)
- Training and inference engine
- System prompt (but keep it zero-shot!)

# World record history

TBA

# Why Sokoban?

* Sokoban is PSPACE-complete; it can't be brute-forced and genuinely requires strong reasoning capabilities.
* Small contamination risk: We generate fresh puzzles, so unlike Go or GSM8k there is little risk the base model has memorized them.
* Diverse reasoning paths are encouraged, as Puzzles typically permit multiple solutions. Ideal for measuring the model's diversity. 

# How to run

```bash
torchrun --standalone --nproc_per_node=4 -m run_rl
```

If you'd rather not manage a node yourself, `modal_app.py` rents the same 8×H100 box on [Modal](https://modal.com) and runs the identical launcher:

```bash
modal run --detach modal_app.py
```

The run uses the `sokoban-speedrun` Modal volume (mounted at `/vol`) as its working directory, so push the datasets there once with `modal volume put sokoban-speedrun datasets /datasets`.

# Shoutouts 

* [nanochat](https://github.com/karpathy/nanochat): We forked [chat_rl.py](https://github.com/karpathy/nanochat/blob/master/scripts/chat_rl.py) 
* [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt): The OG of LLM speedruns
* [nanoRL](https://joshuaharrissite.substack.com/p/nanorl): Fundoku RL LLM speedrun
* [ScaleRL](https://arxiv.org/abs/2510.13786): Our initial recipe took a lot of inspiration from theirs