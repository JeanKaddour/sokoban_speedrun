# Sokoban Speedrun

Goal: the fastest recipe to RL [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) up to a `{TARGET}` solve-rate on held-out Sokoban puzzles, using a single 8xH100 node.

Sokoban: If you don't know it, the best way to familiarize yourself with it is [to play it here](https://www.jeankaddour.com/sokoban). 

Motivation: A lot of LLM RL papers don't reproduce and there are many high-variance moving parts in their pipelines. This is an attempt to standardize things.

## How it works

`speedrun.py` is a self-contained async RL pipeline in one file. It is designed to run on one 8×H100 node, using `torch` and `vllm`. 

Two roles share the node:

- **Generator**: one vLLM process samples completions for each puzzle.
- **Trainers**: data-parallel ranks score the rollouts (solutions verified by ReasoningGym), update the policy, and broadcast fresh weights to the generator over NCCL.

The two run concurrently, so the generator never idles waiting for a step.

## Rules

The primary metric is the pass@1 solve-rate, averaged over `5` seeds, but contributors are encouraged to also report pass@k for {4,8,16}. 

A new record must reach `TARGET` with statistical significance `p<0.01` over the prior record.

What is fixed: 
- Base model: Qwen3-4B
- Training dataset: A fixed set of `10,000` puzzles, frozen and published [here](https://github.com/JeanKaddour/sokoban_speedrun/blob/main/datasets/sokoban_train.jsonl).
- Eval: A fixed, held-out set of `128` puzzles, disjoint from the training set (regenerable larger via `generate_sokoban_datasets.py --eval-size N --eval-seed 10000000 --assert-disjoint`). 

What can be changed: Pretty much anything else.
- RL algorithm, loss function, etc.
- Dataset-agnostic auxiliary rewards (eg. entropy or uncertainty proxies)
- Training and inference engine
- System prompt (but keep it zero-shot!)

# World record history

| # | Record time | Description | Date | Log | Contributors |
| - | - | - | - | - | - |
| 1 | TBD | Async CISPO baseline | TBD | TBD | @JeanKaddour |

# How to run

On an 8-GPU node (the trainer takes GPU 0, vLLM generators take the rest):

```bash
torchrun --standalone --nproc_per_node=4 -m speedrun
```

## Modal

If you'd rather not manage a node yourself, `modal_app.py` rents an 8×H100 box on [Modal](https://modal.com) and launches `speedrun.py` with its defaults (it passes no arguments):

```bash
# one-time: push the fixed datasets to the Modal volume
modal volume put nanochat-rl-hf datasets/sokoban_train.jsonl /datasets/sokoban_train.jsonl
modal volume put nanochat-rl-hf datasets/sokoban_eval.jsonl  /datasets/sokoban_eval.jsonl

# launch (use --detach so the run survives the client disconnecting)
modal run --detach modal_app.py
```

The function runs `python -m speedrun` from the `nanochat-rl-hf` volume (mounted at `/vol`), so its relative paths resolve there: datasets at `/vol/datasets/`, with checkpoints and rollouts written to `/vol/outputs/` and committed every 60s.

# FAQ

## Why Sokoban?

* Small contamination risk: We generate fresh puzzles, so unlike Go or GSM8k there is little risk the base model has memorized them.
* Simple enough to yield RL gains in `O(10)` GPU hours
* Hard enough for gains to be meaningful: Sokoban is PSPACE-complete and can't be brute-forced; it requires genuine reasoning capabilities.
* Diverse reasoning paths are encouraged. Puzzles naturally permit multiple solutions.

## Why handroll your own asnyc RL stack?
* Using an existing framework might help in the short term, but it comes with a lot of machinery and overhead that isn't needed here. General-purpose frameworks are built to support every RL scenario and layers of abstraction between you and the training loop. 
* Our (initial) setup is minimal: torchrun launches a handful of data-parallel trainer ranks alongside one vLLM generator on its own GPUs of the same 8×H100 node, and the trainer broadcasts fresh weights to vLLM over NCCL. This makes it small enough to read end to end. 

# Shoutouts 

* [nanochat](https://github.com/karpathy/nanochat): We forked [chat_rl.py](https://github.com/karpathy/nanochat/blob/master/scripts/chat_rl.py) 
* [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt): The OG of LLM speedruns
* [nanoRL](https://joshuaharrissite.substack.com/p/nanorl): Fundoku RL LLM speedrun
* [ScaleRL](https://arxiv.org/abs/2510.13786): Our initial recipe took a lot of inspiration from theirs
* [ReasoningGym](https://github.com/open-thought/reasoning-gym) provided the Sokoban implementation
