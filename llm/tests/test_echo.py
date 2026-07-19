import json
import unittest
from pathlib import Path

import torch

from speedrun import (
    build_rl_batch_varprefix_cpu,
    policy_gradient_loss_from_token_logprobs,
    process_rollout_sample,
)


class _TinyTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return "<answer>R</answer>"

    def encode(self, text, add_special_tokens=False):
        return [1000 + ord(char) for char in text]


class ECHOTests(unittest.TestCase):
    def test_echo_and_policy_gradients_are_disjoint_and_independently_normalized(self):
        log_probs = torch.tensor([[-1.0, -2.0]], requires_grad=True)
        loss = policy_gradient_loss_from_token_logprobs(
            log_probs,
            torch.tensor([[10, 11]]),
            torch.tensor([1.0]),
            behavior_logprobs=torch.tensor([[-1.0, 0.0]]),
            loss_fn="dppo",
            dppo_divergence="binary_tv",
            dppo_delta=0.2,
            dppo_is_cap=5.0,
            rl_mask=torch.tensor([[True, False]]),
            echo_mask=torch.tensor([[False, True]]),
            echo_alpha=0.05,
            echo_token_normalizer=1,
        )
        loss.backward()
        self.assertAlmostEqual(log_probs.grad[0, 0].item(), -1.0, places=6)
        self.assertAlmostEqual(log_probs.grad[0, 1].item(), -0.05, places=6)

    def test_batch_builder_keeps_rl_and_echo_targets_separate(self):
        sequence = torch.tensor([1, 2, 3, 4, 5, 6, 7])
        _, _, labels, rl_mask, echo_mask = build_rl_batch_varprefix_cpu(
            [sequence],
            [2],
            pad_token_id=0,
            masks=[[True, False, False, False, False]],
            echo_masks=[[False, False, False, True, False]],
        )
        self.assertEqual(rl_mask.sum().item(), 1)
        self.assertEqual(echo_mask.sum().item(), 1)
        self.assertFalse(torch.any(rl_mask & echo_mask))
        self.assertEqual((labels != -100).sum().item(), 2)

    def test_rollout_appends_simulator_observation_without_changing_action_text(self):
        record = json.loads(
            Path("datasets/sokoban_train.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        sample = process_rollout_sample(
            _TinyTokenizer(),
            torch.tensor([1, 2]),
            2,
            {"token_ids": [3, 4], "logprobs": [-1.0, -1.0], "finish_reason": "stop"},
            record,
            trim_after_answer=False,
            echo_alpha=0.05,
        )
        self.assertEqual(sample.completion, "<answer>R</answer>")
        self.assertEqual(sample.action_token_count, 2)
        self.assertEqual(len(sample.loss_mask), len(sample.behavior_logprobs))
        self.assertEqual(len(sample.echo_mask), len(sample.behavior_logprobs))
        self.assertTrue(any(sample.echo_mask))
        self.assertFalse(any(a and b for a, b in zip(sample.loss_mask, sample.echo_mask)))


if __name__ == "__main__":
    unittest.main()
