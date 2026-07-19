import math
import unittest

import torch

from speedrun import (
    IGNORE_INDEX,
    _resolve_loss_args,
    build_parser,
    chunked_token_logprobs,
    policy_gradient_loss_from_token_logprobs,
)


class DPPOLossTests(unittest.TestCase):
    def _run_tv_case(self, advantage, behavior_prob, policy_prob):
        log_prob = torch.tensor([[math.log(policy_prob)]], requires_grad=True)
        behavior_log_prob = torch.tensor([[math.log(behavior_prob)]])
        stats = {}
        loss = policy_gradient_loss_from_token_logprobs(
            log_prob,
            torch.tensor([[1]]),
            torch.tensor([advantage]),
            behavior_logprobs=behavior_log_prob,
            loss_fn="dppo",
            dppo_divergence="binary_tv",
            dppo_delta=0.2,
            dppo_is_cap=5.0,
            stats=stats,
        )
        loss.backward()
        return stats["is_clipped_count"].item() == 1, log_prob.grad.item()

    def test_tv_masks_only_large_moves_away_from_behavior(self):
        for advantage, behavior_prob, policy_prob, should_mask in [
            (+1.0, 0.60, 0.90, True),
            (+1.0, 0.60, 0.30, False),
            (-1.0, 0.60, 0.30, True),
            (-1.0, 0.60, 0.90, False),
        ]:
            with self.subTest(
                advantage=advantage,
                behavior_prob=behavior_prob,
                policy_prob=policy_prob,
            ):
                masked, grad = self._run_tv_case(advantage, behavior_prob, policy_prob)
                self.assertEqual(masked, should_mask)
                self.assertEqual(grad == 0.0, should_mask)

    def test_tv_retains_large_ratio_with_small_probability_shift(self):
        masked, grad = self._run_tv_case(+1.0, 0.001, 0.01)
        self.assertFalse(masked)
        self.assertNotEqual(grad, 0.0)

    def test_binary_kl_is_finite_and_padding_is_inert(self):
        log_prob = torch.tensor([[math.log(0.99), 0.0]], requires_grad=True)
        stats = {}
        loss = policy_gradient_loss_from_token_logprobs(
            log_prob,
            torch.tensor([[1, IGNORE_INDEX]]),
            torch.tensor([1.0]),
            behavior_logprobs=torch.tensor([[math.log(0.20), 0.0]]),
            loss_fn="dppo",
            dppo_divergence="binary_kl",
            dppo_delta=0.05,
            dppo_is_cap=5.0,
            stats=stats,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(stats["is_clipped_count"].item(), 1)
        self.assertTrue(torch.equal(log_prob.grad, torch.zeros_like(log_prob.grad)))

    def test_loss_defaults_resolve_per_divergence(self):
        args = build_parser().parse_args(
            ["--loss-fn", "dppo", "--dppo-divergence", "binary_kl"]
        )
        _resolve_loss_args(args)
        self.assertEqual(args.advantage_mode, "centered")
        self.assertEqual(args.dppo_delta, 0.05)
        self.assertIsNone(args.clip_ratio_c)

    def test_bf16_tied_head_uses_fp32_logits_without_cloning_weight(self):
        torch.manual_seed(7)
        head = torch.nn.Linear(5, 11, bias=False, dtype=torch.bfloat16)
        hidden = torch.randn(2, 4, 5, dtype=torch.bfloat16, requires_grad=True)
        labels = torch.randint(0, 11, (2, 4))

        actual = chunked_token_logprobs(head, hidden, labels, chunk_size=2)
        expected = -torch.nn.functional.cross_entropy(
            torch.nn.functional.linear(hidden.float(), head.weight.float()).reshape(-1, 11),
            labels.reshape(-1),
            reduction="none",
        ).view_as(labels)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

        actual.sum().backward()
        self.assertIsNotNone(head.weight.grad)
        self.assertEqual(head.weight.grad.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
