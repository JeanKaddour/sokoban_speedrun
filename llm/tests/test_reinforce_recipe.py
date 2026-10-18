import unittest

import torch

import speedrun


class ReinforceRecipeTests(unittest.TestCase):
    def test_baked_recipe(self):
        args = speedrun.build_parser().parse_args(speedrun.RECIPE)
        speedrun._resolve_loss_args(args)
        speedrun._validate_sampling_args(args)
        speedrun._validate_pipeline_args(args)
        expected = dict(loss_fn='reinforce', advantage_mode='centered', max_steps=33,
                        adv_difficulty_schedule_steps=35, lr_decay_steps=60,
                        warmup_steps=5, learning_rate=1.6e-6, device_batch_size=2,
                        logits_chunk_size=1024, vllm_kv_cache_dtype='fp8',
                        behavior_logprobs_mode='processed', loss_normalization='sequence',
                        wandb=False, save_rollouts=False, final_rollout_steps=0,
                        wandb_rollout_samples=0, log_entropy=False, entropy_coef=0.,
                        kl_tau=0., save_final=True, save_every=0, clip_ratio_c=None)
        for key, value in expected.items():
            self.assertEqual(getattr(args, key), value, key)

    def test_reinforce_gradient_independent_of_behavior(self):
        labels = torch.tensor([[1, 2, speedrun.IGNORE_INDEX], [1, 2, 3]])
        valid = labels != speedrun.IGNORE_INDEX
        advantages = torch.tensor([.7, -.4])
        reference = None
        for offset in (-10., 0., 10.):
            leaf = torch.tensor([[-.2, -2., -.4], [-1., -.5, -3.]], requires_grad=True)
            logp = leaf.masked_fill(~valid, 0.)
            behavior = (logp.detach() + offset).requires_grad_()
            stats = {}
            loss = speedrun.policy_gradient_loss_from_token_logprobs(
                logp, labels, advantages, behavior_logprobs=behavior,
                loss_fn='reinforce', sequence_normalize=True, stats=stats)
            loss.backward()
            expected = -advantages[:, None] * valid / valid.sum(-1)[:, None] / 2
            torch.testing.assert_close(leaf.grad, expected)
            self.assertIsNone(behavior.grad)
            self.assertEqual(stats['is_clipped_count'].item(), 0)
            if reference is not None:
                self.assertTrue(torch.equal(loss.detach(), reference))
            reference = loss.detach()

    def test_global_sequence_normalizer_including_empty_padding(self):
        labels = torch.tensor([[1, 2], [-100, -100]])
        logp = torch.tensor([[-.5, -1.], [0., 0.]], requires_grad=True)
        loss = speedrun.policy_gradient_loss_from_token_logprobs(
            logp, labels, torch.tensor([1., 0.]), behavior_logprobs=logp.detach(),
            loss_fn='reinforce', sequence_normalize=True, valid_sample_normalizer=8.)
        self.assertAlmostEqual(loss.item(), .75 / 8)
        loss.backward()
        torch.testing.assert_close(logp.grad, torch.tensor([[-1/16, -1/16], [0., 0.]]))

    def test_fixed_horizon_matches_first_33_steps(self):
        rewards = torch.tensor([0., .2, .8, 1.])
        adv = rewards - rewards.mean()
        for step in range(33):
            kw = dict(step=step, centered_blend=.04, difficulty_weight=.18,
                      difficulty_ramp=1., variance_boost=1.1)
            original = speedrun.shape_group_advantages(rewards, adv, num_steps=35, **kw)
            shortened = speedrun.shape_group_advantages(
                rewards, adv, num_steps=33, schedule_steps=35, **kw)
            self.assertTrue(torch.equal(original, shortened))

    def test_invalid_horizon(self):
        args = speedrun.build_parser().parse_args(
            [*speedrun.RECIPE, '--adv-difficulty-schedule-steps', '0'])
        with self.assertRaisesRegex(ValueError, 'schedule-steps'):
            speedrun._validate_pipeline_args(args)

    def test_cache_tuning_does_not_quantize_weights(self):
        args = speedrun.build_parser().parse_args(speedrun.RECIPE)
        tuning = speedrun.vllm_tuning_from_args(args)
        self.assertEqual(tuning['kv_cache_dtype'], 'fp8')
        self.assertNotIn('quantization', tuning)

    def test_other_losses_still_resolve(self):
        for loss in ('cispo', 'grpo', 'gspo'):
            args = speedrun.build_parser().parse_args(['--loss-fn', loss])
            speedrun._resolve_loss_args(args)
            self.assertEqual(args.advantage_mode, 'grpo' if loss == 'grpo' else 'centered')


if __name__ == '__main__':
    unittest.main()
