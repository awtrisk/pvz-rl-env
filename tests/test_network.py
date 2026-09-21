"""
Unit tests for Phase 4 neural network (network.py).

Run from repo root:
    python tests/test_phase4_network.py

Requires:
    - torch
    - mambapy
    - network.py
    - pvz_portable_gym.py and compiled pvz_env (for smoke test section)
"""

import importlib.util
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_BUILD_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "pvz-portable", "build")
if os.path.isdir(_BUILD_DIR):
    sys.path.insert(0, _BUILD_DIR)

from network import GLOBAL_SIZE, NUM_ACTIONS, PvZActorCritic  # noqa: E402


class TestNetworkSequence(unittest.TestCase):
    """Tests for full-sequence forward mode."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.net = PvZActorCritic().to(self.device)
        self.net.eval()
        self.B, self.L = 2, 8
        self.spatial = torch.randn(self.B, self.L, 5, 9, 36, device=self.device)
        self.global_vec = torch.rand(self.B, self.L, GLOBAL_SIZE, device=self.device)
        self.mask = torch.randint(
            0, 2, (self.B, self.L, NUM_ACTIONS), dtype=torch.bool, device=self.device
        )
        self.mask[:, :, 0] = True  # wait is always legal

    def test_forward_sequence_shape(self):
        latent = self.net.forward_sequence(self.spatial, self.global_vec)
        self.assertEqual(latent.shape, (self.B, self.L, 512))

    def test_get_value_shape(self):
        value = self.net.get_value(self.spatial, self.global_vec)
        self.assertEqual(value.shape, (self.B, self.L, 1))

    def test_get_action_and_value_outputs(self):
        action, log_prob, entropy, value = self.net.get_action_and_value(
            self.spatial, self.global_vec, self.mask
        )
        self.assertEqual(action.shape, (self.B, self.L))
        self.assertTrue((action >= 0).all() and (action < NUM_ACTIONS).all())
        self.assertEqual(log_prob.shape, (self.B, self.L))
        self.assertEqual(entropy.shape, (self.B, self.L))
        self.assertEqual(value.shape, (self.B, self.L, 1))

    def test_action_mask_enforces_wait_only(self):
        mask = torch.zeros(
            self.B, self.L, NUM_ACTIONS, dtype=torch.bool, device=self.device
        )
        mask[:, :, 0] = True
        action, _, _, _ = self.net.get_action_and_value(
            self.spatial, self.global_vec, mask
        )
        self.assertTrue((action == 0).all())

    def test_deterministic_action_uses_masked_argmax(self):
        logits = self.net._apply_action_mask(
            self.net.actor(self.net.forward_sequence(self.spatial, self.global_vec)),
            self.mask,
        )
        expected = logits.argmax(dim=-1)
        action, _, _, _ = self.net.get_action_and_value(
            self.spatial, self.global_vec, self.mask, deterministic=True
        )
        self.assertTrue(torch.equal(action, expected))

        repeated, _, _, _ = self.net.get_action_and_value(
            self.spatial, self.global_vec, self.mask, deterministic=True
        )
        self.assertTrue(torch.equal(action, repeated))

    def test_action_mask_rejects_invalid_contracts(self):
        logits = torch.zeros(2, NUM_ACTIONS)
        with self.assertRaises(ValueError):
            self.net._apply_action_mask(logits, torch.ones(2, NUM_ACTIONS - 1))
        with self.assertRaises(ValueError):
            self.net._apply_action_mask(logits, torch.zeros(2, NUM_ACTIONS))

    def test_action_mask_blocks_illegal_actions(self):
        torch.manual_seed(42)
        for _ in range(20):
            action, _, _, _ = self.net.get_action_and_value(
                self.spatial, self.global_vec, self.mask
            )
            for batch in range(self.B):
                for index in range(self.L):
                    selected = self.mask[batch, index].gather(
                        0, action[batch, index].reshape(1)
                    )
                    self.assertTrue(selected.item())

    def test_deterministic_with_given_action(self):
        action = torch.zeros(self.B, self.L, dtype=torch.long, device=self.device)
        _, log_prob, entropy, value = self.net.get_action_and_value(
            self.spatial, self.global_vec, self.mask, action=action
        )
        self.assertEqual(log_prob.shape, (self.B, self.L))
        self.assertEqual(entropy.shape, (self.B, self.L))
        self.assertEqual(value.shape, (self.B, self.L, 1))


class TestNetworkStep(unittest.TestCase):
    """Tests for single-step cached mode."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.net = PvZActorCritic().to(self.device)
        self.net.eval()
        self.B = 3
        self.spatial = torch.randn(self.B, 5, 9, 36, device=self.device)
        self.global_vec = torch.rand(self.B, GLOBAL_SIZE, device=self.device)
        self.mask = torch.ones(
            self.B, NUM_ACTIONS, dtype=torch.bool, device=self.device
        )
        self.caches = self.net.init_caches(self.B, self.device)

    def test_init_caches_shape(self):
        self.assertEqual(len(self.caches), self.net.mamba_config.n_layers)
        for h, inputs in self.caches:
            self.assertEqual(
                h.shape,
                (self.B, self.net.mamba_config.d_inner, self.net.mamba_config.d_state),
            )
            self.assertEqual(
                inputs.shape,
                (
                    self.B,
                    self.net.mamba_config.d_inner,
                    self.net.mamba_config.d_conv - 1,
                ),
            )

    def test_forward_step_shape(self):
        action, log_prob, entropy, value, new_caches = self.net.forward_step(
            self.spatial, self.global_vec, self.mask, self.caches
        )
        self.assertEqual(action.shape, (self.B,))
        self.assertEqual(log_prob.shape, (self.B,))
        self.assertEqual(entropy.shape, (self.B,))
        self.assertEqual(value.shape, (self.B, 1))
        self.assertEqual(len(new_caches), len(self.caches))

    def test_forward_step_updates_recurrent_caches(self):
        before = [(h.clone(), inputs.clone()) for h, inputs in self.caches]
        _, _, _, _, new_caches = self.net.forward_step(
            self.spatial, self.global_vec, self.mask, self.caches
        )

        self.assertTrue(
            any(
                not torch.equal(old_h, new_h) or not torch.equal(old_inputs, new_inputs)
                for (old_h, old_inputs), (new_h, new_inputs) in zip(
                    before, new_caches, strict=True
                )
            )
        )

    def test_zero_memory_gate_matches_legacy_feed_forward(self):
        with torch.no_grad():
            features = self.net._encode_features(self.spatial, self.global_vec)
            expected = self.net.actor(self.net.latent_proj(features))
            action, _, _, value, _ = self.net.forward_step(
                self.spatial,
                self.global_vec,
                self.mask,
                self.caches,
                deterministic=True,
            )
            expected_action = expected.argmax(dim=-1)
            expected_value = self.net.critic(self.net.latent_proj(features))
        self.assertTrue(torch.equal(action, expected_action))
        self.assertTrue(torch.equal(value, expected_value))

    def test_legacy_state_loads_with_zero_memory_gate(self):
        legacy = dict(self.net.state_dict())
        del legacy["memory_gate"]
        restored = PvZActorCritic()
        restored.load_state_dict(legacy)
        self.assertTrue(
            torch.equal(restored.memory_gate, torch.zeros_like(restored.memory_gate))
        )

    def test_legacy_33_channel_state_is_exact_noop(self):
        legacy = dict(self.net.state_dict())
        del legacy["actor.residual.weight"]
        del legacy["actor.residual.bias"]
        legacy["spatial_encoder.0.weight"] = legacy["spatial_encoder.0.weight"][
            :, :33
        ].clone()
        restored = PvZActorCritic().eval()
        restored.load_state_dict(legacy)

        spatial = self.spatial.clone()
        spatial[..., 33:] = 0
        with torch.no_grad():
            expected = self.net.actor(
                self.net.latent_proj(
                    self.net._encode_features(spatial, self.global_vec)
                )
            )
            actual = restored.actor(
                restored.latent_proj(
                    restored._encode_features(spatial, self.global_vec)
                )
            )
        self.assertTrue(torch.equal(actual, expected))
        migrated_weight = restored.state_dict()["spatial_encoder.0.weight"]
        self.assertEqual(migrated_weight[:, 33:].count_nonzero().item(), 0)
        self.assertEqual(restored.actor.residual.weight.count_nonzero().item(), 0)
        self.assertEqual(restored.actor.residual.bias.count_nonzero().item(), 0)

    def test_forward_step_masked_wait(self):
        mask = torch.zeros(1, NUM_ACTIONS, dtype=torch.bool, device=self.device)
        mask[:, 0] = True
        caches = self.net.init_caches(1, self.device)
        spatial = torch.randn(1, 5, 9, 36, device=self.device)
        global_vec = torch.rand(1, GLOBAL_SIZE, device=self.device)
        action, _, _, _, _ = self.net.forward_step(spatial, global_vec, mask, caches)
        self.assertEqual(action.item(), 0)

    def test_forward_step_deterministic_action_uses_masked_argmax(self):
        logits = self.net._apply_action_mask(
            self.net.actor(
                self.net.latent_proj(
                    self.net._encode_features(self.spatial, self.global_vec)
                )
            ),
            self.mask,
        )
        expected = logits.argmax(dim=-1)
        action, _, _, _, _ = self.net.forward_step(
            self.spatial, self.global_vec, self.mask, self.caches, deterministic=True
        )
        self.assertTrue(torch.equal(action, expected))

    def test_sequence_resets_memory_at_episode_boundary(self):
        self.net.memory_gate.data.fill_(0.1)
        spatial = torch.randn(1, 6, 5, 9, 36)
        global_vec = torch.rand(1, 6, GLOBAL_SIZE)
        starts = torch.tensor([[1, 0, 0, 1, 0, 0]], dtype=torch.bool)
        whole = self.net.forward_sequence(spatial, global_vec, starts)
        suffix = self.net.forward_sequence(spatial[:, 3:], global_vec[:, 3:])
        self.assertTrue(torch.allclose(whole[:, 3:], suffix, atol=1e-5, rtol=1e-5))

    def test_sequence_accepts_cache_from_prior_rollout(self):
        self.net.memory_gate.data.fill_(0.1)
        spatial = torch.randn(1, 6, 5, 9, 36)
        global_vec = torch.rand(1, 6, GLOBAL_SIZE)
        mask = torch.ones(1, 496, dtype=torch.bool)
        caches = self.net.init_caches(1, self.device)
        for index in range(3):
            _, _, caches = self.net.step_logits(
                spatial[:, index], global_vec[:, index], mask, caches
            )
        initial_caches = [(h.clone(), inputs.clone()) for h, inputs in caches]
        expected = []
        for index in range(3, 6):
            logits, _, caches = self.net.step_logits(
                spatial[:, index], global_vec[:, index], mask, caches
            )
            expected.append(logits)

        latent = self.net.forward_sequence(
            spatial[:, 3:],
            global_vec[:, 3:],
            torch.zeros(1, 3, dtype=torch.bool),
            initial_caches,
        )
        actual = self.net._apply_action_mask(
            self.net.actor(latent), mask.unsqueeze(1).expand(-1, 3, -1)
        )
        self.assertTrue(
            torch.allclose(actual, torch.stack(expected, dim=1), atol=1e-5, rtol=1e-5)
        )


class TestNetworkGradients(unittest.TestCase):
    """Tests for gradient flow and masking."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.net = PvZActorCritic().to(self.device)
        self.net.train()
        self.spatial = torch.randn(2, 8, 5, 9, 36, device=self.device)
        self.global_vec = torch.rand(2, 8, GLOBAL_SIZE, device=self.device)
        self.mask = torch.randint(
            0, 2, (2, 8, NUM_ACTIONS), dtype=torch.bool, device=self.device
        )
        self.mask[:, :, 0] = True

    def test_gradients_flow(self):
        _, log_prob, entropy, value = self.net.get_action_and_value(
            self.spatial, self.global_vec, self.mask
        )
        loss = -log_prob.mean() + value.mean() - entropy.mean()
        loss.backward()

        found_grad = False
        for p in self.net.parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                found_grad = True
                break
        self.assertTrue(
            found_grad, "Expected non-zero gradient on at least one parameter"
        )

    def test_residual_can_express_seed_cell_interactions(self):
        latent = torch.zeros(1, 512)
        sunflower_back = 46
        squash_front = 46 + 8 * 45 + 8
        with torch.no_grad():
            self.net.actor.residual.bias[sunflower_back] = 2.0
            self.net.actor.residual.bias[squash_front] = -2.0
        logits = self.net.actor(latent)
        base = logits - self.net.actor.residual(latent)
        self.assertGreater((logits - base)[0, sunflower_back].item(), 0)
        self.assertLess((logits - base)[0, squash_front].item(), 0)

    def test_masked_actions_no_nan_grad(self):
        mask = torch.zeros(2, 8, NUM_ACTIONS, dtype=torch.bool, device=self.device)
        mask[:, :, 0] = True
        _, log_prob, _, _ = self.net.get_action_and_value(
            self.spatial, self.global_vec, mask
        )
        loss = -log_prob.mean()
        loss.backward()
        for p in self.net.parameters():
            if p.grad is not None:
                self.assertFalse(torch.isnan(p.grad).any())


@unittest.skipUnless(
    importlib.util.find_spec("pvz_env") is not None,
    "requires a compiled pvz_env extension on sys.path",
)
class TestNetworkRealEnv(unittest.TestCase):
    """Smoke test against the real PvZ C++ Gymnasium environment."""

    def test_forward_with_real_env_observation(self):
        from pvz_portable_gym import PvZGymEnv

        env = PvZGymEnv()
        try:
            obs, info = env.reset()
            _, _, _, _, _ = env.step(0)  # warm-up one step
            mask = info["action_mask"]

            net = PvZActorCritic()
            net.eval()

            # Sequence mode
            spatial = torch.from_numpy(obs["spatial"]).unsqueeze(0).unsqueeze(0).float()
            global_vec = (
                torch.from_numpy(obs["global"]).unsqueeze(0).unsqueeze(0).float()
            )
            action_mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).bool()

            a, _, _, val = net.get_action_and_value(spatial, global_vec, action_mask)
            self.assertEqual(a.shape, (1, 1))
            self.assertEqual(val.shape, (1, 1, 1))
            self.assertTrue(mask[a.item()])

            # Step mode
            caches = net.init_caches(1, spatial.device)
            a2, _, _, val2, _ = net.forward_step(
                spatial.squeeze(1),
                global_vec.squeeze(1),
                action_mask.squeeze(1),
                caches,
            )
            self.assertEqual(a2.shape, (1,))
            self.assertEqual(val2.shape, (1, 1))
            self.assertTrue(mask[a2.item()])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
