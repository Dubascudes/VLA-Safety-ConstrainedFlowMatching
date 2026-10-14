"""CPU regressions against real extracted policy/solver code (no checkpoint).

The tiny model is a test double for neural inference, not a benchmark result.
Run using an OpenPI-capable environment after materialization.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.environ.get('VLA_RUNTIME', ROOT/'runtime'))
sys.path[:0] = [str(RUNTIME/'openpi/src'), str(RUNTIME/'openpi/packages/openpi-client/src')]
READY = all(importlib.util.find_spec(m) for m in ['torch', 'numpy', 'flax', 'jax']) and (RUNTIME/'openpi/src').is_dir()
if READY:
    import numpy as np
    import torch
    from openpi.policies.policy import Policy
    from openpi.models_pytorch.pi0_pytorch import SafetyConfig, SafetyGuidance

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.safety_guide = SafetyGuidance(SafetyConfig())
            self._last_trace = None
            self.seen = []

        def sample_actions(self, device, observation, **kwargs):
            cfg = self.safety_guide.config
            values = [cfg.safety_margin, getattr(cfg, 'guidance_scale', 1.0)]
            self.seen.append((values, kwargs))
            if kwargs.get('collect_trace'):
                self._last_trace = types.SimpleNamespace(to_dict=lambda: {'nested': {'array': np.array(values)}, 'step': np.int64(len(self.seen))})
            return torch.tensor([[values]], dtype=torch.float32)

    def observation(**safety):
        return {'state': np.zeros(3, dtype=np.float32), 'image': {}, 'image_mask': {}, '_safety_eef_pos': [0., 0., 0.], **safety}


@unittest.skipUnless(READY, 'requires materialized source and OpenPI CPU dependencies')
class PolicyRegressionTests(unittest.TestCase):
    def test_viewer_parameters_reach_sampling_without_mutating_input(self):
        model = TinyModel()
        policy = Policy(model, is_pytorch=True)
        obs = observation(_safety_margin=0.04, _safety_guidance_scale=0.5)
        result = policy.infer(obs)
        np.testing.assert_allclose(result['actions'][0], [0.04, 0.5])
        self.assertIn('_safety_margin', obs)
        self.assertAlmostEqual(model.safety_guide.config.safety_margin, 0.01)
        policy.infer(observation())
        np.testing.assert_allclose(model.seen[-1][0], [0.01, 1.0])

    def test_scale_changes_actual_solver_correction(self):
        from openpi.models_pytorch.pi0_pytorch import ObstacleGeometry
        cfg = SafetyConfig(eef_ellipsoid_radii=(0.01, 0.01, 0.01))
        guide = SafetyGuidance(cfg)
        obstacle = ObstacleGeometry(2, np.array([0.1, 0., 0.]), np.eye(3), np.array([0.02, 0., 0.]))
        actions = torch.zeros((1, 3, 7), dtype=torch.float64)
        actions[:, :, 0] = 4.
        origin = torch.zeros(3, dtype=torch.float64)
        full = guide.compute_guidance(actions, 0.5, origin, [obstacle])
        self.assertGreater(torch.linalg.vector_norm(full).item(), 0.)
        cfg.guidance_scale = 0.5
        half = guide.compute_guidance(actions, 0.5, origin, [obstacle])
        torch.testing.assert_close(half, full * 0.5)
        cfg.guidance_scale = 0.
        torch.testing.assert_close(guide.compute_guidance(actions, 0.5, origin, [obstacle]), torch.zeros_like(full))

    def test_invalid_controls_do_not_change_defaults(self):
        model = TinyModel()
        policy = Policy(model, is_pytorch=True)
        for key in ['_safety_margin', '_safety_guidance_scale']:
            for value in [-1., float('nan'), float('inf')]:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    policy.infer(observation(**{key: value}))
        self.assertEqual(model.seen, [])
        self.assertEqual(model.safety_guide.config.safety_margin, 0.01)

    def test_trace_is_opt_in_and_current_call_is_saved(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)/'traces'
            model = TinyModel()
            policy = Policy(model, is_pytorch=True, trace_dir=target)
            policy.infer(observation(_safety_margin=0.04))
            paths = list(target.glob('*.json'))
            self.assertEqual(len(paths), 1)
            saved = json.loads(paths[0].read_text())
            self.assertEqual(saved['step'], 1)
            self.assertEqual(saved['nested']['array'], [0.04, 1.0])
            self.assertIsNone(model._last_trace)
            policy.infer(observation())
            self.assertEqual(len(list(target.glob('*.json'))), 2)
            default_model = TinyModel()
            Policy(default_model, is_pytorch=True).infer(observation())
            self.assertFalse(default_model.seen[0][1]['collect_trace'])

    def test_server_trace_option_and_neutral_defaults(self):
        import importlib.util
        from unittest.mock import patch
        spec = importlib.util.spec_from_file_location('release_serve_policy', RUNTIME/'openpi/scripts/serve_policy.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        args = module.Args()
        self.assertEqual(args.guidance_scale, 1.0)
        self.assertEqual(args.safety_margin, 0.01)
        with tempfile.TemporaryDirectory() as temp:
            args.trace_dir = temp
            model = TinyModel()
            policy = Policy(model, is_pytorch=True)
            with patch.object(module, 'create_policy', return_value=policy), patch.object(module.websocket_policy_server, 'WebsocketPolicyServer') as server, patch.object(module.socket, 'gethostbyname', return_value='127.0.0.1'):
                module.main(args)
            server.return_value.serve_forever.assert_called_once()
            self.assertEqual(policy._trace_dir, Path(temp).resolve())

    def test_trace_toggle_preserves_actual_sampler_output(self):
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, ObstacleGeometry
        obstacle = ObstacleGeometry(2, np.array([0.1, 0., 0.]), np.eye(3), np.array([0.02, 0., 0.]))
        noise = torch.zeros((1, 3, 7), dtype=torch.float32)
        noise[:, :, 0] = 4.
        obs = types.SimpleNamespace(state=torch.zeros((1, 3)))
        expert = types.SimpleNamespace(
            paligemma=types.SimpleNamespace(language_model=types.SimpleNamespace(config=types.SimpleNamespace())),
            forward=lambda **kwargs: (None, None))
        # Only neural forward operations are replaced; execute the real sampler
        # loops, schedules, geometry, optimization, and tracing branches.
        model = types.SimpleNamespace(
            paligemma_with_expert=expert,
            _preprocess_observation=lambda observation, **kwargs: (None, None, None, None, observation.state),
            embed_prefix=lambda *args: (torch.zeros((1, 1, 2)), torch.ones((1, 1), dtype=torch.bool), torch.zeros((1, 1), dtype=torch.bool)),
            _prepare_attention_masks_4d=lambda mask: mask,
            denoise_step=lambda state, pads, cache, actions, time: torch.zeros_like(actions))
        for schedule in ['flat', 'posthoc']:
            for scale in [0., 0.5, 1.]:
                with self.subTest(schedule=schedule, scale=scale):
                    model.safety_guide = SafetyGuidance(SafetyConfig(eef_ellipsoid_radii=(0.01, 0.01, 0.01), guidance_schedule=schedule, guidance_scale=scale))
                    kwargs = dict(noise=noise.clone(), num_steps=2, eef_pos=np.zeros(3), obstacles=[obstacle])
                    plain = PI0Pytorch.sample_actions(model, 'cpu', obs, collect_trace=False, **kwargs)
                    traced = PI0Pytorch.sample_actions(model, 'cpu', obs, collect_trace=True, **kwargs)
                    torch.testing.assert_close(traced, plain)

    def test_invalid_server_defaults_are_rejected(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('validated_serve_policy', RUNTIME/'openpi/scripts/serve_policy.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        for field in ['guidance_scale', 'safety_margin']:
            for value in [-1., float('nan'), float('inf')]:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    module.Args(**{field: value})
                with self.assertRaises(ValueError):
                    SafetyConfig(**{field: value})

if __name__ == '__main__':
    unittest.main()
