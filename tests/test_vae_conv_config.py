"""CPU/meta tests for decoder-only configuration, atomic validation and routing."""

import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
from vae_loader import load_vae_module

VAE = load_vae_module()
FIRST = 'decoder.upsamples.2.upsamples.0.residual.2'
SECOND = 'decoder.upsamples.2.upsamples.0.residual.6'


def config(layers=None, enabled=True):
    return dict(schema_version=1, enabled=enabled, layers=layers or {})


class DecoderConfigTest(unittest.TestCase):
    def setUp(self):
        with torch.device('meta'):
            self.model = VAE.WanVAE_(z_dim=48, temperal_downsample=[False, True, True]).eval()

    def test_exact_selection_preserves_checkpoint_and_cache_contract(self):
        parameters = dict(self.model.named_parameters())
        keys = list(self.model.state_dict())
        self.model.clear_cache()
        cache = self.model._feat_map
        plan = self.model.configure_decoder_convolutions(config({FIRST: 'winograd_2d', SECOND: 'winograd_3d'}))
        self.assertEqual(len(plan), 34)
        self.assertEqual({n: b for n, b in plan.items() if b != 'native'},
                         {FIRST: 'winograd_2d', SECOND: 'winograd_3d'})
        self.assertTrue(all(m._conv_backend == 'native' for m in self.model.encoder.modules()
                            if isinstance(m, VAE.CausalConv3d)))
        self.assertEqual(self.model.conv2._conv_backend, 'native')
        self.assertEqual(list(self.model.state_dict()), keys)
        self.assertTrue(all(p is parameters[n] for n, p in self.model.named_parameters()))
        self.assertIs(self.model._feat_map, cache)
        self.assertEqual(len(cache), 34)

    def test_new_config_replaces_previous_selection_and_disabled_resets(self):
        self.model.configure_decoder_convolutions(config({FIRST: 'winograd_2d'}))
        plan = self.model.configure_decoder_convolutions(config({SECOND: 'winograd_3d'}))
        self.assertEqual(plan[FIRST], 'native')
        self.assertEqual(plan[SECOND], 'winograd_3d')
        plan = self.model.configure_decoder_convolutions(config({FIRST: 'winograd_2d'}, enabled=False))
        self.assertEqual(set(plan.values()), {'native'})
        self.assertEqual(set(self.model.configure_decoder_convolutions(None).values()), {'native'})

    def test_bad_target_is_atomic_even_with_valid_targets_first(self):
        before = self.model.configure_decoder_convolutions(config({SECOND: 'winograd_3d'}))
        with self.assertRaisesRegex(ValueError, 'Unknown decoder'):
            self.model.configure_decoder_convolutions(config({FIRST: 'winograd_2d', 'decoder.misspelled': 'winograd_2d'}))
        self.assertEqual(self.model.decoder_conv_plan(), before)

    def test_rejects_non_target_layers_and_unsupported_shapes(self):
        for name in ['encoder.conv1', 'conv2', 'decoder.middle.0',
                     'decoder.upsamples.*.upsamples.0.residual.2',
                     'decoder.upsamples.2.upsamples.0.shortcut',
                     'decoder.upsamples.0.upsamples.3.time_conv']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.model.configure_decoder_convolutions(config({name: 'winograd_2d'}))
        self.assertEqual(set(self.model.decoder_conv_plan().values()), {'native'})

    def test_disabled_config_still_validates_layer_names(self):
        with self.assertRaises(ValueError):
            self.model.configure_decoder_convolutions(config({'decoder.typo': 'winograd_3d'}, enabled=False))

    def test_schema_validation(self):
        bad = [{}, [], config({FIRST: '2d'}), dict(config(), extra=True),
               dict(config(), schema_version=True), dict(config(), schema_version=2),
               dict(config(), enabled='false'), dict(config(), layers=[])]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                VAE.load_conv_config(value)

    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'bad.json'
            path.write_text('{"schema_version":1,"enabled":false,"enabled":true,"layers":{}}')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                VAE.load_conv_config(path)

    def test_configuration_does_not_alias_caller_dict(self):
        original = config({FIRST: 'winograd_2d'})
        self.model.configure_decoder_convolutions(original)
        original['layers'][FIRST] = 'winograd_3d'
        self.assertEqual(self.model.decoder_conv_plan()[FIRST], 'winograd_2d')

    def test_spatial_full_and_mixed_decode_preserve_meta_shapes(self):
        for modes in [('winograd_2d', 'winograd_2d'),
                      ('winograd_3d', 'winograd_3d'),
                      ('winograd_2d', 'winograd_3d')]:
            self.model.configure_decoder_convolutions(config(dict(zip((FIRST, SECOND), modes))))
            with self.subTest(modes=modes), torch.no_grad():
                output = self.model.decode(torch.empty(1, 48, 3, 2, 3, device='meta'), [0, 1])
                self.assertEqual(tuple(output.shape), (1, 3, 9, 32, 48))

    def test_example_configs_resolve(self):
        for path in sorted((ROOT / 'configs/vae_conv').glob('*.json')):
            with self.subTest(path=path):
                self.model.configure_decoder_convolutions(path)


if __name__ == '__main__':
    unittest.main()
