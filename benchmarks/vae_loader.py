"""Load just the VAE modules, without importing the full generation pipeline."""

import importlib
import sys
import types
from pathlib import Path


def load_vae_module():
    name = '_wan22_benchmark_modules'
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(Path(__file__).resolve().parents[1] / 'wan/modules')]
        sys.modules[name] = package
    return importlib.import_module(name + '.vae2_2')
