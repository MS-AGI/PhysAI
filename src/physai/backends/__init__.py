"""
physai/backends/__init__.py

Backend registry and factory.

Usage
-----
    from physai.backends import get_backend, list_backends

    backend = get_backend("torch", device="cuda")
    backend = get_backend("jax",   enable_x64=True)
    backend = get_backend("tensorflow", memory_growth=True)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

from .base import AbstractBackend, BackendCapabilities

# Backend name → (import path, class name)
_BACKEND_REGISTRY: Dict[str, tuple] = {
    "torch":       ("physai.backends.torch_backend",       "TorchBackend"),
    "pytorch":     ("physai.backends.torch_backend",       "TorchBackend"),
    "jax":         ("physai.backends.jax_backend",         "JAXBackend"),
    "tensorflow":  ("physai.backends.tensorflow_backend",  "TensorFlowBackend"),
    "tf":          ("physai.backends.tensorflow_backend",  "TensorFlowBackend"),
    "paddlepaddle":     ("physai.backends.paddle_backend",     "PaddleBackend"),
    "paddle":           ("physai.backends.paddle_backend",     "PaddleBackend"),
}

# Singleton cache — one instance per (name, canonical_key)
_INSTANCES: Dict[str, AbstractBackend] = {}


def get_backend(name: str, *, cache: bool = True, **kwargs: Any) -> AbstractBackend:
    """
    Instantiate and return a PhysAI backend by name.

    Parameters
    ----------
    name   : "torch" | "pytorch" | "jax" | "tensorflow" | "tf"
    cache  : If True (default), return the same instance for identical kwargs.
    **kwargs: Passed directly to the backend constructor.

    Returns
    -------
    AbstractBackend subclass instance.

    Raises
    ------
    ValueError  : Unknown backend name.
    ImportError : Required framework not installed.
    """
    key = name.lower()
    if key not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend '{name}'. "
            f"Available: {list(_BACKEND_REGISTRY)}."
        )

    cache_key = f"{key}:{sorted(kwargs.items())}"
    if cache and cache_key in _INSTANCES:
        return _INSTANCES[cache_key]

    module_path, class_name = _BACKEND_REGISTRY[key]
    import importlib
    module = importlib.import_module(module_path)
    cls: Type[AbstractBackend] = getattr(module, class_name)
    instance = cls(**kwargs)

    if cache:
        _INSTANCES[cache_key] = instance

    return instance


def list_backends() -> list:
    """Return a list of registered backend names (deduplicated)."""
    seen = set()
    result = []
    for name, (_, cls_name) in _BACKEND_REGISTRY.items():
        if cls_name not in seen:
            seen.add(cls_name)
            result.append(name)
    return result


def clear_backend_cache() -> None:
    """Evict all cached backend instances."""
    _INSTANCES.clear()


__all__ = [
    "AbstractBackend",
    "BackendCapabilities",
    "get_backend",
    "list_backends",
    "clear_backend_cache",
]
