"""
tests/test_register_pde.py

Tests for the user-extensible PDE registry (register_pde / unregister_pde).
Backend-free: the registry only needs a placeholder object as ``backend``.
"""

import warnings

import pytest

from physai.core import auto_optimizer as ao
from physai.core.pde_residual import (
    PDE_REGISTRY,
    PDEResidual,
    build_residual,
    register_pde,
    unregister_pde,
)

_BACKEND = object()


@pytest.fixture(autouse=True)
def _clean_registry():
    """Other suites iterate over PDE_REGISTRY: leave it exactly as found."""
    reg_before = dict(PDE_REGISTRY)
    meta_before = dict(ao._PDE_META)
    yield
    PDE_REGISTRY.clear()
    PDE_REGISTRY.update(reg_before)
    ao._PDE_META.clear()
    ao._PDE_META.update(meta_before)
    from physai.core import pde_residual as pr
    pr._USER_PDE_NAMES.intersection_update(reg_before)
    pr._BUILTIN_META_BACKUP.clear()


def _make(name="MyPDE"):
    class _C(PDEResidual):
        def __call__(self, model_fn, points):
            return model_fn(points)
    _C.__name__ = _C.__qualname__ = name
    return _C


def test_register_function_and_build():
    C = _make()
    assert register_pde("my_pde", C, meta={"order": 3, "nonlinear": True}) is C
    r = build_residual("MY_PDE", _BACKEND, k=2.0)
    assert isinstance(r, C) and r.params == {"k": 2.0}
    m = ao.get_pde_meta("my_pde")
    assert (m.name, m.order, m.nonlinear, m.recommended_arch) == ("my_pde", 3, True, "pinn")


def test_decorator_form_and_default_meta_no_warning():
    @register_pde("deco_pde")
    class D(PDEResidual):
        def __call__(self, model_fn, points):
            return points
    assert PDE_REGISTRY["deco_pde"] is D
    assert ao.get_pde_meta("deco_pde") is not None   # defaults -> no fallback warning


def test_class_level_PDE_META_attribute_used():
    C = _make()
    C.PDE_META = ao.PDEMeta("ignored", 4, True, 2, True, "high", "fno")
    register_pde("attr_pde", C)
    m = ao.get_pde_meta("attr_pde")
    assert (m.name, m.order, m.n_components, m.recommended_arch) == ("attr_pde", 4, 2, "fno")


def test_aliases_share_class_and_get_own_meta():
    C = _make()
    register_pde("main_pde", C, aliases=("alt_pde", "Other_PDE"))
    for k in ("main_pde", "alt_pde", "other_pde"):
        assert PDE_REGISTRY[k] is C and ao.get_pde_meta(k).name == k


@pytest.mark.parametrize("bad", ["", "  ", "1abc", "has space", "a-b", "a.b", "é"])
def test_invalid_names(bad):
    with pytest.raises(ValueError):
        register_pde(bad, _make())


def test_non_string_name_and_bad_classes():
    with pytest.raises(TypeError):
        register_pde(123, _make())
    with pytest.raises(TypeError):
        register_pde("x1", _make()(_BACKEND))            # instance, not class
    with pytest.raises(TypeError):
        register_pde("x2", int)                           # not a PDEResidual
    with pytest.raises(TypeError):
        register_pde("x3", PDEResidual)                   # base class
    class NoCall(PDEResidual):
        pass
    with pytest.raises(TypeError, match="__call__"):
        register_pde("x4", NoCall)
    class BadInit(PDEResidual):
        def __init__(self):
            pass
        def __call__(self, model_fn, points):
            return points
    with pytest.raises(TypeError, match="backend"):
        register_pde("x5", BadInit)
    assert not {"x1", "x2", "x3", "x4", "x5"} & set(PDE_REGISTRY)


def test_collision_and_overwrite():
    A, B = _make("A"), _make("B")
    register_pde("dup_pde", A)
    with pytest.raises(ValueError, match="already registered"):
        register_pde("dup_pde", B)
    assert PDE_REGISTRY["dup_pde"] is A
    register_pde("dup_pde", B, overwrite=True)
    assert PDE_REGISTRY["dup_pde"] is B


def test_notebook_style_redefinition_allowed():
    register_pde("nb_pde", _make("Same"))
    new = _make("Same")                                   # same module + qualname
    register_pde("nb_pde", new)
    assert PDE_REGISTRY["nb_pde"] is new


def test_builtin_protected_and_restorable():
    orig_cls, orig_meta = PDE_REGISTRY["burgers"], ao.get_pde_meta("burgers")
    with pytest.raises(ValueError, match="built-in"):
        register_pde("burgers", _make())
    with pytest.raises(ValueError, match="built-in"):
        unregister_pde("burgers")
    C = _make()
    register_pde("burgers", C, overwrite=True, meta={"order": 5})
    assert PDE_REGISTRY["burgers"] is C and ao.get_pde_meta("burgers").order == 5
    unregister_pde("burgers")
    assert PDE_REGISTRY["burgers"] is orig_cls and ao.get_pde_meta("burgers") == orig_meta


def test_overwrite_without_meta_keeps_existing_meta():
    register_pde("keep_pde", _make(), meta={"order": 4, "stiff": True})
    register_pde("keep_pde", _make("Other"), overwrite=True)
    m = ao.get_pde_meta("keep_pde")
    assert m.order == 4 and m.stiff is True


def test_atomic_alias_collision():
    with pytest.raises(ValueError):
        register_pde("fresh_pde", _make(), aliases=("burgers",))
    assert "fresh_pde" not in PDE_REGISTRY and ao.get_pde_meta("fresh_pde") is None


def test_atomic_duplicate_names():
    with pytest.raises(ValueError, match="Duplicate"):
        register_pde("same_pde", _make(), aliases=("SAME_PDE",))
    assert "same_pde" not in PDE_REGISTRY


@pytest.mark.parametrize("meta", [
    {"order": 0}, {"order": True}, {"n_components": 0}, {"nonlinear": "yes"},
    {"spectral_bias": "huge"}, {"recommended_arch": "cnn"}, {"bogus_field": 1},
    42, "pinn",
])
def test_invalid_meta_rejected_atomically(meta):
    with pytest.raises((TypeError, ValueError)):
        register_pde("meta_pde", _make(), meta=meta)
    assert "meta_pde" not in PDE_REGISTRY and ao.get_pde_meta("meta_pde") is None


def test_unregister_user_pde():
    register_pde("gone_pde", _make(), aliases=["gone_alias"])
    unregister_pde("gone_pde")
    assert "gone_pde" not in PDE_REGISTRY and ao.get_pde_meta("gone_pde") is None
    assert "gone_alias" in PDE_REGISTRY                    # alias is its own entry
    unregister_pde("gone_alias")
    with pytest.raises(KeyError):
        unregister_pde("gone_alias")
    unregister_pde("gone_alias", missing_ok=True)


def test_decorator_fails_fast_on_bad_name():
    with pytest.raises(ValueError):
        register_pde("bad name")
    with pytest.raises(TypeError):
        register_pde(_make())                              # forgot the name


def test_build_residual_errors_are_helpful():
    with pytest.raises(ValueError, match="Did you mean"):
        build_residual("burger", _BACKEND)
    with pytest.raises(TypeError):
        build_residual(None, _BACKEND)
    assert isinstance(build_residual("  Burgers ", _BACKEND), PDE_REGISTRY["burgers"])


def test_autooptimizer_no_fallback_warning_for_registered_pde():
    register_pde("opt_pde", _make(), meta={"order": 2})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ao.get_pde_meta("opt_pde") is not None


def test_registered_spectral_element_recommendation_is_auto_selectable():
    register_pde(
        "spectral_opt_pde", _make(),
        meta={"recommended_arch": "spectral_element"},
    )
    meta = ao.get_pde_meta("spectral_opt_pde")
    timed = ao.ProblemSpec(
        pde_name="spectral_opt_pde",
        domain=ao.DomainSpec(spatial_dims=1, bounds=[(-1.0, 1.0)], time_domain=(0.0, 1.0)),
    )
    steady = ao.ProblemSpec(
        pde_name="spectral_opt_pde",
        domain=ao.DomainSpec(spatial_dims=1, bounds=[(-1.0, 1.0)]),
    )
    assert ao._select_model_arch(timed, meta, _BACKEND) == "spectral_element"
    assert ao._select_model_arch(steady, meta, _BACKEND) == "pinn"
