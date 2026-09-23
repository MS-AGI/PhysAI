"""
tests/test_fno.py

Tests for physai.models.fno: ComplexWeight, SpectralConv, FNOLayer, FNO,
build_fno, and the complex-tensor helper functions. TorchBackend only.
"""
import numpy as np
import pytest
import torch

from physai.backends.torch_backend import TorchBackend
from physai.models.fno import (
    FNO,
    build_fno,
    FNOLayer,
    SpectralConv,
    ComplexWeight,
    _split_complex,
    _join_complex,
    _embed_modes,
)

backend = TorchBackend(device="cpu")


# ---------------------------------------------------------------------------
# ComplexWeight
# ---------------------------------------------------------------------------

class TestComplexWeight:
    def test_shapes(self):
        w = ComplexWeight(backend, shape=(3, 4, 5))
        assert w.re.shape == (3, 4, 5)
        assert w.im.shape == (3, 4, 5)

    def test_as_complex_returns_re_im_tuple(self):
        w = ComplexWeight(backend, shape=(2, 2))
        re, im = w.as_complex()
        assert re is w.re and im is w.im

    def test_parameters_returns_re_and_im(self):
        w = ComplexWeight(backend, shape=(2, 2))
        params = w.parameters()
        assert params == [w.re, w.im]

    def test_parameters_are_leaf_and_require_grad(self):
        w = ComplexWeight(backend, shape=(2, 2))
        assert w.re.requires_grad and w.im.requires_grad
        assert isinstance(w.re, torch.nn.Parameter)


# ---------------------------------------------------------------------------
# Complex helpers
# ---------------------------------------------------------------------------

class TestComplexHelpers:
    def test_split_join_roundtrip(self):
        z = torch.randn(4, 5, dtype=torch.cfloat)
        re, im = _split_complex(backend, z)
        z2 = _join_complex(backend, re, im)
        torch.testing.assert_close(z, z2)

    def test_split_matches_real_imag_attributes(self):
        z = torch.tensor([1.0 + 2.0j, 3.0 - 1.0j])
        re, im = _split_complex(backend, z)
        np.testing.assert_allclose(re.numpy(), [1.0, 3.0])
        np.testing.assert_allclose(im.numpy(), [2.0, -1.0])

    def test_embed_modes_places_low_block_and_zero_pads(self):
        low = torch.ones((1, 1, 2, 3))
        full = torch.zeros((1, 1, 5, 7))
        out = _embed_modes(backend, low, full, modes=(2, 3))
        assert out.shape == (1, 1, 5, 7)
        # low block preserved
        np.testing.assert_allclose(out[0, 0, :2, :3].numpy(), np.ones((2, 3)))
        # everything else zero-padded
        assert torch.all(out[0, 0, 2:, :] == 0)
        assert torch.all(out[0, 0, :, 3:] == 0)


# ---------------------------------------------------------------------------
# SpectralConv
# ---------------------------------------------------------------------------

class TestSpectralConv1D:
    def test_forward_shape_preserved(self):
        conv = SpectralConv(backend, in_channels=4, out_channels=6, modes=(8,))
        v = torch.randn(2, 4, 32)
        out = conv(v)
        assert out.shape == (2, 6, 32)

    def test_output_is_real_and_finite(self):
        conv = SpectralConv(backend, in_channels=2, out_channels=2, modes=(4,))
        v = torch.randn(1, 2, 16)
        out = conv(v)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()

    def test_parameters_length_two(self):
        conv = SpectralConv(backend, in_channels=3, out_channels=3, modes=(4,))
        assert len(conv.parameters()) == 2


class TestSpectralConv2D:
    def test_forward_shape_preserved(self):
        conv = SpectralConv(backend, in_channels=3, out_channels=5, modes=(4, 4))
        v = torch.randn(2, 3, 16, 16)
        out = conv(v)
        assert out.shape == (2, 5, 16, 16)

    def test_output_finite(self):
        conv = SpectralConv(backend, in_channels=2, out_channels=2, modes=(3, 3))
        v = torch.randn(1, 2, 12, 12)
        out = conv(v)
        assert torch.isfinite(out).all()


class TestSpectralConvND:
    def test_3d_forward_shape_via_generic_path(self):
        conv = SpectralConv(backend, in_channels=2, out_channels=2, modes=(3, 3, 3))
        v = torch.randn(1, 2, 8, 8, 8)
        out = conv(v)
        assert out.shape == (1, 2, 8, 8, 8)
        assert torch.isfinite(out).all()

    def test_ndim_dispatch(self):
        conv1 = SpectralConv(backend, 2, 2, modes=(4,))
        conv2 = SpectralConv(backend, 2, 2, modes=(4, 4))
        conv3 = SpectralConv(backend, 2, 2, modes=(2, 2, 2))
        assert conv1.ndim == 1
        assert conv2.ndim == 2
        assert conv3.ndim == 3


# ---------------------------------------------------------------------------
# FNOLayer
# ---------------------------------------------------------------------------

class TestFNOLayer:
    def test_forward_shape_preserved_1d(self):
        layer = FNOLayer(backend, width=8, modes=(6,), activation="gelu")
        v = torch.randn(2, 8, 20)
        out = layer(v)
        assert out.shape == (2, 8, 20)

    def test_forward_shape_preserved_2d(self):
        layer = FNOLayer(backend, width=6, modes=(4, 4), activation="relu")
        v = torch.randn(2, 6, 10, 10)
        out = layer(v)
        assert out.shape == (2, 6, 10, 10)

    @pytest.mark.parametrize("act", ["tanh", "relu", "gelu", "silu"])
    def test_all_activation_names(self, act):
        layer = FNOLayer(backend, width=4, modes=(3,), activation=act)
        v = torch.randn(1, 4, 12)
        out = layer(v)
        assert torch.isfinite(out).all()

    def test_unknown_activation_falls_back_to_tanh(self):
        layer = FNOLayer(backend, width=4, modes=(3,), activation="not_a_real_one")
        v = torch.randn(1, 4, 12)
        out = layer(v)
        assert torch.isfinite(out).all()

    def test_parameters_include_spectral_and_local_linear(self):
        layer = FNOLayer(backend, width=4, modes=(3,), activation="tanh")
        params = layer.parameters()
        spectral_params = layer.spectral.parameters()
        w_params = backend.parameters(layer._W)
        assert len(params) == len(spectral_params) + len(w_params)


# ---------------------------------------------------------------------------
# FNO — full model
# ---------------------------------------------------------------------------

class TestFNO1D:
    def test_forward_shape(self):
        fno = FNO(backend, n_input_channels=1, n_output_channels=1, modes=(8,), width=16, n_layers=2)
        v = torch.randn(2, 1, 32)
        out = fno.forward(v)
        assert out.shape == (2, 1, 32)

    def test_append_grid_false_still_runs(self):
        fno = FNO(backend, n_input_channels=1, n_output_channels=1, modes=(8,), width=16,
                  n_layers=2, append_grid=False)
        v = torch.randn(2, 1, 32)
        out = fno.forward(v)
        assert out.shape == (2, 1, 32)

    def test_multi_channel_io(self):
        fno = FNO(backend, n_input_channels=3, n_output_channels=2, modes=(6,), width=8, n_layers=2)
        v = torch.randn(1, 3, 24)
        out = fno.forward(v)
        assert out.shape == (1, 2, 24)

    def test_output_finite(self):
        fno = FNO(backend, n_input_channels=1, n_output_channels=1, modes=(4,), width=8, n_layers=1)
        v = torch.randn(1, 1, 16)
        out = fno.forward(v)
        assert torch.isfinite(out).all()


class TestFNO2D:
    def test_forward_shape(self):
        fno = FNO(backend, n_input_channels=1, n_output_channels=1, modes=(4, 4), width=8, n_layers=2)
        v = torch.randn(1, 1, 16, 16)
        out = fno.forward(v)
        assert out.shape == (1, 1, 16, 16)


class TestFNOModelFn:
    def test_pointwise_model_fn_shape(self):
        # model_fn treats each point as a 1-point grid (X=1), so
        # X//2+1 == 1 — modes must not exceed that or SpectralConv's
        # zero-pad size (X//2+1-m) goes negative.
        fno = FNO(backend, n_input_channels=2, n_output_channels=1, modes=(1,), width=8,
                  n_layers=1, append_grid=False)
        x = torch.randn(10, 2)
        out = fno.model_fn(x)
        assert out.shape == (10, 1)


class TestFNOParametersAndRepr:
    def test_parameters_nonempty(self):
        fno = FNO(backend, n_input_channels=1, n_output_channels=1, modes=(4,), width=8, n_layers=2)
        params = fno.parameters()
        assert len(params) > 0
        assert all(isinstance(p, torch.Tensor) for p in params)

    def test_repr_contains_key_fields(self):
        fno = FNO(backend, n_input_channels=1, n_output_channels=1, modes=(4, 4), width=16, n_layers=3)
        r = repr(fno)
        assert "torch" in r
        assert "ndim=2" in r
        assert "layers=3" in r


# ---------------------------------------------------------------------------
# build_fno factory
# ---------------------------------------------------------------------------

class TestBuildFnoFactory:
    def test_factory_matches_direct_construction_shape(self):
        fno = build_fno(backend, n_input_channels=1, n_output_channels=1, modes=(4, 4), width=8, n_layers=2)
        v = torch.randn(1, 1, 12, 12)
        out = fno.forward(v)
        assert out.shape == (1, 1, 12, 12)

    def test_factory_default_kwargs(self):
        fno = build_fno(backend, n_input_channels=1, n_output_channels=1, modes=(4,))
        assert fno.width == 64
        assert fno.n_layers == 4