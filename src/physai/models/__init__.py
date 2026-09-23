"""
physai/models/__init__.py
"""
from .pinn import PINN, build_pinn, FourierEmbedding, DirichletConstraint, OutputScaler
from .fno  import FNO,  build_fno,  FNOLayer, SpectralConv, ComplexWeight
from .spectral_element import (
    TensorCPMath, DynamicLossBalancer, USENOCPModule, build_separable_linear_operator,
)
from .spectralpinn import USENOPINNTrainer, build_spectral_element_trainer

__all__ = [
    "PINN", "build_pinn", "FourierEmbedding", "DirichletConstraint", "OutputScaler",
    "FNO",  "build_fno",  "FNOLayer", "SpectralConv", "ComplexWeight",
    "TensorCPMath", "DynamicLossBalancer", "USENOCPModule", "build_separable_linear_operator",
    "USENOPINNTrainer", "build_spectral_element_trainer",
]