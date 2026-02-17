"""
Turbulence model registry.

Usage:
    from dolfinx_rans.models import create_model
    model = create_model("wilcox2006")
"""

from dolfinx_rans.models.base import FieldSpec, FormCoefficients, RANSModel
from dolfinx_rans.models.kepsilon import KepsilonModel
from dolfinx_rans.models.sst import SSTModel
from dolfinx_rans.models.wilcox2006 import Wilcox2006Model

_REGISTRY: dict[str, type[RANSModel]] = {
    "wilcox2006": Wilcox2006Model,
    "sst": SSTModel,
    "kepsilon": KepsilonModel,
}


def create_model(name: str) -> RANSModel:
    """Factory: instantiate a turbulence model by config name."""
    key = name.lower()
    cls = _REGISTRY.get(key)
    if cls is None:
        supported = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown turbulence model '{name}'. Supported: {supported}"
        )
    return cls()


__all__ = [
    "RANSModel",
    "FormCoefficients",
    "FieldSpec",
    "create_model",
    "Wilcox2006Model",
    "SSTModel",
    "KepsilonModel",
]
