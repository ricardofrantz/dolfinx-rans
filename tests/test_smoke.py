"""
Smoke tests for dolfinx-rans.

These tests verify basic functionality without running full simulations.
"""

import numpy as np
import pytest


def test_validation_data_loads():
    """Verify DNS validation data is accessible."""
    from dolfinx_rans.validation import (
        RE_TAU_590,
        MEAN_VELOCITY_590,
        REYNOLDS_STRESSES_590,
        get_k_profile_590,
    )

    assert RE_TAU_590 == 590
    assert MEAN_VELOCITY_590.shape[1] == 2  # y+, U+
    assert REYNOLDS_STRESSES_590.shape[1] == 5  # y+, uu+, vv+, ww+, uv+

    y_plus, k_plus = get_k_profile_590()
    assert len(y_plus) == len(k_plus)
    assert np.max(k_plus) > 0  # TKE should be positive


def test_validation_data_re180():
    """Verify Re_τ=180 data loads."""
    from dolfinx_rans.validation import RE_TAU_180, get_k_profile_180

    assert RE_TAU_180 == 180
    y_plus, k_plus = get_k_profile_180()
    assert len(y_plus) > 0


def test_dataclass_validation():
    """Verify config dataclass validation works."""
    from dolfinx_rans.utils import dc_from_dict
    from dolfinx_rans.solver import ChannelGeom

    # Valid config
    valid = {
        "Lx": 6.28,
        "Ly": 2.0,
        "Nx": 32,
        "Ny": 48,
        "mesh_type": "triangle",
        "y_first": 0.002,
        "growth_rate": 1.1,
    }
    geom = dc_from_dict(ChannelGeom, valid, name="geom")
    assert geom.Lx == 6.28

    # Missing key should raise
    invalid = {"Lx": 6.28}  # Missing required fields
    with pytest.raises(ValueError, match="Missing keys"):
        dc_from_dict(ChannelGeom, invalid, name="geom")

    # Unknown key should raise
    unknown = dict(valid)
    unknown["extra"] = 123
    with pytest.raises(ValueError, match="Unknown keys"):
        dc_from_dict(ChannelGeom, unknown, name="geom")


def test_diagnostics_helpers():
    """Verify formatting helpers work."""
    from dolfinx_rans.utils import fmt_sci, fmt_pair_sci

    assert "1.0e+00" in fmt_sci(1.0, prec=1)
    assert "nan" in fmt_sci(float("nan"))

    pair = fmt_pair_sci(1e-3, 1e3, prec=1)
    assert "," in pair


def test_mesh_stretch_coords():
    """Verify stretched coordinate generation."""
    from dolfinx_rans.solver import _generate_stretched_coords

    y = _generate_stretched_coords(y_first=0.01, H=1.0, N=10, growth=1.2)

    assert y[0] == 0.0
    assert y[-1] == 1.0
    assert len(y) == 11  # N+1 points
    assert np.all(np.diff(y) > 0)  # Monotonically increasing


def _can_import_dolfinx():
    """Check if DOLFINx is available."""
    try:
        import dolfinx
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _can_import_dolfinx(),
    reason="DOLFINx not available"
)
def test_mesh_creation():
    """Test mesh creation (requires DOLFINx)."""
    from dolfinx_rans.solver import ChannelGeom, create_channel_mesh

    geom = ChannelGeom(
        Lx=1.0, Ly=1.0, Nx=4, Ny=4,
        mesh_type="triangle", y_first=0.0, growth_rate=1.0
    )
    domain = create_channel_mesh(geom)
    assert domain.topology.dim == 2
