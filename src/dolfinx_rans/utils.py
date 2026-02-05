"""
Utility functions for dolfinx-rans.

Provides config loading, diagnostics, and logging helpers.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, TypeVar

T = TypeVar("T")


def load_json_config(config_path: str | Path) -> dict[str, Any]:
    """Load JSON configuration file."""
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    return json.loads(p.read_text())


def dc_from_dict(cls: type[T], data: Mapping[str, Any] | None, *, name: str = "config") -> T:
    """
    Build a dataclass instance from a mapping with strict validation.

    Checks for unknown keys and missing required fields.
    Keys starting with "_" are ignored (allows JSON comments).
    """
    data = {} if data is None else dict(data)
    # Allow JSON "comments" by ignoring keys starting with "_"
    data = {k: v for k, v in data.items() if not str(k).startswith("_")}

    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data.keys()) - field_names)
    if unknown:
        raise ValueError(f"Unknown keys in {name}: {unknown}")

    required = {
        f.name
        for f in dataclasses.fields(cls)
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    }
    missing = sorted(required - set(data.keys()))
    if missing:
        raise ValueError(f"Missing keys in {name}: {missing}")

    return cls(**data)


def print_dc_json(obj: Any) -> None:
    """Print a dataclass (or dict) as stable, sorted JSON."""
    if dataclasses.is_dataclass(obj):
        payload = dataclasses.asdict(obj)
    else:
        payload = obj
    print(json.dumps(payload, indent=2, sort_keys=True))


@dataclasses.dataclass(frozen=True)
class CasePaths:
    case_dir: Path
    snps_dir: Path
    history_csv: Path
    run_info_json: Path
    config_used_json: Path


def _try_git_info(start_dir: Path) -> dict[str, str] | None:
    try:
        top = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(start_dir),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=top,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=top,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return {"root": top, "sha": sha, "dirty": "1" if dirty else "0"}
    except Exception:
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def prepare_case_dir(
    out_dir: str | Path,
    *,
    config_path: Path | None,
    cfg: Mapping[str, Any],
    snps_subdir: str = "snps",
) -> CasePaths:
    """
    Create results folder structure and write reproducibility metadata.

    Creates:
      - <out_dir>/config_used.json
      - <out_dir>/run_info.json
      - <out_dir>/<snps_subdir>/
    """
    case_dir = Path(out_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    snps_dir = case_dir / snps_subdir
    snps_dir.mkdir(parents=True, exist_ok=True)

    config_used_json = case_dir / "config_used.json"
    run_info_json = case_dir / "run_info.json"
    history_csv = case_dir / "history.csv"

    write_json(config_used_json, dict(cfg))

    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": os.getcwd(),
        "config_path": str(config_path) if config_path else None,
        "python": {"executable": sys.executable, "version": sys.version},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "dolfinx_rans_version": "0.1.0",
    }
    git = _try_git_info(Path(__file__).parent)
    if git:
        info["git"] = git

    write_json(run_info_json, info)

    return CasePaths(
        case_dir=case_dir,
        snps_dir=snps_dir,
        history_csv=history_csv,
        run_info_json=run_info_json,
        config_used_json=config_used_json,
    )


def fmt_sci(x: float, *, prec: int = 1, sign: bool = False) -> str:
    """Scientific notation with NaN/Inf handling."""
    import math

    if not math.isfinite(float(x)):
        return "nan"
    s = "+" if sign else ""
    return f"{float(x):{s}.{prec}e}"


def fmt_pair_sci(a: float, b: float, *, prec: int = 1, sign: bool = True) -> str:
    """Format a min,max pair as 'a,b' in scientific notation."""
    return f"{fmt_sci(a, prec=prec, sign=sign)},{fmt_sci(b, prec=prec, sign=sign)}"


class StepTablePrinter:
    """
    Compact step logs for solver iteration output.

    Example:
        table = StepTablePrinter([("iter", 6), ("dt", 9), ("res", 9)])
        table.row(["100", "1.0e-02", "1.2e-04"])
    """

    def __init__(self, columns: list[tuple[str, int]], *, gap: str = " ") -> None:
        self.columns = list(columns)
        self.gap = gap
        self._printed_header = False

    def header(self) -> None:
        if self._printed_header:
            return
        self._printed_header = True
        parts = [label.rjust(width) for label, width in self.columns]
        print(self.gap.join(parts), flush=True)

    def row(self, values: list[object]) -> None:
        if not self._printed_header:
            self.header()
        if len(values) != len(self.columns):
            raise ValueError(f"Expected {len(self.columns)} columns, got {len(values)} values")
        parts = []
        for (label, width), v in zip(self.columns, values):
            s = str(v)
            parts.append(s if len(s) > width else s.rjust(width))
        print(self.gap.join(parts), flush=True)


class HistoryWriterCSV:
    """Append-only CSV writer for per-step scalar diagnostics."""

    def __init__(self, path: Path, fieldnames: list[str], *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.path = path
        self.fieldnames = list(fieldnames)
        self._fh = None
        self._writer = None

        if not self.enabled:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists()
        self._fh = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames)
        if new_file:
            self._writer.writeheader()
            self._fh.flush()

    def write(self, row: Mapping[str, object]) -> None:
        if not self.enabled or self._writer is None or self._fh is None:
            return
        cleaned: dict[str, object] = {}
        for k in self.fieldnames:
            if k in row:
                v = row[k]
                if isinstance(v, float):
                    cleaned[k] = f"{v:.16e}"
                else:
                    cleaned[k] = v
        self._writer.writerow(cleaned)
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None
                self._writer = None


def compute_wall_distance_channel(S, use_symmetry: bool = True):
    """
    Compute wall distance for channel flow geometry.

    For channel flow:
    - Half-channel (use_symmetry=True): wall at y=0, symmetry at y=Ly
      → wall distance = y
    - Full channel (use_symmetry=False): walls at y=0 and y=Ly
      → wall distance = min(y, Ly-y)

    Args:
        S: Scalar function space
        use_symmetry: True for half-channel, False for full channel

    Returns:
        y_wall: Function containing wall distance at each DOF
    """
    from dolfinx.fem import Function
    import numpy as np

    domain = S.mesh
    y_wall = Function(S, name="wall_distance")

    # Get DOF coordinates
    x_dofs = S.tabulate_dof_coordinates()
    y_coords = x_dofs[:, 1]

    if use_symmetry:
        # Half-channel: wall at y=0, wall distance = y
        y_wall.x.array[:] = y_coords
    else:
        # Full channel: walls at y=0 and y=Ly
        Ly = np.max(y_coords)
        y_wall.x.array[:] = np.minimum(y_coords, Ly - y_coords)

    # Ensure positive (minimum distance = small epsilon for numerical stability)
    y_wall.x.array[:] = np.maximum(y_wall.x.array, 1e-10)
    y_wall.x.scatter_forward()

    return y_wall


def prepare_wall_shear_stress(u, domain, nu: float, wall_tag: int = 1):
    """
    Precompute forms for wall shear stress evaluation.

    Call once during setup. Returns precomputed state for
    eval_wall_shear_stress() which is cheap to call each iteration.

    Args:
        u: Velocity vector function (the Function object whose .x.array changes)
        domain: Mesh
        nu: Kinematic viscosity (1/Re_tau for nondimensional)
        wall_tag: Boundary tag for wall (default 1)

    Returns:
        dict with precomputed forms and wall_length
    """
    from dolfinx.fem import assemble_scalar, form
    from dolfinx.mesh import locate_entities_boundary, meshtags
    import ufl
    from mpi4py import MPI
    import numpy as np

    comm = domain.comm
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)

    def wall_boundary(x):
        return np.isclose(x[1], 0.0)

    wall_facets = locate_entities_boundary(domain, fdim, wall_boundary)

    num_facets = domain.topology.index_map(fdim).size_local
    facet_values = np.zeros(num_facets, dtype=np.int32)
    facet_values[wall_facets] = wall_tag
    facet_tags = meshtags(domain, fdim, np.arange(num_facets, dtype=np.int32), facet_values)

    ds_wall = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags, subdomain_id=wall_tag)

    # Precompile forms (JIT compilation happens here, not at eval time)
    grad_u = ufl.grad(u)
    du_dy = grad_u[0, 1]
    tau_form = form(nu * du_dy * ds_wall)

    # Wall length is constant — compute once
    wall_length = assemble_scalar(form(1.0 * ds_wall))
    wall_length = comm.allreduce(wall_length, op=MPI.SUM)

    return {"tau_form": tau_form, "wall_length": wall_length, "comm": comm}


def eval_wall_shear_stress(wss_ctx: dict) -> float:
    """
    Evaluate wall shear stress using precomputed forms.

    The velocity Function's .x.array is read automatically during assembly
    (it was captured by reference in the UFL form).

    Args:
        wss_ctx: dict returned by prepare_wall_shear_stress()

    Returns:
        τ_wall: Wall shear stress (averaged over wall)
    """
    from dolfinx.fem import assemble_scalar
    from mpi4py import MPI

    tau_integral = assemble_scalar(wss_ctx["tau_form"])
    tau_integral = wss_ctx["comm"].allreduce(tau_integral, op=MPI.SUM)

    wall_length = wss_ctx["wall_length"]
    if wall_length > 0:
        return tau_integral / wall_length
    return 0.0


def compute_wall_shear_stress(u, domain, nu: float, wall_tag: int = 1) -> float:
    """
    Compute wall shear stress τ_wall = ν * (∂u_x/∂y)|_wall.

    Convenience wrapper: prepares + evaluates in one call.
    For repeated calls, use prepare_wall_shear_stress() + eval_wall_shear_stress().
    """
    ctx = prepare_wall_shear_stress(u, domain, nu, wall_tag)
    return eval_wall_shear_stress(ctx)


def compute_bulk_velocity(u, Lx: float, Ly: float) -> float:
    """
    Compute bulk velocity U_bulk = (1/A) * integral(u_x dA).

    Args:
        u: Velocity vector function
        Lx: Domain length in x (streamwise)
        Ly: Domain height in y (wall-normal)

    Returns:
        U_bulk: Bulk (area-averaged) streamwise velocity
    """
    from dolfinx.fem import assemble_scalar, form
    import ufl
    from mpi4py import MPI

    domain = u.function_space.mesh
    comm = domain.comm

    # Extract x-component of velocity
    # For a vector function u = (u_x, u_y), we need the x-component
    # u.sub(0) returns a view, collapse() creates independent function
    u_x = u.sub(0).collapse()

    # Integrate u_x over entire domain
    integral = assemble_scalar(form(u_x * ufl.dx))
    integral = comm.allreduce(integral, op=MPI.SUM)

    # Domain area
    A = Lx * Ly

    return integral / A


def diagnostics_scalar(f) -> dict[str, float | bool]:
    """Global min/max of a scalar Function (MPI-safe)."""
    import numpy as np
    from mpi4py import MPI

    comm = f.function_space.mesh.comm
    a = np.asarray(f.x.array, dtype=float)
    finite_local = bool(np.isfinite(a).all())

    if a.size:
        local_min = float(np.nanmin(a))
        local_max = float(np.nanmax(a))
    else:
        local_min = float("inf")
        local_max = float("-inf")

    finite = bool(comm.allreduce(finite_local, op=MPI.LAND))
    vmin = float(comm.allreduce(local_min, op=MPI.MIN))
    vmax = float(comm.allreduce(local_max, op=MPI.MAX))
    return {"min": vmin, "max": vmax, "finite": finite}


def diagnostics_vector(u) -> dict[str, float | bool]:
    """Global component ranges + max magnitude of a vector Function (MPI-safe)."""
    import numpy as np
    from mpi4py import MPI

    comm = u.function_space.mesh.comm
    a = np.asarray(u.x.array, dtype=float)
    finite_local = bool(np.isfinite(a).all())

    ncomp = int(getattr(u.function_space.dofmap, "index_map_bs", 1))
    if ncomp <= 0:
        raise ValueError(f"Invalid vector value_size: {ncomp}")

    if a.size:
        vec = a.reshape(-1, ncomp)
        mins = np.nanmin(vec, axis=0)
        maxs = np.nanmax(vec, axis=0)
        mag = np.sqrt(np.sum(vec**2, axis=1))
        umax_local = float(np.nanmax(mag)) if mag.size else 0.0
    else:
        mins = np.full((ncomp,), float("inf"))
        maxs = np.full((ncomp,), float("-inf"))
        umax_local = 0.0

    finite = bool(comm.allreduce(finite_local, op=MPI.LAND))
    umax = float(comm.allreduce(umax_local, op=MPI.MAX))
    out: dict[str, float | bool] = {"umax": umax, "finite": finite}
    for i in range(ncomp):
        out[f"c{i}_min"] = float(comm.allreduce(float(mins[i]), op=MPI.MIN))
        out[f"c{i}_max"] = float(comm.allreduce(float(maxs[i]), op=MPI.MAX))
    return out
