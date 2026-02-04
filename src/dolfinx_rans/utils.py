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
