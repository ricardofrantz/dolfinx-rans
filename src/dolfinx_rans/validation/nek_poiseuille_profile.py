#!/usr/bin/env python3
"""
Extract and compare NekStab Poiseuille baseflow profiles.

This script reads a Nek5000 field from a poiseuille_RANS case, exports:
  - outer-scaled profile CSV/DAT
  - inner-unit profile CSV (optional compatibility)
  - run.sh-compatible reference CSV with columns: y_over_delta,u_over_ubulk
  - optional overlay plots against one or more dolfinx-rans profiles.csv files

Usage example:
  python -m dolfinx_rans.validation.nek_poiseuille_profile \
    --nek-case-dir ../nekStab/example/poiseuille_RANS \
    --out-dir nek_poiseuille_profile \
    --dolfinx-profiles re5200_nek_long/profiles.csv re5200_nek_refined/profiles.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parse_viscosity_from_par(par_path: Path) -> tuple[float, float]:
    """
    Parse viscosity entry in Nek .par file.

    Convention in this case:
      viscosity < 0  -> interpreted as -Re in Nek examples
      viscosity > 0  -> interpreted as nu directly
    """
    nu = 1.0e-5
    re = 1.0 / nu
    for raw in par_path.read_text().splitlines():
        line = raw.strip().replace(" ", "")
        if not line.lower().startswith("viscosity="):
            continue
        val = float(line.split("=", 1)[1])
        if val < 0:
            re = abs(val)
            nu = 1.0 / re
        else:
            nu = val
            re = 1.0 / max(nu, 1e-30)
        break
    return nu, re


def _find_default_field(case_dir: Path) -> Path:
    # Match pattern used in the local Nek plot script first.
    candidates = sorted(case_dir.glob("BF_*0.f*"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No baseflow field matching BF_*0.f* in {case_dir}")


def _extract_nek_profile(
    nek_case_dir: Path,
    nek_field: Path | None,
    ngrid: int,
) -> dict:
    # Reuse the same helper module used by ../nekStab/example/poiseuille_RANS/plot.py
    nek_example_parent = nek_case_dir.parent
    sys.path.insert(0, str(nek_example_parent.resolve()))
    import nekplot as nk  # type: ignore

    field_path = nek_field if nek_field is not None else _find_default_field(nek_case_dir)
    if not field_path.is_absolute():
        field_path = (nek_case_dir / field_path).resolve()
    if not field_path.exists():
        raise FileNotFoundError(f"Nek field file not found: {field_path}")

    x, y, fields, time = nk.read_field(field_path)
    if "vx" not in fields:
        raise RuntimeError(f"Nek field has no 'vx': {field_path}")

    xi, yi, _Xi, _Yi, interp = nk.sem_to_grid(
        x,
        y,
        {"vx": fields["vx"]},
        xlim=(float(np.min(x)), float(np.max(x))),
        ylim=(float(np.min(y)), float(np.max(y))),
        ngrid=ngrid,
    )

    # Mean in streamwise direction -> function of y only
    ux_grid = interp["vx"]
    u_of_y = np.mean(ux_grid, axis=1)

    # Convert full channel y in [-1,1] to wall-distance coordinate y_wall in [0,1]
    mask_lo = yi <= 0.0
    mask_up = yi >= 0.0
    y_lo = yi[mask_lo] + 1.0
    u_lo = u_of_y[mask_lo]
    y_up = 1.0 - yi[mask_up]
    u_up = u_of_y[mask_up]

    ilo = np.argsort(y_lo)
    iup = np.argsort(y_up)
    y_lo, u_lo = y_lo[ilo], u_lo[ilo]
    y_up, u_up = y_up[iup], u_up[iup]

    y_wall = np.linspace(0.0, 1.0, 401)
    u_lo_i = np.interp(y_wall, y_lo, u_lo)
    u_up_i = np.interp(y_wall, y_up, u_up)
    u_mean = 0.5 * (u_lo_i + u_up_i)

    u_bulk = float(np.trapz(u_mean, y_wall) / (y_wall[-1] - y_wall[0]))

    nu, re = _parse_viscosity_from_par(nek_case_dir / "poiseuille_RANS.par")

    # Estimate wall quantities from near-wall slope (first ~5% of points)
    nfit = max(12, int(0.05 * len(y_wall)))
    slope = float(np.polyfit(y_wall[:nfit], u_mean[:nfit], 1)[0])
    tau_wall = nu * slope
    u_tau = float(np.sqrt(max(tau_wall, 0.0)))
    re_tau = u_tau / nu if nu > 0 else float("nan")

    y_plus = y_wall * u_tau / nu if u_tau > 0 else np.full_like(y_wall, np.nan)
    u_plus = u_mean / u_tau if u_tau > 0 else np.full_like(y_wall, np.nan)

    return {
        "field_path": str(field_path),
        "time": float(time),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "nu_from_par": float(nu),
        "re_from_par": float(re),
        "estimated_du_dy_wall": float(slope),
        "estimated_tau_wall": float(tau_wall),
        "estimated_u_tau": float(u_tau),
        "estimated_re_tau": float(re_tau),
        "y_wall": y_wall,
        "u_mean": u_mean,
        "u_bulk": float(u_bulk),
        "y_plus": y_plus,
        "u_plus": u_plus,
    }


def _load_dolfinx_profile(path: Path) -> dict:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise ValueError(f"Empty DOLFINx profile CSV: {path}")
    required = ("y",)
    for col in required:
        if col not in rows[0]:
            raise ValueError(f"{path} missing required column '{col}'")
    y = np.array([float(r["y"]) for r in rows])

    y_over_delta = np.array([float(r["y_over_delta"]) for r in rows], dtype=float) if "y_over_delta" in rows[0] else None
    if y_over_delta is not None and not np.all(np.isfinite(y_over_delta)):
        y_over_delta = None

    if "y_plus" in rows[0]:
        y_plus = np.array([float(r["y_plus"]) for r in rows], dtype=float)
    else:
        y_plus = np.array([np.nan] * len(rows), dtype=float)

    u_over_ubulk = None
    if "u_over_ubulk" in rows[0]:
        u_over_ubulk = np.array([float(r["u_over_ubulk"]) for r in rows], dtype=float)
    elif "u" in rows[0]:
        u_data = np.array([float(r["u"]) for r in rows], dtype=float)
        y_for_bulk = y_over_delta
        if y_for_bulk is None:
            re_tau = float(rows[0].get("Re_tau", float("nan"))) if "Re_tau" in rows[0] else float("nan")
            if np.isfinite(re_tau) and re_tau > 0:
                y_for_bulk = y_plus / re_tau
        if y_for_bulk is not None and np.all(np.isfinite(y_for_bulk)):
            bulk = float(np.trapezoid(u_data, y_for_bulk) / max(y_for_bulk[-1] - y_for_bulk[0], 1e-30))
            u_over_ubulk = u_data / max(bulk, 1e-30)
    elif "u_plus" in rows[0]:
        u_data = np.array([float(r["u_plus"]) for r in rows], dtype=float)
        if y_plus is not None and np.all(np.isfinite(y_plus)):
            re_tau = float(rows[0].get("Re_tau", np.nan))
            if np.isfinite(re_tau) and re_tau > 0:
                y_for_bulk = y_plus / re_tau
                bulk = float(np.trapezoid(u_data, y_for_bulk) / max(y_for_bulk[-1] - y_for_bulk[0], 1e-30))
                u_over_ubulk = u_data / max(bulk, 1e-30)
    else:
        raise ValueError(f"{path} missing velocity column (expected 'u' or 'u_plus')")
    if u_over_ubulk is None:
        u_over_ubulk = np.full_like(y, np.nan, dtype=float)

    return {
        "path": str(path),
        "label": path.parent.name if path.parent.name else path.stem,
        "y": y,
        "y_over_delta": y_over_delta,
        "y_plus": y_plus,
        "u_plus": np.array([float(r["u_plus"]) for r in rows], dtype=float) if "u_plus" in rows[0] else np.full_like(y, np.nan, dtype=float),
        "u_over_ubulk": u_over_ubulk,
    }


def _write_csv(path: Path, header: list[str], rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow([f"{float(v):.16e}" for v in row])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nek-case-dir", type=Path, required=True, help="Path to poiseuille_RANS case directory")
    ap.add_argument("--nek-field", type=Path, default=None, help="Optional field filename (default: BF_*0.f*)")
    ap.add_argument("--out-dir", type=Path, default=Path("nek_poiseuille_profile"), help="Output folder")
    ap.add_argument("--ngrid", type=int, default=600, help="Interpolation grid along x")
    ap.add_argument(
        "--dolfinx-profiles",
        type=Path,
        nargs="*",
        default=[],
        help="Optional list of dolfinx-rans profiles.csv to overlay",
    )
    args = ap.parse_args()

    case_dir = args.nek_case_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    nek = _extract_nek_profile(case_dir, args.nek_field, args.ngrid)

    y_wall = nek["y_wall"]
    u_mean = nek["u_mean"]
    u_bulk = nek["u_bulk"]
    y_plus = nek["y_plus"]
    u_plus = nek["u_plus"]

    # 1) Outer scaled CSV + DAT
    _write_csv(
        out_dir / "nek_profile_outer.csv",
        ["y_over_delta", "u", "u_over_ubulk"],
        [[yy, uu, uu / u_bulk] for yy, uu in zip(y_wall, u_mean)],
    )
    with (out_dir / "nek_profile_outer.dat").open("w") as f:
        f.write("# y_over_delta u_over_ubulk\n")
        for yy, uu in zip(y_wall, u_mean):
            f.write(f"{yy:.16e} {(uu / u_bulk):.16e}\n")

    # 2) Inner-unit diagnostics + 3) run.sh-compatible outer-profile gate
    _write_csv(
        out_dir / "nek_profile_wall_units.csv",
        ["y_over_delta", "y_plus", "u_plus"],
        [[yy, yp, up] for yy, yp, up in zip(y_wall, y_plus, u_plus)],
    )
    # Gate input format expected by run.sh
    _write_csv(
        out_dir / "nek_reference_profile.csv",
        ["y_over_delta", "u_over_ubulk"],
        [[yy, uu / u_bulk] for yy, uu in zip(y_wall, u_mean)],
    )

    dolfinx = []
    for p in args.dolfinx_profiles:
        pp = p.resolve()
        if not pp.exists():
            raise FileNotFoundError(f"DOLFINx profile not found: {pp}")
        dolfinx.append(_load_dolfinx_profile(pp))

    # Outer-scale overlay
    plt.figure(figsize=(7, 5))
    plt.plot(y_wall, u_mean / u_bulk, linewidth=2.0, label="Nek (outer)")
    for d in dolfinx:
        u = d["u_over_ubulk"]
        if d["y_over_delta"] is not None and np.any(np.isfinite(d["y_over_delta"])):
            y_profile = d["y_over_delta"]
        else:
            y_profile = d["y"]
        valid = np.isfinite(y_profile) & np.isfinite(u)
        if np.any(valid):
            plt.plot(np.asarray(y_profile)[valid], np.asarray(u)[valid], linewidth=1.5, label=f"{d['label']} (outer)")
    plt.xlabel("y/delta")
    plt.ylabel("U/U_bulk")
    plt.title("Mean Velocity Profile (Outer Scaling)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_outer.png", dpi=180)
    plt.close()

    # Wall-unit overlay
    plt.figure(figsize=(7, 5))
    plt.semilogx(y_plus[1:], u_plus[1:], linewidth=2.0, label="Nek (derived wall units)")
    for d in dolfinx:
        if np.all(np.isfinite(d["y_plus"])):
            plt.semilogx(d["y_plus"][1:], d["u_plus"][1:], linewidth=1.5, label=d["label"])
    plt.xlabel("y+")
    plt.ylabel("u+")
    plt.title("Mean Velocity Profile (Wall Units)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_wall_units.png", dpi=180)
    plt.close()

    # Quantify outer-shape error where overlays were supplied
    rmse_outer = {}
    for d in dolfinx:
        u = d["u_over_ubulk"]
        y_profile = d["y_over_delta"] if (d["y_over_delta"] is not None and np.any(np.isfinite(d["y_over_delta"]))) else d["y"]
        valid = np.isfinite(y_profile) & np.isfinite(u)
        if not np.any(valid):
            continue
        interp_d = np.interp(y_wall, np.asarray(y_profile)[valid], np.asarray(u)[valid])
        rmse = float(np.sqrt(np.mean((interp_d - (u_mean / u_bulk)) ** 2)))
        rmse_outer[d["label"]] = rmse

    meta = {
        "nek_case_dir": str(case_dir),
        "nek_field": nek["field_path"],
        "nek_time": nek["time"],
        "domain_bounds": {
            "x_min": nek["x_min"],
            "x_max": nek["x_max"],
            "y_min": nek["y_min"],
            "y_max": nek["y_max"],
        },
        "viscosity_from_par": nek["nu_from_par"],
        "re_from_par": nek["re_from_par"],
        "estimated_wall_quantities": {
            "du_dy_wall": nek["estimated_du_dy_wall"],
            "tau_wall": nek["estimated_tau_wall"],
            "u_tau": nek["estimated_u_tau"],
            "re_tau": nek["estimated_re_tau"],
        },
        "u_bulk_outer_scale": u_bulk,
        "outer_rmse_vs_dolfinx": rmse_outer,
        "notes": [
            "Nek Re is read from poiseuille_RANS.par viscosity convention (negative value means Re).",
            "Nek wall-units are derived from near-wall slope of extracted baseflow and are approximate.",
            "Use nek_reference_profile.csv (y_over_delta,u_over_ubulk) for run.sh reference_profile_csv gate input.",
        ],
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"Saved Nek profile artifacts in {out_dir}")
    print(f"  Re from par: {nek['re_from_par']:.1f}")
    print(f"  Estimated Re_tau from extracted baseflow: {nek['estimated_re_tau']:.2f}")
    print("  Reference CSV for run.sh gate:", out_dir / "nek_reference_profile.csv")


if __name__ == "__main__":
    main()
