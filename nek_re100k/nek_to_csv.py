#!/usr/bin/env python3
"""
NEK_TO_CSV: Extract mean channel profile from Nek baseflow and plot it.

OUTPUTS: nek_to_csv.npz, nek_to_csv.csv, nek_to_csv.png, nek_to_csv.json
USAGE:   python nek_to_csv.py                        # .npz exists -> plot/export only
         rm nek_to_csv.npz && python nek_to_csv.py   # recompute from Nek field
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT = Path(__file__).resolve()
CASE_DIR = SCRIPT.parent

OUTPUT_NPZ = SCRIPT.with_suffix(".npz")
OUTPUT_CSV = SCRIPT.with_suffix(".csv")
OUTPUT_PNG = SCRIPT.with_suffix(".png")
OUTPUT_JSON = SCRIPT.with_suffix(".json")
OUTPUT_SYM_CSV = CASE_DIR / "nek_to_csv_symmetry.csv"
OUTPUT_SYM_PNG = CASE_DIR / "nek_to_csv_symmetry.png"

FIELD_FILE = CASE_DIR / "BF_poiseuille_RANS0.f00001"
SOURCE_CASE_DIR = (CASE_DIR / "../../nekStab/example/poiseuille_RANS").resolve()
SOURCE_PAR = SOURCE_CASE_DIR / "poiseuille_RANS.par"
NEKPLOT_DIR = SOURCE_CASE_DIR.parent

REQUIRED_NPZ_KEYS = {
    "y_wall",
    "u_mean",
    "v_mean",
    "p_mean",
    "temp_mean",
    "scalar_1",
    "scalar_2",
    "u_bulk",
    "y_plus",
    "u_plus",
    "temp_present",
    "scalar_1_present",
    "scalar_2_present",
    "available_field_keys",
    "y_half",
    "y_upper_inverted",
    "u_lower",
    "u_upper_inverted",
    "v_lower",
    "v_upper_inverted",
    "p_lower",
    "p_upper_inverted",
    "temp_lower",
    "temp_upper_inverted",
    "s01_lower",
    "s01_upper_inverted",
    "s02_lower",
    "s02_upper_inverted",
}


def _parse_viscosity(par_path: Path) -> tuple[float, float]:
    nu = float("nan")
    re = float("nan")
    if not par_path.exists():
        return nu, re

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


def _fold_to_wall_distance(yi: np.ndarray, q_of_y: np.ndarray, y_wall: np.ndarray) -> np.ndarray:
    """Fold full-channel profile in yi in [-1,1] to wall-distance y_wall in [0,1]."""
    mask_lo = yi <= 0.0
    mask_up = yi >= 0.0

    y_lo = yi[mask_lo] + 1.0
    q_lo = q_of_y[mask_lo]
    y_up = 1.0 - yi[mask_up]
    q_up = q_of_y[mask_up]

    ilo = np.argsort(y_lo)
    iup = np.argsort(y_up)
    y_lo = y_lo[ilo]
    q_lo = q_lo[ilo]
    y_up = y_up[iup]
    q_up = q_up[iup]

    return 0.5 * (np.interp(y_wall, y_lo, q_lo) + np.interp(y_wall, y_up, q_up))


def _split_lower_upper_inverted(
    yi: np.ndarray,
    q_of_y: np.ndarray,
    y_half: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build branch profiles for symmetry check:
    - lower branch from y=0 to y=1
    - upper branch from y=2 to y=1 (inverted)
    """
    y_full = yi + 1.0  # map [-1,1] -> [0,2]
    mask_lo = y_full <= 1.0
    mask_up = y_full >= 1.0

    y_lo = y_full[mask_lo]
    q_lo = q_of_y[mask_lo]
    y_up = y_full[mask_up]
    q_up = q_of_y[mask_up]

    ilo = np.argsort(y_lo)
    iup = np.argsort(y_up)
    y_lo = y_lo[ilo]
    q_lo = q_lo[ilo]
    y_up = y_up[iup]
    q_up = q_up[iup]

    q_lower = np.interp(y_half, y_lo, q_lo)
    y_upper_inverted = 2.0 - y_half  # 2 -> 1 when y_half goes 0 -> 1
    q_upper_inverted = np.interp(y_upper_inverted, y_up, q_up)
    return q_lower, q_upper_inverted


def _read_field_full(field_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], float]:
    """Read Nek field with velocity, pressure, temperature, and passive scalars."""
    from pymech.neksuite import readnek

    field = readnek(str(field_path))
    nel = field.nel
    nx, ny, nz = field.lr1
    npts = nx * ny * max(nz, 1)
    ntotal = nel * npts

    x = np.empty(ntotal)
    y = np.empty(ntotal)

    for ie, elem in enumerate(field.elem):
        s = slice(ie * npts, (ie + 1) * npts)
        x[s] = elem.pos[0].ravel()
        y[s] = elem.pos[1].ravel()

    fields: dict[str, np.ndarray] = {}

    for comp, key in enumerate(("vx", "vy")):
        if comp >= len(field.elem[0].vel):
            continue
        arr = np.empty(ntotal)
        for ie, elem in enumerate(field.elem):
            s = slice(ie * npts, (ie + 1) * npts)
            arr[s] = elem.vel[comp].ravel()
        fields[key] = arr

    if len(field.elem[0].pres) > 0:
        arr = np.empty(ntotal)
        for ie, elem in enumerate(field.elem):
            s = slice(ie * npts, (ie + 1) * npts)
            arr[s] = elem.pres[0].ravel()
        fields["p"] = arr

    if len(field.elem[0].temp) > 0:
        arr = np.empty(ntotal)
        for ie, elem in enumerate(field.elem):
            s = slice(ie * npts, (ie + 1) * npts)
            arr[s] = elem.temp[0].ravel()
        fields["t"] = arr

    if hasattr(field.elem[0], "scal"):
        n_scal = len(field.elem[0].scal)
        for i in range(n_scal):
            arr = np.empty(ntotal)
            for ie, elem in enumerate(field.elem):
                s = slice(ie * npts, (ie + 1) * npts)
                arr[s] = elem.scal[i].ravel()
            fields[f"s{i + 1:02d}"] = arr

    return x, y, fields, float(field.time)


def compute() -> None:
    if not FIELD_FILE.exists():
        raise FileNotFoundError(f"Missing Nek field file: {FIELD_FILE}")

    nekplot_file = NEKPLOT_DIR / "nekplot.py"
    if not nekplot_file.exists():
        raise FileNotFoundError(f"Missing nekplot helper: {nekplot_file}")

    sys.path.insert(0, str(NEKPLOT_DIR))
    import nekplot as nk  # type: ignore

    x, y, fields, time = _read_field_full(FIELD_FILE)
    if "vx" not in fields:
        raise RuntimeError(f"Field file has no 'vx' velocity component: {FIELD_FILE}")

    fields_to_interp: dict[str, np.ndarray] = {}
    for key in ("vx", "vy", "p", "t", "s01", "s02"):
        if key in fields:
            fields_to_interp[key] = fields[key]

    _, yi, _, _, interp = nk.sem_to_grid(
        x,
        y,
        fields_to_interp,
        xlim=(float(np.min(x)), float(np.max(x))),
        ylim=(float(np.min(y)), float(np.max(y))),
        ngrid=600,
    )

    y_wall = np.linspace(0.0, 1.0, 401)
    y_half = np.linspace(0.0, 1.0, 401)
    y_upper_inverted = 2.0 - y_half

    u_of_y = np.mean(interp["vx"], axis=1)
    u_mean = _fold_to_wall_distance(yi, u_of_y, y_wall)
    u_lower, u_upper_inverted = _split_lower_upper_inverted(yi, u_of_y, y_half)
    if "vy" in interp:
        v_of_y = np.mean(interp["vy"], axis=1)
        v_mean = _fold_to_wall_distance(yi, v_of_y, y_wall)
        v_lower, v_upper_inverted = _split_lower_upper_inverted(yi, v_of_y, y_half)
    else:
        v_mean = np.full_like(y_wall, np.nan)
        v_lower = np.full_like(y_half, np.nan)
        v_upper_inverted = np.full_like(y_half, np.nan)
    if "p" in interp:
        p_of_y = np.mean(interp["p"], axis=1)
        p_mean = _fold_to_wall_distance(yi, p_of_y, y_wall)
        p_lower, p_upper_inverted = _split_lower_upper_inverted(yi, p_of_y, y_half)
    else:
        p_mean = np.full_like(y_wall, np.nan)
        p_lower = np.full_like(y_half, np.nan)
        p_upper_inverted = np.full_like(y_half, np.nan)
    if "t" in interp:
        t_of_y = np.mean(interp["t"], axis=1)
        temp_mean = _fold_to_wall_distance(yi, t_of_y, y_wall)
        temp_lower, temp_upper_inverted = _split_lower_upper_inverted(yi, t_of_y, y_half)
        temp_present = 1
    else:
        temp_mean = np.full_like(y_wall, np.nan)
        temp_lower = np.full_like(y_half, np.nan)
        temp_upper_inverted = np.full_like(y_half, np.nan)
        temp_present = 0
    if "s01" in interp:
        s1_of_y = np.mean(interp["s01"], axis=1)
        s1_mean = _fold_to_wall_distance(yi, s1_of_y, y_wall)
        s01_lower, s01_upper_inverted = _split_lower_upper_inverted(yi, s1_of_y, y_half)
        s1_present = 1
    else:
        s1_mean = np.full_like(y_wall, np.nan)
        s01_lower = np.full_like(y_half, np.nan)
        s01_upper_inverted = np.full_like(y_half, np.nan)
        s1_present = 0
    if "s02" in interp:
        s2_of_y = np.mean(interp["s02"], axis=1)
        s2_mean = _fold_to_wall_distance(yi, s2_of_y, y_wall)
        s02_lower, s02_upper_inverted = _split_lower_upper_inverted(yi, s2_of_y, y_half)
        s2_present = 1
    else:
        s2_mean = np.full_like(y_wall, np.nan)
        s02_lower = np.full_like(y_half, np.nan)
        s02_upper_inverted = np.full_like(y_half, np.nan)
        s2_present = 0

    u_bulk = float(np.trapz(u_mean, y_wall) / (y_wall[-1] - y_wall[0]))

    nu, re = _parse_viscosity(SOURCE_PAR)
    if np.isfinite(nu) and nu > 0:
        nfit = max(12, int(0.05 * len(y_wall)))
        du_dy_wall = float(np.polyfit(y_wall[:nfit], u_mean[:nfit], 1)[0])
        tau_wall = float(nu * du_dy_wall)
        u_tau = float(np.sqrt(max(tau_wall, 0.0)))
        re_tau = float(u_tau / nu) if u_tau > 0 else float("nan")
        y_plus = y_wall * u_tau / nu
        u_plus = u_mean / u_tau if u_tau > 0 else np.full_like(y_wall, np.nan)
    else:
        du_dy_wall = float("nan")
        tau_wall = float("nan")
        u_tau = float("nan")
        re_tau = float("nan")
        y_plus = np.full_like(y_wall, np.nan)
        u_plus = np.full_like(y_wall, np.nan)

    np.savez_compressed(
        OUTPUT_NPZ,
        y_wall=y_wall,
        u_mean=u_mean,
        v_mean=v_mean,
        p_mean=p_mean,
        temp_mean=temp_mean,
        scalar_1=s1_mean,
        scalar_2=s2_mean,
        u_bulk=np.array([u_bulk], dtype=float),
        y_plus=y_plus,
        u_plus=u_plus,
        x_min=np.array([float(np.min(x))], dtype=float),
        x_max=np.array([float(np.max(x))], dtype=float),
        y_min=np.array([float(np.min(y))], dtype=float),
        y_max=np.array([float(np.max(y))], dtype=float),
        time=np.array([float(time)], dtype=float),
        nu=np.array([nu], dtype=float),
        re=np.array([re], dtype=float),
        du_dy_wall=np.array([du_dy_wall], dtype=float),
        tau_wall=np.array([tau_wall], dtype=float),
        u_tau=np.array([u_tau], dtype=float),
        re_tau=np.array([re_tau], dtype=float),
        temp_present=np.array([temp_present], dtype=np.int32),
        scalar_1_present=np.array([s1_present], dtype=np.int32),
        scalar_2_present=np.array([s2_present], dtype=np.int32),
        available_field_keys=np.array(sorted(fields_to_interp.keys()), dtype="U32"),
        y_half=y_half,
        y_upper_inverted=y_upper_inverted,
        u_lower=u_lower,
        u_upper_inverted=u_upper_inverted,
        v_lower=v_lower,
        v_upper_inverted=v_upper_inverted,
        p_lower=p_lower,
        p_upper_inverted=p_upper_inverted,
        temp_lower=temp_lower,
        temp_upper_inverted=temp_upper_inverted,
        s01_lower=s01_lower,
        s01_upper_inverted=s01_upper_inverted,
        s02_lower=s02_lower,
        s02_upper_inverted=s02_upper_inverted,
        field_file=np.array([str(FIELD_FILE)], dtype="U512"),
        source_par=np.array([str(SOURCE_PAR)], dtype="U512"),
    )


def _load_data() -> dict[str, np.ndarray | float | str]:
    data = np.load(OUTPUT_NPZ, allow_pickle = False)
    return {
        "y_wall": data["y_wall"],
        "u_mean": data["u_mean"],
        "v_mean": data["v_mean"],
        "p_mean": data["p_mean"],
        "temp_mean": data["temp_mean"],
        "scalar_1": data["scalar_1"],
        "scalar_2": data["scalar_2"],
        "u_bulk": float(data["u_bulk"][0]),
        "y_plus": data["y_plus"],
        "u_plus": data["u_plus"],
        "x_min": float(data["x_min"][0]),
        "x_max": float(data["x_max"][0]),
        "y_min": float(data["y_min"][0]),
        "y_max": float(data["y_max"][0]),
        "time": float(data["time"][0]),
        "nu": float(data["nu"][0]),
        "re": float(data["re"][0]),
        "du_dy_wall": float(data["du_dy_wall"][0]),
        "tau_wall": float(data["tau_wall"][0]),
        "u_tau": float(data["u_tau"][0]),
        "re_tau": float(data["re_tau"][0]),
        "temp_present": int(data["temp_present"][0]),
        "scalar_1_present": int(data["scalar_1_present"][0]),
        "scalar_2_present": int(data["scalar_2_present"][0]),
        "available_field_keys": [str(k) for k in data["available_field_keys"].tolist()],
        "y_half": data["y_half"],
        "y_upper_inverted": data["y_upper_inverted"],
        "u_lower": data["u_lower"],
        "u_upper_inverted": data["u_upper_inverted"],
        "v_lower": data["v_lower"],
        "v_upper_inverted": data["v_upper_inverted"],
        "p_lower": data["p_lower"],
        "p_upper_inverted": data["p_upper_inverted"],
        "temp_lower": data["temp_lower"],
        "temp_upper_inverted": data["temp_upper_inverted"],
        "s01_lower": data["s01_lower"],
        "s01_upper_inverted": data["s01_upper_inverted"],
        "s02_lower": data["s02_lower"],
        "s02_upper_inverted": data["s02_upper_inverted"],
        "field_file": str(data["field_file"][0]),
        "source_par": str(data["source_par"][0]),
    }


def _npz_needs_recompute() -> bool:
    try:
        data = np.load(OUTPUT_NPZ, allow_pickle = False)
    except FileNotFoundError:
        return True
    keys = set(data.files)
    return not REQUIRED_NPZ_KEYS.issubset(keys)


def write_csv(data: dict[str, np.ndarray | float | str]) -> None:
    y = np.asarray(data["y_wall"], dtype=float)
    u = np.asarray(data["u_mean"], dtype=float)
    v = np.asarray(data["v_mean"], dtype=float)
    p = np.asarray(data["p_mean"], dtype=float)
    t = np.asarray(data["temp_mean"], dtype=float)
    s1 = np.asarray(data["scalar_1"], dtype=float)
    s2 = np.asarray(data["scalar_2"], dtype=float)
    u_bulk = float(data["u_bulk"])
    yp = np.asarray(data["y_plus"], dtype=float)
    up = np.asarray(data["u_plus"], dtype=float)

    with OUTPUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "y_over_delta",
                "u",
                "v",
                "pressure",
                "temp",
                "scalar_1",
                "scalar_2",
                "u_over_ubulk",
                "y_plus",
                "u_plus",
            ]
        )
        for yi, ui, vi, pi, ti, s1i, s2i, ypi, upi in zip(y, u, v, p, t, s1, s2, yp, up):
            w.writerow(
                [
                    f"{yi:.16e}",
                    f"{ui:.16e}",
                    f"{vi:.16e}",
                    f"{pi:.16e}",
                    f"{ti:.16e}",
                    f"{s1i:.16e}",
                    f"{s2i:.16e}",
                    f"{(ui / u_bulk):.16e}",
                    f"{ypi:.16e}",
                    f"{upi:.16e}",
                ]
            )


def write_metadata_json(data: dict[str, np.ndarray | float | str]) -> None:
    payload = {
        "field_file": data["field_file"],
        "source_par": data["source_par"],
        "nek_time": float(data["time"]),
        "domain_bounds": {
            "x_min": float(data["x_min"]),
            "x_max": float(data["x_max"]),
            "y_min": float(data["y_min"]),
            "y_max": float(data["y_max"]),
        },
        "parsed_from_par": {
            "nu": float(data["nu"]),
            "re": float(data["re"]),
        },
        "estimated_wall_quantities": {
            "du_dy_wall": float(data["du_dy_wall"]),
            "tau_wall": float(data["tau_wall"]),
            "u_tau": float(data["u_tau"]),
            "re_tau": float(data["re_tau"]),
        },
        "available_field_keys": list(data["available_field_keys"]),
        "scalar_availability": {
            "temp_present": bool(int(data["temp_present"])),
            "s01_present": bool(int(data["scalar_1_present"])),
            "s02_present": bool(int(data["scalar_2_present"])),
        },
        "u_bulk": float(data["u_bulk"]),
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")


def write_symmetry_csv(data: dict[str, np.ndarray | float | str]) -> None:
    y_low = np.asarray(data["y_half"], dtype=float)
    y_up_inv = np.asarray(data["y_upper_inverted"], dtype=float)
    u_lo = np.asarray(data["u_lower"], dtype=float)
    u_up = np.asarray(data["u_upper_inverted"], dtype=float)
    v_lo = np.asarray(data["v_lower"], dtype=float)
    v_up = np.asarray(data["v_upper_inverted"], dtype=float)
    p_lo = np.asarray(data["p_lower"], dtype=float)
    p_up = np.asarray(data["p_upper_inverted"], dtype=float)
    t_lo = np.asarray(data["temp_lower"], dtype=float)
    t_up = np.asarray(data["temp_upper_inverted"], dtype=float)
    s1_lo = np.asarray(data["s01_lower"], dtype=float)
    s1_up = np.asarray(data["s01_upper_inverted"], dtype=float)
    s2_lo = np.asarray(data["s02_lower"], dtype=float)
    s2_up = np.asarray(data["s02_upper_inverted"], dtype=float)

    with OUTPUT_SYM_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "y_lower_0_to_1",
                "y_upper_2_to_1",
                "u_lower",
                "u_upper_inverted",
                "v_lower",
                "v_upper_inverted",
                "pressure_lower",
                "pressure_upper_inverted",
                "temp_lower",
                "temp_upper_inverted",
                "scalar_1_lower",
                "scalar_1_upper_inverted",
                "scalar_2_lower",
                "scalar_2_upper_inverted",
            ]
        )
        for row in zip(
            y_low,
            y_up_inv,
            u_lo,
            u_up,
            v_lo,
            v_up,
            p_lo,
            p_up,
            t_lo,
            t_up,
            s1_lo,
            s1_up,
            s2_lo,
            s2_up,
        ):
            w.writerow([f"{float(v):.16e}" for v in row])


def _plot_profile_or_missing(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    valid = np.isfinite(x) & np.isfinite(y)
    if np.any(valid):
        ax.plot(x[valid], y[valid], linewidth=1.6, color=color)
    else:
        ax.text(0.5, 0.5, "Not available", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot(data: dict[str, np.ndarray | float | str]) -> None:
    y = np.asarray(data["y_wall"], dtype=float)
    u = np.asarray(data["u_mean"], dtype=float)
    v = np.asarray(data["v_mean"], dtype=float)
    p = np.asarray(data["p_mean"], dtype=float)
    t = np.asarray(data["temp_mean"], dtype=float)
    s1 = np.asarray(data["scalar_1"], dtype=float)
    s2 = np.asarray(data["scalar_2"], dtype=float)
    u_bulk = float(data["u_bulk"])
    yp = np.asarray(data["y_plus"], dtype=float)
    up = np.asarray(data["u_plus"], dtype=float)
    re = float(data["re"])
    re_tau = float(data["re_tau"])

    fig, axes = plt.subplots(2, 4, figsize=(14.0, 6.0))
    axes = axes.ravel()

    _plot_profile_or_missing(
        axes[0],
        y,
        u / u_bulk,
        color="tab:blue",
        title="U/U_bulk",
        xlabel="y/delta",
        ylabel="U/U_bulk",
    )

    valid_up = np.isfinite(yp) & np.isfinite(up) & (yp > 0.0)
    if np.any(valid_up):
        axes[1].semilogx(yp[valid_up], up[valid_up], linewidth=1.6, color="tab:red")
    else:
        axes[1].text(0.5, 0.5, "Not available", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_title("u+")
    axes[1].set_xlabel("y+")
    axes[1].set_ylabel("u+")
    axes[1].grid(True, alpha=0.3)

    _plot_profile_or_missing(
        axes[2],
        y,
        v,
        color="tab:green",
        title="v",
        xlabel="y/delta",
        ylabel="v",
    )
    _plot_profile_or_missing(
        axes[3],
        y,
        p,
        color="tab:purple",
        title="pressure",
        xlabel="y/delta",
        ylabel="p",
    )
    _plot_profile_or_missing(
        axes[4],
        y,
        t,
        color="tab:orange",
        title="temp",
        xlabel="y/delta",
        ylabel="temp",
    )
    _plot_profile_or_missing(
        axes[5],
        y,
        s1,
        color="tab:brown",
        title="scalar_1 (s01)",
        xlabel="y/delta",
        ylabel="scalar_1",
    )
    _plot_profile_or_missing(
        axes[6],
        y,
        s2,
        color="tab:pink",
        title="scalar_2 (s02)",
        xlabel="y/delta",
        ylabel="scalar_2",
    )
    axes[7].axis("off")

    fig.suptitle(f"Nek poiseuille_RANS profile (Re={re:.0f}, Re_tau~{re_tau:.2f})")
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_symmetry_pair(
    ax: plt.Axes,
    y_low: np.ndarray,
    y_up_inv: np.ndarray,
    q_low: np.ndarray,
    q_up_inv: np.ndarray,
    *,
    title: str,
    ylabel: str,
) -> None:
    valid_low = np.isfinite(y_low) & np.isfinite(q_low)
    valid_up = np.isfinite(y_up_inv) & np.isfinite(q_up_inv)
    if np.any(valid_low):
        ax.plot(y_low[valid_low], q_low[valid_low], color="tab:blue", linewidth=1.4, label="lower (0->1)")
    if np.any(valid_up):
        ax.plot(
            y_up_inv[valid_up],
            q_up_inv[valid_up],
            color="tab:red",
            linewidth=1.4,
            linestyle="--",
            label="upper (2->1)",
        )
    if not np.any(valid_low) and not np.any(valid_up):
        ax.text(0.5, 0.5, "Not available", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel("y (full channel)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)


def plot_symmetry(data: dict[str, np.ndarray | float | str]) -> None:
    y_low = np.asarray(data["y_half"], dtype=float)
    y_up_inv = np.asarray(data["y_upper_inverted"], dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.0))
    axes = axes.ravel()

    _plot_symmetry_pair(
        axes[0], y_low, y_up_inv,
        np.asarray(data["u_lower"], dtype=float),
        np.asarray(data["u_upper_inverted"], dtype=float),
        title="u symmetry", ylabel="u"
    )
    _plot_symmetry_pair(
        axes[1], y_low, y_up_inv,
        np.asarray(data["v_lower"], dtype=float),
        np.asarray(data["v_upper_inverted"], dtype=float),
        title="v symmetry", ylabel="v"
    )
    _plot_symmetry_pair(
        axes[2], y_low, y_up_inv,
        np.asarray(data["p_lower"], dtype=float),
        np.asarray(data["p_upper_inverted"], dtype=float),
        title="pressure symmetry", ylabel="p"
    )
    _plot_symmetry_pair(
        axes[3], y_low, y_up_inv,
        np.asarray(data["temp_lower"], dtype=float),
        np.asarray(data["temp_upper_inverted"], dtype=float),
        title="temp symmetry", ylabel="temp"
    )
    _plot_symmetry_pair(
        axes[4], y_low, y_up_inv,
        np.asarray(data["s01_lower"], dtype=float),
        np.asarray(data["s01_upper_inverted"], dtype=float),
        title="scalar_1 symmetry (s01)", ylabel="scalar_1"
    )
    _plot_symmetry_pair(
        axes[5], y_low, y_up_inv,
        np.asarray(data["s02_lower"], dtype=float),
        np.asarray(data["s02_upper_inverted"], dtype=float),
        title="scalar_2 symmetry (s02)", ylabel="scalar_2"
    )

    fig.suptitle("Symmetry check: lower 0->1 vs upper 2->1")
    fig.tight_layout()
    fig.savefig(OUTPUT_SYM_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if _npz_needs_recompute():
        compute()

    data = _load_data()
    write_csv(data)
    write_symmetry_csv(data)
    write_metadata_json(data)
    plot(data)
    plot_symmetry(data)

    print(f"Saved: {OUTPUT_NPZ}")
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_SYM_CSV}")
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_SYM_PNG}")
    print(f"Saved: {OUTPUT_JSON}")
    print(f"U_bulk = {float(data['u_bulk']):.16e}")
    print(f"Field keys = {data['available_field_keys']}")


if __name__ == "__main__":
    main()
