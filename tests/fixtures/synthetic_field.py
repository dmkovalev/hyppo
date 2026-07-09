"""
Synthetic oil field generator for reproducibility of HybridCRM experiments.

Generates a synthetic waterflood field with realistic CRM+Buckley-Leverett
physics, calibrated to match statistical properties of the real AS10 field
(see Table synth_validation in Chapter 4).

Usage::

    from tests.fixtures.synthetic_field import generate_synthetic_field
    data = generate_synthetic_field(n_producers=47, n_injectors=12, n_months=48)

Reference: Kovalev D.Yu., "Methods and tools for managing virtual experiments
in data-intensive domains", Chapter 4, Section 4.4 (HybridCRM domain).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class FieldParams:
    """Physical parameters calibrated to AS10 statistics."""

    swc: float = 0.20  # Connate water saturation
    sor: float = 0.20  # Residual oil saturation
    bl_M: float = 3.0  # Buckley-Leverett mobility ratio
    bl_n: float = 2.5  # Corey exponent
    vp_min: float = 100_000.0  # Min pore volume, m3
    vp_max: float = 500_000.0  # Max pore volume, m3
    sw0_min: float = 0.25  # Min initial water saturation
    sw0_max: float = 0.55  # Max initial water saturation
    inj_min: float = 1_000.0  # Min base injection, m3/month
    inj_max: float = 4_000.0  # Max base injection, m3/month
    noise_inj: float = 200.0  # Injection noise std
    noise_wct: float = 0.02  # Water cut measurement noise std
    f_weight_min: float = 0.3  # Min CRM connectivity weight
    f_weight_max: float = 0.8  # Max CRM connectivity weight


def buckley_leverett(
    sw: np.ndarray,
    swc: float,
    sor: float,
    bl_M: float,
    bl_n: float,
) -> np.ndarray:
    """
    Buckley-Leverett fractional flow: fw(S) = 1 / (1 + kro / (M * krw)).

    :param sw: Water saturation array, any shape.
    :param swc: Connate water saturation.
    :param sor: Residual oil saturation.
    :param bl_M: Mobility ratio.
    :param bl_n: Corey exponent.
    :returns: Fractional flow of water (fw), same shape as *sw*.
    """
    sw_norm = (sw - swc) / (1.0 - swc - sor + 1e-9)
    sw_norm = np.clip(sw_norm, 1e-6, 1.0 - 1e-6)

    krw = sw_norm**bl_n
    kro = (1.0 - sw_norm) ** bl_n

    fw = 1.0 / (1.0 + kro / (bl_M * krw + 1e-9))
    return np.clip(fw, 0.0, 1.0)


def material_balance_sw(
    sw0: np.ndarray,
    cum_inj: np.ndarray,
    vp: np.ndarray,
    sor: float,
    swc: float,
) -> np.ndarray:
    """
    Simplified material balance: Sw(t) = Sw0 + (Sw_max - Sw0) * (1 - exp(-cum_inj/Vp)).

    :param sw0: Initial water saturation, shape ``(n_wells,)`` or ``(n_wells, 1)``.
    :param cum_inj: Cumulative weighted injection, shape ``(n_wells, n_months)``.
    :param vp: Pore volume per well, shape ``(n_wells,)``.
    :param sor: Residual oil saturation.
    :param swc: Connate water saturation.
    :returns: Water saturation trajectory, shape ``(n_wells, n_months)``.
    """
    sw_max = 1.0 - sor - 0.01
    vp_2d = vp[:, np.newaxis]
    sw0_2d = sw0[:, np.newaxis] if sw0.ndim == 1 else sw0

    sw = sw0_2d + (sw_max - sw0_2d) * (1.0 - np.exp(-cum_inj / (vp_2d + 1e-9)))
    return np.clip(sw, swc + 0.01, sw_max)


def generate_synthetic_field(
    n_producers: int = 47,
    n_injectors: int = 12,
    n_months: int = 48,
    seed: int = 42,
    params: FieldParams | None = None,
) -> Dict[str, np.ndarray]:
    """
    Generate a synthetic waterflood field with CRM + Buckley-Leverett physics.

    Default parameters (47 producers, 12 injectors, 48 months) match the
    HybridCRM experiment configuration in Chapter 4.

    :param n_producers: Number of producer wells.
    :param n_injectors: Number of injector wells.
    :param n_months: History length in months.
    :param seed: Random seed for reproducibility.
    :param params: Physical parameters (defaults calibrated to AS10).
    :returns: Dictionary with keys:

        - ``opr``: Oil production rate, shape ``(n_producers, n_months)``
        - ``lpr``: Liquid production rate, shape ``(n_producers, n_months)``
        - ``wct``: Water cut, shape ``(n_producers, n_months)``
        - ``inj``: Injection rates, shape ``(n_injectors, n_months)``
        - ``connectivity``: CRM connectivity, shape ``(n_producers, n_injectors)``
        - ``coords_prod``: Producer coordinates, shape ``(n_producers, 2)``
        - ``coords_inj``: Injector coordinates, shape ``(n_injectors, 2)``
        - ``params``: :class:`FieldParams` used for generation
    """
    if params is None:
        params = FieldParams()

    rng = np.random.RandomState(seed)

    # --- Well coordinates (random placement on 2 km x 2 km field) ---
    coords_prod = rng.uniform(0, 2000, size=(n_producers, 2))
    coords_inj = rng.uniform(0, 2000, size=(n_injectors, 2))

    # --- CRM connectivity matrix based on distance ---
    # f_{ij} ~ exp(-dist / scale), then row-normalize
    dist = np.linalg.norm(
        coords_prod[:, np.newaxis, :] - coords_inj[np.newaxis, :, :],
        axis=2,
    )  # (n_producers, n_injectors)
    scale = 800.0
    f_raw = np.exp(-dist / scale)
    # Row-normalize and scale to [f_weight_min, f_weight_max]
    f_sum = f_raw.sum(axis=1, keepdims=True) + 1e-9
    connectivity = f_raw / f_sum
    connectivity = params.f_weight_min + connectivity * (
        params.f_weight_max - params.f_weight_min
    )

    # --- True physical parameters per producer ---
    vp_true = rng.uniform(params.vp_min, params.vp_max, size=n_producers)
    sw0_true = rng.uniform(params.sw0_min, params.sw0_max, size=n_producers)

    # --- Injection rates (n_injectors, n_months) ---
    base_inj = rng.uniform(params.inj_min, params.inj_max, size=n_injectors)
    time_mod = 1.0 + 0.3 * np.sin(np.arange(n_months, dtype=np.float64) * 0.05)
    inj = base_inj[:, np.newaxis] * time_mod[np.newaxis, :]
    inj += rng.randn(n_injectors, n_months) * params.noise_inj
    inj = np.clip(inj, 0.0, None)

    # --- Weighted injection per producer (CRM physics) ---
    # weighted_inj[p, t] = sum_i connectivity[p, i] * inj[i, t]
    weighted_inj = connectivity @ inj  # (n_producers, n_months)
    cum_inj = np.cumsum(weighted_inj, axis=1)

    # --- Water saturation and fractional flow ---
    sw = material_balance_sw(sw0_true, cum_inj, vp_true, params.sor, params.swc)
    fw = buckley_leverett(sw, params.swc, params.sor, params.bl_M, params.bl_n)

    # --- Observable quantities ---
    wct = fw + rng.randn(n_producers, n_months) * params.noise_wct
    wct = np.clip(wct, 0.0, 1.0)

    lpr = weighted_inj * 0.9 + rng.randn(n_producers, n_months) * 100.0
    lpr = np.clip(lpr, 10.0, None)

    opr = lpr * (1.0 - wct)

    return {
        "opr": opr,
        "lpr": lpr,
        "wct": wct,
        "inj": inj,
        "connectivity": connectivity,
        "coords_prod": coords_prod,
        "coords_inj": coords_inj,
        "params": params,
    }


# ---------------------------------------------------------------------------
# CLI: generate and save to CSV
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Generate synthetic oil field for HybridCRM experiments.",
    )
    parser.add_argument("--out-dir", default="synthetic_data", help="Output directory")
    parser.add_argument("--n-producers", type=int, default=47)
    parser.add_argument("--n-injectors", type=int, default=12)
    parser.add_argument("--n-months", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = generate_synthetic_field(
        n_producers=args.n_producers,
        n_injectors=args.n_injectors,
        n_months=args.n_months,
        seed=args.seed,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    import csv

    # Save production data
    with open(os.path.join(args.out_dir, "production.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["well", "month", "opr", "lpr", "wct"])
        for w in range(args.n_producers):
            for t in range(args.n_months):
                writer.writerow(
                    [
                        f"P{w + 1:03d}",
                        t + 1,
                        f"{data['opr'][w, t]:.2f}",
                        f"{data['lpr'][w, t]:.2f}",
                        f"{data['wct'][w, t]:.4f}",
                    ]
                )

    # Save injection data
    with open(os.path.join(args.out_dir, "injection.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["well", "month", "inj"])
        for w in range(args.n_injectors):
            for t in range(args.n_months):
                writer.writerow([f"I{w + 1:03d}", t + 1, f"{data['inj'][w, t]:.2f}"])

    # Save coordinates
    with open(os.path.join(args.out_dir, "coords.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["well", "x", "y", "type"])
        for w in range(args.n_producers):
            writer.writerow(
                [
                    f"P{w + 1:03d}",
                    f"{data['coords_prod'][w, 0]:.1f}",
                    f"{data['coords_prod'][w, 1]:.1f}",
                    "producer",
                ]
            )
        for w in range(args.n_injectors):
            writer.writerow(
                [
                    f"I{w + 1:03d}",
                    f"{data['coords_inj'][w, 0]:.1f}",
                    f"{data['coords_inj'][w, 1]:.1f}",
                    "injector",
                ]
            )

    # Save connectivity
    with open(os.path.join(args.out_dir, "connectivity.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        header = ["producer"] + [f"I{w + 1:03d}" for w in range(args.n_injectors)]
        writer.writerow(header)
        for w in range(args.n_producers):
            writer.writerow(
                [f"P{w + 1:03d}"]
                + [f"{data['connectivity'][w, i]:.4f}" for i in range(args.n_injectors)]
            )

    print(f"Synthetic field saved to {args.out_dir}/")
    print(
        f"  {args.n_producers} producers, {args.n_injectors} injectors, "
        f"{args.n_months} months"
    )
    print(f"  OPR range: [{data['opr'].min():.0f}, {data['opr'].max():.0f}] m3/month")
    print(f"  WCT range: [{data['wct'].min():.3f}, {data['wct'].max():.3f}]")
