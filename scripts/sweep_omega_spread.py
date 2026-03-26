"""Sweep omega_spread: test different metallic-mean power combinations.

Current default: (beta_3^1, beta_3^2, beta_3^3) = (3.303, 10.908, 36.02)

Sweep:
- Number of omega values (2, 3, 4, 5)
- Which metallic mean to use for powers (golden, silver, bronze, copper)
- Which powers (1-6)
- Mixed: use different metallic means at each position
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
from itertools import combinations
from arsat.solver import (
    AntiResonantSolver, SolverConfig,
    random_3sat, BRONZE_BETA, SILVER_BETA, PHI,
)
import math

COPPER_BETA = (4 + math.sqrt(20)) / 2  # 4.236

def test_spread(formula_cache, n, spread, num_seeds=20):
    """Test a specific omega_spread and return mean rho."""
    cfg = SolverConfig(omega_spread=tuple(spread))
    solver = AntiResonantSolver(cfg)
    rhos = [solver.solve(f, n).satisfaction_ratio for f in formula_cache]
    return np.mean(rhos)


def sweep():
    sizes = [(20, 84), (50, 210), (100, 420), (200, 840)]
    num_seeds = 20

    # Pregenerate formulas
    formula_caches = {}
    for n, m in sizes:
        formula_caches[n] = [random_3sat(n, m, seed=42+s) for s in range(num_seeds)]

    # ── Part 1: Sweep metallic mean bases ──
    print("=" * 80)
    print("Part 1: Different metallic mean bases (powers 1,2,3)")
    print("=" * 80)

    bases = {
        "Golden (phi)": PHI,
        "Silver (B2)": SILVER_BETA,
        "Bronze (B3)": BRONZE_BETA,
        "Copper (B4)": COPPER_BETA,
    }

    header = f"{'Base':>16}"
    for n, m in sizes:
        header += f" | n={n:>4}"
    print(header)
    print("-" * len(header))

    for name, beta in bases.items():
        spread = [beta**1, beta**2, beta**3]
        row = f"{name:>16}"
        for n, m in sizes:
            rho = test_spread(formula_caches[n], n, spread, num_seeds)
            row += f" | {rho:.4f}"
        print(row)

    # ── Part 2: Sweep power ranges ──
    print()
    print("=" * 80)
    print("Part 2: Different power ranges (using Bronze base)")
    print("=" * 80)

    power_sets = {
        "(1,2,3)": [1, 2, 3],
        "(1,2,4)": [1, 2, 4],
        "(1,3,5)": [1, 3, 5],
        "(2,3,4)": [2, 3, 4],
        "(1,2,3,4)": [1, 2, 3, 4],
        "(1,2,3,4,5)": [1, 2, 3, 4, 5],
        "(0.5,1,2,3)": [0.5, 1, 2, 3],
        "(1,1.5,2,3)": [1, 1.5, 2, 3],
    }

    header = f"{'Powers':>16}"
    for n, m in sizes:
        header += f" | n={n:>4}"
    print(header)
    print("-" * len(header))

    for name, powers in power_sets.items():
        spread = [BRONZE_BETA**p for p in powers]
        row = f"{name:>16}"
        for n, m in sizes:
            rho = test_spread(formula_caches[n], n, spread, num_seeds)
            row += f" | {rho:.4f}"
        print(row)

    # ── Part 3: Mixed metallic means ──
    print()
    print("=" * 80)
    print("Part 3: Mixed metallic means at each omega position")
    print("=" * 80)

    mixed_spreads = {
        "Au^2,Ag^2,Br^2": [PHI**2, SILVER_BETA**2, BRONZE_BETA**2],
        "Au^1,Br^2,Cu^3": [PHI, BRONZE_BETA**2, COPPER_BETA**3],
        "Ag^1,Br^2,Br^3": [SILVER_BETA, BRONZE_BETA**2, BRONZE_BETA**3],
        "Au^1,Ag^2,Br^3": [PHI, SILVER_BETA**2, BRONZE_BETA**3],
        "Br^1,Cu^2,Cu^3": [BRONZE_BETA, COPPER_BETA**2, COPPER_BETA**3],
        "Au^1,Br^1,Cu^1": [PHI, BRONZE_BETA, COPPER_BETA],
        "Au^2,Br^2,Cu^2": [PHI**2, BRONZE_BETA**2, COPPER_BETA**2],
        "Au^3,Br^3,Cu^3": [PHI**3, BRONZE_BETA**3, COPPER_BETA**3],
    }

    header = f"{'Spread':>16}"
    for n, m in sizes:
        header += f" | n={n:>4}"
    print(header)
    print("-" * len(header))

    for name, spread in mixed_spreads.items():
        row = f"{name:>16}"
        for n, m in sizes:
            rho = test_spread(formula_caches[n], n, spread, num_seeds)
            row += f" | {rho:.4f}"
        print(row)

    # ── Part 4: Single omega (ablation) ──
    print()
    print("=" * 80)
    print("Part 4: Single omega values (ablation study)")
    print("=" * 80)

    singles = [0.5, 1.0, PHI, SILVER_BETA, BRONZE_BETA, BRONZE_BETA**2,
               BRONZE_BETA**3, COPPER_BETA**2, 50.0, 100.0]

    header = f"{'Omega':>16}"
    for n, m in sizes:
        header += f" | n={n:>4}"
    print(header)
    print("-" * len(header))

    for omega in singles:
        cfg = SolverConfig(multi_omega=False, omega=omega)
        row = f"{omega:>16.3f}"
        for n, m in sizes:
            solver = AntiResonantSolver(cfg)
            rhos = [solver.solve(f, n).satisfaction_ratio for f in formula_caches[n]]
            row += f" | {np.mean(rhos):.4f}"
        print(row)


if __name__ == "__main__":
    sweep()
