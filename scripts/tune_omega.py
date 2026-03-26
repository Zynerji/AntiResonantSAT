"""Omega tuning for AntiResonantSAT.

The advantage compresses at large n because metallic-mean phase angles
cluster together after normalization. This script:
1. Profiles the phase angle distribution vs n
2. Tests omega scaling laws: constant, sqrt(n), n, log(n), n^0.75
3. Grid searches for optimal omega at each problem size
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
from arsat.solver import (
    AntiResonantSolver, SolverConfig, metallic_phase_weights,
    random_3sat, evaluate_sat, BRONZE_BETA, SILVER_BETA, PHI,
)

def profile_phase_angles():
    """Show how phase angle differences compress with n."""
    print("=" * 70)
    print("Phase Angle Profile: delta_theta statistics vs n")
    print("=" * 70)
    print(f"{'n':>6} | {'beta':>6} | {'mean dtheta':>12} | {'max dtheta':>12} | {'omega*mean':>12}")
    print("-" * 70)

    for n in [20, 50, 100, 200, 500, 1000]:
        for beta, name in [(BRONZE_BETA, "Br"), (SILVER_BETA, "Ag"), (PHI, "Au")]:
            phases = metallic_phase_weights(n, beta)
            # Compute pairwise differences for adjacent variables
            diffs = np.abs(np.diff(phases))
            # Also compute all pairwise differences (sampling)
            idx = np.random.RandomState(42).choice(n, size=min(n, 100), replace=False)
            all_diffs = []
            for i in range(len(idx)):
                for j in range(i+1, len(idx)):
                    all_diffs.append(abs(phases[idx[i]] - phases[idx[j]]))
            all_diffs = np.array(all_diffs)

            print(f"{n:>6} | {name:>6} | {all_diffs.mean():>12.6f} | {all_diffs.max():>12.6f} | {1.0*all_diffs.mean():>12.6f}")
    print()


def test_omega_scaling():
    """Test different omega scaling laws."""
    print("=" * 70)
    print("Omega Scaling Laws: rho vs n for different omega(n) functions")
    print("=" * 70)

    sizes = [(20, 84), (50, 210), (100, 420), (200, 840), (500, 2100)]
    num_seeds = 10

    # Scaling laws to test
    scaling_laws = {
        "const(1.0)": lambda n: 1.0,
        "sqrt(n)":    lambda n: np.sqrt(n),
        "n^0.75":     lambda n: n ** 0.75,
        "n":          lambda n: float(n),
        "log(n)*3":   lambda n: 3 * np.log(n),
        "n/log(n)":   lambda n: n / np.log(n),
        "sqrt(n)*2":  lambda n: 2 * np.sqrt(n),
    }

    # Header
    header = f"{'n':>5} {'m':>6}"
    for name in scaling_laws:
        header += f" | {name:>12}"
    print(header)
    print("-" * len(header))

    results = {}
    for n, m in sizes:
        formulas = [random_3sat(n, m, seed=42+s) for s in range(num_seeds)]
        row = f"{n:>5} {m:>6}"

        for law_name, law_fn in scaling_laws.items():
            omega = law_fn(n)
            cfg = SolverConfig(omega=omega)
            solver = AntiResonantSolver(cfg)

            rhos = []
            for f in formulas:
                result = solver.solve(f, n)
                rhos.append(result.satisfaction_ratio)
            mean_rho = np.mean(rhos)
            row += f" | {mean_rho:>12.4f}"
            results[(n, law_name)] = mean_rho

        print(row)

    print()

    # Find best scaling law per size
    print("Best scaling law per size:")
    for n, m in sizes:
        best_law = max(scaling_laws.keys(), key=lambda k: results[(n, k)])
        best_rho = results[(n, best_law)]
        print(f"  n={n}: {best_law} (rho={best_rho:.4f})")
    print()

    # Find best overall (average rank across sizes)
    print("Average rho across all sizes:")
    for law_name in scaling_laws:
        avg = np.mean([results[(n, law_name)] for n, m in sizes])
        print(f"  {law_name:>12}: {avg:.4f}")


def grid_search_omega():
    """Fine-grained grid search for optimal omega at each size."""
    print("=" * 70)
    print("Grid Search: optimal constant omega per problem size")
    print("=" * 70)

    sizes = [(20, 84), (50, 210), (100, 420), (200, 840), (500, 2100)]
    num_seeds = 10

    for n, m in sizes:
        formulas = [random_3sat(n, m, seed=42+s) for s in range(num_seeds)]
        best_omega = 1.0
        best_rho = 0.0

        # Coarse search
        for omega in np.concatenate([np.arange(0.5, 5.0, 0.5), np.arange(5, 50, 5), np.arange(50, 500, 50)]):
            cfg = SolverConfig(omega=omega)
            solver = AntiResonantSolver(cfg)
            rhos = [solver.solve(f, n).satisfaction_ratio for f in formulas]
            mean_rho = np.mean(rhos)
            if mean_rho > best_rho:
                best_rho = mean_rho
                best_omega = omega

        # Fine search around best
        for omega in np.arange(max(0.1, best_omega - 2), best_omega + 2, 0.25):
            cfg = SolverConfig(omega=omega)
            solver = AntiResonantSolver(cfg)
            rhos = [solver.solve(f, n).satisfaction_ratio for f in formulas]
            mean_rho = np.mean(rhos)
            if mean_rho > best_rho:
                best_rho = mean_rho
                best_omega = omega

        print(f"  n={n:>4}: best omega={best_omega:>8.2f}  rho={best_rho:.4f}")

    print()


if __name__ == "__main__":
    profile_phase_angles()
    test_omega_scaling()
    grid_search_omega()
