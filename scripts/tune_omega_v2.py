"""Focused omega tuning with more seeds and parametric fit."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
from arsat.solver import (
    AntiResonantSolver, SolverConfig, metallic_phase_weights,
    random_3sat, evaluate_sat, BRONZE_BETA,
)


def dense_grid_search():
    """Dense grid search with 20 seeds for statistical robustness."""
    sizes = [(20, 84), (50, 210), (100, 420), (200, 840), (500, 2100)]
    num_seeds = 20

    # Test omega range adapted per size
    omega_ranges = {
        20:  np.concatenate([np.arange(0.5, 10, 0.5), np.arange(10, 50, 5), np.arange(50, 200, 10), np.arange(200, 500, 50)]),
        50:  np.concatenate([np.arange(0.5, 10, 0.5), np.arange(10, 100, 5), np.arange(100, 300, 25)]),
        100: np.concatenate([np.arange(0.5, 10, 0.5), np.arange(10, 100, 5), np.arange(100, 500, 25)]),
        200: np.concatenate([np.arange(0.25, 5, 0.25), np.arange(5, 50, 5), np.arange(50, 300, 25)]),
        500: np.concatenate([np.arange(0.25, 5, 0.25), np.arange(5, 50, 5), np.arange(50, 200, 25)]),
    }

    best_omegas = {}

    for n, m in sizes:
        formulas = [random_3sat(n, m, seed=42+s) for s in range(num_seeds)]
        omegas = omega_ranges[n]

        best_omega, best_rho = 1.0, 0.0
        landscape = []

        for omega in omegas:
            cfg = SolverConfig(omega=omega)
            solver = AntiResonantSolver(cfg)
            rhos = [solver.solve(f, n).satisfaction_ratio for f in formulas]
            mean_rho = np.mean(rhos)
            landscape.append((omega, mean_rho))
            if mean_rho > best_rho:
                best_rho = mean_rho
                best_omega = omega

        # Fine search around best
        for omega in np.arange(max(0.1, best_omega * 0.7), best_omega * 1.3, best_omega * 0.05):
            cfg = SolverConfig(omega=omega)
            solver = AntiResonantSolver(cfg)
            rhos = [solver.solve(f, n).satisfaction_ratio for f in formulas]
            mean_rho = np.mean(rhos)
            if mean_rho > best_rho:
                best_rho = mean_rho
                best_omega = omega

        best_omegas[n] = (best_omega, best_rho)

        # Show top 5 omegas for this size
        landscape.sort(key=lambda x: -x[1])
        print(f"\nn={n}, m={m}:")
        print(f"  Best: omega={best_omega:.2f}, rho={best_rho:.4f}")
        print(f"  Top 5:")
        for omega, rho in landscape[:5]:
            print(f"    omega={omega:>8.2f}  rho={rho:.4f}")

    # Fit parametric curve: omega(n) = a * n^b + c
    print("\n" + "=" * 60)
    print("Optimal omega per size:")
    ns = []
    oms = []
    for n in sorted(best_omegas):
        omega, rho = best_omegas[n]
        print(f"  n={n:>4}: omega={omega:>8.2f}  rho={rho:.4f}")
        ns.append(n)
        oms.append(omega)

    # Try to fit omega = a / n^b (inverse power law — larger n, smaller omega)
    # log(omega) = log(a) - b*log(n)
    ns = np.array(ns, dtype=float)
    oms = np.array(oms, dtype=float)

    # Fit log-log regression
    log_ns = np.log(ns)
    log_oms = np.log(oms)

    # Use polyfit for log-log
    coeffs = np.polyfit(log_ns, log_oms, 1)
    b = coeffs[0]
    a = np.exp(coeffs[1])

    print(f"\nPower law fit: omega(n) = {a:.2f} * n^{b:.3f}")
    print("Fitted values:")
    for n_val in ns:
        fitted = a * n_val ** b
        actual = oms[int(np.where(ns == n_val)[0][0])]
        print(f"  n={int(n_val):>4}: fitted={fitted:>8.2f}  actual={actual:>8.2f}")

    return a, b, best_omegas


def validate_scaling(a, b, best_omegas):
    """Validate the parametric scaling against const(1.0) and per-size optima."""
    print("\n" + "=" * 60)
    print("Validation: parametric omega vs const(1.0) vs per-size optima")
    print("=" * 60)

    sizes = [(20, 84), (50, 210), (100, 420), (200, 840), (500, 2100)]
    num_seeds = 20

    header = f"{'n':>5} | {'const(1.0)':>12} | {'parametric':>12} | {'per-size opt':>12} | {'omega_param':>12}"
    print(header)
    print("-" * len(header))

    for n, m in sizes:
        formulas = [random_3sat(n, m, seed=42+s) for s in range(num_seeds)]

        # const(1.0)
        cfg = SolverConfig(omega=1.0)
        r1 = np.mean([AntiResonantSolver(cfg).solve(f, n).satisfaction_ratio for f in formulas])

        # parametric
        omega_p = a * n ** b
        cfg = SolverConfig(omega=omega_p)
        r2 = np.mean([AntiResonantSolver(cfg).solve(f, n).satisfaction_ratio for f in formulas])

        # per-size optimal
        omega_opt, r3 = best_omegas[n]

        print(f"{n:>5} | {r1:>12.4f} | {r2:>12.4f} | {r3:>12.4f} | {omega_p:>12.2f}")


if __name__ == "__main__":
    a, b, best_omegas = dense_grid_search()
    validate_scaling(a, b, best_omegas)
