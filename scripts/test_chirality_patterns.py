"""Test different chirality patterns for the 3 shells.

Current: Bronze(+1), Silver(-1), Golden(+1) — "right-left-right"
Test:    Bronze(-1), Silver(+1), Golden(-1) — "left-right-left" (mirror)
Also:    All same (+1,+1,+1), all same (-1,-1,-1), mixed patterns
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import time
import numpy as np
from arsat.solver import (
    SolverConfig, metallic_phase_weights, build_laplacian,
    apply_mobius_closure, evaluate_sat, random_3sat,
    BRONZE_BETA, SILVER_BETA, PHI,
)
from scipy.sparse.linalg import eigsh


def solve_with_chirality(formula, n_vars, chirality_pattern, config=None):
    """Run 3-shell pipeline with specified chirality pattern.
    chirality_pattern: tuple of 3 ints, e.g. (+1, -1, +1)
    """
    if config is None:
        config = SolverConfig()

    betas = [BRONZE_BETA, SILVER_BETA, PHI]
    shell_assigns = []
    shell_rhos = []

    for beta, chirality in zip(betas, chirality_pattern):
        for omega in config.omega_spread:
            phases = metallic_phase_weights(n_vars, beta)
            L = build_laplacian(formula, n_vars, phases, omega, chirality)

            k = min(config.k_eigenvectors, n_vars - 1)
            try:
                vals, vecs = eigsh(L, k=k, which='SM', maxiter=1000)
            except Exception:
                vecs = np.random.randn(n_vars, k)

            if config.use_mobius:
                vecs = apply_mobius_closure(vecs)

            # Try all eigenvectors + negations
            for col in range(vecs.shape[1]):
                for sign in [1, -1]:
                    assign = np.sign(sign * vecs[:, col])
                    assign[assign == 0] = 1
                    rho = evaluate_sat(formula, assign)
                    shell_assigns.append(assign)
                    shell_rhos.append(rho)

    # Best individual
    best_idx = np.argmax(shell_rhos)
    best_assign = shell_assigns[best_idx]
    best_rho = shell_rhos[best_idx]

    # Also try compound vote of best from each shell group
    # (each shell group = 3 omegas * k eigenvectors * 2 signs = 12 candidates)
    per_shell = len(config.omega_spread) * config.k_eigenvectors * 2
    shell_bests = []
    for s in range(3):
        start = s * per_shell
        end = start + per_shell
        group_rhos = shell_rhos[start:end]
        best_in_group = np.argmax(group_rhos)
        shell_bests.append(shell_assigns[start + best_in_group])

    # Compound vote
    vote = (config.bronze_weight * shell_bests[0]
            + config.silver_weight * shell_bests[1]
            + config.golden_weight * shell_bests[2])
    voted = np.sign(vote)
    voted[voted == 0] = 1
    voted_rho = evaluate_sat(formula, voted)

    if voted_rho > best_rho:
        best_rho = voted_rho
        best_assign = voted

    return best_rho


def main():
    sizes = [(20, 84), (50, 210), (100, 420), (200, 840)]
    num_seeds = 20

    patterns = {
        "(+1,-1,+1) RLR": (+1, -1, +1),
        "(-1,+1,-1) LRL": (-1, +1, -1),
        "(+1,+1,+1) RRR": (+1, +1, +1),
        "(-1,-1,-1) LLL": (-1, -1, -1),
        "(+1,-1,-1) RLL": (+1, -1, -1),
        "(-1,+1,+1) LRR": (-1, +1, +1),
        "(-1,-1,+1) LLR": (-1, -1, +1),
        "(+1,+1,-1) RRL": (+1, +1, -1),
    }

    print("=" * 90)
    print("Chirality Pattern Comparison")
    print("Shells: Bronze, Silver, Golden")
    print("Pendulum omega: (3.303, 10.908, 36.02)")
    print("=" * 90)

    header = f"{'n':>5} {'m':>6}"
    for name in patterns:
        header += f" | {name:>15}"
    print(header)
    print("-" * len(header))

    for n, m in sizes:
        formulas = [random_3sat(n, m, seed=42+s) for s in range(num_seeds)]
        row = f"{n:>5} {m:>6}"

        for name, pattern in patterns.items():
            rhos = [solve_with_chirality(f, n, pattern) for f in formulas]
            mean_rho = np.mean(rhos)
            row += f" | {mean_rho:>15.4f}"

        print(row)

    print("=" * 90)


if __name__ == "__main__":
    main()
