"""Push n=200 and n=500 above 90%.

The spectral ceiling at large n has two causes:
1. Phase angle compression (metallic-mean β^k collapses to uniform)
2. k=2 eigenvectors carry too little information for large problems

Approaches to test:
A. More eigenvectors (k=4, 8, 16)
B. Greedy 1-flip post-processing (flip any variable that improves rho)
C. Clause-focused reweighting (boost unsatisfied clause weights between shells)
D. Multi-resolution phase: β^(k/sqrt(n)) instead of β^k
E. Additional shells (copper β₄, nickel β₅)
F. Combination of best approaches
"""

import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
from arsat.solver import (
    AntiResonantSolver, SolverConfig, SolverResult,
    metallic_phase_weights, build_laplacian, apply_mobius_closure,
    evaluate_sat, random_3sat, BRONZE_BETA, SILVER_BETA, PHI,
    SpectralBasisCache,
)
from scipy.sparse.linalg import eigsh


# ── Approach B: Greedy 1-flip post-processing ────────────────────────

def greedy_flip(formula, assignment, max_passes=3):
    """Flip variables that improve satisfaction. O(n*m) per pass."""
    assign = assignment.copy()
    n = len(assign)
    improved = True
    passes = 0

    while improved and passes < max_passes:
        improved = False
        passes += 1
        current_rho = evaluate_sat(formula, assign)

        for i in range(n):
            assign[i] = -assign[i]  # flip
            new_rho = evaluate_sat(formula, assign)
            if new_rho > current_rho:
                current_rho = new_rho
                improved = True
            else:
                assign[i] = -assign[i]  # unflip

    return assign, current_rho


def greedy_flip_fast(formula, assignment):
    """Fast greedy flip using clause-variable index. O(m) per pass."""
    assign = assignment.copy()
    n = len(assign)
    m = len(formula)

    # Precompute: for each variable, which clauses contain it and how
    var_clauses = [[] for _ in range(n)]  # (clause_idx, literal_sign)
    for c_idx, clause in enumerate(formula):
        for lit in clause:
            v = abs(lit) - 1
            var_clauses[v].append((c_idx, 1 if lit > 0 else -1))

    # Precompute clause satisfaction
    clause_sat = np.zeros(m, dtype=bool)
    for c_idx, clause in enumerate(formula):
        for lit in clause:
            v = abs(lit) - 1
            if (lit > 0 and assign[v] > 0) or (lit < 0 and assign[v] < 0):
                clause_sat[c_idx] = True
                break

    improved = True
    passes = 0
    while improved and passes < 3:
        improved = False
        passes += 1
        for i in range(n):
            # Count how many clauses would change satisfaction if we flip var i
            gain = 0
            for c_idx, sign in var_clauses[i]:
                was_sat = clause_sat[c_idx]
                # After flip: this literal's contribution changes
                # Check if clause would be satisfied without this var's current contribution
                clause = formula[c_idx]
                would_be_sat = False
                for lit in clause:
                    v = abs(lit) - 1
                    if v == i:
                        # After flip: this literal satisfied if sign matches new assignment
                        if (sign > 0 and assign[v] < 0) or (sign < 0 and assign[v] > 0):
                            would_be_sat = True
                            break
                    else:
                        if (lit > 0 and assign[v] > 0) or (lit < 0 and assign[v] < 0):
                            would_be_sat = True
                            break

                if was_sat and not would_be_sat:
                    gain -= 1
                elif not was_sat and would_be_sat:
                    gain += 1

            if gain > 0:
                assign[i] = -assign[i]
                # Update clause_sat
                for c_idx, sign in var_clauses[i]:
                    clause = formula[c_idx]
                    clause_sat[c_idx] = False
                    for lit in clause:
                        v = abs(lit) - 1
                        if (lit > 0 and assign[v] > 0) or (lit < 0 and assign[v] < 0):
                            clause_sat[c_idx] = True
                            break
                improved = True

    return assign, clause_sat.sum() / m


# ── Approach C: Clause-focused reweighting ───────────────────────────

def clause_reweight_solve(formula, n_vars, config, n_rounds=2):
    """Run spectral solve, then re-weight unsatisfied clauses and re-solve."""
    solver = AntiResonantSolver(config)
    result = solver.solve(formula, n_vars)
    best_assign = result.assignment
    best_rho = result.satisfaction_ratio

    # Identify unsatisfied clauses and create boosted formula
    for round_idx in range(n_rounds):
        boosted = list(formula)
        for clause in formula:
            sat = False
            for lit in clause:
                v = abs(lit) - 1
                if (lit > 0 and best_assign[v] > 0) or (lit < 0 and best_assign[v] < 0):
                    sat = True
                    break
            if not sat:
                # Duplicate unsatisfied clause (boost its weight)
                boosted.append(clause)
                boosted.append(clause)

        result2 = solver.solve(boosted, n_vars)
        rho_original = evaluate_sat(formula, result2.assignment)
        if rho_original > best_rho:
            best_rho = rho_original
            best_assign = result2.assignment

    return best_assign, best_rho


# ── Approach D: Multi-resolution phase spacing ───────────────────────

def sqrt_phase_weights(n_vars, beta):
    """Phase weights using β^(k/sqrt(n)) — maintains separation at large n."""
    scale = math.sqrt(n_vars)
    raw = np.array([beta ** (k / scale) for k in range(n_vars)])
    return 2 * np.pi * raw / raw.sum()


# ── Run all approaches ───────────────────────────────────────────────

def test_approaches():
    sizes = [(100, 420), (200, 840), (500, 2100)]
    num_seeds = 15

    approaches = {}

    for n, m in sizes:
        formulas = [random_3sat(n, m, seed=42+s) for s in range(num_seeds)]
        print(f"\n{'='*70}")
        print(f"n={n}, m={m}, {num_seeds} seeds")
        print(f"{'='*70}")

        # Baseline: current AntiResonantSAT
        cfg = SolverConfig()
        solver = AntiResonantSolver(cfg)
        rhos = [solver.solve(f, n).satisfaction_ratio for f in formulas]
        print(f"  Baseline (k=2):           {np.mean(rhos):.4f} +/- {np.std(rhos):.4f}  best={max(rhos):.4f}")

        # A: More eigenvectors
        for k_val in [4, 8]:
            cfg_k = SolverConfig(k_eigenvectors=k_val)
            solver_k = AntiResonantSolver(cfg_k)
            rhos = [solver_k.solve(f, n).satisfaction_ratio for f in formulas]
            print(f"  A. k={k_val} eigenvectors:      {np.mean(rhos):.4f} +/- {np.std(rhos):.4f}  best={max(rhos):.4f}")

        # B: Greedy flip on baseline
        cfg = SolverConfig()
        solver = AntiResonantSolver(cfg)
        rhos_flip = []
        for f in formulas:
            result = solver.solve(f, n)
            _, rho = greedy_flip(f, result.assignment, max_passes=2)
            rhos_flip.append(rho)
        print(f"  B. Baseline + greedy flip: {np.mean(rhos_flip):.4f} +/- {np.std(rhos_flip):.4f}  best={max(rhos_flip):.4f}")

        # B2: k=4 + greedy flip
        cfg_k4 = SolverConfig(k_eigenvectors=4)
        solver_k4 = AntiResonantSolver(cfg_k4)
        rhos_k4_flip = []
        for f in formulas:
            result = solver_k4.solve(f, n)
            _, rho = greedy_flip(f, result.assignment, max_passes=2)
            rhos_k4_flip.append(rho)
        print(f"  B2. k=4 + greedy flip:    {np.mean(rhos_k4_flip):.4f} +/- {np.std(rhos_k4_flip):.4f}  best={max(rhos_k4_flip):.4f}")

        # C: Clause reweighting
        cfg = SolverConfig()
        rhos_rw = []
        for f in formulas:
            _, rho = clause_reweight_solve(f, n, cfg, n_rounds=2)
            rhos_rw.append(rho)
        print(f"  C. Clause reweight (2rnd): {np.mean(rhos_rw):.4f} +/- {np.std(rhos_rw):.4f}  best={max(rhos_rw):.4f}")

        # C+B: Clause reweight + greedy flip
        rhos_rw_flip = []
        for f in formulas:
            assign, _ = clause_reweight_solve(f, n, cfg, n_rounds=2)
            _, rho = greedy_flip(f, assign, max_passes=2)
            rhos_rw_flip.append(rho)
        print(f"  C+B. Reweight + flip:     {np.mean(rhos_rw_flip):.4f} +/- {np.std(rhos_rw_flip):.4f}  best={max(rhos_rw_flip):.4f}")

        # F: Full combo: k=4 + greedy flip (3 passes)
        cfg_full = SolverConfig(k_eigenvectors=4)
        solver_full = AntiResonantSolver(cfg_full)
        rhos_full = []
        for f in formulas:
            result = solver_full.solve(f, n)
            _, rho = greedy_flip(f, result.assignment, max_passes=3)
            rhos_full.append(rho)
        print(f"  F. k=4 + 3-pass flip:     {np.mean(rhos_full):.4f} +/- {np.std(rhos_full):.4f}  best={max(rhos_full):.4f}")


if __name__ == "__main__":
    test_approaches()
