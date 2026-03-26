"""Benchmark suite for AntiResonantSAT.

Compares anti-resonant solver against baselines:
  - Random assignment (7/8 = 87.5%)
  - Uniform spectral (single Fiedler vector)
  - Original helical SAT (log-phase weighting)
  - AntiResonantSAT (3-shell chiral pipeline)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh

from .solver import (
    AntiResonantSolver,
    SolverConfig,
    evaluate_sat,
    random_3sat,
)


def random_baseline(formula, n_vars: int, seed: int = 0) -> float:
    """Random assignment baseline."""
    rng = np.random.RandomState(seed)
    assign = rng.choice([1, -1], size=n_vars).astype(float)
    return evaluate_sat(formula, assign)


def uniform_spectral_baseline(formula, n_vars: int) -> float:
    """Uniform spectral baseline — single Fiedler vector with constant edge weights."""
    from scipy.sparse import diags
    W = lil_matrix((n_vars, n_vars))
    for clause in formula:
        vars_in = sorted(set(abs(lit) - 1 for lit in clause))
        for i in range(len(vars_in)):
            for j in range(i + 1, len(vars_in)):
                u, v = vars_in[i], vars_in[j]
                W[u, v] -= 1.0
                W[v, u] -= 1.0

    W_csc = W.tocsc()
    degrees = np.array(np.abs(W_csc).sum(axis=1)).flatten()
    L = diags(degrees) - W_csc

    try:
        _, vec = eigsh(L.tocsc(), k=1, which='SM', maxiter=500)
        assign = np.sign(vec[:, 0])
        assign[assign == 0] = 1
    except Exception:
        assign = np.ones(n_vars)

    return evaluate_sat(formula, assign)


def helical_baseline(formula, n_vars: int, omega: float = 0.3, N: float = 20000) -> float:
    """Original helical SAT heuristic — log-phase weighting with Mobius closure."""
    from scipy.sparse import diags
    W = lil_matrix((n_vars, n_vars))
    for clause in formula:
        vars_in = sorted(set(abs(lit) - 1 for lit in clause))
        for i in range(len(vars_in)):
            for j in range(i + 1, len(vars_in)):
                u, v = vars_in[i], vars_in[j]
                theta_u = 2 * np.pi * np.log(u + 1) / N
                theta_v = 2 * np.pi * np.log(v + 1) / N
                w = np.cos(omega * (theta_u - theta_v))
                W[u, v] += w
                W[v, u] += w

    W_csc = W.tocsc()
    degrees = np.array(np.abs(W_csc).sum(axis=1)).flatten()
    L = diags(degrees) - W_csc

    try:
        _, vec = eigsh(L.tocsc(), k=1, which='SM', maxiter=500)
        # Mobius closure
        n = vec.shape[0]
        phases = 4 * np.pi * np.arange(n) / n
        vec = vec * np.cos(phases)[:, np.newaxis]
        assign = np.sign(vec[:, 0])
        assign[assign == 0] = 1
    except Exception:
        assign = np.ones(n_vars)

    return evaluate_sat(formula, assign)


@dataclass
class BenchmarkResult:
    n_vars: int
    m_clauses: int
    method: str
    mean_rho: float
    std_rho: float
    best_rho: float
    mean_ms: float


def run_benchmarks(
    sizes: Optional[List[tuple]] = None,
    num_seeds: int = 10,
    config: Optional[SolverConfig] = None,
):
    """Run full benchmark suite."""
    if sizes is None:
        sizes = [(20, 84), (50, 210), (100, 420), (200, 840), (500, 2100)]

    if config is None:
        config = SolverConfig()

    print("=" * 90)
    print("AntiResonantSAT Benchmark — Chiral Multi-Shell Spectral SAT")
    print(f"Shells: Bronze(+1, B=3.303) -> Silver(-1, B=2.414) -> Golden(+1, phi=1.618)")
    print(f"k={config.k_eigenvectors}, w={config.omega:.2f}, "
          f"Mobius={'ON' if config.use_mobius else 'OFF'}, "
          f"adaptive={'ON' if config.adaptive_voting else 'OFF'}")
    print("=" * 90)

    header = f"{'n':>5} {'m':>6} | {'Method':<18} | {'rho mean':>8} {'+/-std':>7} {'rho best':>8} | {'ms':>8} | {'D% vs rand':>10}"
    print(header)
    print("-" * len(header))

    for n, m in sizes:
        methods = {}

        # Generate instances
        formulas = [random_3sat(n, m, seed=42 + s) for s in range(num_seeds)]

        # Random baseline
        rhos = [random_baseline(f, n, seed=s) for s, f in enumerate(formulas)]
        methods["Random (7/8)"] = (np.mean(rhos), np.std(rhos, ddof=1), max(rhos), 0.0)

        # Uniform spectral
        t0 = time.perf_counter()
        rhos = [uniform_spectral_baseline(f, n) for f in formulas]
        ms = (time.perf_counter() - t0) * 1000 / num_seeds
        methods["Uniform spectral"] = (np.mean(rhos), np.std(rhos, ddof=1), max(rhos), ms)

        # Helical (original)
        t0 = time.perf_counter()
        rhos = [helical_baseline(f, n) for f in formulas]
        ms = (time.perf_counter() - t0) * 1000 / num_seeds
        methods["Helical (log-phase)"] = (np.mean(rhos), np.std(rhos, ddof=1), max(rhos), ms)

        # AntiResonantSAT
        solver = AntiResonantSolver(config)
        rhos = []
        total_ms = 0
        for f in formulas:
            result = solver.solve(f, n)
            rhos.append(result.satisfaction_ratio)
            total_ms += result.runtime_ms
        ms = total_ms / num_seeds
        methods["AntiResonant (3-shell)"] = (np.mean(rhos), np.std(rhos, ddof=1), max(rhos), ms)

        # Print results for this size
        rand_mean = methods["Random (7/8)"][0]
        first = True
        for method_name, (mean, std, best, runtime) in methods.items():
            delta = ((mean - rand_mean) / rand_mean) * 100 if rand_mean > 0 else 0
            prefix = f"{n:>5} {m:>6}" if first else " " * 12
            marker = " *" if method_name == "AntiResonant (3-shell)" and mean == max(v[0] for v in methods.values()) else ""
            print(f"{prefix} | {method_name:<18} | {mean:>8.4f} {std:>7.4f} {best:>8.4f} | {runtime:>7.1f}  | {delta:>+9.2f}%{marker}")
            first = False
        print("-" * len(header))

    print("=" * 90)
    print("* = best method for that problem size")


if __name__ == "__main__":
    run_benchmarks()
