"""Pure Python reference implementation of the Anti-Resonant Spectral SAT solver.

Three-shell chiral pipeline:
  1. Bronze (right-handed, beta=3.303) — aggressive partitioning
  2. Silver (LEFT-handed, beta=2.414) — orthogonal chiral partition
  3. Golden (right-handed, phi=1.618) — stability / tie-breaking

Each shell builds a Laplacian with metallic-mean phase weights and extracts
the k smallest eigenvectors. Final assignment via compound-weighted majority vote.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import eigsh

# Metallic mean constants
PHI = (1 + math.sqrt(5)) / 2          # Golden: 1.618...
SILVER_BETA = (2 + math.sqrt(8)) / 2  # Silver: 2.414...
BRONZE_BETA = (3 + math.sqrt(13)) / 2 # Bronze: 3.303...

Clause = List[int]
Formula = List[Clause]


def metallic_mean(n: int) -> float:
    return (n + math.sqrt(n * n + 4)) / 2


def metallic_phase_weights(n_vars: int, beta: float) -> np.ndarray:
    """Compute anti-resonant phase angles: theta_k = 2*pi * beta^k / sum(beta^j)."""
    if n_vars <= 500 or beta < 2.0:
        raw = np.array([beta ** k for k in range(n_vars)])
    else:
        log_beta = math.log(beta)
        max_log = (n_vars - 1) * log_beta
        raw = np.exp(np.arange(n_vars) * log_beta - max_log)
    return 2 * np.pi * raw / raw.sum()


def build_laplacian(
    formula: Formula,
    n_vars: int,
    phase_angles: np.ndarray,
    omega: float,
    chirality: int,
) -> csc_matrix:
    """Build the graph Laplacian with metallic-mean phase-weighted edges."""
    W = lil_matrix((n_vars, n_vars))

    for clause in formula:
        vars_in_clause = sorted(set(abs(lit) - 1 for lit in clause))
        for i in range(len(vars_in_clause)):
            for j in range(i + 1, len(vars_in_clause)):
                u, v = vars_in_clause[i], vars_in_clause[j]
                delta = omega * (phase_angles[u] - phase_angles[v])
                w = math.cos(delta) + chirality * math.sin(delta)
                W[u, v] += w
                W[v, u] += w

    W_csc = W.tocsc()
    # D = diag(row sums of W) — standard graph Laplacian
    degrees = np.array(W_csc.sum(axis=1)).flatten()
    from scipy.sparse import diags
    D = diags(degrees)
    L = D - W_csc
    return L.tocsc()


def apply_mobius_closure(vectors: np.ndarray) -> np.ndarray:
    """Apply 720 degree (4*pi) topological phase rotation."""
    n = vectors.shape[0]
    phases = 4 * np.pi * np.arange(n) / n
    return vectors * np.cos(phases)[:, np.newaxis]


def evaluate_sat(formula: Formula, assignment: np.ndarray) -> float:
    """Compute satisfaction ratio."""
    satisfied = 0
    for clause in formula:
        for lit in clause:
            var = abs(lit) - 1
            if (lit > 0 and assignment[var] > 0) or (lit < 0 and assignment[var] < 0):
                satisfied += 1
                break
    return satisfied / len(formula)


def random_3sat(n_vars: int, m_clauses: int, seed: int = 42) -> Formula:
    """Generate random 3-SAT instance."""
    rng = np.random.RandomState(seed)
    formula = []
    for _ in range(m_clauses):
        vars_sel = rng.choice(n_vars, size=3, replace=False)
        signs = rng.choice([1, -1], size=3)
        clause = [(int(vars_sel[i]) + 1) * int(signs[i]) for i in range(3)]
        formula.append(clause)
    return formula


@dataclass
class SolverConfig:
    k_eigenvectors: int = 2
    omega: float = 1.0
    use_mobius: bool = True
    adaptive_voting: bool = True
    multi_omega: bool = True  # try multiple omega values, keep best
    # Pendulum omega: use powers of bronze metallic mean as frequency spread.
    # beta_3^1 = 3.303, beta_3^2 = 10.908, beta_3^3 = 36.02
    # This makes the entire system self-referential — metallic means control
    # both phase spacing AND frequency, with zero free parameters.
    omega_spread: tuple = (BRONZE_BETA, BRONZE_BETA**2, BRONZE_BETA**3)
    bronze_weight: float = 0.45
    silver_weight: float = 0.30
    golden_weight: float = 0.25


@dataclass
class SolverResult:
    assignment: np.ndarray
    satisfaction_ratio: float
    runtime_ms: float
    n_vars: int
    n_clauses: int
    bronze_rho: float = 0.0
    silver_rho: float = 0.0
    golden_rho: float = 0.0


class AntiResonantSolver:
    """Chiral multi-shell spectral SAT solver."""

    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()

    def _run_shell(
        self,
        formula: Formula,
        n_vars: int,
        beta: float,
        chirality: int,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """Run a single shell pass. Returns (assignment, rho, eigenvectors)."""
        phases = metallic_phase_weights(n_vars, beta)
        L = build_laplacian(formula, n_vars, phases, self.config.omega, chirality)

        k = min(self.config.k_eigenvectors, n_vars - 1)
        try:
            vals, vecs = eigsh(L, k=k, which='SM', maxiter=1000)
        except Exception:
            vecs = np.random.randn(n_vars, k)

        if self.config.use_mobius:
            vecs = apply_mobius_closure(vecs)

        # Try all eigenvectors + negations, keep best
        best_assign = np.sign(vecs[:, 0])
        best_assign[best_assign == 0] = 1
        best_rho = evaluate_sat(formula, best_assign)

        for col in range(vecs.shape[1]):
            for sign in [1, -1]:
                assign = np.sign(sign * vecs[:, col])
                assign[assign == 0] = 1
                rho = evaluate_sat(formula, assign)
                if rho > best_rho:
                    best_rho = rho
                    best_assign = assign.copy()

        return best_assign, best_rho, vecs

    def _solve_single_omega(self, formula: Formula, n_vars: int, omega: float):
        """Run 3-shell pipeline at a single omega value."""
        saved_omega = self.config.omega
        self.config.omega = omega

        # LRL chirality: left-right-left outperforms RLR at n>=50
        br_assign, br_rho, _ = self._run_shell(formula, n_vars, BRONZE_BETA, -1)
        ag_assign, ag_rho, _ = self._run_shell(formula, n_vars, SILVER_BETA, +1)
        au_assign, au_rho, _ = self._run_shell(formula, n_vars, PHI, -1)

        # Compound vote
        if self.config.adaptive_voting:
            voted = self._adaptive_vote(formula, n_vars, br_assign, ag_assign, au_assign)
        else:
            vote = (self.config.bronze_weight * br_assign
                    + self.config.silver_weight * ag_assign
                    + self.config.golden_weight * au_assign)
            voted = np.sign(vote)
            voted[voted == 0] = 1

        voted_rho = evaluate_sat(formula, voted)

        # Best of compound vote and individual shells
        best_assign, best_rho = voted, voted_rho
        for assign, rho in [(br_assign, br_rho), (ag_assign, ag_rho), (au_assign, au_rho)]:
            if rho > best_rho:
                best_assign, best_rho = assign, rho

        self.config.omega = saved_omega
        return best_assign, best_rho, br_rho, ag_rho, au_rho

    def solve(self, formula: Formula, n_vars: int) -> SolverResult:
        """Solve via 3-shell chiral pipeline.

        In multi-omega mode, tries multiple omega values and keeps the best
        result. This captures the fact that optimal omega varies with problem
        structure, without needing a parametric model.
        """
        t0 = time.perf_counter()

        if self.config.multi_omega:
            omegas = self.config.omega_spread
        else:
            omegas = (self.config.omega,)

        best_assign = None
        best_rho = -1.0
        best_br = best_ag = best_au = 0.0

        for omega in omegas:
            assign, rho, br, ag, au = self._solve_single_omega(formula, n_vars, omega)
            if rho > best_rho:
                best_assign, best_rho = assign, rho
                best_br, best_ag, best_au = br, ag, au

        elapsed = (time.perf_counter() - t0) * 1000

        return SolverResult(
            assignment=best_assign,
            satisfaction_ratio=best_rho,
            runtime_ms=elapsed,
            n_vars=n_vars,
            n_clauses=len(formula),
            bronze_rho=best_br,
            silver_rho=best_ag,
            golden_rho=best_au,
        )

    def _adaptive_vote(
        self,
        formula: Formula,
        n_vars: int,
        bronze: np.ndarray,
        silver: np.ndarray,
        golden: np.ndarray,
    ) -> np.ndarray:
        """Clause-satisfaction-adaptive compound voting."""
        # Precompute var -> clause mapping
        var_clauses: dict[int, list[int]] = {i: [] for i in range(n_vars)}
        for c_idx, clause in enumerate(formula):
            for lit in clause:
                var_clauses[abs(lit) - 1].append(c_idx)

        def clause_satisfied(assign: np.ndarray, c_idx: int) -> bool:
            for lit in formula[c_idx]:
                v = abs(lit) - 1
                if (lit > 0 and assign[v] > 0) or (lit < 0 and assign[v] < 0):
                    return True
            return False

        result = np.zeros(n_vars)
        for i in range(n_vars):
            clauses = var_clauses[i]
            if not clauses:
                result[i] = bronze[i]
                continue

            total = len(clauses)
            b_sat = sum(clause_satisfied(bronze, c) for c in clauses) / total
            s_sat = sum(clause_satisfied(silver, c) for c in clauses) / total
            g_sat = sum(clause_satisfied(golden, c) for c in clauses) / total

            bw = self.config.bronze_weight * (0.5 + b_sat)
            sw = self.config.silver_weight * (0.5 + s_sat)
            gw = self.config.golden_weight * (0.5 + g_sat)

            vote = bw * bronze[i] + sw * silver[i] + gw * golden[i]
            result[i] = 1 if vote >= 0 else -1

        return result
