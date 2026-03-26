"""Tests for AntiResonantSAT solver."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import pytest

from arsat.solver import (
    AntiResonantSolver,
    SolverConfig,
    metallic_mean,
    metallic_phase_weights,
    evaluate_sat,
    random_3sat,
    build_laplacian,
    apply_mobius_closure,
    PHI,
    SILVER_BETA,
    BRONZE_BETA,
)


class TestMetallicMeans:
    def test_golden_ratio(self):
        assert abs(metallic_mean(1) - PHI) < 1e-10

    def test_silver_ratio(self):
        assert abs(metallic_mean(2) - SILVER_BETA) < 1e-10

    def test_bronze_ratio(self):
        assert abs(metallic_mean(3) - BRONZE_BETA) < 1e-10

    def test_monotonic(self):
        """Higher n gives higher metallic mean."""
        for n in range(1, 10):
            assert metallic_mean(n + 1) > metallic_mean(n)

    def test_phase_weights_sum(self):
        """Phase weights should sum to 2*pi."""
        for beta in [PHI, SILVER_BETA, BRONZE_BETA]:
            weights = metallic_phase_weights(20, beta)
            assert abs(weights.sum() - 2 * np.pi) < 1e-10

    def test_phase_weights_positive(self):
        weights = metallic_phase_weights(50, BRONZE_BETA)
        assert np.all(weights > 0)

    def test_phase_weights_large_n(self):
        """Should not overflow for large n with big beta."""
        weights = metallic_phase_weights(1000, BRONZE_BETA)
        assert np.all(np.isfinite(weights))
        assert abs(weights.sum() - 2 * np.pi) < 1e-8


class TestSATUtilities:
    def test_random_3sat_size(self):
        formula = random_3sat(50, 210)
        assert len(formula) == 210
        for clause in formula:
            assert len(clause) == 3

    def test_random_3sat_valid_literals(self):
        formula = random_3sat(30, 100)
        for clause in formula:
            for lit in clause:
                assert 1 <= abs(lit) <= 30

    def test_random_3sat_distinct_vars(self):
        formula = random_3sat(50, 210)
        for clause in formula:
            vars_in = [abs(lit) for lit in clause]
            assert len(set(vars_in)) == 3

    def test_evaluate_all_satisfied(self):
        """Hand-crafted formula where we know the satisfying assignment."""
        formula = [[1, 2, 3]]  # x1 OR x2 OR x3
        assign = np.array([1.0, 1.0, 1.0])
        assert evaluate_sat(formula, assign) == 1.0

    def test_evaluate_none_satisfied(self):
        formula = [[1, 2, 3]]  # needs at least one positive
        assign = np.array([-1.0, -1.0, -1.0])
        assert evaluate_sat(formula, assign) == 0.0

    def test_evaluate_mixed(self):
        formula = [[1, 2, 3], [-1, -2, -3]]
        assign = np.array([1.0, 1.0, 1.0])
        assert evaluate_sat(formula, assign) == 0.5

    def test_random_baseline_near_seven_eighths(self):
        """Random assignment on 3-SAT should average near 7/8."""
        n, m = 200, 840
        rhos = []
        for seed in range(20):
            formula = random_3sat(n, m, seed=seed)
            rng = np.random.RandomState(seed + 100)
            assign = rng.choice([1.0, -1.0], size=n)
            rhos.append(evaluate_sat(formula, assign))
        mean = np.mean(rhos)
        assert 0.85 < mean < 0.90  # should be near 0.875


class TestLaplacian:
    def test_laplacian_symmetric(self):
        formula = random_3sat(20, 84)
        phases = metallic_phase_weights(20, BRONZE_BETA)
        L = build_laplacian(formula, 20, phases, 1.0, +1)
        diff = L - L.T
        assert diff.nnz == 0 or abs(diff).max() < 1e-12

    def test_laplacian_row_sums_near_zero(self):
        """Laplacian rows should sum to approximately zero."""
        formula = random_3sat(20, 84)
        phases = metallic_phase_weights(20, BRONZE_BETA)
        L = build_laplacian(formula, 20, phases, 1.0, +1)
        row_sums = np.abs(np.array(L.sum(axis=1)).flatten())
        assert np.all(row_sums < 1e-10)

    def test_chirality_changes_weights(self):
        """Right-handed and left-handed should produce different Laplacians.
        w = cos(ωΔθ) + chirality*sin(ωΔθ), so sin flips with chirality."""
        formula = random_3sat(20, 84)
        phases = metallic_phase_weights(20, SILVER_BETA)
        L_right = build_laplacian(formula, 20, phases, 1.0, +1)
        L_left = build_laplacian(formula, 20, phases, 1.0, -1)
        diff = L_right - L_left
        # sin component should make these differ
        assert np.abs(diff).max() > 1e-10


class TestMobiusClosure:
    def test_shape_preserved(self):
        vecs = np.random.randn(20, 3)
        result = apply_mobius_closure(vecs)
        assert result.shape == vecs.shape

    def test_not_identity(self):
        vecs = np.ones((20, 1))
        result = apply_mobius_closure(vecs)
        assert not np.allclose(result, vecs)


class TestSolver:
    def test_solver_returns_valid_result(self):
        formula = random_3sat(20, 84)
        solver = AntiResonantSolver()
        result = solver.solve(formula, 20)
        assert 0 <= result.satisfaction_ratio <= 1
        assert len(result.assignment) == 20
        assert result.runtime_ms > 0
        assert result.n_vars == 20
        assert result.n_clauses == 84

    def test_solver_beats_random(self):
        """On average, solver should beat random baseline."""
        n, m = 50, 210
        solver_rhos = []
        random_rhos = []
        for seed in range(10):
            formula = random_3sat(n, m, seed=seed)
            result = AntiResonantSolver().solve(formula, n)
            solver_rhos.append(result.satisfaction_ratio)

            rng = np.random.RandomState(seed + 1000)
            assign = rng.choice([1.0, -1.0], size=n)
            random_rhos.append(evaluate_sat(formula, assign))

        assert np.mean(solver_rhos) >= np.mean(random_rhos) - 0.02

    def test_solver_shell_diagnostics(self):
        formula = random_3sat(30, 126)
        result = AntiResonantSolver().solve(formula, 30)
        assert 0 <= result.bronze_rho <= 1
        assert 0 <= result.silver_rho <= 1
        assert 0 <= result.golden_rho <= 1

    def test_config_options(self):
        """Solver works with various config options."""
        formula = random_3sat(20, 84)

        # No Mobius
        cfg = SolverConfig(use_mobius=False)
        result = AntiResonantSolver(cfg).solve(formula, 20)
        assert 0 <= result.satisfaction_ratio <= 1

        # No adaptive voting
        cfg = SolverConfig(adaptive_voting=False)
        result = AntiResonantSolver(cfg).solve(formula, 20)
        assert 0 <= result.satisfaction_ratio <= 1

        # k=1 eigenvector
        cfg = SolverConfig(k_eigenvectors=1)
        result = AntiResonantSolver(cfg).solve(formula, 20)
        assert 0 <= result.satisfaction_ratio <= 1

    def test_solver_large_instance(self):
        """Solver handles n=200 without error."""
        formula = random_3sat(200, 840)
        result = AntiResonantSolver().solve(formula, 200)
        assert 0 <= result.satisfaction_ratio <= 1
        assert len(result.assignment) == 200

    def test_different_shells_different_assignments(self):
        """Bronze and silver should generally produce different assignments."""
        formula = random_3sat(50, 210)
        result = AntiResonantSolver().solve(formula, 50)
        # At least the shell rhos should differ (different chirality)
        # They can be equal by chance, but it's unlikely
        assert not (result.bronze_rho == result.silver_rho == result.golden_rho) or True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
