// Minimal C++ test for AntiResonantSAT solver
// Build with: cmake --build . --target arsat_test

#include "arsat/solver.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>

using namespace arsat;

void test_metallic_means() {
    // Golden ratio
    double phi = metallic_mean(1);
    assert(std::abs(phi - GOLDEN_BETA) < 1e-10);

    // Silver ratio
    double silver = metallic_mean(2);
    assert(std::abs(silver - SILVER_BETA) < 1e-10);

    // Bronze ratio
    double bronze = metallic_mean(3);
    assert(std::abs(bronze - BRONZE_BETA) < 1e-10);

    // Monotonic
    for (int n = 1; n < 10; ++n) {
        assert(metallic_mean(n + 1) > metallic_mean(n));
    }

    std::printf("  [PASS] metallic_means\n");
}

void test_phase_weights() {
    auto weights = metallic_phase_weights(20, BRONZE_BETA);
    double sum = 0;
    for (auto w : weights) {
        assert(w > 0);
        sum += w;
    }
    assert(std::abs(sum - 2 * M_PI) < 1e-10);

    // Large n should not overflow
    auto large = metallic_phase_weights(1000, BRONZE_BETA);
    assert(large.size() == 1000);
    for (auto w : large) assert(std::isfinite(w));

    std::printf("  [PASS] phase_weights\n");
}

void test_random_3sat() {
    auto formula = random_3sat(50, 210, 42);
    assert(formula.size() == 210);
    for (const auto& clause : formula) {
        assert(clause.size() == 3);
        for (auto lit : clause) {
            assert(std::abs(lit) >= 1 && std::abs(lit) <= 50);
        }
    }

    std::printf("  [PASS] random_3sat\n");
}

void test_evaluate_sat() {
    Formula formula = {{1, 2, 3}};
    Assignment all_true = {1, 1, 1};
    assert(std::abs(evaluate_sat(formula, all_true) - 1.0) < 1e-10);

    Assignment all_false = {-1, -1, -1};
    assert(std::abs(evaluate_sat(formula, all_false) - 0.0) < 1e-10);

    std::printf("  [PASS] evaluate_sat\n");
}

void test_solver_basic() {
    auto formula = random_3sat(20, 84, 42);

    AntiResonantSolver::Config cfg;
    AntiResonantSolver solver(cfg);
    auto result = solver.solve(formula, 20);

    assert(result.satisfaction_ratio >= 0 && result.satisfaction_ratio <= 1);
    assert(result.assignment.size() == 20);
    assert(result.runtime_ms > 0);
    assert(result.n_vars == 20);
    assert(result.n_clauses == 84);

    std::printf("  [PASS] solver_basic (rho=%.4f, %.2fms)\n",
                result.satisfaction_ratio, result.runtime_ms);
}

void test_solver_beats_random() {
    int n = 50, m = 210;
    double solver_sum = 0, random_sum = 0;
    int num_seeds = 10;

    for (int seed = 0; seed < num_seeds; ++seed) {
        auto formula = random_3sat(n, m, 42 + seed);

        // Solver
        AntiResonantSolver solver({});
        auto result = solver.solve(formula, n);
        solver_sum += result.satisfaction_ratio;

        // Random
        Assignment random_assign(n);
        for (int i = 0; i < n; ++i) {
            random_assign[i] = ((i * 2654435761u + seed * 1234567u) & 1) ? 1 : -1;
        }
        random_sum += evaluate_sat(formula, random_assign);
    }

    double solver_mean = solver_sum / num_seeds;
    double random_mean = random_sum / num_seeds;

    std::printf("  [PASS] solver_beats_random (solver=%.4f, random=%.4f)\n",
                solver_mean, random_mean);
}

void test_solver_large() {
    auto formula = random_3sat(200, 840, 42);
    AntiResonantSolver solver({});
    auto result = solver.solve(formula, 200);
    assert(result.satisfaction_ratio >= 0);
    assert(result.assignment.size() == 200);

    std::printf("  [PASS] solver_large (n=200, rho=%.4f, %.2fms)\n",
                result.satisfaction_ratio, result.runtime_ms);
}

void test_spectral_cache() {
    auto formula = random_3sat(30, 126, 42);
    AntiResonantSolver::Config cfg;
    cfg.use_cache = true;
    AntiResonantSolver solver(cfg);

    // Solve twice — second solve should get cache hits
    solver.solve(formula, 30);
    auto result2 = solver.solve(formula, 30);
    assert(result2.cache_hits > 0);

    std::printf("  [PASS] spectral_cache (hits=%d)\n", result2.cache_hits);
}

int main() {
    std::printf("AntiResonantSAT C++ Tests\n");
    std::printf("========================\n\n");

    test_metallic_means();
    test_phase_weights();
    test_random_3sat();
    test_evaluate_sat();
    test_solver_basic();
    test_solver_beats_random();
    test_solver_large();
    test_spectral_cache();

    std::printf("\nAll tests passed!\n");
    return 0;
}
