#include "arsat/solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

namespace arsat {

AntiResonantSolver::AntiResonantSolver(Config config)
    : config_(config), cache_(64) {}

// ── Single shell pass ───────────────────────────────────────────────────

AntiResonantSolver::ShellResult AntiResonantSolver::run_shell(
    const Formula& formula,
    int n_vars,
    double beta,
    Chirality chirality
) {
    // 1. Compute metallic-mean phase angles
    auto phases = metallic_phase_weights(n_vars, beta);

    // 2. Build adjacency + Laplacian
    auto adj = build_adjacency(
        formula, n_vars, phases, config_.omega,
        static_cast<int>(chirality)
    );
    auto L = build_laplacian(adj);

    // 3. Solve with basis cache (v9.3 incremental caching)
    // First solve: full eigsh → cache. Subsequent: Rayleigh-Ritz refinement.
    EigenResult eigen_result;
    if (config_.use_cache) {
        eigen_result = solve_with_basis_cache(cache_, L, config_.k_eigenvectors);
    } else {
        eigen_result = smallest_eigenpairs(L, config_.k_eigenvectors);
    }

    // 4. Apply Mobius closure
    if (config_.use_mobius) {
        apply_mobius_closure(eigen_result.vectors);
    }

    // 5. Assignment from first (smallest) eigenvector
    Assignment assign(n_vars);
    if (eigen_result.vectors.cols() > 0) {
        for (int i = 0; i < n_vars; ++i) {
            double v = eigen_result.vectors(i, 0);
            assign[i] = (v >= 0) ? 1 : -1;
        }
    } else {
        // Fallback: random
        for (int i = 0; i < n_vars; ++i) {
            assign[i] = ((i * 2654435761u) & 1) ? 1 : -1;
        }
    }

    // 6. If k > 1, try other eigenvectors and keep best
    double best_rho = evaluate_sat(formula, assign);
    Assignment best_assign = assign;

    for (int col = 1; col < eigen_result.vectors.cols(); ++col) {
        Assignment alt(n_vars);
        for (int i = 0; i < n_vars; ++i) {
            alt[i] = (eigen_result.vectors(i, col) >= 0) ? 1 : -1;
        }
        double alt_rho = evaluate_sat(formula, alt);
        if (alt_rho > best_rho) {
            best_rho = alt_rho;
            best_assign = alt;
        }

        // Also try negated eigenvector (breaks sign degeneracy)
        for (int i = 0; i < n_vars; ++i) {
            alt[i] = -alt[i];
        }
        alt_rho = evaluate_sat(formula, alt);
        if (alt_rho > best_rho) {
            best_rho = alt_rho;
            best_assign = alt;
        }
    }

    // Also try negation of first eigenvector
    {
        Assignment neg(n_vars);
        for (int i = 0; i < n_vars; ++i) neg[i] = -best_assign[i];
        double neg_rho = evaluate_sat(formula, neg);
        if (neg_rho > best_rho) {
            best_rho = neg_rho;
            best_assign = neg;
        }
    }

    return {best_assign, best_rho, eigen_result.vectors};
}

// ── Compound voting ─────────────────────────────────────────────────────

Assignment AntiResonantSolver::compound_vote(
    const Formula& formula,
    int n_vars,
    const ShellResult& bronze,
    const ShellResult& silver,
    const ShellResult& golden
) {
    if (!config_.adaptive_voting) {
        // Simple weighted majority
        Assignment result(n_vars);
        for (int i = 0; i < n_vars; ++i) {
            double vote = config_.bronze_weight * bronze.assignment[i]
                        + config_.silver_weight * silver.assignment[i]
                        + config_.golden_weight * golden.assignment[i];
            result[i] = (vote >= 0) ? 1 : -1;
        }
        return result;
    }

    // Adaptive voting: weight shells differently per variable based on
    // clause satisfaction context.
    //
    // For each variable, compute how many clauses containing it are
    // satisfied by each shell's assignment. Weight shells by their
    // local clause satisfaction.

    // Precompute: for each variable, which clauses contain it
    std::vector<std::vector<int>> var_to_clauses(n_vars);
    for (int c = 0; c < static_cast<int>(formula.size()); ++c) {
        for (auto lit : formula[c]) {
            int v = std::abs(lit) - 1;
            var_to_clauses[v].push_back(c);
        }
    }

    // Precompute clause satisfaction for each shell
    auto clause_sat = [&](const Assignment& assign, int clause_idx) -> bool {
        for (auto lit : formula[clause_idx]) {
            int v = std::abs(lit) - 1;
            if ((lit > 0 && assign[v] > 0) || (lit < 0 && assign[v] < 0))
                return true;
        }
        return false;
    };

    Assignment result(n_vars);
    for (int i = 0; i < n_vars; ++i) {
        // Count clauses satisfied by each shell for this variable's clauses
        double b_sat = 0, s_sat = 0, g_sat = 0;
        double total = var_to_clauses[i].size();
        if (total == 0) {
            result[i] = bronze.assignment[i];  // no clauses → use bronze
            continue;
        }

        for (int c : var_to_clauses[i]) {
            b_sat += clause_sat(bronze.assignment, c) ? 1.0 : 0.0;
            s_sat += clause_sat(silver.assignment, c) ? 1.0 : 0.0;
            g_sat += clause_sat(golden.assignment, c) ? 1.0 : 0.0;
        }

        // Normalize to [0,1]
        b_sat /= total;
        s_sat /= total;
        g_sat /= total;

        // Adaptive weights: shell that satisfies more of this var's clauses
        // gets a higher vote weight
        double bw = config_.bronze_weight * (0.5 + b_sat);
        double sw = config_.silver_weight * (0.5 + s_sat);
        double gw = config_.golden_weight * (0.5 + g_sat);

        double vote = bw * bronze.assignment[i]
                    + sw * silver.assignment[i]
                    + gw * golden.assignment[i];
        result[i] = (vote >= 0) ? 1 : -1;
    }

    return result;
}

// ── Single-omega pipeline ────────────────────────────────────────────────

struct OmegaResult {
    Assignment assignment;
    double rho;
    double bronze_rho, silver_rho, golden_rho;
};

OmegaResult AntiResonantSolver::solve_single_omega(
    const Formula& formula, int n_vars, double omega
) {
    double saved = config_.omega;
    config_.omega = omega;

    // LRL chirality: left-right-left outperforms RLR at n>=50
    auto bronze = run_shell(formula, n_vars, BRONZE_BETA, Chirality::Left);
    auto silver = run_shell(formula, n_vars, SILVER_BETA, Chirality::Right);
    auto golden = run_shell(formula, n_vars, GOLDEN_BETA, Chirality::Left);

    auto voted = compound_vote(formula, n_vars, bronze, silver, golden);
    double voted_rho = evaluate_sat(formula, voted);

    Assignment best = voted;
    double best_rho = voted_rho;

    if (bronze.rho > best_rho) { best = bronze.assignment; best_rho = bronze.rho; }
    if (silver.rho > best_rho) { best = silver.assignment; best_rho = silver.rho; }
    if (golden.rho > best_rho) { best = golden.assignment; best_rho = golden.rho; }

    config_.omega = saved;
    return {best, best_rho, bronze.rho, silver.rho, golden.rho};
}

// ── Main solver ─────────────────────────────────────────────────────────

SolverResult AntiResonantSolver::solve(const Formula& formula, int n_vars) {
    auto t0 = std::chrono::high_resolution_clock::now();

    Assignment best;
    double best_rho = -1.0;
    double best_br = 0, best_ag = 0, best_au = 0;

    if (config_.multi_omega) {
        // Multi-omega: try each pendulum frequency, keep best
        for (int i = 0; i < Config::N_OMEGA_SPREAD; ++i) {
            auto result = solve_single_omega(formula, n_vars, config_.omega_spread[i]);
            if (result.rho > best_rho) {
                best = result.assignment;
                best_rho = result.rho;
                best_br = result.bronze_rho;
                best_ag = result.silver_rho;
                best_au = result.golden_rho;
            }
        }
    } else {
        auto result = solve_single_omega(formula, n_vars, config_.omega);
        best = result.assignment;
        best_rho = result.rho;
        best_br = result.bronze_rho;
        best_ag = result.silver_rho;
        best_au = result.golden_rho;
    }

    // Greedy flip refinement: spectral gives ~90% warm start,
    // greedy flip closes the gap to ~97-98% in 2 passes.
    if (config_.greedy_refine) {
        best = greedy_flip(formula, best, config_.greedy_passes);
        best_rho = evaluate_sat(formula, best);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return {
        best,
        best_rho,
        ms,
        n_vars,
        static_cast<int>(formula.size()),
        best_br,
        best_ag,
        best_au,
        cache_.hits()
    };
}

}  // namespace arsat
