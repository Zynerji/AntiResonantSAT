#pragma once
#include "graph.hpp"
#include "metallic_means.hpp"
#include "sat_types.hpp"
#include "spectral.hpp"

namespace arsat {

// ── Multi-Shell Chiral SAT Solver ───────────────────────────────────────
//
// Pipeline:
//   1. Bronze shell (right-handed, β₃=3.303): aggressive partitioning
//   2. Silver shell (LEFT-handed, β₂=2.414): orthogonal chiral partition
//   3. Golden shell (right-handed, β₁=φ):    stability / tie-breaking
//
// Each shell:
//   - Builds Laplacian with metallic-mean phase weights + chirality
//   - Extracts k smallest eigenvectors
//   - Applies Mobius closure
//   - Produces candidate assignment from sign(eigenvector)
//
// Final assignment: compound-weighted majority vote across shells,
// with clause-satisfaction-adaptive weighting.

class AntiResonantSolver {
public:
    struct Config {
        int k_eigenvectors = 2;    // eigenvectors per shell
        double omega = 1.0;        // frequency parameter (used when multi_omega=false)
        bool use_mobius = true;     // apply Mobius closure
        bool use_cache = true;     // enable spectral cache
        bool adaptive_voting = true; // clause-satisfaction-adaptive weights
        bool multi_omega = true;    // try multiple omega values, keep best

        // Pendulum omega: powers of bronze metallic mean as frequency spread.
        // beta_3^1 = 3.303, beta_3^2 = 10.908, beta_3^3 = 36.02
        // Zero free parameters — metallic means control everything.
        static constexpr int N_OMEGA_SPREAD = 3;
        double omega_spread[N_OMEGA_SPREAD] = {
            BRONZE_BETA,
            BRONZE_BETA * BRONZE_BETA,
            BRONZE_BETA * BRONZE_BETA * BRONZE_BETA,
        };

        // Shell weights for compound voting (before adaptation)
        double bronze_weight = 0.45;
        double silver_weight = 0.30;
        double golden_weight = 0.25;
    };

    explicit AntiResonantSolver(Config config = {});

    // Solve a formula. Returns best assignment found.
    SolverResult solve(const Formula& formula, int n_vars);

    // Access cache stats
    const SpectralCache& cache() const { return cache_; }

private:
    Config config_;
    SpectralCache cache_;

    // Run a single shell pass
    struct ShellResult {
        Assignment assignment;
        double rho;
        DenseMat eigenvectors;
    };

    ShellResult run_shell(
        const Formula& formula,
        int n_vars,
        double beta,
        Chirality chirality
    );

    // Compound voting across shells with adaptive weighting
    Assignment compound_vote(
        const Formula& formula,
        int n_vars,
        const ShellResult& bronze,
        const ShellResult& silver,
        const ShellResult& golden
    );

    // Single-omega pipeline (used by multi-omega solve)
    struct OmegaResult;
    OmegaResult solve_single_omega(const Formula& formula, int n_vars, double omega);
};

}  // namespace arsat
