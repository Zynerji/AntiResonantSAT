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
        int k_eigenvectors = 4;    // eigenvectors per shell
        double omega = 1.0;        // frequency parameter (used when multi_omega=false)
        bool use_mobius = true;     // apply Mobius closure
        bool use_cache = true;     // enable spectral cache
        bool adaptive_voting = true; // clause-satisfaction-adaptive weights
        bool greedy_refine = true;  // greedy 1-flip post-processing
        int greedy_passes = 2;      // max passes of greedy flip
        bool multi_omega = true;    // try multiple omega values, keep best

        // Pendulum omega: powers of bronze metallic mean as frequency spread.
        // More powers = more coverage = better. Sweep confirmed (1,2,3,4,5) wins.
        // Zero free parameters — metallic means control everything.
        static constexpr int N_OMEGA_SPREAD = 5;
        double omega_spread[N_OMEGA_SPREAD] = {
            3.3027756377319946,     // BRONZE_BETA^1
            10.908326913559146,     // BRONZE_BETA^2
            36.02186539041965,      // BRONZE_BETA^3
            118.95346881596149,     // BRONZE_BETA^4
            392.79610325667474,     // BRONZE_BETA^5
        };

        // Shell weights for compound voting (before adaptation)
        double bronze_weight = 0.45;
        double silver_weight = 0.30;
        double golden_weight = 0.25;
    };

    explicit AntiResonantSolver(Config config);
    AntiResonantSolver() : AntiResonantSolver(Config{}) {}

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

    // Single-omega pipeline result
    struct OmegaResult {
        Assignment assignment;
        double rho;
        double bronze_rho, silver_rho, golden_rho;
    };

    OmegaResult solve_single_omega(const Formula& formula, int n_vars, double omega);
};

}  // namespace arsat
