#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace arsat {

// A literal: positive = variable true, negative = variable negated
// Variable indices are 1-based (DIMACS convention)
using Literal = int32_t;
using Clause = std::vector<Literal>;
using Formula = std::vector<Clause>;

// Variable assignment: +1 = true, -1 = false
using Assignment = std::vector<int8_t>;

struct SolverResult {
    Assignment assignment;
    double satisfaction_ratio;  // fraction of clauses satisfied
    double runtime_ms;
    int n_vars;
    int n_clauses;

    // Per-shell diagnostics
    double bronze_rho;
    double silver_rho;
    double golden_rho;
    int cache_hits;
};

struct BenchmarkConfig {
    int n_vars = 100;
    int m_clauses = 420;  // 4.2 * n_vars for phase transition
    int num_seeds = 10;
    int k_eigenvectors = 2;  // eigenvectors per shell
    double omega = 1.0;      // frequency parameter
    bool verbose = false;
};

// Shell chirality: +1 = right-handed, -1 = left-handed
enum class Chirality : int8_t {
    Right = +1,
    Left  = -1,
};

}  // namespace arsat
