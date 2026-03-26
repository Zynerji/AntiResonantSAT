#pragma once
#include <Eigen/Sparse>
#include "sat_types.hpp"
#include <vector>

namespace arsat {

using SparseMat = Eigen::SparseMatrix<double>;

// Build the weighted adjacency matrix for a formula using metallic-mean phase weights.
// For each clause, adds edges between all variable pairs with weight:
//   w(u,v) = cos(chirality * omega * (θ_u - θ_v))
// where θ_k are metallic-mean phase angles.
//
// Returns the adjacency matrix (symmetric, n_vars x n_vars).
SparseMat build_adjacency(
    const Formula& formula,
    int n_vars,
    const std::vector<double>& phase_angles,
    double omega,
    int chirality  // +1 or -1
);

// Build graph Laplacian L = D - W from adjacency matrix W.
SparseMat build_laplacian(const SparseMat& adjacency);

// Parse DIMACS CNF format. Returns (formula, n_vars).
struct DimacsResult {
    Formula formula;
    int n_vars;
};
DimacsResult parse_dimacs(const std::string& filename);

// Generate random 3-SAT instance at clause ratio alpha = m/n
Formula random_3sat(int n_vars, int m_clauses, uint64_t seed);

// Evaluate satisfaction ratio
double evaluate_sat(const Formula& formula, const Assignment& assignment);

}  // namespace arsat
