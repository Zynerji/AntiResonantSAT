#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace arsat {

using SparseMat = Eigen::SparseMatrix<double>;
using DenseMat = Eigen::MatrixXd;
using DenseVec = Eigen::VectorXd;

// Result of eigendecomposition
struct EigenResult {
    DenseMat vectors;  // n_vars x k columns (smallest eigenvectors)
    DenseVec values;   // k eigenvalues
};

// Compute the k smallest eigenpairs of a symmetric sparse matrix.
// Uses Spectra (ARPACK-style) for sparse eigenvalue problems.
EigenResult smallest_eigenpairs(const SparseMat& matrix, int k);

// Apply Mobius closure: 720 degree (4*pi) topological phase rotation.
// Modifies eigenvectors in-place by multiplying by exp(i * 4pi * idx / n)
// and extracting the real part.
void apply_mobius_closure(DenseMat& vectors);

// ── Spectral Cache ──────────────────────────────────────────────────────
// LRU cache keyed by a hash of the Laplacian's structure + values.
// Thread-safe for concurrent shell evaluations.

class SpectralCache {
public:
    explicit SpectralCache(size_t max_entries = 64);

    // Compute hash of a sparse matrix (structure + values)
    static uint64_t hash_matrix(const SparseMat& mat);

    // Look up cached result. Returns true if found.
    bool lookup(uint64_t key, EigenResult& out) const;

    // Insert result into cache (evicts oldest if full).
    void insert(uint64_t key, const EigenResult& result);

    // Get cache hit count
    int hits() const { return hits_; }

    // Clear cache
    void clear();

private:
    size_t max_entries_;
    mutable std::mutex mutex_;
    // Simple map + insertion-order tracking for LRU
    struct Entry {
        EigenResult result;
        uint64_t order;
    };
    std::unordered_map<uint64_t, Entry> cache_;
    uint64_t insert_order_ = 0;
    mutable int hits_ = 0;
};

}  // namespace arsat
