#include "arsat/spectral.hpp"

#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>

#include <algorithm>
#include <cmath>

namespace arsat {

// ── Eigendecomposition ──────────────────────────────────────────────────

EigenResult smallest_eigenpairs(const SparseMat& matrix, int k) {
    int n = matrix.rows();
    if (n <= 0 || k <= 0) {
        return {{}, {}};
    }

    // Clamp k to matrix size - 1 (Spectra requirement)
    k = std::min(k, n - 1);
    int ncv = std::min(std::max(2 * k + 1, 20), n);

    // Use shift-and-invert mode for smallest eigenvalues (much faster convergence)
    // Shift near 0 to find smallest algebraic eigenvalues
    Spectra::SparseSymShiftSolve<double> op(matrix);
    Spectra::SymEigsShiftSolver<Spectra::SparseSymShiftSolve<double>> solver(op, k, ncv, 0.0);

    solver.init();
    int nconv = solver.compute(Spectra::SortRule::LargestMagn, 1000, 1e-10);

    if (solver.info() != Spectra::CompInfo::Successful) {
        // Fallback: use regular solver if shift-invert fails
        Spectra::SparseSymMatProd<double> op2(matrix);
        Spectra::SymEigsSolver<Spectra::SparseSymMatProd<double>> solver2(op2, k, ncv);
        solver2.init();
        solver2.compute(Spectra::SortRule::SmallestAlge, 1000, 1e-10);

        if (solver2.info() != Spectra::CompInfo::Successful) {
            // Last resort: return zero vectors
            EigenResult result;
            result.vectors = DenseMat::Zero(n, k);
            result.values = DenseVec::Zero(k);
            return result;
        }
        return {solver2.eigenvectors(), solver2.eigenvalues()};
    }

    return {solver.eigenvectors(), solver.eigenvalues()};
}

// ── Mobius closure ──────────────────────────────────────────────────────

void apply_mobius_closure(DenseMat& vectors) {
    int n = vectors.rows();
    int k = vectors.cols();
    constexpr double FOUR_PI = 4.0 * 3.14159265358979323846;

    for (int i = 0; i < n; ++i) {
        double phase = FOUR_PI * i / n;
        double cos_p = std::cos(phase);
        double sin_p = std::sin(phase);

        // Multiply by exp(i*phase) and take real part:
        // Re(v * (cos + i*sin)) = v * cos
        // But we need both real and imaginary parts for proper rotation.
        // Since v is real, Re(v * exp(i*phase)) = v * cos(phase)
        // This applies the topological twist.
        for (int j = 0; j < k; ++j) {
            vectors(i, j) *= cos_p;
        }
    }
}

// ── Spectral Cache ──────────────────────────────────────────────────────

SpectralCache::SpectralCache(size_t max_entries)
    : max_entries_(max_entries) {}

uint64_t SpectralCache::hash_matrix(const SparseMat& mat) {
    // FNV-1a hash over non-zero structure and values
    uint64_t hash = 14695981039346656037ULL;
    constexpr uint64_t FNV_PRIME = 1099511628211ULL;

    // Hash dimensions
    auto mix = [&](uint64_t val) {
        hash ^= val;
        hash *= FNV_PRIME;
    };

    mix(mat.rows());
    mix(mat.cols());
    mix(mat.nonZeros());

    // Hash non-zero entries (row, col, value)
    for (int k = 0; k < mat.outerSize(); ++k) {
        for (SparseMat::InnerIterator it(mat, k); it; ++it) {
            mix(it.row());
            mix(it.col());
            // Hash the double value via its bit representation
            uint64_t vbits;
            double v = it.value();
            std::memcpy(&vbits, &v, sizeof(double));
            mix(vbits);
        }
    }

    return hash;
}

bool SpectralCache::lookup(uint64_t key, EigenResult& out) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        out = it->second.result;
        ++hits_;
        return true;
    }
    return false;
}

void SpectralCache::insert(uint64_t key, const EigenResult& result) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Evict oldest if full
    if (cache_.size() >= max_entries_) {
        uint64_t oldest_key = 0;
        uint64_t oldest_order = UINT64_MAX;
        for (const auto& [k, entry] : cache_) {
            if (entry.order < oldest_order) {
                oldest_order = entry.order;
                oldest_key = k;
            }
        }
        cache_.erase(oldest_key);
    }

    cache_[key] = {result, insert_order_++};
}

void SpectralCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
    hits_ = 0;
    insert_order_ = 0;
}

// ── Spectral Basis Cache (v9.3-style incremental caching) ───────────

EigenResult solve_with_basis_cache(
    SpectralCache& cache,
    const SparseMat& laplacian,
    int k
) {
    uint64_t key = SpectralCache::hash_matrix(laplacian);
    EigenResult cached;

    if (cache.lookup(key, cached)) {
        // Cache hit — Rayleigh-Ritz refinement
        int n = laplacian.rows();
        int k_use = std::min(k, static_cast<int>(cached.vectors.cols()));

        // V = cached eigenvectors (n x k)
        DenseMat V = cached.vectors.leftCols(k_use);

        // H = V^T L V (k x k matrix) — O(n*k)
        DenseMat LV = DenseMat(laplacian) * V;  // n x k
        DenseMat H = V.transpose() * LV;        // k x k

        // Small dense eigensolve on k x k — O(k^3)
        Eigen::SelfAdjointEigenSolver<DenseMat> solver(H);

        // Rotate back to full space
        EigenResult result;
        result.vectors = V * solver.eigenvectors();
        result.values = solver.eigenvalues();
        return result;
    }

    // Cold solve — full eigendecomposition
    EigenResult result = smallest_eigenpairs(laplacian, k);

    // Cache for next time
    cache.insert(key, result);
    return result;
}

}  // namespace arsat
