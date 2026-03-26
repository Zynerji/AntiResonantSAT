#include "arsat/metallic_means.hpp"
#include <numeric>

namespace arsat {

std::vector<double> metallic_phase_weights(int n_vars, double beta) {
    // θ_k = 2π · β^k / Σβ^j
    // Compute raw powers and normalize
    std::vector<double> weights(n_vars);
    double sum = 0.0;

    // Use modular exponentiation to prevent overflow for large n
    // β^k mod (sum) doesn't help, so we normalize in log space for large k
    if (n_vars <= 500 || beta < 2.0) {
        // Direct computation safe for moderate sizes
        for (int k = 0; k < n_vars; ++k) {
            weights[k] = std::pow(beta, k);
            sum += weights[k];
        }
    } else {
        // For large n with big beta, use geometric series normalization
        // β^k / Σβ^j = β^k * (β-1) / (β^n - 1)
        // But compute in log space to avoid overflow
        double log_beta = std::log(beta);
        double max_log = (n_vars - 1) * log_beta;
        for (int k = 0; k < n_vars; ++k) {
            // Shift by -max_log to prevent overflow
            weights[k] = std::exp(k * log_beta - max_log);
            sum += weights[k];
        }
    }

    // Convert to phase angles: θ_k = 2π * w_k / Σw
    constexpr double TWO_PI = 2.0 * 3.14159265358979323846;
    for (int k = 0; k < n_vars; ++k) {
        weights[k] = TWO_PI * weights[k] / sum;
    }

    return weights;
}

}  // namespace arsat
