#pragma once
#include <cmath>
#include <string>
#include <vector>

namespace arsat {

// β_n = (n + sqrt(n² + 4)) / 2
inline double metallic_mean(int n) {
    return (n + std::sqrt(n * n + 4.0)) / 2.0;
}

// The seven anti-resonant elements
struct Element {
    std::string symbol;
    std::string name;
    int n;           // metallic mean index (-1 = transcendental, -2 = chaotic)
    double beta;     // metallic mean value
    double energy;   // measured or predicted from E(n) = -0.705n - 4.416
};

// Compile-time constants for the three shells we use
inline constexpr double GOLDEN_BETA = 1.6180339887498949;  // (1+√5)/2
inline constexpr double SILVER_BETA = 2.4142135623730951;  // (2+√8)/2
inline constexpr double BRONZE_BETA = 3.3027756377319946;  // (3+√13)/2

// Generate anti-resonant phase weights for n variables using metallic mean β
// θ_k = 2π · β^k / Σβ^j, normalized to sum to 1
std::vector<double> metallic_phase_weights(int n_vars, double beta);

// Generate edge weight between variables u, v given their phase angles.
// w(u,v) = cos(ω·Δθ) + chirality · sin(ω·Δθ)
//
// cos is even (no chirality information), sin is odd (flips with chirality).
// Right-handed (+1): w = cos + sin   (favors positive phase differences)
// Left-handed  (-1): w = cos - sin   (favors negative phase differences)
// This creates genuinely different Laplacians and eigenvector partitions.
inline double phase_edge_weight(double theta_u, double theta_v,
                                 double omega, int chirality) {
    double delta = omega * (theta_u - theta_v);
    return std::cos(delta) + chirality * std::sin(delta);
}

}  // namespace arsat
