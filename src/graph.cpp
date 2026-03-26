#include "arsat/graph.hpp"
#include "arsat/metallic_means.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace arsat {

// ── Adjacency matrix construction ───────────────────────────────────────

SparseMat build_adjacency(
    const Formula& formula,
    int n_vars,
    const std::vector<double>& phase_angles,
    double omega,
    int chirality
) {
    // Accumulate edge weights in a map for efficiency
    // Key: (min(u,v), max(u,v)) packed into uint64
    std::unordered_map<uint64_t, double> edge_weights;

    auto pack_edge = [](int u, int v) -> uint64_t {
        int lo = std::min(u, v);
        int hi = std::max(u, v);
        return (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
    };

    for (const auto& clause : formula) {
        // Get unique variable indices (0-based)
        std::vector<int> vars;
        vars.reserve(clause.size());
        for (auto lit : clause) {
            int v = std::abs(lit) - 1;  // DIMACS is 1-based
            vars.push_back(v);
        }
        // Remove duplicates within clause
        std::sort(vars.begin(), vars.end());
        vars.erase(std::unique(vars.begin(), vars.end()), vars.end());

        // Add edges for all pairs
        for (size_t i = 0; i < vars.size(); ++i) {
            for (size_t j = i + 1; j < vars.size(); ++j) {
                int u = vars[i], v = vars[j];
                double w = phase_edge_weight(
                    phase_angles[u], phase_angles[v], omega, chirality
                );
                edge_weights[pack_edge(u, v)] += w;
            }
        }
    }

    // Build sparse matrix from accumulated weights
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(edge_weights.size() * 2);

    for (const auto& [key, w] : edge_weights) {
        int u = static_cast<int>(key >> 32);
        int v = static_cast<int>(key & 0xFFFFFFFF);
        triplets.emplace_back(u, v, w);
        triplets.emplace_back(v, u, w);  // symmetric
    }

    SparseMat adj(n_vars, n_vars);
    adj.setFromTriplets(triplets.begin(), triplets.end());
    return adj;
}

// ── Laplacian ───────────────────────────────────────────────────────────

SparseMat build_laplacian(const SparseMat& adjacency) {
    int n = adjacency.rows();
    // D = diag(row sums of W) — standard graph Laplacian
    Eigen::VectorXd degrees(n);
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (SparseMat::InnerIterator it(adjacency, i); it; ++it) {
            sum += it.value();
        }
        degrees(i) = sum;
    }

    // L = D - W
    SparseMat L = -adjacency;
    for (int i = 0; i < n; ++i) {
        L.coeffRef(i, i) += degrees(i);
    }
    L.makeCompressed();
    return L;
}

// ── DIMACS parser ───────────────────────────────────────────────────────

DimacsResult parse_dimacs(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open DIMACS file: " + filename);
    }

    Formula formula;
    int n_vars = 0, n_clauses = 0;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == 'c') continue;  // comment
        if (line[0] == 'p') {
            std::istringstream iss(line);
            std::string p, cnf;
            iss >> p >> cnf >> n_vars >> n_clauses;
            formula.reserve(n_clauses);
            continue;
        }
        // Clause line: literals terminated by 0
        std::istringstream iss(line);
        Clause clause;
        int lit;
        while (iss >> lit && lit != 0) {
            clause.push_back(lit);
        }
        if (!clause.empty()) {
            formula.push_back(std::move(clause));
        }
    }

    return {std::move(formula), n_vars};
}

// ── Random 3-SAT generator ─────────────────────────────────────────────

Formula random_3sat(int n_vars, int m_clauses, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> var_dist(0, n_vars - 1);
    std::uniform_int_distribution<int> sign_dist(0, 1);

    Formula formula;
    formula.reserve(m_clauses);

    for (int i = 0; i < m_clauses; ++i) {
        Clause clause;
        clause.reserve(3);

        // Select 3 distinct variables
        int vars[3];
        vars[0] = var_dist(rng);
        do { vars[1] = var_dist(rng); } while (vars[1] == vars[0]);
        do { vars[2] = var_dist(rng); } while (vars[2] == vars[0] || vars[2] == vars[1]);

        for (int j = 0; j < 3; ++j) {
            int sign = sign_dist(rng) ? 1 : -1;
            clause.push_back((vars[j] + 1) * sign);  // 1-based
        }
        formula.push_back(std::move(clause));
    }

    return formula;
}

// ── Evaluation ──────────────────────────────────────────────────────────

double evaluate_sat(const Formula& formula, const Assignment& assignment) {
    int satisfied = 0;
    for (const auto& clause : formula) {
        for (auto lit : clause) {
            int var = std::abs(lit) - 1;
            bool val = (lit > 0) ? (assignment[var] > 0) : (assignment[var] < 0);
            if (val) {
                ++satisfied;
                break;
            }
        }
    }
    return static_cast<double>(satisfied) / formula.size();
}

}  // namespace arsat
