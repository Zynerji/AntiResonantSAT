#include "arsat/solver.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace arsat;

static void print_usage(const char* prog) {
    std::printf("AntiResonantSAT — Chiral Multi-Shell Spectral SAT Solver\n\n");
    std::printf("Usage:\n");
    std::printf("  %s --benchmark              Run full benchmark suite\n", prog);
    std::printf("  %s -n 200 -m 840            Solve random 3-SAT instance\n", prog);
    std::printf("  %s --dimacs file.cnf         Solve DIMACS CNF file\n", prog);
    std::printf("\nOptions:\n");
    std::printf("  -n NUM        Number of variables (default: 100)\n");
    std::printf("  -m NUM        Number of clauses (default: 4.2*n)\n");
    std::printf("  -k NUM        Eigenvectors per shell (default: 2)\n");
    std::printf("  -w FLOAT      Omega frequency (default: 1.0)\n");
    std::printf("  --seeds NUM   Number of random seeds (default: 10)\n");
    std::printf("  --no-mobius   Disable Mobius closure\n");
    std::printf("  --no-cache    Disable spectral cache\n");
    std::printf("  --no-adaptive Disable adaptive voting\n");
    std::printf("  -v            Verbose output\n");
}

static void run_single(const Formula& formula, int n_vars,
                        AntiResonantSolver::Config& cfg, bool verbose) {
    AntiResonantSolver solver(cfg);
    auto result = solver.solve(formula, n_vars);

    std::printf("  rho = %.4f  (bronze=%.4f  silver=%.4f  golden=%.4f)  "
                "%.2f ms  cache_hits=%d\n",
                result.satisfaction_ratio,
                result.bronze_rho, result.silver_rho, result.golden_rho,
                result.runtime_ms, result.cache_hits);
}

static void run_benchmark(AntiResonantSolver::Config& cfg, int num_seeds, bool verbose) {
    struct BenchSize { int n; int m; };
    std::vector<BenchSize> sizes = {
        {20, 84}, {50, 210}, {100, 420}, {200, 840}, {500, 2100}
    };

    std::printf("╔═══════════════════════════════════════════════════════════════════════════════╗\n");
    std::printf("║  AntiResonantSAT Benchmark — Chiral Multi-Shell Spectral SAT                 ║\n");
    std::printf("║  Shells: Bronze(+1, β=3.303) → Silver(-1, β=2.414) → Golden(+1, φ=1.618)    ║\n");
    std::printf("║  k=%d eigenvectors, ω=%.2f, Möbius=%s, adaptive=%s                      ║\n",
                cfg.k_eigenvectors, cfg.omega,
                cfg.use_mobius ? "ON " : "OFF",
                cfg.adaptive_voting ? "ON " : "OFF");
    std::printf("╠═══════════════════════════════════════════════════════════════════════════════╣\n");
    std::printf("║  n    │  m     │  ρ mean  │  ρ std  │  ρ best  │  random │  Δ%%    │  ms     ║\n");
    std::printf("╠═══════════════════════════════════════════════════════════════════════════════╣\n");

    for (auto& sz : sizes) {
        std::vector<double> rhos;
        double total_ms = 0;
        double best_rho = 0;

        for (int seed = 0; seed < num_seeds; ++seed) {
            auto formula = random_3sat(sz.n, sz.m, 42 + seed);
            AntiResonantSolver solver(cfg);
            auto result = solver.solve(formula, sz.n);
            rhos.push_back(result.satisfaction_ratio);
            total_ms += result.runtime_ms;
            best_rho = std::max(best_rho, result.satisfaction_ratio);

            if (verbose) {
                std::printf("  [n=%d seed=%d] rho=%.4f (Br=%.4f Ag=%.4f Au=%.4f) %.2fms\n",
                    sz.n, seed, result.satisfaction_ratio,
                    result.bronze_rho, result.silver_rho, result.golden_rho,
                    result.runtime_ms);
            }
        }

        double mean = 0, var = 0;
        for (double r : rhos) mean += r;
        mean /= rhos.size();
        for (double r : rhos) var += (r - mean) * (r - mean);
        double stddev = std::sqrt(var / (rhos.size() - 1));

        double random_baseline = 0.875;  // 7/8 for 3-SAT
        double delta_pct = ((mean - random_baseline) / random_baseline) * 100.0;

        std::printf("║ %4d  │ %5d  │  %.4f  │ %.4f  │  %.4f  │  %.3f  │ %+5.2f  │ %6.1f  ║\n",
                    sz.n, sz.m, mean, stddev, best_rho, random_baseline, delta_pct,
                    total_ms / num_seeds);
    }

    std::printf("╚═══════════════════════════════════════════════════════════════════════════════╝\n");
}

int main(int argc, char** argv) {
    AntiResonantSolver::Config cfg;
    int n_vars = 100;
    int m_clauses = -1;  // auto = 4.2*n
    int num_seeds = 10;
    bool do_benchmark = false;
    bool verbose = false;
    std::string dimacs_file;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--benchmark") == 0) { do_benchmark = true; }
        else if (std::strcmp(argv[i], "--dimacs") == 0 && i+1 < argc) { dimacs_file = argv[++i]; }
        else if (std::strcmp(argv[i], "-n") == 0 && i+1 < argc) { n_vars = std::atoi(argv[++i]); }
        else if (std::strcmp(argv[i], "-m") == 0 && i+1 < argc) { m_clauses = std::atoi(argv[++i]); }
        else if (std::strcmp(argv[i], "-k") == 0 && i+1 < argc) { cfg.k_eigenvectors = std::atoi(argv[++i]); }
        else if (std::strcmp(argv[i], "-w") == 0 && i+1 < argc) { cfg.omega = std::atof(argv[++i]); }
        else if (std::strcmp(argv[i], "--seeds") == 0 && i+1 < argc) { num_seeds = std::atoi(argv[++i]); }
        else if (std::strcmp(argv[i], "--no-mobius") == 0) { cfg.use_mobius = false; }
        else if (std::strcmp(argv[i], "--no-cache") == 0) { cfg.use_cache = false; }
        else if (std::strcmp(argv[i], "--no-adaptive") == 0) { cfg.adaptive_voting = false; }
        else if (std::strcmp(argv[i], "-v") == 0) { verbose = true; }
        else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]); return 0;
        }
    }

    if (m_clauses < 0) m_clauses = static_cast<int>(4.2 * n_vars);

    if (do_benchmark) {
        run_benchmark(cfg, num_seeds, verbose);
        return 0;
    }

    if (!dimacs_file.empty()) {
        std::printf("Loading DIMACS: %s\n", dimacs_file.c_str());
        auto [formula, nv] = parse_dimacs(dimacs_file);
        std::printf("  %d variables, %zu clauses\n", nv, formula.size());
        run_single(formula, nv, cfg, verbose);
        return 0;
    }

    // Single random instance
    std::printf("Random 3-SAT: n=%d, m=%d\n", n_vars, m_clauses);
    auto formula = random_3sat(n_vars, m_clauses, 42);
    run_single(formula, n_vars, cfg, verbose);

    return 0;
}
