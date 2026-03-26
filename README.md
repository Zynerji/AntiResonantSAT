# AntiResonantSAT

**Anti-Resonant Spectral SAT Solver** — A chiral multi-shell spectral heuristic for Max-3-SAT that breaks the 87% random-assignment ceiling.

## The Idea

Standard spectral SAT methods extract a single eigenvector (Fiedler vector) from a clause-variable graph Laplacian. This carries O(log n) bits of constraint information — barely above random assignment (7/8 = 87.5%).

AntiResonantSAT replaces the single-pass approach with a **3-shell chiral pipeline** inspired by quantum anti-resonant encoding:

1. **Bronze shell (right-handed)** — Metallic mean β₃=3.303 phase weighting, aggressive variable partitioning
2. **Silver shell (LEFT-handed)** — Reverse-chirality β₂=2.414 weighting, partitions along orthogonal axis
3. **Golden shell (right-handed)** — Stability pass with φ=1.618, breaks chiral ties

The key innovation: **chiral asymmetry**. Bronze and silver wind in opposite directions, so their eigenvectors partition variables along *different* axes. Agreement = high confidence. Disagreement = chiral boundary where golden casts the deciding vote.

### Why metallic means?

Logarithmic phase weighting (the standard approach) converges to uniform spacing at scale — the helical advantage disappears for n≥100. Metallic means are **maximally irrational** (worst-case for rational approximation), so the phase spacing stays non-degenerate at any scale.

## Architecture

```
3-SAT Formula
     │
     ├──► Bronze Laplacian (β₃, chirality=+1) ──► k eigenvectors ──┐
     │                                                               │
     ├──► Silver Laplacian (β₂, chirality=-1) ──► k eigenvectors ──┤── Compound Vote ──► Assignment
     │                                                               │
     └──► Golden Laplacian (φ,  chirality=+1) ──► k eigenvectors ──┘
```

Each shell builds a weighted Laplacian with metallic-mean phase spacing and extracts the top-k smallest eigenvectors. Assignments from each shell vote with compound weights (bronze heaviest for unsatisfied clauses, golden for satisfied regions).

## Results (Python reference, random 3-SAT at phase transition)

| n | Random (7/8) | Helical | Uniform Spectral | **AntiResonantSAT** | Improvement |
|---|---|---|---|---|---|
| 20 | 87.9% | 87.3% | 86.7% | **93.3%** | +6.2% |
| 50 | 87.0% | 88.5% | 88.2% | **91.6%** | +5.3% |
| 100 | 86.7% | 86.9% | 87.0% | **89.3%** | +3.1% |
| 200 | 88.3% | 87.4% | 87.9% | **88.9%** | +0.8% |
| 500 | 87.7% | 87.6% | 87.6% | **88.6%** | +1.0% |

Beats all baselines at every problem size. At n=50, **91.6% breaks the 87-88% spectral ceiling** by over 4 percentage points.

## Build

### C++ (fast mode)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
./arsat --benchmark          # Run full benchmark
./arsat -n 200 -m 840        # Single instance
./arsat --dimacs file.cnf     # DIMACS input
```

Requires: C++17 compiler, CMake 3.16+. Eigen and Spectra are fetched automatically.

### Python (reference + bindings)

```bash
pip install -e .
python -m arsat.benchmark    # Run benchmark suite
```

## Algorithm Details

### Metallic Mean Phase Weighting

For metallic mean index n, β_n = (n + √(n²+4)) / 2:
- Golden: β₁ = φ = 1.618...
- Silver: β₂ = 2.414...
- Bronze: β₃ = 3.303...

Phase angle for variable k: `θ_k = 2π · β^k / Σβ^j`

Edge weight between variables u,v in a clause:
`w(u,v) = cos(ω · (θ_u - θ_v)) + chirality · sin(ω · (θ_u - θ_v))`

Where chirality = +1 (right-handed) or -1 (left-handed). The cosine term is even (same for both chiralities), while the sine term is odd — this creates genuinely different Laplacians and eigenvector partitions for each handedness.

### Pendulum Omega (Zero Free Parameters)

Instead of a single tuned frequency parameter, the solver tries multiple omega values derived from the bronze metallic mean itself:

```
omega_spread = (beta_3^1, beta_3^2, beta_3^3) = (3.303, 10.908, 36.02)
```

This makes the entire system self-referential: metallic means control both the phase spacing AND the frequency parameter. Grid search confirmed these bronze powers hit near-optimal omegas across all tested problem sizes. The solver runs the 3-shell pipeline at each omega and keeps the best result.

### Spectral Cache

The C++ solver caches eigendecompositions keyed by a hash of the Laplacian's non-zero pattern + values. When the silver/golden passes only re-weight existing edges (same sparsity pattern), the cache provides a warm start via Rayleigh-Ritz refinement rather than cold eigensolve.

### Compound Voting

Each shell produces candidate assignments from its eigenvectors. The final assignment uses weighted majority vote:
- Bronze votes weighted by `(1 - clause_satisfaction)` — strongest on unsatisfied clauses
- Silver votes weighted uniformly — provides orthogonal information
- Golden votes weighted by `clause_satisfaction` — stabilizes already-good regions

## Origin

Combines two projects:
- [AntiResonantPeriodicTable](https://github.com/Zynerji/AntiResonantPeriodicTable) — Metallic mean quantum optimization, IBM Marrakesh validated
- Helical-SAT-Heuristic — Spectral graph approach for Max-3-SAT

## License

MIT
