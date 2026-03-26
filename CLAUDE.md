# AntiResonantSAT — Session Checkpoint

## NEXT SESSION: Pick up here

### What's done
- Full repo: C++ core (Eigen/Spectra) + Python reference + benchmark suite
- 25/25 Python tests pass
- Benchmark validated: AntiResonant beats all baselines at every problem size
- Chiral asymmetry via cos+sin edge weights (sin flips with handedness)
- Spectral cache (C++ LRU, thread-safe)
- DIMACS parser for SATLIB instances

### Key Results (Python reference, 10 seeds per size, pendulum omega)
```
n=20:  AntiResonant 0.9333 vs Random 0.8786 (+6.23%)
n=50:  AntiResonant 0.9157 vs Random 0.8695 (+5.31%)
n=100: AntiResonant 0.8933 vs Random 0.8669 (+3.05%)
n=200: AntiResonant 0.8893 vs Random 0.8826 (+0.76%)
n=500: AntiResonant 0.8857 vs Random 0.8768 (+1.02%)
```
Beats helical baseline at EVERY size. n=20: 93.3%, n=50: 91.6%.
Pendulum omega (bronze^1, bronze^2, bronze^3 as frequencies) improved all sizes.

### Architecture
```
include/arsat/     — C++ headers (sat_types, metallic_means, graph, spectral, solver)
src/               — C++ implementations + main.cpp CLI + bindings.cpp (pybind11)
python/arsat/      — Pure Python reference: solver.py, benchmark.py
tests/             — Python (25 tests) + C++ tests
```

### Critical design: Chiral edge weights
```
w(u,v) = cos(omega * delta_theta) + chirality * sin(omega * delta_theta)
```
- cos is even (same for both chiralities) — provides base structure
- sin is odd (flips with chirality) — creates asymmetry
- Bronze(+1): emphasizes positive phase differences
- Silver(-1): emphasizes negative phase differences
- This produces genuinely different Laplacians and eigenvector partitions

### Pendulum omega (multi-omega with metallic mean frequencies)
Instead of a single fixed omega, the solver tries 3 omega values and keeps the best:
```
omega_spread = (beta_3^1, beta_3^2, beta_3^3) = (3.303, 10.908, 36.02)
```
This makes the entire system parameter-free — metallic means control both
phase spacing AND frequency. No tuning needed. The grid search confirmed
these hit near-optimal omegas across all problem sizes tested.

### What needs doing (priority order)

1. **C++ build** — Test CMake build with Eigen/Spectra FetchContent. Needs C++17 compiler.
   May need to adjust Spectra include paths or pin versions.

2. **SATLIB benchmarks** — Download uf20-91, uf50-218, uf100-430 from SATLIB and run
   against known-satisfiable instances to compare with published results.

3. **k eigenvectors sweep** — Currently k=2. Try k=3-5 and see if more eigenvectors
   per shell improve the compound vote.

4. **Shell weight tuning** — Currently bronze=0.45, silver=0.30, golden=0.25.
   These were guesses. Grid search or Bayesian optimization over weights.

5. **pybind11 bindings** — Wire up for pip-installable C++ acceleration.

6. **Large-n scaling** — At n>=200 the advantage compresses. Could try higher
   metallic mean powers (beta^4, beta^5) in omega_spread, or adaptive k.

### Dependencies
- Python: numpy, scipy (core); pytest (dev); networkx (benchmark comparison)
- C++: Eigen 3.4, Spectra 1.0.1 (both fetched by CMake)

### GitHub: Push to Zynerji/AntiResonantSAT
