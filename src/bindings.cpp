#ifdef ARSAT_BUILD_PYTHON

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "arsat/solver.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_arsat_cpp, m) {
    m.doc() = "AntiResonantSAT C++ core — chiral multi-shell spectral SAT solver";

    // SolverResult
    py::class_<arsat::SolverResult>(m, "SolverResult")
        .def_readonly("assignment", &arsat::SolverResult::assignment)
        .def_readonly("satisfaction_ratio", &arsat::SolverResult::satisfaction_ratio)
        .def_readonly("runtime_ms", &arsat::SolverResult::runtime_ms)
        .def_readonly("n_vars", &arsat::SolverResult::n_vars)
        .def_readonly("n_clauses", &arsat::SolverResult::n_clauses)
        .def_readonly("bronze_rho", &arsat::SolverResult::bronze_rho)
        .def_readonly("silver_rho", &arsat::SolverResult::silver_rho)
        .def_readonly("golden_rho", &arsat::SolverResult::golden_rho)
        .def_readonly("cache_hits", &arsat::SolverResult::cache_hits);

    // Config
    py::class_<arsat::AntiResonantSolver::Config>(m, "SolverConfig")
        .def(py::init<>())
        .def_readwrite("k_eigenvectors", &arsat::AntiResonantSolver::Config::k_eigenvectors)
        .def_readwrite("omega", &arsat::AntiResonantSolver::Config::omega)
        .def_readwrite("use_mobius", &arsat::AntiResonantSolver::Config::use_mobius)
        .def_readwrite("use_cache", &arsat::AntiResonantSolver::Config::use_cache)
        .def_readwrite("adaptive_voting", &arsat::AntiResonantSolver::Config::adaptive_voting)
        .def_readwrite("bronze_weight", &arsat::AntiResonantSolver::Config::bronze_weight)
        .def_readwrite("silver_weight", &arsat::AntiResonantSolver::Config::silver_weight)
        .def_readwrite("golden_weight", &arsat::AntiResonantSolver::Config::golden_weight);

    // Solver
    py::class_<arsat::AntiResonantSolver>(m, "AntiResonantSolver")
        .def(py::init<arsat::AntiResonantSolver::Config>(),
             py::arg("config") = arsat::AntiResonantSolver::Config{})
        .def("solve", &arsat::AntiResonantSolver::solve,
             py::arg("formula"), py::arg("n_vars"));

    // Utility functions
    m.def("random_3sat", &arsat::random_3sat,
          py::arg("n_vars"), py::arg("m_clauses"), py::arg("seed") = 42);
    m.def("evaluate_sat", &arsat::evaluate_sat);
    m.def("parse_dimacs", &arsat::parse_dimacs);
    m.def("metallic_mean", &arsat::metallic_mean);
}

#endif  // ARSAT_BUILD_PYTHON
