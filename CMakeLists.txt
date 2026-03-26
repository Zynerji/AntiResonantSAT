cmake_minimum_required(VERSION 3.16)
project(AntiResonantSAT VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Options ──────────────────────────────────────────────────────────────
option(ARSAT_BUILD_PYTHON "Build pybind11 bindings" OFF)
option(ARSAT_BUILD_TESTS  "Build C++ tests"         OFF)

# ── Fetch Eigen & Spectra (header-only) ──────────────────────────────────
include(FetchContent)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)
FetchContent_Declare(
    spectra
    GIT_REPOSITORY https://github.com/yixuan/spectra.git
    GIT_TAG        v1.0.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(eigen spectra)

# ── Core library ─────────────────────────────────────────────────────────
add_library(arsat_core STATIC
    src/metallic_means.cpp
    src/graph.cpp
    src/spectral.cpp
    src/solver.cpp
)
target_include_directories(arsat_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${eigen_SOURCE_DIR}
    ${spectra_SOURCE_DIR}/include
)
target_compile_definitions(arsat_core PRIVATE
    $<$<CONFIG:Release>:NDEBUG>
)
if(MSVC)
    target_compile_options(arsat_core PRIVATE /O2 /fp:fast /arch:AVX2)
else()
    target_compile_options(arsat_core PRIVATE -O3 -march=native -ffast-math)
endif()

# ── CLI executable ───────────────────────────────────────────────────────
add_executable(arsat src/main.cpp)
target_link_libraries(arsat PRIVATE arsat_core)

# ── pybind11 bindings (optional) ─────────────────────────────────────────
if(ARSAT_BUILD_PYTHON)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(_arsat_cpp src/bindings.cpp)
    target_link_libraries(_arsat_cpp PRIVATE arsat_core)
endif()

# ── Tests (optional) ────────────────────────────────────────────────────
if(ARSAT_BUILD_TESTS)
    enable_testing()
    add_executable(arsat_test tests/test_solver.cpp)
    target_link_libraries(arsat_test PRIVATE arsat_core)
    add_test(NAME arsat_test COMMAND arsat_test)
endif()
