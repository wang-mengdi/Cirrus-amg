// Optional approximate linear backend for the independent Aphros validation
// executable. The C ABI carries only CSR integers/doubles and opaque ownership;
// no Eigen, STL or long-double object crosses the MinGW/MSVC boundary.
#define AMGCL_NO_BOOST
#include <amgcl/backend/cuda.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/make_solver.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <tuple>

namespace {
using Backend = amgcl::backend::cuda<double>;
using Preconditioner = amgcl::amg<Backend,
    amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>;
using SymmetricSolver = amgcl::make_solver<Preconditioner, amgcl::solver::cg<Backend>>;
using GeneralSolver = amgcl::make_solver<Preconditioner, amgcl::solver::bicgstab<Backend>>;

void checkCuda(cudaError_t value) {
    if (value != cudaSuccess) throw std::runtime_error(cudaGetErrorString(value));
}
struct SparseHandle {
    cusparseHandle_t value = nullptr;
    SparseHandle() {
        if (cusparseCreate(&value) != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error("Cannot create cuSPARSE handle");
    }
    ~SparseHandle() { if (value) cusparseDestroy(value); }
};
struct Solver {
    // Destroy solvers and their arrays before destroying the cuSPARSE handle.
    SparseHandle sparse;
    int32_t rows;
    std::unique_ptr<SymmetricSolver> symmetric;
    std::unique_ptr<GeneralSolver> general;
    Backend::vector rhs, solution;
    double setupSeconds = 0;

    Solver(int32_t n, int32_t nz, const int32_t* ptr, const int32_t* col,
           const double* val, bool symmetricMatrix): rows(n), rhs(n), solution(n) {
        const auto start = std::chrono::steady_clock::now();
        const auto matrix = std::make_tuple(n,
            amgcl::make_iterator_range(ptr, ptr + n + 1),
            amgcl::make_iterator_range(col, col + nz),
            amgcl::make_iterator_range(val, val + nz));
        Backend::params backend(sparse.value);
        if (symmetricMatrix) {
            SymmetricSolver::params params;
            params.solver.tol = 1e-13;
            params.solver.maxiter = 500;
            symmetric.reset(new SymmetricSolver(matrix, params, backend));
        } else {
            GeneralSolver::params params;
            params.solver.tol = 1e-13;
            params.solver.maxiter = 500;
            general.reset(new GeneralSolver(matrix, params, backend));
        }
        checkCuda(cudaDeviceSynchronize());
        setupSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
    }
};

void message(char* buffer, uint64_t capacity, const char* text) noexcept {
    if (!buffer || !capacity) return;
    const size_t size = std::min<size_t>(std::strlen(text), size_t(capacity - 1));
    std::memcpy(buffer, text, size);
    buffer[size] = 0;
}
template<class Function>
int guarded(Function function, char* error, uint64_t capacity) noexcept {
    message(error, capacity, "");
    try { function(); return 0; }
    catch (const std::exception& value) { message(error, capacity, value.what()); }
    catch (...) { message(error, capacity, "Unknown CUDA AMG error"); }
    return 1;
}
}

#define API extern "C" __declspec(dllexport)
API int __cdecl twisted_cuda_amg_abi_version() noexcept { return 1; }

API int __cdecl twisted_cuda_amg_create(int32_t n, int32_t nz,
    const int32_t* ptr, const int32_t* col, const double* val,
    int32_t symmetric, void** result, double* setupSeconds,
    char* error, uint64_t capacity) noexcept {
    if (result) *result = nullptr;
    return guarded([&] {
        if (!result || !setupSeconds || !ptr || !col || !val || n <= 0 || nz <= 0 ||
            (symmetric != 0 && symmetric != 1))
            throw std::invalid_argument("Invalid CUDA AMG creation arguments");
        if (ptr[0] != 0 || ptr[n] != nz)
            throw std::invalid_argument("Invalid CSR row endpoints");
        for (int32_t row = 0; row < n; ++row) {
            if (ptr[row] < 0 || ptr[row] >= ptr[row + 1] || ptr[row + 1] > nz)
                throw std::invalid_argument("Invalid or empty CSR row");
            for (int32_t j = ptr[row]; j < ptr[row + 1]; ++j) {
                if (col[j] < 0 || col[j] >= n || !std::isfinite(val[j]) ||
                    (j > ptr[row] && col[j - 1] >= col[j]))
                    throw std::invalid_argument("Invalid CSR column order or coefficient");
            }
        }
        auto solver = std::make_unique<Solver>(n, nz, ptr, col, val, symmetric != 0);
        *setupSeconds = solver->setupSeconds;
        *result = solver.release();
    }, error, capacity);
}

API int __cdecl twisted_cuda_amg_solve(void* handle, int32_t n,
    const double* rhs, double* answer, int32_t* iterations,
    double* relativeResidual, double* solveSeconds,
    char* error, uint64_t capacity) noexcept {
    return guarded([&] {
        if (!handle || !rhs || !answer || !iterations || !relativeResidual || !solveSeconds)
            throw std::invalid_argument("Invalid CUDA AMG solve arguments");
        auto& solver = *static_cast<Solver*>(handle);
        if (n != solver.rows) throw std::invalid_argument("CUDA AMG vector size differs");
        for (int32_t i = 0; i < n; ++i)
            if (!std::isfinite(rhs[i])) throw std::invalid_argument("Nonfinite CUDA AMG RHS");
        const auto start = std::chrono::steady_clock::now();
        thrust::copy(rhs, rhs + n, solver.rhs.begin());
        thrust::fill(solver.solution.begin(), solver.solution.end(), 0.0);
        const auto info = solver.symmetric ? (*solver.symmetric)(solver.rhs, solver.solution)
                                           : (*solver.general)(solver.rhs, solver.solution);
        thrust::copy(solver.solution.begin(), solver.solution.end(), answer);
        checkCuda(cudaDeviceSynchronize());
        *iterations = int32_t(std::get<0>(info));
        *relativeResidual = std::get<1>(info);
        *solveSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        if (!std::isfinite(*relativeResidual))
            throw std::runtime_error("Nonfinite CUDA AMG residual estimate");
        for (int32_t i = 0; i < n; ++i)
            if (!std::isfinite(answer[i])) throw std::runtime_error("Nonfinite CUDA AMG solution");
        // This is only the approximate solve. Aphros must still check and
        // refine its original extended equations before accepting the result.
    }, error, capacity);
}

API void __cdecl twisted_cuda_amg_destroy(void* handle) noexcept {
    delete static_cast<Solver*>(handle);
}
