// Standalone Anderson algebra tests. These do not exercise the coupled SIMPLE
// solver or its physical-residual candidate acceptance/backtracking.
#include "AndersonAcceleration.h"

#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/QR>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using Json = nlohmann::json;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

Json solveLinearMap(const Matrix& matrix, const Vector& rhs, const Matrix& preconditioner,
                    bool accelerated, Vector& solution) {
    simple::AndersonAcceleration accelerator(5);
    Vector x = Vector::Zero(rhs.size());
    int iteration = 0, accepted = 0;
    double relativeResidual = 1.0, maximumGamma = 0.0;
    for (iteration = 1; iteration <= 50000; ++iteration) {
        const Vector g = x + preconditioner * (rhs - matrix * x);
        x = accelerated ? accelerator.update(x, g) : g;
        if (accelerated) {
            accepted += accelerator.accepted() ? 1 : 0;
            maximumGamma = std::max(maximumGamma, accelerator.coefficientNorm());
        }
        // Check the actual matrix equation, not a small update or an internal
        // least-squares residual. Both accelerated and bare runs use this gate.
        relativeResidual = (matrix * x - rhs).norm() / rhs.norm();
        if (relativeResidual < 1e-10) break;
    }
    require(relativeResidual < 1e-10, "Linear map failed to meet the true residual threshold");
    solution = x;
    return {{"iterations", iteration}, {"relative_true_residual", relativeResidual},
            {"algebraically_accepted_candidates", accepted}, {"maximum_gamma_norm", maximumGamma}};
}

Json testDiagonalContraction() {
    constexpr int n = 6;
    Vector contraction(n);
    contraction << 0.99, 0.98, 0.94, 0.85, 0.5, 0.2;
    const Matrix matrix = Matrix::Identity(n, n) - contraction.asDiagonal().toDenseMatrix();
    const Vector exact = Vector::LinSpaced(n, 1.0, 2.0);
    const Vector rhs = matrix * exact;
    Vector bareSolution, acceleratedSolution;
    const auto bare = solveLinearMap(matrix, rhs, Matrix::Identity(n, n), false, bareSolution);
    const auto accelerated = solveLinearMap(matrix, rhs, Matrix::Identity(n, n), true, acceleratedSolution);
    require(accelerated.at("iterations").get<int>() < bare.at("iterations").get<int>(),
            "Anderson did not reduce diagonal-contraction iterations");
    require((acceleratedSolution - bareSolution).norm() < 1e-7,
            "Accelerated and bare diagonal-contraction solutions differ");
    return {{"bare", bare}, {"anderson", accelerated},
            {"solution_difference", (acceleratedSolution - bareSolution).norm()},
            {"accelerated_exact_error", (acceleratedSolution - exact).norm()}};
}

Json testCoupledSpd() {
    constexpr int n = 20;
    Matrix seed(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            seed(i, j) = std::sin(0.17 * (i + 1) * (j + 3))
                + std::cos(0.29 * (i + 2) * (j + 1));
    const Matrix orthogonal = seed.householderQr().householderQ() * Matrix::Identity(n, n);
    Vector eigenvalues(n);
    for (int i = 0; i < n; ++i)
        eigenvalues[i] = 0.02 * std::pow(100.0, static_cast<double>(i) / (n - 1));
    const Matrix matrix = orthogonal * eigenvalues.asDiagonal() * orthogonal.transpose();
    const Vector inverseSqrtDiagonal = matrix.diagonal().cwiseSqrt().cwiseInverse();
    const Matrix scaled = inverseSqrtDiagonal.asDiagonal() * matrix * inverseSqrtDiagonal.asDiagonal();
    const Eigen::SelfAdjointEigenSolver<Matrix> spectrum(scaled);
    require(spectrum.info() == Eigen::Success, "SPD test spectral setup failed");
    const double omega = 0.9 / spectrum.eigenvalues().maxCoeff();
    const Matrix preconditioner = omega * matrix.diagonal().cwiseInverse().asDiagonal();
    Vector exact(n);
    for (int i = 0; i < n; ++i) exact[i] = std::sin(0.7 * i) + 0.5;
    const Vector rhs = matrix * exact;
    Vector bareSolution, acceleratedSolution;
    const auto bare = solveLinearMap(matrix, rhs, preconditioner, false, bareSolution);
    const auto accelerated = solveLinearMap(matrix, rhs, preconditioner, true, acceleratedSolution);
    require(accelerated.at("iterations").get<int>() < bare.at("iterations").get<int>(),
            "Anderson did not reduce coupled-SPD iterations");
    require((acceleratedSolution - bareSolution).norm() < 1e-7,
            "Accelerated and bare coupled-SPD solutions differ");
    return {{"bare", bare}, {"anderson", accelerated},
            {"solution_difference", (acceleratedSolution - bareSolution).norm()},
            {"accelerated_exact_error", (acceleratedSolution - exact).norm()},
            {"matrix_symmetry_error", (matrix - matrix.transpose()).norm()},
            {"minimum_eigenvalue", eigenvalues.minCoeff()}};
}

Json testAffineConstraints() {
    Matrix constraints(2, 5);
    constraints << 1.0, 2.0, -0.7, 0.3, 1.1,
                  -0.2, 0.9, 1.7, 1.0, -0.4;
    Vector rhs(2);
    rhs << 0.37, -0.21;
    const Matrix inverse = (constraints * constraints.transpose()).inverse();
    const Matrix projector = Matrix::Identity(5, 5) - constraints.transpose() * inverse * constraints;
    const Vector particular = constraints.transpose() * inverse * rhs;
    Vector x = particular, drive(5);
    drive << 0.1, -0.2, 0.4, 0.3, -0.7;
    double maximumError = 0.0;
    simple::AndersonAcceleration accelerator;
    for (int iteration = 0; iteration < 30; ++iteration) {
        // Every map output obeys B*g=rhs, with nonzero rhs. Therefore a valid
        // affine output combination must preserve both constraints.
        const Vector g = particular + 0.83 * projector * x + projector * drive;
        x = accelerator.update(x, g);
        maximumError = std::max(maximumError, (constraints * x - rhs).norm());
    }
    require(maximumError < 1e-12, "Anderson violated an affine conservation constraint");
    return {{"maximum_Bx_minus_rhs", maximumError}, {"independent_constraints", 2}, {"nonzero_rhs", true}};
}

Json testSafeguards() {
    Json result;
    simple::AndersonAcceleration accelerator;
    Vector x = Vector::Zero(1), g = Vector::Ones(1);
    accelerator.update(x, g);
    g[0] += 1e-8;
    Vector candidate = accelerator.update(x, g);
    require(!accelerator.accepted() && (candidate - g).norm() == 0.0
            && accelerator.coefficientNorm() > 1e6, "Large-gamma safeguard did not fall back to g");
    result["extreme_gamma_guard"] = {{"rejected", !accelerator.accepted()},
        {"gamma_norm", accelerator.coefficientNorm()}, {"fallback_error", (candidate - g).norm()}};

    bool caught = false;
    g[0] = std::numeric_limits<double>::quiet_NaN();
    try { accelerator.update(x, g); }
    catch (const std::invalid_argument&) { caught = true; }
    require(caught, "NaN input was not rejected");
    result["nan_input_rejected"] = caught;

    accelerator.reset();
    x = Vector::Zero(2);
    g = Vector::Ones(2);
    accelerator.update(x, g);
    x = Vector::Ones(2);
    g = 2.0 * Vector::Ones(2); // Identical residual f=1, hence zero-rank DeltaF.
    candidate = accelerator.update(x, g);
    require(!accelerator.accepted() && (candidate - g).norm() == 0.0,
            "Zero-rank history did not safely fall back to g");
    result["zero_rank_history_rejected"] = true;
    return result;
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "Usage: simple_anderson_test [output.json]\n";
        return 2;
    }
    Json report{{"scope", "Standalone fixed-point acceleration tests; no CFD residual safeguard or coupled SIMPLE acceptance is exercised"},
                {"passed", false}};
    try {
        report["diagonal_contraction"] = testDiagonalContraction();
        report["coupled_spd_jacobi_fixed_point"] = testCoupledSpd();
        report["affine_linear_constraints"] = testAffineConstraints();
        report.update(testSafeguards());
        report["passed"] = true;
    } catch (const std::exception& error) {
        report["error"] = error.what();
    }
    if (argc == 2) {
        try {
            const std::filesystem::path output(argv[1]);
            if (output.has_parent_path()) std::filesystem::create_directories(output.parent_path());
            std::ofstream file(output);
            file << report.dump(2) << '\n';
            if (!file.good()) throw std::runtime_error("Could not write the Anderson test report");
        } catch (const std::exception& error) {
            report["passed"] = false;
            report["output_error"] = error.what();
        }
    }
    std::cout << report.dump(2) << '\n';
    return report.at("passed").get<bool>() ? 0 : 1;
}
