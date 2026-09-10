#include "OperatorChecks.h"
#include "QuadraticReconstruction.h"

#include <Eigen/LU>
#include <Eigen/Sparse>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>

namespace simple {
namespace {

using Json = nlohmann::json;
using Matrix3 = Eigen::Matrix3d;
using Sparse = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;

struct Analytic {
    std::string name;
    std::function<double(const Vec3&)> value;
    std::function<Vec3(const Vec3&)> gradient;
    Matrix3 hessian = Matrix3::Zero();
};

struct Norms {
    int count = 0;
    double area = 0.0;
    double sumAbs = 0.0;
    double sumSquare = 0.0;
    double maximum = 0.0;
    void add(double error, double faceArea) {
        ++count;
        area += faceArea;
        sumAbs += faceArea * std::abs(error);
        sumSquare += faceArea * error * error;
        maximum = std::max(maximum, std::abs(error));
    }
    Json json() const {
        return {{"count", count}, {"area", area},
                {"L1", area > 0 ? sumAbs / area : 0.0},
                {"L2", area > 0 ? std::sqrt(sumSquare / area) : 0.0},
                {"Linf", maximum}};
    }
};

Vec3 normal(const Face& f) { return Vec3::Unit(f.axis) * f.sign; }

double weight(const Mesh& mesh, const Face& f) {
    return mesh.cells[f.neighbor].h / (mesh.cells[f.owner].h + mesh.cells[f.neighbor].h);
}

Eigen::VectorXd divergence(const Mesh& mesh, const Eigen::VectorXd& flux) {
    Eigen::VectorXd result = Eigen::VectorXd::Zero(mesh.cells.size());
    for (int j = 0; j < flux.size(); ++j) {
        const auto& f = mesh.faces[j];
        result[f.owner] += flux[j];
        if (f.neighbor >= 0) result[f.neighbor] -= flux[j];
    }
    return result;
}

// Each side of a periodic face evaluates the manufactured function in its own
// local unwrapped chart. This avoids treating a nonperiodic polynomial as though
// its samples jumped across the torus. Analytic Dirichlet samples are used on all
// exterior faces here; they are not Solver's physical wall/opening conditions.
std::vector<Vec3> manufacturedLeastSquares(const Mesh& mesh, const Analytic& field) {
    std::vector<Matrix3> metric(mesh.cells.size(), Matrix3::Zero());
    std::vector<Vec3> rhs(mesh.cells.size(), Vec3::Zero());
    auto add = [&](int i, const Vec3& delta, double difference, double area) {
        const double factor = area / (mesh.cells[i].h * mesh.cells[i].h * delta.squaredNorm());
        metric[i] += factor * delta * delta.transpose();
        rhs[i] += factor * delta * difference;
    };
    for (const auto& f : mesh.faces) {
        const auto& owner = mesh.cells[f.owner].center;
        if (f.neighbor >= 0) {
            const auto& neighbor = mesh.cells[f.neighbor].center;
            add(f.owner, f.delta, field.value(owner + f.delta) - field.value(owner), f.area);
            add(f.neighbor, -f.delta, field.value(neighbor - f.delta) - field.value(neighbor), f.area);
        } else {
            add(f.owner, f.ownerOffset, field.value(f.center) - field.value(owner), f.area);
        }
    }
    for (int i = 0; i < static_cast<int>(rhs.size()); ++i) {
        if (!(metric[i].determinant() > 1e-12))
            throw std::runtime_error("Operator check encountered a singular least-squares metric");
        rhs[i] = metric[i].inverse() * rhs[i];
    }
    return rhs;
}

Json quadraticFluxErrors(const Mesh& mesh, const Analytic& field, bool exactGradients) {
    std::vector<Vec3> gradients;
    if (!exactGradients) gradients = manufacturedLeastSquares(mesh, field);
    Norms derivative[2], integratedFlux[2];
    for (const auto& f : mesh.faces) {
        if (f.neighbor < 0) continue;
        const auto& a = mesh.cells[f.owner];
        const auto& b = mesh.cells[f.neighbor];
        const Vec3 image = a.center + f.delta;
        const double qa = field.value(a.center);
        const double qb = field.value(image);
        const Vec3 ga = exactGradients ? field.gradient(a.center) : gradients[f.owner];
        // A quadratic gradient changes by Hessian * translation under unwrapping.
        const Vec3 gb = exactGradients ? field.gradient(image) :
            (gradients[f.neighbor] + field.hessian * (image - b.center)).eval();
        const double w = weight(mesh, f);
        const Vec3 tangent = f.delta - f.distance * normal(f);
        const double calculated = (qb - qa - (w * ga + (1 - w) * gb).dot(tangent)) / f.distance;
        const double exact = field.gradient(f.center).dot(normal(f));
        const int group = a.level == b.level ? 0 : 1;
        derivative[group].add(calculated - exact, f.area);
        // Outward diffusive flux for unit diffusivity: -A * grad(q).n.
        integratedFlux[group].add(-f.area * (calculated - exact), f.area);
    }
    Json result;
    const char* names[] = {"uniform", "coarse_fine"};
    for (int group = 0; group < 2; ++group)
        result[names[group]] = {{"normal_derivative_error", derivative[group].json()},
                                {"unit_diffusivity_integrated_flux_error", integratedFlux[group].json()}};
    return result;
}

} // namespace

void runOperatorChecks(const Mesh& mesh, const std::string& output) {
    if (mesh.cells.empty()) throw std::invalid_argument("Operator checks require a nonempty mesh");
    const int nc = static_cast<int>(mesh.cells.size());
    const int nf = static_cast<int>(mesh.faces.size());
    Json report{{"scope", "Independent mesh and replicated operator-formula checks, not calls to Solver internals or end-to-end CFD validation"},
                {"cells", nc}, {"faces", nf}, {"coarse_fine_faces", mesh.coarseFineFaces},
                {"periodic_z", mesh.periodicZ},
                {"quadratic_policy", "Report truncation error without a pass threshold; uniform excludes physical boundary faces"},
                {"norm_definition", "L1 and L2 are area-weighted means over each face class; Linf is the maximum absolute face error"},
                {"periodic_policy", "Manufactured polynomials use adjacent unwrapped analytic values and translated gradients, not discontinuous modulo-domain samples"},
                {"least_squares_boundary_policy", "Analytic Dirichlet samples on every exterior face, independent of physical solver boundary conditions"}};
    std::vector<std::string> failures;
    auto check = [&](bool condition, const std::string& name) {
        if (!condition) failures.push_back(name);
    };

    std::vector<Matrix3> moment(nc, Matrix3::Zero());
    int periodicFaces = 0;
    for (const auto& f : mesh.faces) {
        const Vec3 sf = f.area * normal(f);
        moment[f.owner] += sf * f.ownerOffset.transpose();
        if (f.neighbor >= 0) {
            moment[f.neighbor] -= sf * f.neighborOffset.transpose();
            if ((mesh.cells[f.owner].center + f.delta - mesh.cells[f.neighbor].center).norm() > 1e-12)
                ++periodicFaces;
        }
    }
    double momentError = 0.0;
    for (int i = 0; i < nc; ++i)
        momentError = std::max(momentError,
            (moment[i] / mesh.cells[i].volume - Matrix3::Identity()).cwiseAbs().maxCoeff());
    report["first_geometric_moment_relative_linf"] = momentError;
    report["periodic_faces_checked"] = periodicFaces;
    check(momentError < 1e-11, "Gauss first geometric moment");

    std::mt19937_64 rng(20260906);
    std::uniform_real_distribution<double> random(-1.0, 1.0);
    Eigen::VectorXd randomFlux(nf);
    long double boundaryFlux = 0.0L;
    long double totalAbs = 0.0L;
    for (int j = 0; j < nf; ++j) {
        randomFlux[j] = random(rng) * mesh.faces[j].area;
        totalAbs += std::abs(randomFlux[j]);
        if (mesh.faces[j].neighbor < 0) boundaryFlux += randomFlux[j];
    }
    const auto randomDiv = divergence(mesh, randomFlux);
    long double summedDiv = 0.0L;
    for (int i = 0; i < nc; ++i) summedDiv += randomDiv[i];
    const double conservationError = static_cast<double>(
        std::abs(summedDiv - boundaryFlux) / std::max(totalAbs, 1e-30L));
    report["random_flux_global_conservation_relative_error"] = conservationError;
    report["random_flux_test_note"] = "Arbitrary oriented fluxes include physical boundaries; sum of integrated cell divergence must equal total boundary flux";
    check(conservationError < 1e-13, "global random-flux conservation");

    for (int test = 0; test < 5; ++test) {
        Vec3 slope = Vec3::Zero();
        if (test >= 1 && test <= 3) slope[test - 1] = 1.0;
        if (test == 4) slope = Vec3(0.71, -1.31, 2.17);
        const double offset = 0.371;
        Analytic field;
        field.name = test == 0 ? "constant" : test == 4 ? "mixed_affine" : "affine_axis_" + std::to_string(test - 1);
        field.value = [=](const Vec3& x) { return offset + slope.dot(x); };
        field.gradient = [=](const Vec3&) { return slope; };
        const auto gradients = manufacturedLeastSquares(mesh, field);
        double gradientError = 0.0, interpolationError = 0.0, compactError = 0.0;
        for (int i = 0; i < nc; ++i)
            gradientError = std::max(gradientError, (gradients[i] - slope).norm());
        for (const auto& f : mesh.faces) {
            if (f.neighbor < 0) continue;
            const auto& a = mesh.cells[f.owner];
            const Vec3 image = a.center + f.delta;
            const double w = weight(mesh, f);
            const Vec3 gf = w * gradients[f.owner] + (1 - w) * gradients[f.neighbor];
            const Vec3 skew = w * f.ownerOffset + (1 - w) * f.neighborOffset;
            const double qa = field.value(a.center), qb = field.value(image);
            const double interpolated = w * qa + (1 - w) * qb + gf.dot(skew);
            const double compact = (qb - qa - gf.dot(f.delta - f.distance * normal(f))) / f.distance;
            interpolationError = std::max(interpolationError, std::abs(interpolated - field.value(f.center)));
            compactError = std::max(compactError, std::abs(compact - slope.dot(normal(f))));
        }
        const double scale = std::max(1.0, slope.norm());
        report["affine"][field.name] = {{"least_squares_gradient_linf", gradientError},
            {"face_interpolation_linf", interpolationError}, {"compact_normal_gradient_linf", compactError}};
        check(gradientError < 1e-10 * scale, field.name + " reconstructed gradient");
        check(interpolationError < 1e-11 * scale, field.name + " face interpolation");
        check(compactError < 1e-10 * scale, field.name + " compact normal gradient");
    }

    const double height = mesh.extent[1];
    std::vector<Analytic> quadratic;
    Analytic channel;
    channel.name = "y_times_H_minus_y";
    channel.value = [=](const Vec3& x) { return x[1] * (height - x[1]); };
    channel.gradient = [=](const Vec3& x) { return Vec3(0.0, height - 2 * x[1], 0.0); };
    channel.hessian(1, 1) = -2.0;
    quadratic.push_back(channel);
    Analytic cross;
    cross.name = "x_times_y";
    cross.value = [](const Vec3& x) { return x[0] * x[1]; };
    cross.gradient = [](const Vec3& x) { return Vec3(x[1], x[0], 0.0); };
    cross.hessian(0, 1) = cross.hessian(1, 0) = 1.0;
    quadratic.push_back(cross);
    Analytic sphere;
    sphere.name = "x_squared_plus_y_squared_plus_z_squared";
    sphere.value = [](const Vec3& x) { return x.squaredNorm(); };
    sphere.gradient = [](const Vec3& x) { return (2.0 * x).eval(); };
    sphere.hessian = 2.0 * Matrix3::Identity();
    quadratic.push_back(sphere);
    Analytic mixed;
    mixed.name = "full_mixed_quadratic";
    Matrix3 mixedHessian;
    mixedHessian << 1.1, -0.7, 0.33, -0.7, 2.0, 0.8, 0.33, 0.8, -0.9;
    const Vec3 mixedSlope(0.23, -0.61, 1.47);
    mixed.value = [=](const Vec3& x) { return 0.17 + mixedSlope.dot(x) + 0.5 * x.dot(mixedHessian * x); };
    mixed.gradient = [=](const Vec3& x) { return (mixedSlope + mixedHessian * x).eval(); };
    mixed.hessian = mixedHessian;
    quadratic.push_back(mixed);
    for (const auto& field : quadratic) {
        report["quadratic_flux_errors"][field.name] = {
            {"exact_cell_gradients", quadraticFluxErrors(mesh, field, true)},
            {"least_squares_cell_gradients", quadraticFluxErrors(mesh, field, false)}};
    }

    // Exercise the actual cached quadratic-reconstruction module and its shared
    // deferred face helper. A coupled SIMPLE solve still requires separate tests.
    QuadraticReconstruction reconstruction(mesh);
    report["quadratic_reconstruction_module"] = {
        {"scope", "Calls the production QuadraticReconstruction cache/evaluation and deferred face helper; coupled SIMPLE remains a separate end-to-end validation"},
        {"active_cells", reconstruction.activeCount()},
        {"minimum_stencil_samples", reconstruction.minimumStencilSize()},
        {"maximum_stencil_samples", reconstruction.maximumStencilSize()},
        {"maximum_weighted_design_condition_number", reconstruction.maximumConditionNumber()},
        {"applicable", reconstruction.activeCount() > 0}};
    for (const auto& field : quadratic) {
        const auto derivatives = reconstruction.evaluateManufactured(field.value);
        double gradientError = 0.0, hessianError = 0.0, interpolationError = 0.0;
        Norms normalError, fluxError;
        for (int i = 0; i < nc; ++i) {
            if (!reconstruction.active(i)) {
                check(derivatives.gradient[i].squaredNorm() == 0.0 && derivatives.hessian[i].squaredNorm() == 0.0,
                      "inactive quadratic reconstruction must remain zero");
                continue;
            }
            gradientError = std::max(gradientError, (derivatives.gradient[i] - field.gradient(mesh.cells[i].center)).norm());
            hessianError = std::max(hessianError, (derivatives.hessian[i] - field.hessian).norm());
        }
        for (const auto& f : mesh.faces) {
            if (f.neighbor < 0 || mesh.cells[f.owner].level == mesh.cells[f.neighbor].level) continue;
            check(reconstruction.active(f.owner) && reconstruction.active(f.neighbor), "both coarse/fine cells must have quadratic stencils");
            const auto& a = mesh.cells[f.owner];
            const auto& b = mesh.cells[f.neighbor];
            const Vec3 image = a.center + f.delta;
            const Vec3 gp = derivatives.gradient[f.owner];
            const Matrix3 hp = derivatives.hessian[f.owner];
            const Matrix3 hn = derivatives.hessian[f.neighbor];
            const Vec3 gn = derivatives.gradient[f.neighbor] + hn * (image - b.center);
            const double qa = field.value(a.center), qb = field.value(image);
            const double calculated = (qb - qa) / f.distance
                + quadraticDeferredNormalGradient(f, gp, gn, hp, hn);
            const double error = calculated - field.gradient(f.center).dot(normal(f));
            normalError.add(error, f.area);
            fluxError.add(-f.area * error, f.area);
            const double w = weight(mesh, f);
            const double qpf = qa + gp.dot(f.ownerOffset) + 0.5 * f.ownerOffset.dot(hp * f.ownerOffset);
            const double qnf = qb + gn.dot(f.neighborOffset) + 0.5 * f.neighborOffset.dot(hn * f.neighborOffset);
            interpolationError = std::max(interpolationError, std::abs(w * qpf + (1 - w) * qnf - field.value(f.center)));
        }
        report["quadratic_reconstruction_module"]["fields"][field.name] = {
            {"reconstructed_gradient_linf", gradientError}, {"reconstructed_hessian_linf", hessianError},
            {"two_sided_taylor_face_value_linf", interpolationError},
            {"coarse_fine_normal_derivative_error", normalError.json()},
            {"coarse_fine_unit_diffusivity_flux_error", fluxError.json()}};
        check(gradientError < 1e-10, field.name + " quadratic-module gradient");
        check(hessianError < 1e-9, field.name + " quadratic-module Hessian");
        check(normalError.maximum < 1e-10, field.name + " quadratic-module compact face gradient");
        check(interpolationError < 1e-11, field.name + " quadratic-module Taylor face interpolation");
    }
    // This field is periodic in x and z, so the production evaluate(q) path can
    // be checked directly without a manufactured sample override.
    Eigen::VectorXd channelValues(nc);
    for (int i = 0; i < nc; ++i) channelValues[i] = channel.value(mesh.cells[i].center);
    const auto channelDerivatives = reconstruction.evaluate(channelValues);
    double productionChannelError = 0.0;
    double productionChannelInterpolationError = 0.0;
    for (const auto& f : mesh.faces) {
        if (f.neighbor < 0 || mesh.cells[f.owner].level == mesh.cells[f.neighbor].level) continue;
        productionChannelError = std::max(productionChannelError,
            std::abs(quadraticFaceGradient(mesh, f, channelValues, channelDerivatives)
                - channel.gradient(f.center).dot(normal(f))));
        productionChannelInterpolationError = std::max(productionChannelInterpolationError,
            std::abs(quadraticFaceValue(mesh, f, channelValues, channelDerivatives) - channel.value(f.center)));
    }
    report["quadratic_reconstruction_module"]["production_cell_values_channel_gradient_linf"] = productionChannelError;
    report["quadratic_reconstruction_module"]["production_cell_values_channel_interpolation_linf"] = productionChannelInterpolationError;
    check(productionChannelError < 1e-10 && productionChannelInterpolationError < 1e-11,
          "production quadratic evaluate/face helpers on periodic channel polynomial");

    // Positive compact pressure matrix L = -D k Delta. The deferred tangential
    // term E has +k*grad.tangent. No reference row/column replacement is applied:
    // this checks the conservation identity before the solver's gauge operation.
    Eigen::VectorXd p(nc), mobility(nc);
    std::vector<Vec3> randomGradient(nc);
    for (int i = 0; i < nc; ++i) {
        p[i] = random(rng);
        mobility[i] = 1.25 + 0.5 * random(rng);
        randomGradient[i] = Vec3(random(rng), random(rng), random(rng));
    }
    std::vector<Triplet> entries;
    entries.reserve(nf * 4);
    Eigen::VectorXd orthogonalFlux = Eigen::VectorXd::Zero(nf);
    Eigen::VectorXd explicitFlux = Eigen::VectorXd::Zero(nf);
    int pressureBoundaries = 0;
    for (int j = 0; j < nf; ++j) {
        const auto& f = mesh.faces[j];
        if (f.boundary == 1) continue;
        const int a = f.owner, b = f.neighbor;
        const double w = b >= 0 ? weight(mesh, f) : 1.0;
        const double mf = w * mobility[a] + (b >= 0 ? (1 - w) * mobility[b] : 0.0);
        const double k = mf * f.area / f.distance;
        entries.emplace_back(a, a, k);
        if (b >= 0) {
            entries.emplace_back(a, b, -k);
            entries.emplace_back(b, a, -k);
            entries.emplace_back(b, b, k);
            orthogonalFlux[j] = -k * (p[b] - p[a]);
            explicitFlux[j] = k * (w * randomGradient[a] + (1 - w) * randomGradient[b]).dot(
                f.delta - f.distance * normal(f));
        } else {
            ++pressureBoundaries;
            orthogonalFlux[j] = k * p[a]; // homogeneous pressure-correction boundary
        }
    }
    Sparse matrix(nc, nc);
    matrix.setFromTriplets(entries.begin(), entries.end());
    const Eigen::VectorXd lp = matrix * p;
    const Eigen::VectorXd divOrth = divergence(mesh, orthogonalFlux);
    const Eigen::VectorXd divExplicit = divergence(mesh, explicitFlux);
    const Eigen::VectorXd correctedFlux = randomFlux + orthogonalFlux + explicitFlux;
    const Eigen::VectorXd correctedDiv = divergence(mesh, correctedFlux);
    const double scale = std::max(1.0, lp.cwiseAbs().maxCoeff());
    const double lpError = (lp - divOrth).cwiseAbs().maxCoeff() / scale;
    const double fullError = (correctedDiv - randomDiv - lp - divExplicit).cwiseAbs().maxCoeff() / scale;
    const double rowSum = (matrix * Eigen::VectorXd::Ones(nc)).cwiseAbs().maxCoeff();
    const Sparse difference = matrix - Sparse(matrix.transpose());
    const double symmetryError = difference.norm() / std::max(1.0, matrix.norm());
    report["pressure_correction_identity"] = {
        {"definition", "L=-D*k*Delta; E=+k*g.tangent; Fcorrected=Fstar-k*Delta(p)+E; D(Fcorrected)=D(Fstar)+L*p+D(E)"},
        {"scope", "Independent sparse assembly with random p, positive mobility and random deferred gradients; no pressure solve or gauge modification"},
        {"Lp_minus_D_orthogonal_flux_relative_linf", lpError},
        {"corrected_divergence_identity_relative_linf", fullError},
        {"matrix_relative_symmetry_error", symmetryError},
        {"row_sum_linf", rowSum}, {"homogeneous_pressure_boundary_faces", pressureBoundaries},
        {"random_pressure_energy", p.dot(lp)}};
    check(lpError < 1e-12, "pressure compact matrix/flux identity");
    check(fullError < 1e-12, "pressure corrected-flux divergence identity");
    check(symmetryError < 1e-13, "pressure matrix symmetry");
    check(p.dot(lp) >= -1e-12, "pressure matrix positive energy");
    if (pressureBoundaries == 0) check(rowSum < 1e-12, "periodic pressure constant nullspace");

    report["passed"] = failures.empty();
    report["failures"] = failures;
    const std::filesystem::path path(output);
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path);
    stream << report.dump(2) << '\n';
    if (!stream.good()) throw std::runtime_error("Cannot write operator-check JSON: " + output);
    std::cout << "Independent operator checks: " << (failures.empty() ? "PASS" : "FAIL")
              << " geometry=" << momentError << " fluxConservation=" << conservationError
              << " pressureIdentity=" << fullError << " output=" << output << '\n';
    if (!failures.empty())
        throw std::runtime_error("Independent SIMPLE operator checks failed: " + failures.front());
}

} // namespace simple
