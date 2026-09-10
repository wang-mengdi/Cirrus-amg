#include "QuadraticReconstruction.h"

#include <Eigen/QR>
#include <Eigen/SVD>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

namespace simple {
namespace {

void requireInternal(const Face& face) {
    if (face.owner < 0 || face.neighbor < 0 || !(face.distance > 0.0))
        throw std::invalid_argument("Quadratic face formula requires an internal face");
}

void requireDerivatives(const Face& face, const QuadraticReconstruction::Derivatives& d) {
    requireInternal(face);
    const auto count = static_cast<int>(d.gradient.size());
    if (face.owner >= count || face.neighbor >= count || d.hessian.size() != d.gradient.size())
        throw std::invalid_argument("Quadratic derivative array size mismatch");
}

} // namespace

QuadraticReconstruction::QuadraticReconstruction(const Mesh& mesh) : mesh_(mesh) {
    const int count = static_cast<int>(mesh.cells.size());
    cellToStencil_.assign(count, -1);
    std::vector<std::vector<Sample>> adjacent(count);
    std::vector<bool> required(count, false);
    for (const auto& face : mesh.faces) {
        if (face.neighbor < 0) continue;
        adjacent[face.owner].push_back(Sample{face.neighbor, face.delta});
        adjacent[face.neighbor].push_back(Sample{face.owner, -face.delta});
        if (mesh.cells[face.owner].level != mesh.cells[face.neighbor].level)
            required[face.owner] = required[face.neighbor] = true;
    }
    stencils_.resize(std::count(required.begin(), required.end(), true));
    // Coefficients depend on h and the ordered displacements, not cell IDs.
    // Match the original input bits, including signed zero, without rounding
    // or permuting rows. Release these lookup keys after construction.
    using GeometryKey = std::vector<std::uint64_t>;
    std::map<GeometryKey, int> geometryLookup;
    auto bits = [](double value) {
        std::uint64_t result;
        static_assert(sizeof(result) == sizeof(value));
        std::memcpy(&result, &value, sizeof(value));
        return result;
    };
    std::uint64_t sampleCount = 0, uniqueSampleCount = 0;

    for (int cell = 0; cell < count; ++cell) {
        if (!required[cell]) continue;
        const double h = mesh.cells[cell].h;
        if (!(h > 0.0)) throw std::invalid_argument("Quadratic reconstruction cell size must be positive");
        // Keep distinct periodic images of a cell, but remove duplicate paths
        // reaching the same image. Dyadic native octree offsets are exactly
        // represented by this dimensionless integer key at the supported 2:1 ratio.
        using Key = std::array<std::int64_t, 4>;
        std::map<Key, Sample> samples;
        auto add = [&](int neighbor, const Vec3& displacement) {
            if (displacement.norm() < h * 1e-12) return;
            Key key{neighbor, 0, 0, 0};
            for (int axis = 0; axis < 3; ++axis)
                key[axis + 1] = std::llround(displacement[axis] / h * 1048576.0);
            auto result = samples.emplace(key, Sample{neighbor, displacement});
            if (!result.second && (result.first->second.displacement - displacement).norm() > h * 1e-10)
                throw std::runtime_error("Quadratic stencil periodic-image key collision");
        };
        for (const auto& first : adjacent[cell]) {
            add(first.cell, first.displacement);
            for (const auto& second : adjacent[first.cell])
                add(second.cell, first.displacement + second.displacement);
        }

        cellToStencil_[cell] = activeCount_;
        auto& stencil = stencils_[activeCount_];
        const int rows = static_cast<int>(samples.size());
        if (rows < 9)
            throw std::runtime_error("Quadratic two-ring stencil has fewer than 9 samples at cell " + std::to_string(cell));
        stencil.cells.reserve(samples.size());
        GeometryKey geometryKey;
        geometryKey.reserve(1 + 3 * samples.size());
        geometryKey.push_back(bits(h));
        for (const auto& entry : samples) {
            stencil.cells.push_back(entry.second.cell);
            for (int axis = 0; axis < 3; ++axis)
                geometryKey.push_back(bits(entry.second.displacement[axis]));
        }
        auto cached = geometryLookup.find(geometryKey);
        if (cached == geometryLookup.end()) {
            stencil.geometry = static_cast<int>(geometries_.size());
            geometries_.emplace_back();
            auto& geometry = geometries_.back();
            geometry.displacements.reserve(samples.size());
            for (const auto& entry : samples) geometry.displacements.push_back(entry.second.displacement);
            Eigen::MatrixXd weightedDesign(rows, 9);
            Eigen::VectorXd weights(rows);
            for (int row = 0; row < rows; ++row) {
                const Vec3 eta = geometry.displacements[row] / h;
                const double x = eta[0], y = eta[1], z = eta[2];
                weightedDesign.row(row) << x, y, z, 0.5 * x * x, 0.5 * y * y, 0.5 * z * z,
                    x * y, x * z, y * z;
                // This scales the residual; the resulting LS objective weight is its square.
                weights[row] = 1.0 / std::pow(std::max(eta.norm(), 0.5), 2);
                weightedDesign.row(row) *= weights[row];
            }
            Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(weightedDesign);
            qr.setThreshold(1e-12);
            if (qr.rank() != 9)
                throw std::runtime_error("Rank-deficient quadratic two-ring stencil at cell " + std::to_string(cell));
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(weightedDesign);
            const auto singular = svd.singularValues();
            const double condition = singular[0] / singular[8];
            if (!std::isfinite(condition) || condition > 1e10)
                throw std::runtime_error("Ill-conditioned quadratic two-ring stencil at cell " + std::to_string(cell));
            // Solve all sample basis vectors at setup time. The stored coefficients
            // already include spatial scaling, so evaluation is only a matrix product.
            geometry.coefficients = qr.solve(weights.asDiagonal().toDenseMatrix());
            geometry.coefficients.topRows(3) /= h;
            geometry.coefficients.bottomRows(6) /= h * h;
            if (!geometry.coefficients.allFinite())
                throw std::runtime_error("Nonfinite quadratic reconstruction coefficients");
            geometry.condition = condition;
            uniqueSampleCount += rows;
            geometryLookup.emplace(std::move(geometryKey), stencil.geometry);
        } else stencil.geometry = cached->second;
        sampleCount += rows;
        ++activeCount_;
        minimumStencilSize_ = activeCount_ == 1 ? rows : std::min(minimumStencilSize_, rows);
        maximumStencilSize_ = std::max(maximumStencilSize_, rows);
        maximumConditionNumber_ = std::max(maximumConditionNumber_, geometries_[stencil.geometry].condition);
    }
    if (std::getenv("SIMPLE_CONSTRUCTION_MEMORY")) {
        std::printf("Quadratic geometry cache: active_cells=%d unique_templates=%llu sample_entries=%llu unique_sample_entries=%llu\n",
            activeCount_, static_cast<unsigned long long>(geometries_.size()),
            static_cast<unsigned long long>(sampleCount), static_cast<unsigned long long>(uniqueSampleCount));
        std::fflush(stdout);
    }
}

bool QuadraticReconstruction::active(int cell) const {
    return cell >= 0 && cell < static_cast<int>(cellToStencil_.size()) && cellToStencil_[cell] >= 0;
}

std::vector<std::pair<int,double>> QuadraticReconstruction::derivativeRow(int cell,int derivative) const {
    if(!active(cell)||derivative<0||derivative>=9)throw std::invalid_argument("Inactive quadratic derivative row");
    const auto& stencil=stencils_[cellToStencil_[cell]];
    const auto& geometry=geometries_[stencil.geometry];
    std::map<int,double> row;double sum=0;
    for(int j=0;j<int(stencil.cells.size());++j) {
        const double coefficient=geometry.coefficients(derivative,j);
        row[stencil.cells[j]]+=coefficient;sum+=coefficient;
    }
    row[cell]-=sum;
    return {row.begin(),row.end()};
}

QuadraticReconstruction::Derivatives QuadraticReconstruction::evaluateSamples(
    const std::function<double(int, const Sample&)>& difference) const {
    Derivatives output;
    output.gradient.assign(cellToStencil_.size(), Vec3::Zero());
    output.hessian.assign(cellToStencil_.size(), Matrix3::Zero());
    for (int cell = 0; cell < static_cast<int>(cellToStencil_.size()); ++cell) {
        const int index = cellToStencil_[cell];
        if (index < 0) continue;
        const auto& stencil = stencils_[index];
        const auto& geometry = geometries_[stencil.geometry];
        Eigen::VectorXd rhs(stencil.cells.size());
        for (int row = 0; row < rhs.size(); ++row)
            rhs[row] = difference(cell, Sample{stencil.cells[row], geometry.displacements[row]});
        const Eigen::Matrix<double, 9, 1> coefficients = geometry.coefficients * rhs;
        if (!coefficients.allFinite()) throw std::runtime_error("Nonfinite quadratic field reconstruction");
        output.gradient[cell] = coefficients.head<3>();
        output.hessian[cell] << coefficients[3], coefficients[6], coefficients[7],
                               coefficients[6], coefficients[4], coefficients[8],
                               coefficients[7], coefficients[8], coefficients[5];
    }
    return output;
}

QuadraticReconstruction::Derivatives QuadraticReconstruction::evaluate(const Eigen::VectorXd& values) const {
    if (values.size() != static_cast<int>(mesh_.cells.size()) || !values.allFinite())
        throw std::invalid_argument("Quadratic reconstruction requires finite cell values of matching size");
    return evaluateSamples([&](int cell, const Sample& sample) { return values[sample.cell] - values[cell]; });
}

QuadraticReconstruction::Derivatives QuadraticReconstruction::evaluateManufactured(
    const std::function<double(const Vec3&)>& value) const {
    return evaluateSamples([&](int cell, const Sample& sample) {
        const auto& center = mesh_.cells[cell].center;
        return value(center + sample.displacement) - value(center);
    });
}

double quadraticDeferredNormalGradient(const Face& face, const Vec3& ownerGradient,
    const Vec3& neighborGradient, const Eigen::Matrix3d& ownerHessian,
    const Eigen::Matrix3d& neighborHessian) {
    requireInternal(face);
    const Vec3 n = Vec3::Unit(face.axis) * face.sign;
    const Vec3 tangent = face.delta - face.distance * n;
    const Vec3 midpointToFace = 0.5 * (face.ownerOffset + face.neighborOffset);
    const Vec3 midpointGradient = 0.5 * (ownerGradient + neighborGradient);
    const Eigen::Matrix3d faceHessian = 0.5 * (ownerHessian + neighborHessian);
    return -midpointGradient.dot(tangent) / face.distance + n.dot(faceHessian * midpointToFace);
}

double quadraticDeferredNormalGradient(const Face& face,
    const QuadraticReconstruction::Derivatives& derivatives) {
    requireDerivatives(face, derivatives);
    return quadraticDeferredNormalGradient(face, derivatives.gradient[face.owner],
        derivatives.gradient[face.neighbor], derivatives.hessian[face.owner],
        derivatives.hessian[face.neighbor]);
}

double quadraticFaceGradient(const Mesh& mesh, const Face& face, const Eigen::VectorXd& values,
    const QuadraticReconstruction::Derivatives& derivatives) {
    if (values.size() != static_cast<int>(mesh.cells.size()))
        throw std::invalid_argument("Quadratic face-gradient value array size mismatch");
    requireDerivatives(face, derivatives);
    return (values[face.neighbor] - values[face.owner]) / face.distance
        + quadraticDeferredNormalGradient(face, derivatives);
}

double quadraticFaceValue(const Mesh& mesh, const Face& face, const Eigen::VectorXd& values,
    const QuadraticReconstruction::Derivatives& derivatives) {
    if (values.size() != static_cast<int>(mesh.cells.size()))
        throw std::invalid_argument("Quadratic face-interpolation value array size mismatch");
    requireDerivatives(face, derivatives);
    auto taylor = [&](int cell, const Vec3& offset) {
        return values[cell] + derivatives.gradient[cell].dot(offset)
            + 0.5 * offset.dot(derivatives.hessian[cell] * offset);
    };
    const double w = mesh.cells[face.neighbor].h / (mesh.cells[face.owner].h + mesh.cells[face.neighbor].h);
    return w * taylor(face.owner, face.ownerOffset) + (1 - w) * taylor(face.neighbor, face.neighborOffset);
}

} // namespace simple
