#pragma once

#include "SimpleMesh.h"
#include <Eigen/Core>
#include <functional>
#include <vector>

namespace simple {

// Quadratic point-value reconstruction on the two-ring face-neighbor stencil.
// Only cells touching coarse/fine faces are active. Geometry is immutable for
// this object's lifetime; rebuild it after changing the mesh topology.
class QuadraticReconstruction {
public:
    using Matrix3 = Eigen::Matrix3d;
    struct Derivatives {
        std::vector<Vec3> gradient;
        std::vector<Matrix3> hessian;
    };

    explicit QuadraticReconstruction(const Mesh& mesh);
    Derivatives evaluate(const Eigen::VectorXd& cellValues) const;
    bool active(int cell) const;
    // Linear coefficients of one cached derivative in terms of cell values.
    // 0..2: gradient; 3..8: xx, yy, zz, xy, xz, yz Hessian entries.
    std::vector<std::pair<int,double>> derivativeRow(int cell,int derivative) const;
    int activeCount() const { return activeCount_; }
    int minimumStencilSize() const { return minimumStencilSize_; }
    int maximumStencilSize() const { return maximumStencilSize_; }
    double maximumConditionNumber() const { return maximumConditionNumber_; }

    // Verification helper: uses exactly the cached production coefficients but
    // evaluates manufactured data at the unwrapped sample coordinates. This is
    // required for nonperiodic polynomials; it is not a solver boundary model.
    Derivatives evaluateManufactured(const std::function<double(const Vec3&)>& value) const;

private:
    struct Sample {
        int cell;
        Vec3 displacement;
    };
    struct Stencil {
        std::vector<int> cells;
        int geometry = -1;
    };
    struct Geometry {
        std::vector<Vec3> displacements;
        // Maps differences q_sample-q_center directly to [gradient, Hessian].
        Eigen::MatrixXd coefficients;
        double condition = 0.0;
    };
    const Mesh& mesh_;
    // Preserve public cell IDs while allocating objects only for active cells.
    std::vector<int> cellToStencil_;
    std::vector<Stencil> stencils_;
    std::vector<Geometry> geometries_;
    int activeCount_ = 0;
    int minimumStencilSize_ = 0;
    int maximumStencilSize_ = 0;
    double maximumConditionNumber_ = 0.0;
    Derivatives evaluateSamples(const std::function<double(int, const Sample&)>& difference) const;
};

// Add this to (qN-qP)/distance. All inputs refer to the same adjacent periodic
// chart; actual periodic solver gradients already satisfy that convention.
double quadraticDeferredNormalGradient(const Face& face, const Vec3& ownerGradient,
    const Vec3& neighborGradient, const Eigen::Matrix3d& ownerHessian,
    const Eigen::Matrix3d& neighborHessian);
double quadraticDeferredNormalGradient(const Face& face,
    const QuadraticReconstruction::Derivatives& derivatives);

double quadraticFaceGradient(const Mesh& mesh, const Face& face, const Eigen::VectorXd& values,
    const QuadraticReconstruction::Derivatives& derivatives);

// Two-sided Taylor interpolation; intended for faces whose two cells are active.
double quadraticFaceValue(const Mesh& mesh, const Face& face, const Eigen::VectorXd& values,
    const QuadraticReconstruction::Derivatives& derivatives);

} // namespace simple
