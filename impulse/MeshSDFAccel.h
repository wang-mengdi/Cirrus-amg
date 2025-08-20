#pragma once
#include <Eigen/Core>
#include <igl/AABB.h>
#include <igl/readOBJ.h>


class MeshSDFAccel {
public:
    // Switch numeric precision here (float for single-precision).
    using Scalar = float;

    // Public API
    // Build once: copy V/F, build AABB, precompute per-face normals.
    void build(const Eigen::Matrix<Scalar, -1, 3>& V_in, const Eigen::Matrix<int, -1, 3>& F_in);

    // Batch query signed distance for points P (N x 3).
    // Negative inside, positive outside. Uses pseudonormal sign rule.
    Eigen::Matrix<Scalar, -1, 1> querySDF(const Eigen::Matrix<Scalar, -1, 3>& P) const;

private:
    // Stored mesh
    Eigen::Matrix<Scalar, -1, 3> V_;   // (nV x 3)
    Eigen::Matrix<int, -1, 3> F_;   // (nF x 3)

    // AABB over (V_, F_). igl::AABB is header-only; safe to keep as a member.
    igl::AABB<Eigen::Matrix<Scalar, -1, 3>, 3> tree_;

    // Per-face unit normals (nF x 3), for pseudonormal sign
    Eigen::Matrix<Scalar, -1, 3> FN_;
};
