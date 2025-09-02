#pragma once
#include <Eigen/Core>
#include <igl/AABB.h>
#include <igl/readOBJ.h>

#include "PoissonTile.h"

class MeshSDFAccel {
public:
    MeshSDFAccel(){}
	MeshSDFAccel(const fs::path& obj_path) { init(obj_path); }

    void init(const fs::path& obj_path);

    // Build once: copy V/F, build AABB, precompute per-face normals.
    void build(const Eigen::Matrix<T, -1, 3>& V_in, const Eigen::Matrix<int, -1, 3>& F_in);

    // Batch query signed distance for points P (N x 3).
    // Negative inside, positive outside. Uses pseudonormal sign rule.
	// xform is the affine transform from mesh-local to world coordinates.
    std::vector<T> querySDF(const std::vector<Vec>& points, const Eigen::Transform<T, 3, Eigen::Affine>& xform) const;

public:
    // Stored mesh
    Eigen::Matrix<T, -1, 3> V_;   // (nV x 3)
    Eigen::Matrix<int, -1, 3> F_; // (nF x 3)

    // AABB over (V_, F_). igl::AABB is header-only; safe to keep as a member.
    igl::AABB<Eigen::Matrix<T, -1, 3>, 3> tree_;

    // Per-face unit normals (nF x 3), for pseudonormal sign
    Eigen::Matrix<T, -1, 3> FN_;
};