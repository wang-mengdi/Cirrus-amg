#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <igl/AABB.h>
#include <igl/readOBJ.h>
#include <igl/fast_winding_number.h>

#include "PoissonTile.h"

void AssertRigidTransform(const Eigen::Transform<T, 3, Eigen::Affine>& xform);

class MeshSDFAccel {
public:
    MeshSDFAccel() {}
    MeshSDFAccel(const fs::path& obj_path) { init(obj_path); }

    void init(const fs::path& obj_path);

    // Build once: copy V/F, build AABB, precompute normals and fast winding BVH.
    void build(const Eigen::Matrix<T, -1, 3>& V_in, const Eigen::Matrix<int, -1, 3>& F_in);

    // Negative inside, positive outside.
    // xform is the affine transform from mesh-local to world coordinates.
    std::vector<T> querySDF(
        const std::vector<Vec>& points,
        const Eigen::Transform<T, 3, Eigen::Affine>& xform) const;

public:
    // Stored mesh
    Eigen::Matrix<T, -1, 3> V_;   // (nV x 3)
    Eigen::Matrix<int, -1, 3> F_; // (nF x 3)

    // Prebuilt AABB for closest-point distance
    igl::AABB<Eigen::Matrix<T, -1, 3>, 3> tree_;

    // Per-face unit normals
    Eigen::Matrix<T, -1, 3> FN_;

    Eigen::Matrix<T, -1, 1> face_area_; // (nF x 1)
    T total_area_ = T(0);

    // Prebuilt fast winding number hierarchy
    igl::FastWindingNumberBVH fwn_bvh_;
    bool has_fwn_ = false;
};