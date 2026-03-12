#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <igl/AABB.h>
#include <igl/readOBJ.h>
#include <igl/fast_winding_number.h>

#include "PoissonTile.h"

void AssertRigidTransform(const Eigen::Transform<T, 3, Eigen::Affine>& xform);

class SDFAccelBase {
public:
    virtual ~SDFAccelBase() = default;

    // Negative inside, positive outside.
    // xform is the affine transform from local coordinates to world coordinates.
    virtual std::vector<T> querySDF(
        const std::vector<Vec>& points,
        const Eigen::Transform<T, 3, Eigen::Affine>& xform) const = 0;
};

class SphereSDFAccel : public SDFAccelBase {
public:
    // default: unit sphere centered at origin
    SphereSDFAccel()
        : center_(Vec(0, 0, 0)), radius_(T(0.5)) {
    }

    explicit SphereSDFAccel(T radius)
        : center_(Vec(0, 0, 0)), radius_(radius) {
    }

    SphereSDFAccel(const Vec& center, T radius)
        : center_(center), radius_(radius) {
    }

    void setCenter(const Vec& c) { center_ = c; }
    void setRadius(T radius) { radius_ = radius; }

    Vec center() const { return center_; }
    T radius() const { return radius_; }

    // Negative inside, positive outside.
    // xform maps sphere-local coordinates to world coordinates.
    std::vector<T> querySDF(
        const std::vector<Vec>& points,
        const Eigen::Transform<T, 3, Eigen::Affine>& xform) const override;

private:
    Vec center_;  // sphere center in local coordinates
    T radius_;
};

class MeshSDFAccel: public SDFAccelBase {
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
        const Eigen::Transform<T, 3, Eigen::Affine>& xform) const override;

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