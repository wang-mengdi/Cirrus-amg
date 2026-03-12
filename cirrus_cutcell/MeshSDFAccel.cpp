#include "MeshSDFAccel.h"
#include <cmath>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <igl/per_face_normals.h>
#include <igl/fast_winding_number.h>
#include <igl/signed_distance.h>

void AssertRigidTransform(const Eigen::Transform<T, 3, Eigen::Affine>& xform)
{
    const auto& R = xform.linear();
    ASSERT(R.isUnitary(T(1e-6)));
    ASSERT(std::abs(R.determinant() - T(1)) < T(1e-6));
}

void MeshSDFAccel::init(const fs::path& obj_path)
{
    ASSERT(fs::exists(obj_path));

    Eigen::MatrixXd Vd;
    Eigen::MatrixXi Fi;
    ASSERT(igl::readOBJ(obj_path.string(), Vd, Fi));

    Eigen::Matrix<T, -1, 3> V = Vd.cast<T>();
    Eigen::Matrix<int, -1, 3> F = Fi;
    build(V, F);
}

void MeshSDFAccel::build(const Eigen::Matrix<T, -1, 3>& V_in,
    const Eigen::Matrix<int, -1, 3>& F_in)
{
    V_ = V_in;
    F_ = F_in;

    ASSERT(V_.cols() == 3);
    ASSERT(F_.cols() == 3);
    ASSERT(V_.rows() > 0);
    ASSERT(F_.rows() > 0);

    // Build AABB once. Reused by all future distance queries.
    tree_.init(V_, F_);

    // Per-face unit normals.
    igl::per_face_normals(V_, F_, FN_);
    ASSERT(FN_.rows() == F_.rows());
    ASSERT(FN_.cols() == 3);

    // Per-face areas and total area.
    face_area_.resize(F_.rows(), 1);
    total_area_ = T(0);

    for (int i = 0; i < F_.rows(); ++i) {
        const Eigen::Matrix<T, 1, 3> a = V_.row(F_(i, 0));
        const Eigen::Matrix<T, 1, 3> b = V_.row(F_(i, 1));
        const Eigen::Matrix<T, 1, 3> c = V_.row(F_(i, 2));

        const T area = T(0.5) * ((b - a).cross(c - a)).norm();
        face_area_(i) = area;
        total_area_ += area;
    }

    // Build fast winding number BVH once.
    // libigl's precompute path uses FastWindingNumberBVH as the reusable hierarchy.
    igl::fast_winding_number(V_.template cast<float>().eval(), F_, 2, fwn_bvh_);
    has_fwn_ = true;
}

std::vector<T> MeshSDFAccel::querySDF(
    const std::vector<Vec>& points,
    const Eigen::Transform<T, 3, Eigen::Affine>& xform) const
{
    AssertRigidTransform(xform);
    ASSERT(has_fwn_);

    const int N = static_cast<int>(points.size());
    std::vector<T> sdf(N);
    if (N == 0) return sdf;

    const auto invX = xform.inverse();
    const auto R = invX.linear();
    const auto t = invX.translation();

    tbb::parallel_for(
        tbb::blocked_range<int>(0, N, 4096),
        [&](const tbb::blocked_range<int>& r)
        {
            for (int i = r.begin(); i != r.end(); ++i) {
                const T x = static_cast<T>(points[i][0]);
                const T y = static_cast<T>(points[i][1]);
                const T z = static_cast<T>(points[i][2]);

                Eigen::Matrix<T, 1, 3> q;
                q(0) = R(0, 0) * x + R(0, 1) * y + R(0, 2) * z + t(0);
                q(1) = R(1, 0) * x + R(1, 1) * y + R(1, 2) * z + t(1);
                q(2) = R(2, 0) * x + R(2, 1) * y + R(2, 2) * z + t(2);

                sdf[i] = igl::signed_distance_fast_winding_number(
                    q, V_, F_, tree_, fwn_bvh_);
            }
        });

    return sdf;
}