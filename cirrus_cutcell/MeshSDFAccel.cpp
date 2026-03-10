#include "MeshSDFAccel.h"
#include <cmath>

// oneTBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/signed_distance.h>


void AssertRigidTransform(const Eigen::Transform<T, 3, Eigen::Affine>& xform)
{
    const auto& R = xform.linear();
    ASSERT(R.isUnitary(T(1e-6)));
    ASSERT(std::abs(R.determinant() - T(1)) < T(1e-6));
}

void MeshSDFAccel::init(const fs::path& obj_path)
{
	//fmt::print("current working directory: {}\n", fs::current_path().string());
	//fmt::print("MeshSDFAccel::init: Checking if file exists at {}\n", obj_path.string());
	// Check if file exists
    if (!fs::exists(obj_path)) {
        throw std::runtime_error("MeshSDFAccel::init: File does not exist: " + obj_path.string());
	}
	//fmt::print("MeshSDFAccel::init: Reading mesh from {}\n", obj_path.string());

    Eigen::MatrixXd Vd;
    Eigen::MatrixXi Fi;
    if (!igl::readOBJ(obj_path.string(), Vd, Fi)) {
        throw std::runtime_error("MeshSDFAccel::init: Failed to read OBJ file " + obj_path.string());
    }
    // Convert to T
    Eigen::Matrix<T, -1, 3> V = Vd.cast<T>();
    Eigen::Matrix<int, -1, 3> F = Fi;
    build(V, F);
}

void MeshSDFAccel::build(const Eigen::Matrix<T, -1, 3>& V_in, const Eigen::Matrix<int, -1, 3>& F_in)
{
    V_ = V_in;
    F_ = F_in;

    ASSERT(V_.cols() == 3);
    ASSERT(F_.cols() == 3);

    // Build AABB once (read-only after this).
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
}

////self version use tbb but face some sign flipping issues
//std::vector<T> MeshSDFAccel::querySDF(const std::vector<Vec>& points, const Eigen::Transform<T, 3, Eigen::Affine>& xform) const
//{
//    AssertRigidTransform(xform);
//
//    const int N = static_cast<int>(points.size());
//    std::vector<T> sdf(N);
//
//    // Inverse transform: world -> mesh-local
//    const Eigen::Transform<T, 3, Eigen::Affine> invX = xform.inverse();
//    const Eigen::Matrix<T, 4, 4> M = invX.matrix();
//
//    tbb::parallel_for(
//        tbb::blocked_range<int>(0, N, 2048),
//        [&](const tbb::blocked_range<int>& r) {
//            for (int i = r.begin(); i != r.end(); ++i) {
//                // Read point (Vec must support .x()/.y()/.z() or [0],[1],[2])
//                const T px = static_cast<T>(points[i][0]);
//                const T py = static_cast<T>(points[i][1]);
//                const T pz = static_cast<T>(points[i][2]);
//
//                // Homogeneous transform
//                Eigen::Matrix<T, 4, 1> pw; pw << px, py, pz, T(1);
//                Eigen::Matrix<T, 4, 1> pl4 = M * pw;
//                Eigen::Matrix<T, 1, 3> p = pl4.template head<3>().transpose();
//
//                // Closest point in mesh-local
//                int fid = -1;
//                Eigen::Matrix<T, 1, 3> c;
//                const T d2 = tree_.squared_distance(V_, F_, p, fid, c);
//                const T d = std::sqrt(std::max<T>(d2, T(0)));
//
//                // Pseudonormal sign
//                int s = 1;
//                if (fid >= 0) {
//                    const Eigen::Matrix<T, 1, 3> n = FN_.row(fid);
//                    const T dot = (p - c).dot(n);
//                    s = (dot >= T(0)) ? +1 : -1;
//                }
//                sdf[i] = s * d;
//            }
//        });
//
//    return sdf;
//}

std::vector<T> MeshSDFAccel::querySDF(const std::vector<Vec>& points,
    const Eigen::Transform<T, 3, Eigen::Affine>& xform) const
{
    AssertRigidTransform(xform);
    const int N = static_cast<int>(points.size());
    std::vector<T> sdf(N);
    if (N == 0) return sdf;

    const auto invX = xform.inverse();
    Eigen::Matrix<T, Eigen::Dynamic, 3> P(N, 3);
    for (int i = 0; i < N; ++i) {
        Eigen::Matrix<T, 3, 1> pw(points[i][0], points[i][1], points[i][2]);
        Eigen::Matrix<T, 3, 1> pl = invX * pw;
        P.row(i) = pl.transpose();
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> S;
    Eigen::VectorXi I;
    Eigen::Matrix<T, Eigen::Dynamic, 3> C, Nrm;

    igl::signed_distance(P, V_, F_, igl::SIGNED_DISTANCE_TYPE_WINDING_NUMBER, S, I, C, Nrm);

    for (int i = 0; i < N; ++i) sdf[i] = S(i);
    return sdf;
}