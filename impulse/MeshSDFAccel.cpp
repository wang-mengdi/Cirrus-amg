#include "MeshSDFAccel.h"
#include <cmath>

// oneTBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

void MeshSDFAccel::init(const fs::path& obj_path)
{
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

    // Build AABB once (read-only after this).
    tree_.init(V_, F_);

    // Precompute per-face unit normals.
    FN_.resize(F_.rows(), 3);
    for (int i = 0; i < F_.rows(); ++i) {
        const Eigen::Matrix<T, 1, 3> a = V_.row(F_(i, 0));
        const Eigen::Matrix<T, 1, 3> b = V_.row(F_(i, 1));
        const Eigen::Matrix<T, 1, 3> c = V_.row(F_(i, 2));
        FN_.row(i) = (b - a).cross(c - a).normalized();
    }
}

std::vector<T> MeshSDFAccel::querySDF(const std::vector<Vec>& points, const Eigen::Transform<T, 3, Eigen::Affine>& xform) const
{
    const int N = static_cast<int>(points.size());
    std::vector<T> sdf(N);

    // Inverse transform: world -> mesh-local
    const Eigen::Transform<T, 3, Eigen::Affine> invX = xform.inverse();
    const Eigen::Matrix<T, 4, 4> M = invX.matrix();

    tbb::parallel_for(
        tbb::blocked_range<int>(0, N, 2048),
        [&](const tbb::blocked_range<int>& r) {
            for (int i = r.begin(); i != r.end(); ++i) {
                // Read point (Vec must support .x()/.y()/.z() or [0],[1],[2])
                const T px = static_cast<T>(points[i][0]);
                const T py = static_cast<T>(points[i][1]);
                const T pz = static_cast<T>(points[i][2]);

                // Homogeneous transform
                Eigen::Matrix<T, 4, 1> pw; pw << px, py, pz, T(1);
                Eigen::Matrix<T, 4, 1> pl4 = M * pw;
                Eigen::Matrix<T, 1, 3> p = pl4.template head<3>().transpose();

                // Closest point in mesh-local
                int fid = -1;
                Eigen::Matrix<T, 1, 3> c;
                const T d2 = tree_.squared_distance(V_, F_, p, fid, c);
                const T d = std::sqrt(std::max<T>(d2, T(0)));

                // Pseudonormal sign
                int s = 1;
                if (fid >= 0) {
                    const Eigen::Matrix<T, 1, 3> n = FN_.row(fid);
                    const T dot = (p - c).dot(n);
                    s = (dot >= T(0)) ? +1 : -1;
                }
                sdf[i] = s * d;
            }
        });

    return sdf;
}