#include "MeshSDFAccel.h"
#include <cmath>

// oneTBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

void MeshSDFAccel::build(const Eigen::Matrix<Scalar, -1, 3>& V_in, const Eigen::Matrix<int, -1, 3>& F_in)
{
    V_ = V_in;
    F_ = F_in;

    // Build AABB once (read-only after this).
    tree_.init(V_, F_);

    // Precompute per-face unit normals.
    FN_.resize(F_.rows(), 3);
    for (int i = 0; i < F_.rows(); ++i) {
        const Eigen::Matrix<Scalar, 1, 3> a = V_.row(F_(i, 0));
        const Eigen::Matrix<Scalar, 1, 3> b = V_.row(F_(i, 1));
        const Eigen::Matrix<Scalar, 1, 3> c = V_.row(F_(i, 2));
        Eigen::Matrix<Scalar, 1, 3> n = (b - a).cross(c - a);
        const Scalar len = static_cast<Scalar>(n.norm());
        FN_.row(i) = (len > Scalar(0)) ? (n / len) : Eigen::Matrix<Scalar, 1, 3>::Zero();
    }
}

Eigen::Matrix<MeshSDFAccel::Scalar, -1, 1> MeshSDFAccel::querySDF(const Eigen::Matrix<Scalar, -1, 3>& P) const
{
    const int N = static_cast<int>(P.rows());
    Eigen::Matrix<Scalar, -1, 1> sdf(N);

    // Parallel over contiguous blocks to reduce scheduling overhead.
    tbb::parallel_for(
        tbb::blocked_range<int>(0, N, /*grainsize*/ 2048),
        [&](const tbb::blocked_range<int>& r) {
            for (int i = r.begin(); i != r.end(); ++i) {
                int fid = -1;
                Eigen::Matrix<Scalar, 1, 3> c; // closest point

                // AABB nearest triangle and squared distance
                // Note: this overload writes fid, c, d2 by reference.
                auto d2 = tree_.squared_distance(V_, F_, P.row(i), fid, c);

                // Unsigned distance
                const auto d = std::sqrt(std::max(d2, decltype(d2)(0)));

                // Pseudonormal sign using the nearest face normal
                int s = 1; // default outside
                if (fid >= 0) {
                    const Eigen::Matrix<Scalar, 1, 3> n = FN_.row(fid);
                    const auto dot = (P.row(i) - c).dot(n);
                    s = (dot >= 0) ? +1 : -1;
                }
                sdf[i] = s * d;
            }
        }
    );

    return sdf;
}
