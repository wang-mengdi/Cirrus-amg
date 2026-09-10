#pragma once
#include <Eigen/Sparse>
#include <memory>
namespace simple {
class AmgPressure {
public:
    AmgPressure();
    ~AmgPressure();
    void compute(const Eigen::SparseMatrix<double>& matrix,double tolerance);
    Eigen::VectorXd solve(const Eigen::VectorXd& rhs) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
