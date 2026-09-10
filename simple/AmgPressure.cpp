#include "AmgPressure.h"
#include <stdexcept>
#ifdef SIMPLE_HAVE_AMGCL
#define AMGCL_NO_BOOST
#include <amgcl/adapter/eigen.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/make_solver.hpp>
#endif
namespace simple {
struct AmgPressure::Impl {
#ifdef SIMPLE_HAVE_AMGCL
    using Backend=amgcl::backend::builtin<double>;
    using Solver=amgcl::make_solver<amgcl::amg<Backend,
        amgcl::coarsening::smoothed_aggregation,amgcl::relaxation::spai0>,amgcl::solver::cg<Backend>>;
    std::unique_ptr<Solver> solver;
#endif
    Eigen::SparseMatrix<double> matrix;
    double tolerance=0;
};
AmgPressure::AmgPressure():impl_(new Impl) {}
AmgPressure::~AmgPressure()=default;
void AmgPressure::compute(const Eigen::SparseMatrix<double>& matrix,double tolerance) {
#ifdef SIMPLE_HAVE_AMGCL
    impl_->matrix=matrix;impl_->tolerance=tolerance;
    Eigen::SparseMatrix<double> transpose=matrix.transpose(),asymmetry=matrix-transpose;
    if(asymmetry.norm()>1e-13*matrix.norm())throw std::runtime_error("AMG-CG pressure matrix must be symmetric");
    Impl::Solver::params p;p.solver.tol=tolerance;p.solver.maxiter=1000;
    Eigen::SparseMatrix<double,Eigen::RowMajor> rowmajor=matrix;
    impl_->solver=std::make_unique<Impl::Solver>(rowmajor,p);
#else
    throw std::runtime_error("pressure_solver=amg requires xmake option --simple_amgcl_root");
#endif
}
Eigen::VectorXd AmgPressure::solve(const Eigen::VectorXd& rhs) const {
#ifdef SIMPLE_HAVE_AMGCL
    if(!impl_->solver)throw std::runtime_error("AMG pressure setup missing");
    auto solve=[&](const Eigen::VectorXd& b) {
        const double scale=b.cwiseAbs().maxCoeff();
        Eigen::VectorXd x=Eigen::VectorXd::Zero(b.size());
        if(scale==0)return x;
        std::vector<double> source(b.data(),b.data()+b.size()),answer(b.size(),0.);
        for(auto& v:source)v/=scale;
        (*impl_->solver)(source,answer);
        x=Eigen::Map<Eigen::VectorXd>(answer.data(),answer.size())*scale;
        return x;
    };
    Eigen::VectorXd x=solve(rhs);
    for(int pass=0;pass<3;++pass) {
        Eigen::VectorXd residual=rhs-impl_->matrix*x;
        if(residual.norm()<=impl_->tolerance*rhs.norm())break;
        x+=solve(residual);
    }
    if(!x.allFinite() || (rhs-impl_->matrix*x).norm()>10*impl_->tolerance*rhs.norm())
        throw std::runtime_error("AMG pressure original-matrix residual check failed");
    return x;
#else
    throw std::runtime_error("AMG pressure support was not built");
#endif
}
}
