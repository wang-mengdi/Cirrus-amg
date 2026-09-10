// Optional single-block linear backends for the validation worktree.
// Consume Aphros' assembled coefficients and check the ORIGINAL equations.
#pragma once
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <array>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <fstream>
#include "twisted_pressure_snapshot.h"
#include "twisted_pressure_geometry.h"
#ifdef APHROS_TWISTED_HAVE_AMGCL
#define AMGCL_NO_BOOST
#include <amgcl/adapter/eigen.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/make_solver.hpp>
#endif

template<class M>
class TwistedSerialDirect {
  using Sparse=Eigen::SparseMatrix<double>;
  const M* mesh_=nullptr;
  std::vector<IdxCell> cells_;
  std::vector<std::array<int,2*M::dim>> neighbors_;
#ifdef APHROS_TWISTED_HAVE_AMGCL
  using Backend=amgcl::backend::builtin<double>;
  using Amg=amgcl::make_solver<amgcl::amg<Backend,
      amgcl::coarsening::smoothed_aggregation,amgcl::relaxation::spai0>,
      amgcl::solver::cg<Backend>>;
  using GeneralAmg=amgcl::make_solver<amgcl::amg<Backend,
      amgcl::coarsening::smoothed_aggregation,amgcl::relaxation::spai0>,
      amgcl::solver::bicgstab<Backend>>;
#endif
  struct Factorization {
    Sparse matrix;
    Eigen::SimplicialLDLT<Sparse> ldlt;
    Eigen::SparseLU<Sparse> lu;
    bool symmetric=true,use_amg=false;
#ifdef APHROS_TWISTED_HAVE_AMGCL
    std::unique_ptr<Amg> amg;
    std::unique_ptr<GeneralAmg> general_amg;
#endif
  };
  // Least recently used first. A hit requires identical compressed matrix bytes;
  // no coefficient, RHS, initial guess, tolerance, or solve ordering is changed.
  std::vector<std::unique_ptr<Factorization>> factors_;
  size_t factor_hits_=0,factor_misses_=0;
  std::ofstream compatibility_log_;
  size_t compatibility_calls_=0;
  void InitTopology(M& m) {
    if(mesh_==&m)return;
    mesh_=&m;cells_.clear();neighbors_.clear();factors_.clear();
    factor_hits_=factor_misses_=0;
    const auto shape=m.GetGlobalSize();
    const auto h=m.GetCellSize();
    auto key=[&](IdxCell c) {
      auto x=m.GetCenter(c);int index=0,stride=1;
      for(size_t d=0;d<M::dim;++d) {
        int q=int(std::llround(x[d]/h[d]-.5));q=(q%shape[d]+shape[d])%shape[d];
        index+=stride*q;stride*=shape[d];
      }
      return index;
    };
    std::vector<int> lookup(shape.prod(),-1);
    for(auto c:m.Cells()) {lookup.at(key(c))=int(cells_.size());cells_.push_back(c);}
    if(cells_.size()!=size_t(shape.prod()))throw std::runtime_error("Twisted backend requires one full-domain block");
    neighbors_.resize(cells_.size());
    for(int i=0;i<int(cells_.size());++i)for(auto q:m.Nci(cells_[i])) {
      int j=lookup.at(key(m.GetCell(cells_[i],q)));
      if(j<0)throw std::runtime_error("Twisted backend missing neighbor");
      neighbors_[i][q.raw()]=j;
    }
  }
 public:
  typename linear::Solver<M>::Info Solve(const FieldCell<typename M::Expr>& system,
      FieldCell<typename M::Scal>& out, M& m, double tolerance) {
    InitTopology(m);
    const int full=int(cells_.size());const double volume=m.GetCellSize().prod();
    const bool pressure=system.GetName()=="pressure";
    if(pressure)if(const char* value=std::getenv("APHROS_TWISTED_PRESSURE_TOLERANCE")) {
      // Proj shares one linear-solver instance between pressure and velocity.
      // Tighten only the pressure solve without changing either assembled
      // operator, the velocity tolerance, or the projection iteration order.
      char* end=nullptr;const double target=std::strtod(value,&end);
      if(end==value || *end || !std::isfinite(target) || target<=0 || target>tolerance)
        throw std::runtime_error("Pressure tolerance override must be positive and no looser than the original tolerance");
      tolerance=target;
    }
    // Isolated scalar equations are solved exactly below; any incident row stays.
    std::vector<unsigned char> incident(full,0);
    for(int i=0;i<full;++i)for(int q=0;q<2*M::dim;++q)
      if(system[cells_[i]][1+q]!=0)incident[i]=incident[neighbors_[i][q]]=1;
    std::vector<int> index(full,-1),active;
    for(int i=0;i<full;++i)if(incident[i]) {index[i]=int(active.size());active.push_back(i);}
    const int n=int(active.size());
    if(!n)throw std::runtime_error("Twisted backend has no coupled fluid equations");
    const bool volume_compatibility=pressure && std::getenv("APHROS_TWISTED_VOLUME_COMPATIBILITY");
    Eigen::VectorXd physical_volumes;double fluid_volume=0.;
    if(volume_compatibility) {
      const auto& physical=TwistedPressureVolumes<M>();
      if(physical.mesh!=&m)throw std::runtime_error("Pressure compatibility requires this mesh's actual Embed volumes");
      physical_volumes.resize(n);double compensation=0.;
      for(int r=0;r<n;++r) {
        const double v=physical.volume[cells_[active[r]]];
        if(!(v>0) || !std::isfinite(v))throw std::runtime_error("Invalid pressure compatibility volume");
        physical_volumes[r]=v;
        const double y=v-compensation,t=fluid_volume+y;compensation=(t-fluid_volume)-y;fluid_volume=t;
      }
      if(!(fluid_volume>0) || !std::isfinite(fluid_volume))throw std::runtime_error("Invalid total pressure fluid volume");
      if(!compatibility_log_.is_open()) {
        compatibility_log_.open("pressure_compatibility.csv");
        compatibility_log_<<std::setprecision(17)<<"call,rhs_sum,fluid_volume,cell_count,regular_cell_volume,minimum_cell_volume,compatibility_divergence,effective_tolerance\n";
      }
    }
    std::vector<Eigen::Triplet<double>> entries;entries.reserve(n*(2*M::dim+1));
    Eigen::VectorXd rhs(n);int gauge=-1;double gauge_diagonal=-1;
    for(int r=0;r<n;++r) {
      const int i=active[r];const auto& e=system[cells_[i]];
      entries.emplace_back(r,r,e[0]);rhs[r]=-e.back();
      if(pressure && e[0]>gauge_diagonal) {gauge=r;gauge_diagonal=e[0];}
      for(int q=0;q<2*M::dim;++q)if(e[1+q]!=0) {
        int j=index[neighbors_[i][q]];
        if(j<0)throw std::runtime_error("Coupled row incorrectly excluded");
        entries.emplace_back(r,j,e[1+q]);
      }
    }
    Sparse original(n,n);original.setFromTriplets(entries.begin(),entries.end());
    if(pressure) {
      std::vector<Eigen::Triplet<double>> pinned;
      for(const auto& e:entries)if(e.row()!=gauge && e.col()!=gauge)pinned.push_back(e);
      pinned.emplace_back(gauge,gauge,1.);entries.swap(pinned);
    }
    Sparse matrix(n,n);matrix.setFromTriplets(entries.begin(),entries.end());matrix.makeCompressed();
    int capacity=1;
    if(const char* value=std::getenv("APHROS_TWISTED_FACTOR_CACHE")) {
      char* end=nullptr;const long parsed=std::strtol(value,&end,10);
      if(end==value || *end || parsed<1 || parsed>8)
        throw std::runtime_error("APHROS_TWISTED_FACTOR_CACHE must be between 1 and 8");
      capacity=int(parsed);
    }
    const bool requested_amg=std::getenv("APHROS_TWISTED_AMG")!=nullptr;
    auto identical=[&](const Factorization& factor) {
      if(factor.use_amg!=requested_amg)return false;
      const auto& cached_=factor.matrix;
      if(cached_.rows()!=n || cached_.nonZeros()!=matrix.nonZeros())return false;
      return std::memcmp(cached_.valuePtr(),matrix.valuePtr(),matrix.nonZeros()*sizeof(double))==0 &&
        std::memcmp(cached_.innerIndexPtr(),matrix.innerIndexPtr(),matrix.nonZeros()*sizeof(int))==0 &&
        std::memcmp(cached_.outerIndexPtr(),matrix.outerIndexPtr(),(n+1)*sizeof(int))==0;
    };
    size_t found=0;
    while(found<factors_.size() && !identical(*factors_[found]))++found;
    const bool cache_hit=found<factors_.size();
    if(cache_hit) {
      auto entry=std::move(factors_[found]);factors_.erase(factors_.begin()+found);
      factors_.push_back(std::move(entry));++factor_hits_;
    } else {
      while(factors_.size()>=size_t(capacity))factors_.erase(factors_.begin());
      factors_.emplace_back(new Factorization);++factor_misses_;
    }
    auto& factor=*factors_.back();
    auto& symmetric_=factor.symmetric;auto& use_amg_=factor.use_amg;
    auto& ldlt_=factor.ldlt;auto& lu_=factor.lu;
#ifdef APHROS_TWISTED_HAVE_AMGCL
    auto& amg_=factor.amg;auto& general_amg_=factor.general_amg;
#endif
    if(!cache_hit) {
      factor.matrix=matrix;
      Sparse transpose=matrix.transpose(),asymmetry=matrix-transpose;
      symmetric_=asymmetry.norm()<1e-13*matrix.norm();
      use_amg_=requested_amg;
      if(use_amg_) {
#ifdef APHROS_TWISTED_HAVE_AMGCL
        Eigen::SparseMatrix<double,Eigen::RowMajor> rowmajor=matrix;
        if(symmetric_) {
          general_amg_.reset();
          typename Amg::params p;p.solver.tol=1e-14;p.solver.maxiter=500;
          amg_.reset(new Amg(rowmajor,p));
        } else {
          amg_.reset();
          typename GeneralAmg::params p;p.solver.tol=1e-14;p.solver.maxiter=500;
          general_amg_.reset(new GeneralAmg(rowmajor,p));
        }
#else
        throw std::runtime_error("AMG requested without APHROS_TWISTED_HAVE_AMGCL build option");
#endif
      } else if(symmetric_) {
        ldlt_.compute(matrix);if(ldlt_.info()!=Eigen::Success)throw std::runtime_error("Reference LDLT failed");
      } else {
        lu_.compute(matrix);if(lu_.info()!=Eigen::Success)throw std::runtime_error("Reference LU failed");
      }
    }
    if(std::getenv("APHROS_TWISTED_FACTOR_TRACE")) {
      std::cerr<<"factor-cache name="<<system.GetName()<<" hit="<<cache_hit
        <<" entries="<<factors_.size()<<" hits="<<factor_hits_<<" misses="<<factor_misses_
        <<" rows="<<n<<" nnz="<<matrix.nonZeros()<<'\n';
    }
    int linear_iterations=0;
    auto solve=[&](Eigen::VectorXd b) {
      if(pressure) {
        // A periodic pressure system has a constant nullspace. Floating-point
        // face summation leaves a tiny incompatible mean; concentrating it in
        // the pinned row amplifies its residual by the fluid-cell count.
        // Remove only a mean already below the requested per-volume tolerance.
        // Every final residual is still checked against the UNMODIFIED RHS.
        double sum=0,compensation=0;
        for(int i=0;i<n;++i) {double y=b[i]-compensation,t=sum+y;compensation=(t-sum)-y;sum=t;}
        const double mean=sum/n;
        if(std::abs(mean)/volume>tolerance)
          throw std::runtime_error("Pressure RHS has a significant incompatible mean");
        if(volume_compatibility) {
          // Distribute the roundoff compatibility defect by actual fluid
          // volume. A constant integral correction would amplify divergence
          // in tiny cut cells. Bound BOTH the old per-regular-volume mean and
          // the physical divergence; retain all original-row residual checks.
          const double density=sum/fluid_volume;
          if(!std::isfinite(density) || std::abs(density)>tolerance)
            throw std::runtime_error("Pressure RHS has a significant incompatible fluid-volume source");
          b-=density*physical_volumes;
          compatibility_log_<<++compatibility_calls_<<','<<sum<<','<<fluid_volume<<','<<n<<','<<volume<<','
            <<physical_volumes.minCoeff()<<','<<density<<','<<tolerance<<'\n';
          compatibility_log_.flush();
          if(!compatibility_log_.good())throw std::runtime_error("Pressure compatibility diagnostic write failed");
        } else b.array()-=mean;
        b[gauge]=0.;
      }
      Eigen::VectorXd x=Eigen::VectorXd::Zero(n);
      if(b.squaredNorm()==0)return x;
      if(use_amg_) {
#ifdef APHROS_TWISTED_HAVE_AMGCL
        // AMGCL treats an absolute RHS norm below machine epsilon as zero.
        // Pressure corrections legitimately get much smaller in SI units, so
        // normalize the RHS and undo this exact scalar transformation afterward.
        const double scale=b.cwiseAbs().maxCoeff();
        std::vector<double> source(b.data(),b.data()+n),answer(n,0.);
        for(auto& v:source)v/=scale;
        auto info=symmetric_?(*amg_)(source,answer):(*general_amg_)(source,answer);
        linear_iterations+=int(std::get<0>(info));
        x=Eigen::Map<Eigen::VectorXd>(answer.data(),n)*scale;
#endif
      } else {
        if(symmetric_)x=ldlt_.solve(b);else x=lu_.solve(b);
        ++linear_iterations;
      }
      return x;
    };
    Eigen::VectorXd x=solve(rhs);int refinements=0;
    double residual=0;
    for(;refinements<6;++refinements) {
      const Eigen::VectorXd r=rhs-original*x;
      residual=r.cwiseAbs().maxCoeff()/volume;
      if(residual<=tolerance)break;
      x+=solve(r);
    }
    if(!x.allFinite())throw std::runtime_error("Nonfinite reference linear solution");
    out.Reinit(m,0.);
    for(int i=0;i<full;++i) {
      const auto& e=system[cells_[i]];
      if(index[i]>=0)out[cells_[i]]=x[index[i]];
      else if(e[0]!=0)out[cells_[i]]=-e.back()/e[0];
      else if(e.back()!=0)throw std::runtime_error("Inconsistent isolated zero row");
    }
    // Evaluate ALL original rows, including the unpinned gauge and exterior.
    residual=0;
    for(int i=0;i<full;++i) {
      const auto& e=system[cells_[i]];double r=e.back()+e[0]*out[cells_[i]];
      for(int q=0;q<2*M::dim;++q)r+=e[1+q]*out[cells_[neighbors_[i][q]]];
      residual=std::max(residual,std::abs(r)/volume);
    }
    if(!std::isfinite(residual) || residual>10*tolerance) {
      std::ostringstream message;message<<std::setprecision(17)<<"Reference "<<system.GetName()
          <<" solve failed original-matrix residual check: "<<residual<<", tolerance="<<tolerance;
      throw std::runtime_error(message.str());
    }
    if(pressure && std::getenv("APHROS_TWISTED_CAPTURE_PRESSURE")) {
      auto& snapshot=TwistedLastPressure<M>();
      snapshot.mesh=&m;snapshot.equations=system;snapshot.returned_pressure=out;
      ++snapshot.sequence;snapshot.gauge_raw=cells_[active[gauge]].raw();
      snapshot.tolerance=tolerance;snapshot.reported_residual=residual;
      snapshot.refinements=refinements;
    }
    if(m.flags.linreport) {
      std::ostringstream report;report<<std::scientific<<std::setprecision(17)<<"linear(twisted-"
          <<(use_amg_?"amg":"direct")<<") '"<<system.GetName()<<"': res="<<residual
          <<" iter="<<linear_iterations<<" active="<<n<<'\n';
      std::cerr<<report.str();
    }
    return {residual,linear_iterations};
  }
};
