// Read-only audit of the upstream embedded viscous assembly. This is an affine
// scalar patch in a planar domain, not a replacement for the curved-flow case.
// It calls the original GradientImplicit and RedistributeConstTerms, recording
// their residual on u=y-wall, whose exact Laplacian and source are zero.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include "distr/distrbasic.h"
#include "solver/approx_eb.h"

using M = MeshCartesian<double, 3>;
using EB = Embed<M>;
using U = UEmbed<M>;
using Scal = M::Scal;
using Vect = M::Vect;

void Run(M& m, Vars& var) {
  auto sem=m.GetSem("affine-viscous-audit");
  struct Context {
    FieldNode<Scal> levelset;
    FieldCell<Scal> scalar, before;
    FieldCell<M::Expr> system;
    MapEmbed<BCond<Scal>> bc;
    std::unique_ptr<EB> geometry;
    double wall=0, fraction=0, gradient_error=0;
  } *ctx(sem);
  auto& t=*ctx;
  if(sem("geometry")) {
    if(var.Int["bx"]!=1 || var.Int["by"]!=1 || var.Int["bz"]!=1)
      throw std::runtime_error("Affine audit requires one block");
    m.flags.is_periodic={true,false,true};
    t.fraction=var.Double["audit_offset_fraction"];
    if(!(t.fraction>0 && t.fraction<1))throw std::runtime_error("Expected subcell offset in (0,1)");
    t.wall=(m.GetGlobalSize()[1]/3+t.fraction)*m.GetCellSize()[0];
    t.levelset.Reinit(m,0.);
    for(auto node:m.AllNodes())t.levelset[node]=m.GetNode(node)[1]-t.wall;
    t.levelset.SetHalo(2);
    t.geometry.reset(new EB(m,0));
  }
  if(sem.Nested("embed"))t.geometry->Init(t.levelset);
  if(sem("assemble")) {
    auto& eb=*t.geometry;
    t.scalar.Reinit(m,0.);
    for(auto c:m.AllCells())t.scalar[c]=m.GetCenter(c)[1]-t.wall;
    t.scalar.SetHalo(2);
    for(auto c:eb.CFaces())t.bc[c]=BCond<Scal>(BCondType::dirichlet,0,0.);
    const auto size=m.GetGlobalSize();
    for(auto f:eb.Faces()) {
      const auto key=m.GetIndexFaces().GetMIdxDir(f);
      const auto axis=key.second.raw();
      if(axis==1 && (key.first[axis]==0 || key.first[axis]==size[axis]))
        t.bc[f]=BCond<Scal>(BCondType::dirichlet,key.first[axis]==0?1:0,m.GetCenter(f)[1]-t.wall);
    }
    const auto gradient=U::GradientImplicit(t.scalar,t.bc,eb);
    t.system.Reinit(m,M::Expr::GetUnit(0));
    t.before.Reinit(m,0.);
    eb.LoopFaces([&](auto cf) {
      const double exact=eb.GetNormal(cf)[1];
      t.gradient_error=std::max(t.gradient_error,std::abs(U::Eval(gradient[cf],cf,t.scalar,eb)-exact));
    });
    for(auto c:eb.Cells()) {
      M::Expr sum(0);
      eb.LoopNci(c,[&](auto q) {
        const auto cf=eb.GetFace(c,q);
        const auto flux=gradient[cf]*(-eb.GetArea(cf)*eb.GetOutwardFactor(c,q));
        eb.AppendExpr(sum,flux,q);
      });
      t.system[c]=sum;
      t.before[c]=U::Eval(sum,c,t.scalar,m);
    }
    m.Comm(&t.before);
  }
  if(sem.Nested("redistribute"))U::RedistributeConstTerms(t.system,*t.geometry,m);
  if(sem("report")) {
    const auto& eb=*t.geometry;
    const double h=m.GetCellSize()[0];
    double before_max=0,after_max=0,before_sum=0,after_sum=0,after_square=0;
    // ConvDiffScalExp redistributes the complete flux residual. Evaluate that
    // original helper on the same affine field as a separate control.
    const auto whole=U::RedistributeCutCells(t.before,eb);
    double whole_max=0;
    size_t cells=0,cut=0;
    std::ofstream rows("affine_cells.csv");
    rows<<std::setprecision(17)<<"x,y,z,volume,cut,u,residual_before,residual_after,residual_whole_redistribution\n";
    for(auto c:eb.Cells()) {
      const auto x=m.GetCenter(c);
      const double before=t.before[c],after=U::Eval(t.system[c],c,t.scalar,m);
      before_max=std::max(before_max,std::abs(before));after_max=std::max(after_max,std::abs(after));
      before_sum+=before;after_sum+=after;after_square+=after*after;
      whole_max=std::max(whole_max,std::abs(whole[c]));
      ++cells;cut+=eb.IsCut(c);
      rows<<x[0]<<','<<x[1]<<','<<x[2]<<','<<eb.GetVolume(c)<<','<<eb.IsCut(c)<<','
          <<t.scalar[c]<<','<<before<<','<<after<<','<<whole[c]<<'\n';
    }
    if(!cut || t.gradient_error>1e-10 || before_max/(h*h)>1e-10)
      throw std::runtime_error("Affine audit failed its exact-gradient/pre-redistribution control");
    std::ofstream out("affine_audit.json");
    out<<std::setprecision(17)<<std::boolalpha
       <<"{\n  \"audit_completed\": true,\n  \"scope\": \"Original Aphros embedded viscous expressions on an exact affine scalar; not a CFD solution or curved-wall acceptance\",\n"
       <<"  \"finest_h\": "<<h<<",\n  \"wall_y\": "<<t.wall<<",\n  \"offset_fraction\": "<<t.fraction
       <<",\n  \"cells\": "<<cells<<",\n  \"cut_cells\": "<<cut
       <<",\n  \"face_gradient_absolute_max_error\": "<<t.gradient_error
       <<",\n  \"residual_before_max_per_h2\": "<<before_max/(h*h)
       <<",\n  \"residual_after_max_per_h2\": "<<after_max/(h*h)
       <<",\n  \"residual_after_rms_per_h2\": "<<std::sqrt(after_square/cells)/(h*h)
       <<",\n  \"whole_residual_redistribution_max_per_h2\": "<<whole_max/(h*h)
       <<",\n  \"residual_before_sum\": "<<before_sum<<",\n  \"residual_after_sum\": "<<after_sum
       <<",\n  \"affine_consistent_before\": "<<(before_max/(h*h)<1e-10)
       <<",\n  \"affine_consistent_after\": "<<(after_max/(h*h)<1e-10)
       <<",\n  \"affine_consistent_whole_redistribution\": "<<(whole_max/(h*h)<1e-10)<<"\n}\n";
    std::cout<<"Affine audit completed; before="<<before_max/(h*h)<<" after="<<after_max/(h*h)<<std::endl;
  }
}

int main(int argc,const char** argv) {
  try {
    MpiWrapper mpi(&argc,&argv);
    return RunMpiBasicFile<M>(mpi,Run,argc>1?argv[1]:"a.conf");
  } catch(const std::exception& e) {
    std::cerr<<"Affine audit error: "<<e.what()<<std::endl;
    return 1;
  }
}
