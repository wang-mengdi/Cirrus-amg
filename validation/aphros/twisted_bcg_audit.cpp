// Read-only call to original Aphros BCG on a prescribed nonzero scalar, source,
// and face flux in the curved tube. This is an operator audit, not a flow run.
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include "distr/distrbasic.h"
#include "solver/approx_eb.h"
#include "solver/twisted_diagnostics.h"
using M=MeshCartesian<double,3>;
using EB=Embed<M>;
using U=UEmbed<M>;
using Vect=M::Vect;
void Run(M& m,Vars& var) {
  auto sem=m.GetSem("bcg-audit");
  struct Context {FieldNode<double> phi;std::unique_ptr<EB> eb;} *ctx(sem);
  if(sem("geometry")) {
    if(var.Int["bx"]!=1||var.Int["by"]!=1||var.Int["bz"]!=1)
      throw std::runtime_error("BCG audit requires one full block");
    ctx->phi.Reinit(m,0.);
    for(auto node:m.AllNodes()) {
      const auto x=m.GetNode(node);const double angle=2*M_PI*x[0]/var.Double["twisted_period"];
      const double y=x[1]-var.Double["twisted_center_y"]-var.Double["twisted_amplitude"]*std::sin(angle);
      const double z=x[2]-var.Double["twisted_center_z"]-var.Double["twisted_amplitude"]*std::cos(angle);
      ctx->phi[node]=var.Double["twisted_radius"]-std::sqrt(y*y+z*z);
    }
    ctx->phi.SetHalo(2);ctx->eb.reset(new EB(m,0));
  }
  if(sem.Nested("embed"))ctx->eb->Init(ctx->phi);
  if(sem("audit")) {
    auto& eb=*ctx->eb;TwistedGeometryDump(m,eb);
    FieldCell<double> value(m,0.),source(m,0.);FieldEmbed<double> flux(eb,0.);
    MapEmbed<BCond<double>> bc;
    for(auto c:eb.AllCells()) {
      const auto x=m.GetCenter(c);
      value[c]=.03*std::sin(2*M_PI*x[0]/var.Double["twisted_period"])+x[1]*x[1]+.2*x[2];
      source[c]=.4*std::cos(2*M_PI*x[0]/var.Double["twisted_period"])+x[1]-.1*x[2];
    }
    for(auto f:eb.SuFaces()) {
      const auto x=eb.GetFaceCenter(f);const int d=m.GetDir(f).raw();
      flux[f]=eb.GetArea(f)*(.01*std::sin(2*M_PI*x[0]/var.Double["twisted_period"]+.7*d)+.03*x[1]-.02*x[2]);
    }
    for(auto c:eb.SuCFaces())bc[c]=BCond<double>(BCondType::dirichlet,0,0.);
    value.SetHalo(2);source.SetHalo(2);flux.SetHalo(2);
    const auto gradient=U::Gradient(value,bc,eb);
    const auto result=U::InterpolateBcg(value,bc,flux,source,var.Double["dt0"],eb);
    std::ofstream cells("bcg_cells.csv"),faces("bcg_faces.csv");
    cells<<std::setprecision(17)<<"x,y,z,value,source\n";
    faces<<std::setprecision(17)<<"x,y,z,axis,area,flux,gradient,value\n";
    for(auto c:eb.Cells()) {
      const auto x=m.GetCenter(c);cells<<x[0]<<','<<x[1]<<','<<x[2]<<','<<value[c]<<','<<source[c]<<'\n';
    }
    for(auto f:eb.Faces()) {
      const auto x=eb.GetFaceCenter(f);
      faces<<x[0]<<','<<x[1]<<','<<x[2]<<','<<m.GetDir(f).raw()<<','<<eb.GetArea(f)<<','
           <<flux[f]<<','<<gradient[f]<<','<<result[f]<<'\n';
    }
    cells.flush();faces.flush();if(!cells.good()||!faces.good())throw std::runtime_error("BCG audit output failed");
    std::cout<<"Original BCG operator audit complete"<<std::endl;
  }
}
int main(int argc,const char** argv) {
  try {MpiWrapper mpi(&argc,&argv);return RunMpiBasicFile<M>(mpi,Run,argc>1?argv[1]:"a.conf");}
  catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
