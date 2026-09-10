// Read-only material interpolation on the original curved Embed geometry.
// Calls the same upstream functions and wall conditions as Proj; no flow solve.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include "distr/distrbasic.h"
#include "solver/approx_eb.h"
using M=MeshCartesian<double,3>;
using EB=Embed<M>;
using U=UEmbed<M>;
void Run(M& m,Vars& var) {
  auto sem=m.GetSem("material-audit");
  struct Context {FieldNode<double> phi;std::unique_ptr<EB> eb;} *ctx(sem);
  if(sem("geometry")) {
    if(var.Int["bx"]!=1||var.Int["by"]!=1||var.Int["bz"]!=1)
      throw std::runtime_error("Material audit requires one full block");
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
    auto& eb=*ctx->eb;
    const double rho=var.Double["rho1"],mu=var.Double["mu1"];
    FieldCell<double> ones(m,1.),density(m,rho),viscosity(m,mu);
    ones.SetHalo(2);density.SetHalo(2);viscosity.SetHalo(2);
    MapEmbed<BCond<double>> bc;
    for(auto c:eb.SuCFaces())bc[c]=BCond<double>(BCondType::neumann,0,0.);
    const auto weight=U::Interpolate(ones,bc,eb);
    const auto faceMu=U::Interpolate(viscosity,bc,eb);
    const auto faceRho=U::InterpolateHarmonic(density,bc,eb);
    FieldCell<double> probe(m,0.);
    for(auto c:eb.AllCells()) {const auto x=m.GetCenter(c);probe[c]=.03*std::sin(2*M_PI*x[0]/var.Double["twisted_period"])+x[1]*x[1]+.2*x[2];}
    probe.SetHalo(2);
    const auto fullGradient=U::Gradient(probe,bc,eb);
    const auto compactGradient=U::GradientImplicit(bc,eb);
    MapEmbed<BCond<double>> velocityBc;
    for(auto c:eb.SuCFaces())velocityBc[c]=BCond<double>(BCondType::dirichlet,0,0.);
    const auto velocityGradient=U::Gradient(probe,velocityBc,eb);
    FieldEmbed<double> faceAcceleration(eb,0.);
    for(auto f:eb.Faces())faceAcceleration[f]=((m.GetDir(f).raw()==0?1.:0.)-fullGradient[f])/faceRho[f];
    faceAcceleration.SetHalo(0);
    const auto acceleration=U::AverageGradient(faceAcceleration,eb);
    std::ofstream faces("material_faces.csv"),cells("material_cells.csv");
    faces<<std::setprecision(17)<<"x,y,z,axis,area,weight,mu,rho,compact_gradient,full_gradient,pressure_flux,viscosity_flux,face_acceleration\n";
    cells<<std::setprecision(17)<<"face_x,face_y,face_z,axis,side,x,y,z,accel_x,accel_y,accel_z,pressure_operator,viscosity_operator\n";
    size_t open=0,wall=0,changed=0;double deviation=0,wallDeviation=0,muError=0,rhoError=0;
    for(auto f:eb.Faces()) {
      ++open;const double w=weight[f];
      if(!(w>0&&std::isfinite(faceMu[f])&&std::isfinite(faceRho[f])))throw std::runtime_error("Invalid interpolated material");
      deviation=std::max(deviation,std::abs(w-1));
      muError=std::max(muError,std::abs(faceMu[f]/mu-w));
      rhoError=std::max(rhoError,std::abs(rho/faceRho[f]-w));
      if(std::abs(w-1)>1e-12) {
        ++changed;const auto x=eb.GetFaceCenter(f);
        const double gc=U::Eval(compactGradient[f],f,probe,eb),gf=fullGradient[f];
        faces<<x[0]<<','<<x[1]<<','<<x[2]<<','<<m.GetDir(f).raw()<<','<<eb.GetArea(f)<<','<<w<<','<<faceMu[f]<<','<<faceRho[f]<<','<<gc<<','<<gf<<','<<eb.GetArea(f)/faceRho[f]*gc<<','<<eb.GetArea(f)*faceMu[f]*gf<<','<<faceAcceleration[f]<<'\n';
        for(int side=0;side<2;++side) {
          const auto c=eb.GetCell(f,side);const auto y=m.GetCenter(c);const auto a=acceleration[c];
          double pressureOperator=0,viscosityOperator=0;
          eb.LoopNci(c,[&](auto q) {
            const auto cf=eb.GetFace(c,q);const double factor=-eb.GetArea(cf)*eb.GetOutwardFactor(c,q);
            pressureOperator+=factor/faceRho[cf]*U::Eval(compactGradient[cf],cf,probe,eb);
            viscosityOperator+=factor*faceMu[cf]*velocityGradient[cf];
          });
          cells<<x[0]<<','<<x[1]<<','<<x[2]<<','<<m.GetDir(f).raw()<<','<<side<<','<<y[0]<<','<<y[1]<<','<<y[2]<<','<<a[0]<<','<<a[1]<<','<<a[2]<<','<<pressureOperator<<','<<viscosityOperator<<'\n';
        }
      }
    }
    for(auto c:eb.CFaces()) {
      ++wall;
      wallDeviation=std::max({wallDeviation,std::abs(weight[c]-1),std::abs(faceMu[c]/mu-1),std::abs(rho/faceRho[c]-1)});
    }
    faces.flush();cells.flush();if(!faces.good()||!cells.good())throw std::runtime_error("Material field dump failed");
    std::ofstream report("material_audit.json");
    const bool passed=muError<1e-12&&rhoError<1e-12&&wallDeviation<1e-12;
    report<<std::setprecision(17)<<std::boolalpha
      <<"{\n  \"passed\": "<<passed<<",\n  \"scope\": \"Original Aphros Proj material interpolation, not a flow or accuracy result\",\n"
      <<"  \"open_faces\": "<<open<<",\n  \"wall_faces\": "<<wall<<",\n  \"changed_open_faces\": "<<changed
      <<",\n  \"maximum_constant_deviation\": "<<deviation<<",\n  \"mu_weight_error\": "<<muError
      <<",\n  \"inverse_density_weight_error\": "<<rhoError<<",\n  \"wall_material_deviation\": "<<wallDeviation<<"\n}\n";
    report.flush();if(!report.good()||!passed)throw std::runtime_error("Material interpolation audit failed");
    std::cout<<"Original material audit complete: "<<changed<<" exceptional open faces"<<std::endl;
  }
}
int main(int argc,const char** argv) {
  try {MpiWrapper mpi(&argc,&argv);return RunMpiBasicFile<M>(mpi,Run,argc>1?argv[1]:"a.conf");}
  catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
