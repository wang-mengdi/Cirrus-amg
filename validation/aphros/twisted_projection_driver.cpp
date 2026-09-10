// Independent driver around the original Aphros Proj/Embed implementations.
// Only case construction and read-only field/geometry diagnostics live here.
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include "distr/distrbasic.h"
#include "parse/solver.h"
#include "parse/proj.h"
#include "solver/proj.h"
#include "solver/approx_eb.h"
#include "solver/twisted_diagnostics.h"
#include "util/linear.h"
#include "twisted_pressure_snapshot.h"
#include "twisted_pressure_geometry.h"
#include "twisted_pressure_face_snapshot.h"

using M=MeshCartesian<double,3>;
using Vect=M::Vect;
using Scal=M::Scal;
using EB=Embed<M>;

void DumpPressureRows(const M& m,const EB& eb,const Proj<EB>& solver) {
  if(!std::getenv("APHROS_TWISTED_CAPTURE_PRESSURE"))return;
  const auto& snapshot=TwistedLastPressure<M>();
  if(snapshot.mesh!=&m || !snapshot.sequence)
    throw std::runtime_error("No captured pressure system for this mesh");
  const auto& p=solver.GetPressure();
  std::ofstream rows("proj_final_b0_pressure_rows.csv");
  rows<<std::setprecision(17)<<"cell_raw,x,y,z,volume,p,b,a0,is_gauge";
  for(int q=0;q<6;++q)rows<<",a"<<q+1<<",p"<<q+1<<",x"<<q+1<<",y"<<q+1<<",z"<<q+1;
  rows<<'\n';size_t count=0;
  for(auto c:eb.Cells()) {
    if(p[c]!=snapshot.returned_pressure[c])
      throw std::runtime_error("Captured pressure solution is not the final pressure field");
    const auto x=m.GetCenter(c);const auto& e=snapshot.equations[c];
    rows<<c.raw()<<','<<x[0]<<','<<x[1]<<','<<x[2]<<','<<eb.GetVolume(c)<<','<<p[c]<<','<<e.back()<<','<<e[0]<<','<<(c.raw()==snapshot.gauge_raw);
    for(auto q:m.Nci(c)) {
      const auto neighbor=m.GetCell(c,q);const auto y=m.GetCenter(neighbor);
      rows<<','<<e[1+q.raw()]<<','<<p[neighbor]<<','<<y[0]<<','<<y[1]<<','<<y[2];
    }
    rows<<'\n';++count;
  }
  rows.flush();if(!rows.good())throw std::runtime_error("Pressure system dump failed");
  std::ofstream metadata("proj_final_b0_pressure_snapshot.json");
  metadata<<std::setprecision(17)<<"{\n  \"scope\": \"Actual last original pressure system and returned solution; final-field identity checked\",\n"
    <<"  \"pressure_calls\": "<<snapshot.sequence<<",\n  \"physical_time\": "<<solver.GetTime()
    <<",\n  \"fluid_rows\": "<<count<<",\n  \"regular_cell_volume\": "<<m.GetCellSize().prod()
    <<",\n  \"effective_tolerance\": "<<snapshot.tolerance<<",\n  \"reported_original_residual\": "<<snapshot.reported_residual
    <<",\n  \"refinements\": "<<snapshot.refinements<<"\n}\n";
  metadata.flush();if(!metadata.good())throw std::runtime_error("Pressure snapshot metadata failed");
}

void DumpFlow(const M& m,const EB& eb,const Proj<EB>& solver) {
  const auto& u=solver.GetVelocity();const auto& p=solver.GetPressure();
  const auto& flux=solver.GetVolumeFlux();
  std::ofstream cells("proj_final_b0_cells.csv"),faces("proj_final_b0_faces.csv");
  cells<<std::setprecision(17)<<"x,y,z,volume,u,v,w,p\n";
  faces<<std::setprecision(17)<<"x,y,z,axis,area,flux\n";
  for(auto c:eb.Cells()) {
    const auto x=m.GetCenter(c);
    cells<<x[0]<<','<<x[1]<<','<<x[2]<<','<<eb.GetVolume(c)<<','
         <<u[c][0]<<','<<u[c][1]<<','<<u[c][2]<<','<<p[c]<<'\n';
  }
  for(auto f:eb.Faces()) {
    const auto x=eb.GetFaceCenter(f);
    faces<<x[0]<<','<<x[1]<<','<<x[2]<<','<<m.GetDir(f).raw()<<','<<eb.GetArea(f)<<','<<flux[f]<<'\n';
  }
  cells.flush();faces.flush();
  if(!cells.good()||!faces.good())throw std::runtime_error("Projection field dump failed");
}

void SolveStep(M& m,Proj<EB>& solver,const Vars& var) {
  auto sem=m.GetSem("projection-step");
  if(sem.Nested("start"))solver.StartStep();
  sem.LoopBegin();
  if(sem.Nested("iteration"))solver.MakeIteration();
  if(sem("convergence")) {
    const double error=solver.GetError();
    std::cout<<".....iter="<<solver.GetIter()<<", diff="<<std::setprecision(17)<<error<<std::endl;
    if(!std::isfinite(error))throw std::runtime_error("Nonfinite projection iteration error");
    if(error<var.Double["tol"] && solver.GetIter()>=var.Int["min_iter"])sem.LoopBreak();
    else if(solver.GetIter()>=var.Int["max_iter"])throw std::runtime_error("Projection step did not converge");
  }
  sem.LoopEnd();
  if(sem.Nested("finish"))solver.FinishStep();
}

void Run(M& m,Vars& var) {
  auto sem=m.GetSem("tube-projection-driver");
  struct Context {
    FieldNode<Scal> levelset;
    FieldCell<Scal> rho,mu,volume_source,mass_source;
    FieldCell<Vect> velocity,force;
    FieldEmbed<Scal> balanced_force;
    MapEmbed<BCondFluid<Vect>> boundary;
    MapCell<std::shared_ptr<CondCellFluid>> conditions;
    std::unique_ptr<EB> geometry;
    std::unique_ptr<Proj<EB>> solver;
  } *ctx(sem);
  auto& t=*ctx;
  if(sem("geometry")) {
    if(var.String["fluid_solver"]!="proj" || var.String["conv"]!="imp" || var.Int["stokes"] ||
       var.String["vel_init"]!="zero" || var.Double["rho1"]!=var.Double["rho2"] ||
       var.Double["mu1"]!=var.Double["mu2"] || var.Int["enable_surftens"] ||
       var.Int["enable_advection"] || var.Int["embed_smoothen_iters"] ||
       var.Int["dim"]!=3 || var.Double["dt0"]!=var.Double["dtmax"] ||
       var.Int["bx"]!=1 || var.Int["by"]!=1 || var.Int["bz"]!=1)
      throw std::runtime_error("Projection diagnostic expects the fixed single-block implicit-diffusion NS tube case");
    const double radius=var.Double["twisted_radius"],a=var.Double["twisted_amplitude"];
    const double period=var.Double["twisted_period"],cy=var.Double["twisted_center_y"],cz=var.Double["twisted_center_z"];
    t.levelset.Reinit(m,0.);
    for(auto node:m.AllNodes()) {
      const auto x=m.GetNode(node);const double angle=2*M_PI*x[0]/period;
      const double y=x[1]-cy-a*std::sin(angle),z=x[2]-cz-a*std::cos(angle);
      t.levelset[node]=radius-std::sqrt(y*y+z*z);
    }
    t.levelset.SetHalo(2);t.geometry.reset(new EB(m,0));
  }
  if(sem.Nested("embed"))t.geometry->Init(t.levelset);
  if(sem("solver")) {
    TwistedGeometryDump(m,*t.geometry);
    if(std::getenv("APHROS_TWISTED_VOLUME_COMPATIBILITY")) {
      auto& physical=TwistedPressureVolumes<M>();
      physical.mesh=&m;physical.volume.Reinit(m,0.);
      for(auto c:t.geometry->Cells())physical.volume[c]=t.geometry->GetVolume(c);
    }
    t.rho.Reinit(m,var.Double["rho1"]);t.mu.Reinit(m,var.Double["mu1"]);
    t.volume_source.Reinit(m,0.);t.mass_source.Reinit(m,0.);
    t.velocity.Reinit(m,Vect(0));t.force.Reinit(m,Vect(0));t.balanced_force.Reinit(m,0.);
    const Vect force(var.Vect["force"]),gravity(var.Vect["gravity"]);
    if(gravity.norm()!=0.)throw std::runtime_error("Use the prescribed axial balanced force and zero gravity");
    for(auto f:m.AllFaces())t.balanced_force[f]=force.dot(m.GetNormal(f));
    for(auto c:t.geometry->SuCFaces())t.boundary[c]=BCondFluid<Vect>();
    std::shared_ptr<linear::Solver<M>> linear=ULinear<M>::MakeLinearSolver(var,"symm",m);
    const ProjArgs<M> args{t.velocity,t.boundary,t.conditions,&t.rho,&t.mu,&t.force,&t.balanced_force,
      &t.volume_source,&t.mass_source,0.,var.Double["dt0"],linear,ParsePar<Proj<M>>()(var)};
    t.solver.reset(new Proj<EB>(m,*t.geometry,args));
    TwistedTimeDump(m,*t.geometry,t.velocity,0.,true);
  }
  sem.LoopBegin();
  if(sem.Nested("step"))SolveStep(m,*t.solver,var);
  if(sem("step-done")) {
    TwistedTimeDump(m,*t.geometry,t.solver->GetVelocity(),t.solver->GetTime(),false);
    TwistedWallDump(m,*t.geometry,t.solver->GetVelocity(),t.solver->GetVelocityCond(),t.mu);
    DumpFlow(m,*t.geometry,*t.solver);
    DumpPressureRows(m,*t.geometry,*t.solver);
    TwistedDumpPressureFaces(m,*t.geometry,t.solver->GetVolumeFlux(),t.solver->GetPressure(),t.solver->GetTime());
    if(t.solver->GetTime()>=var.Double["tmax"]*(1-1e-13))sem.LoopBreak();
  }
  sem.LoopEnd();
  if(sem("complete"))std::cout<<"End of simulation: original Aphros Proj with implicit diffusion"<<std::endl;
}

int main(int argc,const char** argv) {
  try {MpiWrapper mpi(&argc,&argv);return RunMpiBasicFile<M>(mpi,Run,argc>1?argv[1]:"a.conf");}
  catch(const std::exception& e){std::cerr<<"Projection driver error: "<<e.what()<<std::endl;return 1;}
}
