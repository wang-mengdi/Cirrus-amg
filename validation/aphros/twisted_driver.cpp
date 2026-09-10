// Minimal benchmark driver around Aphros' original Embed and Simple classes.
// It omits Hydro's unused multiphase/tracer/particle orchestration. No spatial
// operator, SIMPLE update, or linear-system assembly is implemented here.
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include "distr/distrbasic.h"
#include "parse/simple.h"
#include "solver/embed.h"
#include "solver/simple.h"
#include "util/linear.h"
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#endif

using M = MeshCartesian<double, 3>;
using Vect = M::Vect;
using Scal = M::Scal;
using EB = Embed<M>;

void Resources(const char* stage, bool first=false) {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS_EX memory{};
  if (!GetProcessMemoryInfo(GetCurrentProcess(),
          reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&memory), sizeof(memory)))
    throw std::runtime_error("Cannot read benchmark memory counters");
  std::ofstream out("driver_memory.csv", first?std::ios::out:std::ios::app);
  if(first)out<<"stage,working_set_bytes,peak_working_set_bytes,private_bytes\n";
  out<<stage<<','<<memory.WorkingSetSize<<','<<memory.PeakWorkingSetSize<<','<<memory.PrivateUsage<<'\n';
#endif
}

void SolveStep(M& m, Simple<EB>& solver, const Vars& var) {
  auto sem=m.GetSem("tube-step");
  if(sem.Nested("start"))solver.StartStep();
  sem.LoopBegin();
  if(sem.Nested("iteration"))solver.MakeIteration();
  if(sem("convergence")) {
    const double error=solver.GetError();
    std::cout<<".....iter="<<solver.GetIter()<<", diff="<<std::setprecision(17)<<error<<std::endl;
    if(!std::isfinite(error))throw std::runtime_error("Nonfinite SIMPLE iteration error");
    if(error<var.Double["tol"] && solver.GetIter()>=var.Int["min_iter"])sem.LoopBreak();
    else if(solver.GetIter()>=var.Int["max_iter"])throw std::runtime_error("SIMPLE time step did not converge");
  }
  sem.LoopEnd();
  if(sem.Nested("finish"))solver.FinishStep();
}

void Run(M& m, Vars& var) {
  auto sem=m.GetSem("tube-driver");
  struct Context {
    FieldNode<Scal> levelset;
    FieldCell<Scal> rho,mu,volume_source,mass_source;
    FieldCell<Vect> initial_velocity,force;
    FieldEmbed<Scal> balanced_force;
    MapEmbed<BCondFluid<Vect>> boundary;
    MapCell<std::shared_ptr<CondCellFluid>> cell_conditions;
    std::unique_ptr<EB> geometry;
    std::unique_ptr<Simple<EB>> solver;
  } *ctx(sem);
  auto& t=*ctx;
  if(sem("geometry")) {
    if(var.String["fluid_solver"]!="simple" || var.String["vel_init"]!="zero" ||
       var.Double["rho1"]!=var.Double["rho2"] || var.Double["mu1"]!=var.Double["mu2"] ||
       var.Int["enable_surftens"] || var.Int["enable_advection"] ||
       var.Int["embed_smoothen_iters"]!=0 || var.Int["dim"]!=3 ||
       var.Double["dt0"]!=var.Double["dtmax"])
      throw std::runtime_error("Driver only supports the prescribed single-phase, fixed-step tube benchmark");
    if(var.Int["bx"]!=1 || var.Int["by"]!=1 || var.Int["bz"]!=1)
      throw std::runtime_error("Diagnostic driver currently requires one block");
    const double radius=var.Double["twisted_radius"], amplitude=var.Double["twisted_amplitude"];
    const double period=var.Double["twisted_period"],cy=var.Double["twisted_center_y"],cz=var.Double["twisted_center_z"];
    t.levelset.Reinit(m,0.);
    for(auto node:m.AllNodes()) {
      const auto x=m.GetNode(node);const double angle=2*M_PI*x[0]/period;
      const double y=x[1]-cy-amplitude*std::sin(angle),z=x[2]-cz-amplitude*std::cos(angle);
      t.levelset[node]=radius-std::sqrt(y*y+z*z);
    }
    t.levelset.SetHalo(2);
    t.geometry.reset(new EB(m,0));
    Resources("before_embed",true);
  }
  if(sem.Nested("embed-init"))t.geometry->Init(t.levelset);
  if(sem("solver")) {
    t.rho.Reinit(m,var.Double["rho1"]);t.mu.Reinit(m,var.Double["mu1"]);
    t.volume_source.Reinit(m,0.);t.mass_source.Reinit(m,0.);
    t.initial_velocity.Reinit(m,Vect(0));t.force.Reinit(m,Vect(0));
    t.balanced_force.Reinit(m,0.);
    const Vect force(var.Vect["force"]),gravity(var.Vect["gravity"]);
    if(gravity.norm()!=0.)throw std::runtime_error("This driver requires zero gravity, as in the fixed benchmark");
    for(auto f:m.AllFaces())t.balanced_force[f]=force.dot(m.GetNormal(f));
    for(auto c:t.geometry->SuCFaces())t.boundary[c]=BCondFluid<Vect>();
    std::shared_ptr<linear::Solver<M>> symmetric=ULinear<M>::MakeLinearSolver(var,"symm",m);
    std::shared_ptr<linear::Solver<M>> general=ULinear<M>::MakeLinearSolver(var,"gen",m);
    auto par=ParsePar<Simple<M>>()(var);
    const SimpleArgs<M> args{t.initial_velocity,t.boundary,t.cell_conditions,&t.rho,&t.mu,
                            &t.force,&t.balanced_force,&t.volume_source,&t.mass_source,0.,
                            var.Double["dt0"],symmetric,general,par};
    t.solver.reset(new Simple<EB>(m,*t.geometry,args));
    Resources("after_solver_init");
  }
  sem.LoopBegin();
  if(sem.Nested("step"))SolveStep(m,*t.solver,var);
  if(sem("step-done")) {
    Resources("completed_step");
    if(t.solver->GetTime()>=var.Double["tmax"]*(1-1e-13))sem.LoopBreak();
  }
  sem.LoopEnd();
  if(sem("complete")) {
    Resources("final");
    std::cout<<"End of simulation: minimal driver using upstream Embed and Simple"<<std::endl;
  }
}

int main(int argc,const char** argv) {
  try {
    MpiWrapper mpi(&argc,&argv);
    return RunMpiBasicFile<M>(mpi,Run,argc>1?argv[1]:"a.conf");
  } catch(const std::exception& error) {
    std::cerr<<"Tube driver error: "<<error.what()<<std::endl;
    return 1;
  }
}
