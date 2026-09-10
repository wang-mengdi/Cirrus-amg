// Diagnostic-only additions to Aphros. These routines read geometry and fields;
// they do not change boundary conditions, linear systems, or solver updates.
#pragma once
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <map>
#include <vector>
#include <sstream>

template<class M,class EB,class FF,class FE>
void TwistedAdvectionDump(const M& m,const EB& eb,const FieldCell<typename M::Scal>& velocity,
                          const FF& flux,const FE& expression) {
  const char* prefix=std::getenv("APHROS_TWISTED_ADV_DUMP");if(!prefix)return;
  static std::map<std::pair<int,std::string>,int> counters;
  const std::string name=velocity.GetName();const int iteration=counters[{m.GetId(),name}]++;
  if(iteration>1)return;
  std::ofstream out(std::string(prefix)+"_"+name+"_"+std::to_string(iteration)+"_b"+std::to_string(m.GetId())+"_faces.csv");
  out<<std::setprecision(17)<<"x,y,z,axis,area,volume_flux,coefficient_m,coefficient_p,constant,evaluated_flux\n";
  for(auto f:eb.Faces()) {
    auto x=eb.GetFaceCenter(f);const auto& e=expression[f];
    const auto cm=eb.GetCell(f,0),cp=eb.GetCell(f,1);
    for(size_t d=0;d<3;++d)out<<(d<M::dim?x[d]:0.)<<',';
    out<<m.GetDir(f).raw()<<','<<eb.GetArea(f)<<','<<flux[f]<<','<<e[0]<<','<<e[1]<<','<<e[2]<<','
       <<e[0]*velocity[cm]+e[1]*velocity[cp]+e[2]<<'\n';
  }
  std::ofstream raw(std::string(prefix)+"_"+name+"_"+std::to_string(iteration)+"_b"+std::to_string(m.GetId())+"_sources.csv");
  raw<<std::setprecision(17)<<"x,y,z,axis,volume_flux,velocity_m,velocity_p\n";
  for(auto f:eb.SuFaces()) {
    const auto x=m.GetCenter(f);
    for(size_t d=0;d<3;++d)raw<<(d<M::dim?x[d]:0.)<<',';
    raw<<m.GetDir(f).raw()<<','<<flux[f]<<','<<velocity[eb.GetCell(f,0)]<<','<<velocity[eb.GetCell(f,1)]<<'\n';
  }
}

template<class M,class EB>
void TwistedTimeDump(const M& m,const EB& eb,const FieldCell<typename M::Vect>& velocity,
                     double time,bool initialize) {
  const char* prefix=std::getenv("APHROS_TWISTED_TIME_DUMP");if(!prefix)return;
  using Vect=typename M::Vect;
  struct State { double time=0;std::vector<Vect> velocity; };
  static std::map<int,State> states;auto& previous=states[m.GetId()];
  std::vector<Vect> current;double sum=0,volume=0,maximum=0;int index=0;
  for(auto c:eb.Cells()) {
    current.push_back(velocity[c]);
    if(!initialize) {
      if(index>=int(previous.velocity.size()))throw std::runtime_error("Temporal diagnostic cell count changed");
      const auto delta=velocity[c]-previous.velocity[index];const double norm=delta.norm();
      sum+=norm*norm*eb.GetVolume(c);volume+=eb.GetVolume(c);maximum=std::max(maximum,norm);
    }
    ++index;
  }
  std::string path=std::string(prefix)+"_b"+std::to_string(m.GetId())+"_time.csv";
  if(initialize) {
    std::ofstream out(path);out<<"time,time_step,velocity_change_linf,velocity_change_volume_l2,temporal_acceleration_volume_l2\n";
  } else {
    const double dt=time-previous.time;
    if(!(dt>0 && volume>0))throw std::runtime_error("Invalid time diagnostic interval");
    const double change=std::sqrt(sum/volume);
    std::ofstream out(path,std::ios::app);out<<std::setprecision(17)<<time<<','<<dt<<','<<maximum<<','<<change<<','<<change/dt<<'\n';
  }
  previous.time=time;previous.velocity=std::move(current);
}

template<class M>
void TwistedGeometryDump(const M&, const M&) {}

template<class M>
void TwistedGeometryDump(const M& m, const Embed<M>& eb) {
  using Scal=typename M::Scal;
  const char* prefix = std::getenv("APHROS_TWISTED_GEOMETRY");
  if (!prefix) return;
  const std::string base = std::string(prefix) + "_b" + std::to_string(m.GetId());
  const double h = m.GetCellSize()[0];
  auto key = [h](auto x, bool face, size_t axis) {
    std::array<int,3> q{};
    for (size_t d=0; d<M::dim; ++d)
      q[d] = int(std::llround(x[d]/h - ((face && d==axis) ? 0. : .5)));
    return q;
  };
  auto xyz = [](auto& out, auto x) {
    for (size_t d=0; d<3; ++d) out << ',' << (d<M::dim ? x[d] : 0.);
  };
  std::ofstream cells(base + "_geometry_cells.csv"), faces(base + "_geometry_faces.csv"),
      walls(base + "_geometry_walls.csv"), polys(base + "_geometry_polygons.csv");
  for (auto out : {&cells,&faces,&walls,&polys}) *out << std::setprecision(17);
  cells << "i,j,k,x,y,z,h,volume,cut,geometry_x,geometry_y,geometry_z\n";
  faces << "axis,i,j,k,x,y,z,area\n";
  walls << "i,j,k,x,y,z,nx,ny,nz,area,alpha\n";
  polys << "axis,i,j,k,vertex,x,y,z\n";
  for (auto c : eb.Cells()) {
    auto q = key(m.GetCenter(c),false,0);
    cells << q[0] << ',' << q[1] << ',' << q[2]; xyz(cells,m.GetCenter(c));
    cells << ',' << h << ',' << eb.GetVolume(c) << ',' << eb.IsCut(c);
    xyz(cells,eb.GetCellCenter(c)); cells << '\n';
  }
  for (auto f : eb.Faces()) {
    auto axis = m.GetDir(f).raw(); auto q = key(m.GetCenter(f),true,axis);
    faces << axis << ',' << q[0] << ',' << q[1] << ',' << q[2];
    xyz(faces,eb.GetFaceCenter(f)); faces << ',' << eb.GetArea(f) << '\n';
    int vertex=0;
    for (auto x : eb.GetFacePoly(f)) {
      polys << axis << ',' << q[0] << ',' << q[1] << ',' << q[2] << ',' << vertex++;
      xyz(polys,x); polys << '\n';
    }
  }
  for (auto c : eb.CFaces()) {
    auto q = key(m.GetCenter(c),false,0);
    walls << q[0] << ',' << q[1] << ',' << q[2];
    xyz(walls,eb.GetFaceCenter(c)); xyz(walls,eb.GetNormal(c));
    walls << ',' << eb.GetArea(c) << ',' << eb.GetAlpha(c) << '\n';
    int vertex=0;
    for (auto x : eb.GetCutPoly(c)) {
      polys << 3 << ',' << q[0] << ',' << q[1] << ',' << q[2] << ',' << vertex++;
      xyz(polys,x); polys << '\n';
    }
  }
  if(std::getenv("APHROS_TWISTED_INTERPOLATION_CHECK")) {
    FieldCell<Scal> one(m,1.);
    MapEmbed<BCond<Scal>> bc;
    const auto value=UEmbed<M>::Interpolate(one,bc,eb);
    const auto gradient=UEmbed<M>::Gradient(one,bc,eb);
    std::ofstream check(base+"_interpolation_check.csv");
    check<<std::setprecision(17)<<"axis,i,j,k,interpolated_one,gradient_one\n";
    for(auto f:eb.Faces())if(eb.GetType(f)!=Embed<M>::Type::regular) {
      const auto axis=m.GetDir(f).raw();const auto q=key(m.GetCenter(f),true,axis);
      check<<axis<<','<<q[0]<<','<<q[1]<<','<<q[2]<<','<<value[f]<<','<<gradient[f]<<'\n';
    }
    check.flush();if(!check.good())throw std::runtime_error("Interpolation diagnostic write failed");
  }
  if(const char* diagnostic=std::getenv("APHROS_TWISTED_WALL_CONSISTENCY")) {
    if(M::dim!=3)throw std::runtime_error("Manufactured tube diagnostic requires 3D");
    double radius,amplitude,period,cy,cz;
    std::istringstream parameters(diagnostic);
    if(!(parameters>>radius>>amplitude>>period>>cy>>cz) || !(radius>0 && period>0))
      throw std::runtime_error("Expected manufactured scalar radius amplitude period cy cz");
    auto manufactured=[&](const typename M::Vect& x) {
      const double phase=2*M_PI*x[0]/period;
      const double y=x[1]-cy-amplitude*std::sin(phase),z=x[2]-cz-amplitude*std::cos(phase);
      return 1-(y*y+z*z)/(radius*radius);
    };
    FieldCell<Scal> scalar(m,0.);
    for(auto c:eb.AllCells())scalar[c]=manufactured(m.GetCenter(c));
    scalar.SetHalo(2);
    MapEmbed<BCond<Scal>> zero_bc,exact_bc;
    for(auto c:eb.CFaces()) {
      zero_bc[c]=BCond<Scal>(BCondType::dirichlet,0,0.);
      exact_bc[c]=BCond<Scal>(BCondType::dirichlet,0,manufactured(eb.GetFaceCenter(c)));
    }
    const auto zero_gradient=UEmbed<M>::Gradient(scalar,zero_bc,eb);
    const auto exact_gradient=UEmbed<M>::Gradient(scalar,exact_bc,eb);
    std::ofstream check(base+"_wall_consistency.csv");
    check<<std::setprecision(17)<<"x,y,z,zero_boundary_derivative,exact_boundary_derivative\n";
    for(auto c:eb.CFaces()) {
      const auto x=eb.GetFaceCenter(c);
      check<<x[0]<<','<<x[1]<<','<<x[2]<<','<<zero_gradient[c]<<','<<exact_gradient[c]<<'\n';
    }
    check.flush();if(!check.good())throw std::runtime_error("Manufactured wall diagnostic write failed");
  }
  if(std::getenv("APHROS_TWISTED_GEOMETRY_ONLY")) {
    for(auto out:{&cells,&faces,&walls,&polys}) {out->flush();if(!out->good())throw std::runtime_error("Geometry dump failed");}
    std::cerr<<"GEOMETRY ONLY: no flow solution was computed\n";
    std::exit(0);
  }
}

template<class M>
void TwistedWallDump(const M&, const M&, const FieldCell<typename M::Vect>&,
                     const MapEmbed<BCond<typename M::Vect>>&, const FieldCell<typename M::Scal>&) {}

template<class M>
void TwistedWallDump(const M& m, const Embed<M>& eb, const FieldCell<typename M::Vect>& vel,
                     const MapEmbed<BCond<typename M::Vect>>& bc, const FieldCell<typename M::Scal>& mu) {
  const char* prefix = std::getenv("APHROS_TWISTED_GEOMETRY");
  if (!prefix) return;
  auto grad = UEmbed<M>::Gradient(vel,bc,eb);
  std::ofstream out(std::string(prefix)+"_final_b"+std::to_string(m.GetId())+"_walls.csv");
  out << std::setprecision(17) << "i,j,k,x,y,z,nx,ny,nz,area,du_dn,dv_dn,dw_dn,tau_x,tau_y,tau_z\n";
  double h=m.GetCellSize()[0];
  for (auto c : eb.CFaces()) {
    auto x=m.GetCenter(c);
    for (size_t d=0;d<3;++d) { if(d) out << ','; out << (d<M::dim ? int(std::llround(x[d]/h-.5)) : 0); }
    auto normal=eb.GetNormal(c);
    for (auto v : {eb.GetFaceCenter(c),normal})
      for (size_t d=0;d<3;++d) out << ',' << (d<M::dim?v[d]:0.);
    out << ',' << eb.GetArea(c);
    for (auto v : {grad[c],grad[c].orth(normal)*mu[c]})
      for (size_t d=0;d<3;++d) out << ',' << (d<M::dim?v[d]:0.);
    out << '\n';
  }
}
