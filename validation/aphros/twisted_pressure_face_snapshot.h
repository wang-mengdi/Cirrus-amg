// Read-only capture of the expressions evaluated by the original Proj::Project.
// No solver field, coefficient, or floating-point expression is changed.
#pragma once
#include <array>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>

template<class M> struct TwistedPressureFaceSnapshot {
  struct Row {
    size_t face,cell0,cell1,axis;
    std::array<double,3> center;
    double area,e0,e1,b,p0,p1,flux;
  };
  const M* mesh=nullptr;
  size_t sequence=0;
  double dt=0;
  std::vector<Row> rows;
};

template<class M> inline TwistedPressureFaceSnapshot<M>& TwistedLastPressureFaces() {
  static TwistedPressureFaceSnapshot<M> snapshot;
  return snapshot;
}

template<class M,class EB,class Expressions,class Flux,class Pressure>
void TwistedCapturePressureFaces(const M& m,const EB& eb,const Expressions& expressions,
                                const Flux& flux,const Pressure& pressure,double dt) {
  if(!std::getenv("APHROS_TWISTED_CAPTURE_PRESSURE_FACES"))return;
  auto& snapshot=TwistedLastPressureFaces<M>();
  snapshot.mesh=&m;snapshot.dt=dt;++snapshot.sequence;snapshot.rows.clear();
  for(auto f:eb.Faces()) {
    const auto c0=m.GetCell(f,0),c1=m.GetCell(f,1);
    const auto x=eb.GetFaceCenter(f);const auto& e=expressions[f];
    typename TwistedPressureFaceSnapshot<M>::Row row{};
    row.face=f.raw();row.cell0=c0.raw();row.cell1=c1.raw();row.axis=m.GetDir(f).raw();
    for(size_t d=0;d<M::dim;++d)row.center[d]=x[d];
    row.area=eb.GetArea(f);row.e0=e[0];row.e1=e[1];row.b=e[2];
    row.p0=pressure[c0];row.p1=pressure[c1];row.flux=flux[f];
    snapshot.rows.push_back(row);
  }
}

template<class M,class EB,class Flux,class Pressure>
void TwistedDumpPressureFaces(const M& m,const EB& eb,const Flux& flux,
                             const Pressure& pressure,double time) {
  if(!std::getenv("APHROS_TWISTED_CAPTURE_PRESSURE_FACES"))return;
  const auto& snapshot=TwistedLastPressureFaces<M>();
  if(snapshot.mesh!=&m || !snapshot.sequence)
    throw std::runtime_error("No captured original pressure face expressions");
  std::ofstream out("proj_final_b0_pressure_faces.csv");
  out<<std::setprecision(17)<<"face_raw,cell0_raw,cell1_raw,x,y,z,axis,area,e0,e1,b,p0,p1,flux\n";
  size_t i=0;
  for(auto f:eb.Faces()) {
    if(i>=snapshot.rows.size())throw std::runtime_error("Pressure face capture count mismatch");
    const auto& r=snapshot.rows[i++];
    const auto c0=m.GetCell(f,0),c1=m.GetCell(f,1);
    if(r.face!=f.raw() || r.cell0!=c0.raw() || r.cell1!=c1.raw() ||
       r.flux!=flux[f] || r.p0!=pressure[c0] || r.p1!=pressure[c1])
      throw std::runtime_error("Captured pressure face is not the final field");
    out<<r.face<<','<<r.cell0<<','<<r.cell1;
    for(double x:r.center)out<<','<<x;
    out<<','<<r.axis<<','<<r.area<<','<<r.e0<<','<<r.e1<<','<<r.b<<','<<r.p0<<','<<r.p1<<','<<r.flux<<'\n';
  }
  if(i!=snapshot.rows.size())throw std::runtime_error("Pressure face capture count mismatch");
  out.flush();if(!out.good())throw std::runtime_error("Pressure face dump failed");
  std::ofstream meta("proj_final_b0_pressure_faces_snapshot.json");
  meta<<std::setprecision(17)<<"{\n  \"scope\": \"Read-only original face expressions; exact final pressure/flux identity checked\",\n"
      <<"  \"pressure_calls\": "<<snapshot.sequence<<",\n  \"physical_time\": "<<time
      <<",\n  \"projection_dt\": "<<snapshot.dt<<",\n  \"open_faces\": "<<i<<"\n}\n";
  meta.flush();if(!meta.good())throw std::runtime_error("Pressure face metadata failed");
}
