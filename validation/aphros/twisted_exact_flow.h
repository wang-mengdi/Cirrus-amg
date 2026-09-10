// Read-only hexadecimal output of the actual stored flow fields.
#pragma once
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

template<class M, class EB, class Solver>
void TwistedExactFlowDump(const M& m,const EB& eb,const Solver& solver) {
  using S=typename M::Scal;
  const auto& velocity=solver.GetVelocity();
  const auto& pressure=solver.GetPressure();
  const auto& flux=solver.GetVolumeFlux();
  const auto h=m.GetCellSize()[0];
  auto key=[h](auto x,int axis) {
    std::array<int,3> result{};
    for(int d=0;d<3;++d)result[d]=int(std::llround(x[d]/h-(d==axis?S(0):S(.5))));
    return result;
  };
  std::ofstream cells("proj_final_b0_exact_cells.csv"),faces("proj_final_b0_exact_faces.csv");
  cells<<"i,j,k,volume_hex,u_hex,v_hex,w_hex,p_hex\n"<<std::hexfloat;
  faces<<"axis,i,j,k,area_hex,flux_hex\n"<<std::hexfloat;
  size_t nc=0,nf=0;
  for(auto c:eb.Cells()) {
    const auto q=key(m.GetCenter(c),-1);
    cells<<q[0]<<','<<q[1]<<','<<q[2]<<','<<eb.GetVolume(c);
    for(int d=0;d<3;++d)cells<<','<<velocity[c][d];
    cells<<','<<pressure[c]<<'\n';++nc;
  }
  for(auto f:eb.Faces()) {
    const int axis=m.GetDir(f).raw();const auto q=key(m.GetCenter(f),axis);
    faces<<axis<<','<<q[0]<<','<<q[1]<<','<<q[2]<<','<<eb.GetArea(f)<<','<<flux[f]<<'\n';++nf;
  }
  cells.flush();faces.flush();
  if(!cells.good()||!faces.good())throw std::runtime_error("Exact flow dump failed");
  std::ofstream meta("proj_final_b0_exact.json");
  meta<<"{\n  \"format\": \"C++ hexadecimal floating point; actual stored scalar values\",\n"
      <<"  \"mantissa_bits\": "<<std::numeric_limits<S>::digits
      <<",\n  \"scalar_bytes\": "<<sizeof(S)<<",\n  \"cells\": "<<nc<<",\n  \"faces\": "<<nf
      <<",\n  \"physical_time_hex\": \""<<std::hexfloat<<solver.GetTime()<<"\"\n}\n";
  meta.flush();if(!meta.good())throw std::runtime_error("Exact flow metadata failed");
}
