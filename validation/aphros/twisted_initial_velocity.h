// An optional initial guess through the original ProjArgs::fcvel interface.
// Pressure, flux initialization, time integration, and all equation bodies
// remain in the original Aphros implementation.
#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>

template<class M,class EB>
void TwistedLoadInitialVelocity(const char* path,const M& m,const EB& eb,
                               FieldCell<typename M::Vect>& velocity) {
  using S=typename M::Scal;
  using V=typename M::Vect;
  const auto fail=[](const char* text){throw std::runtime_error(text);};
  std::ifstream input(path,std::ios::binary);
  if(!input)fail("Cannot open initial velocity file");
  char magic[8];std::array<uint64_t,3> shape{};std::array<double,3> spacing{};
  input.read(magic,8);input.read(reinterpret_cast<char*>(shape.data()),sizeof(shape));
  input.read(reinterpret_cast<char*>(spacing.data()),sizeof(spacing));
  if(!input || std::memcmp(magic,"TWVEL01",8))fail("Invalid initial velocity header");
  const auto global=m.GetGlobalSize();const auto h=m.GetCellSize();
  for(int d=0;d<3;++d) {
    if(shape[d]!=uint64_t(global[d]) || spacing[d]!=double(h[d]) || !shape[d])
      fail("Initial velocity shape or spacing differs from the mesh");
    if(bool(m.flags.is_periodic[d])!=(d==0))fail("Initial velocity requires the periodic-x tube");
  }
  const size_t count=size_t(shape[0])*size_t(shape[1])*size_t(shape[2]);
  std::vector<unsigned char> mask(count);
  std::vector<std::array<double,3>> values(count);
  input.read(reinterpret_cast<char*>(mask.data()),mask.size());
  input.read(reinterpret_cast<char*>(values.data()),values.size()*sizeof(values[0]));
  if(!input || input.peek()!=std::char_traits<char>::eof())fail("Initial velocity length differs from its header");
  for(size_t i=0;i<count;++i) {
    if(mask[i]>1)fail("Invalid initial fluid mask");
    for(auto value:values[i])if(!std::isfinite(value) || (!mask[i] && value!=0))
      fail("Initial velocity contains nonfinite or nonzero excluded-cell values");
  }
  const auto index=[&](auto c,bool wrap)->size_t {
    const auto x=m.GetCenter(c);std::array<long long,3> w{};
    for(int d=0;d<3;++d) {
      const S pos=x[d]/h[d]-S(.5);
      w[d]=std::llround(pos);
      if(std::abs(pos-S(w[d]))>S(1e-10))fail("Initial velocity mesh does not start at the expected origin");
      if(wrap && d==0)w[d]=(w[d]%static_cast<long long>(shape[d])+static_cast<long long>(shape[d]))%static_cast<long long>(shape[d]);
      if(w[d]<0 || uint64_t(w[d])>=shape[d])return count;
    }
    return (size_t(w[2])*size_t(shape[1])+size_t(w[1]))*size_t(shape[0])+size_t(w[0]);
  };
  size_t fluid=0;
  for(auto c:m.Cells()) {
    const size_t i=index(c,false);
    if(i==count || bool(mask[i])==eb.IsExcluded(c))fail("Initial velocity fluid mask differs from loaded geometry");
    fluid+=mask[i];
  }
  size_t halo=0;
  for(auto c:m.AllCells()) {
    const size_t i=index(c,true);V value(S(0));
    if(i!=count)for(int d=0;d<3;++d)value[d]=S(values[i][d]);
    velocity[c]=value;
    if(index(c,false)==count && i!=count && mask[i])++halo;
  }
  velocity.SetHalo(2);
  // Echo actual interior storage, including excluded cells. Runtime provenance
  // binds the input file; the Python checker compares this echo byte-for-byte.
  for(auto c:m.Cells()) {
    const size_t i=index(c,false);
    for(int d=0;d<3;++d)values[i][d]=double(velocity[c][d]);
  }
  std::ofstream echo("initial_velocity_echo.bin",std::ios::binary);
  echo.write(magic,8);echo.write(reinterpret_cast<const char*>(shape.data()),sizeof(shape));
  echo.write(reinterpret_cast<const char*>(spacing.data()),sizeof(spacing));
  echo.write(reinterpret_cast<const char*>(mask.data()),mask.size());
  echo.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(values[0]));
  echo.flush();if(!echo)fail("Initial velocity echo failed");
  std::ofstream meta("initial_velocity_loaded.json");
  meta<<"{\"scope\":\"Initial velocity only; original Proj constructs pressure and flux\","
      <<"\"fluid_cells\":"<<fluid<<",\"periodic_fluid_halo_cells\":"<<halo
      <<",\"interior_cells\":"<<count<<",\"halo\":2}\n";
  meta.flush();if(!meta)fail("Initial velocity metadata failed");
}
