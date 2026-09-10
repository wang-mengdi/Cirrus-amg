#include "geom/field.h"
#include "twisted_initial_velocity.h"
#include <iostream>
#include <limits>

struct Mesh {
  using Scal=long double;
  using Vect=generic::Vect<Scal,3>;
  struct {std::array<bool,3> is_periodic{{true,false,false}};} flags;
  std::vector<IdxCell> cells,all;
  std::vector<Vect> centers;
  Mesh() {
    for(int z=-2;z<4;++z)for(int y=-2;y<4;++y)for(int x=-2;x<6;++x) {
      const IdxCell c(centers.size());all.push_back(c);
      centers.push_back(Vect((x+.5L)*.125L,(y+.5L)*.125L,(z+.5L)*.125L));
      if(x>=0 && x<4 && y>=0 && y<2 && z>=0 && z<2)cells.push_back(c);
    }
  }
  auto GetGlobalSize() const {return std::array<int,3>{{4,2,2}};}
  Vect GetCellSize() const {return Vect(.125L);}
  Vect GetCenter(IdxCell c) const {return centers[size_t(c)];}
  const auto& Cells() const{return cells;}
  const auto& AllCells() const{return all;}
};
struct Geometry {
  const Mesh& m;
  bool IsExcluded(IdxCell c) const {
    const auto x=m.GetCenter(c);
    return x[0]==.0625L && x[1]==.0625L && x[2]==.0625L;
  }
};
void Require(bool b,const char* message){if(!b)throw std::runtime_error(message);}

int main() {
  Mesh m;Geometry eb{m};
  std::array<uint64_t,3> shape{{4,2,2}};
  std::array<double,3> spacing{{.125,.125,.125}};
  std::vector<unsigned char> mask(16,1);mask[0]=0;
  std::vector<std::array<double,3>> values(16);
  for(size_t i=1;i<16;++i)values[i]={{double(i),-double(i),double(i)*.5}};
  auto write=[&](bool extra=false) {
    std::ofstream f("seed.bin",std::ios::binary);f.write("TWVEL01",8);
    f.write(reinterpret_cast<char*>(shape.data()),sizeof(shape));
    f.write(reinterpret_cast<char*>(spacing.data()),sizeof(spacing));
    f.write(reinterpret_cast<char*>(mask.data()),mask.size());
    f.write(reinterpret_cast<char*>(values.data()),values.size()*sizeof(values[0]));
    if(extra)f.put('x');
  };
  using F=FieldCell<Mesh::Vect>;
  F velocity(F::Range{IdxCell(m.centers.size())},Mesh::Vect(99));
  write();TwistedLoadInitialVelocity("seed.bin",m,eb,velocity);
  for(auto c:m.AllCells()) {
    const auto p=m.GetCenter(c)/.125L;
    int x=int(std::llround(p[0]-.5L)),y=int(std::llround(p[1]-.5L)),z=int(std::llround(p[2]-.5L));
    x=(x%4+4)%4;
    for(int d=0;d<3;++d) {
      const double expected=(y<0 || y>=2 || z<0 || z>=2)?0:values[(z*2+y)*4+x][d];
      Require(velocity[c][d]==expected,"interior and periodic/nonperiodic halo assignment");
    }
  }
  Require(velocity.GetHalo()==2,"halo validity retained");
  std::ifstream input("seed.bin",std::ios::binary),echo("initial_velocity_echo.bin",std::ios::binary);
  Require(std::string(std::istreambuf_iterator<char>(input),{})==std::string(std::istreambuf_iterator<char>(echo),{}),"actual storage echo");
  int rejected=0;
  const auto reject=[&] {
    try {TwistedLoadInitialVelocity("seed.bin",m,eb,velocity);}
    catch(const std::runtime_error&){++rejected;return;}
    throw std::runtime_error("malformed input accepted");
  };
  write(true);reject();
  shape[0]=5;write();reject();shape[0]=4;
  spacing[1]=.25;write();reject();spacing[1]=.125;
  mask[0]=1;write();reject();mask[0]=0;
  mask[1]=2;write();reject();mask[1]=1;
  values[1][0]=std::numeric_limits<double>::quiet_NaN();write();reject();values[1][0]=1;
  values[0][0]=1;write();reject();values[0][0]=0;
  write();m.flags.is_periodic[1]=true;reject();m.flags.is_periodic[1]=false;
  {std::ofstream f("seed.bin",std::ios::binary);f<<"TWVEL01";}reject();
  Require(rejected==9,"all malformed cases rejected");
  std::cout<<"{\"passed\":true,\"malformed_inputs_rejected\":"<<rejected<<"}\n";
}
