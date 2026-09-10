// Dump actual Cartesian center getters, linked to each separately verified
// build. The comparison uses the old implementation as its oracle.
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include "geom/mesh.h"
#include "util/mpi.h"
using M=MeshCartesian<long double,3>;
using Vect=M::Vect;
using I=M::MIdx;
int main(int argc,const char** argv) {
  try {
    MpiWrapper mpi(&argc,&argv);
    if(argc!=2)throw std::runtime_error("Expected a fresh coordinate output path");
    std::ifstream previous(argv[1]);if(previous.good())throw std::runtime_error("Preserve existing coordinate output");
    std::ofstream out(argv[1]);out<<"case,kind,id,x,y,z\n"<<std::hexfloat;
    size_t cells=0,faces=0,test=0;
    const I shapes[]={I(32,16,16),I(7,11,9),I(1,2,3)};
    const I begins[]={I(0,0,0),I(5,-3,12),I(-5,8,-9)};
    const Vect lows[]={Vect(0,0,0),Vect(-.137L,.021L,-2.3L),Vect(-.02L,-.04L,-.06L)};
    const Vect highs[]={Vect(.25L,.125L,.125L),Vect(.9L,1.234L,-.7L),Vect(.01L,.08L,.15L)};
    for(size_t shape=0;shape<3;++shape)for(int halo=1;halo<=3;++halo) {
      M m(begins[shape],shapes[shape],Rect<Vect>(lows[shape],highs[shape]),halo,true,true,shapes[shape],0);
      auto write=[&](const char* kind,auto index) {
        const auto x=m.GetCenter(index);out<<test<<','<<kind<<','<<index.raw();
        for(size_t d=0;d<3;++d) {
          if(!std::isfinite(x[d]))throw std::runtime_error("Nonfinite actual Cartesian center");
          out<<','<<x[d];
        }
        out<<'\n';
      };
      for(auto c:m.AllCells()){write("cell",c);++cells;}
      for(auto f:m.AllFaces()){write("face",f);++faces;}
      ++test;
    }
    out.flush();if(!out.good())throw std::runtime_error("Coordinate output failed");
    std::cout<<"{\"cases\":"<<test<<",\"cells\":"<<cells<<",\"faces\":"<<faces
             <<",\"scalar_bytes\":"<<sizeof(M::Scal)<<",\"scalar_digits\":"<<std::numeric_limits<M::Scal>::digits<<"}\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
