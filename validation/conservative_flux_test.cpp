#include "ConservativeFlux.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace simple;
using Vector=Eigen::VectorXd;
void emit(const char* op,FluxPair a,FluxPair b,FluxPair value) {
    std::cout<<op<<','<<std::hexfloat<<a.high<<','<<a.low<<','<<b.high<<','<<b.low<<','<<value.high<<','<<value.low<<'\n';
}
int main() {
    std::mt19937_64 generator(1788973634);
    auto sample=[&]() {return std::ldexp(double(int(generator()%2001)-1000)/1001.,int(generator()%401)-200);};
    for(int i=0;i<2048;++i) {
        const double ah=sample(),bh=sample();
        FluxPair a=FluxPair::sum(ah,std::ldexp(sample(),-54)),b=FluxPair::sum(bh,std::ldexp(sample(),-54));
        if(i%4==0)b={-a.high,std::ldexp(a.low,-1)};
        emit("add",a,b,a+b);
        const double factor=sample();emit("mul",a,{factor,0.},a*factor);
        if(factor!=0)emit("div",a,{factor,0.},a/factor);
    }
    Mesh mesh;mesh.cells.resize(4);mesh.faces.resize(3);
    mesh.cells[0].volume=6.2169322674883179e-24;
    for(int c=1;c<4;++c)mesh.cells[c].volume=1.;
    mesh.faces[0].owner=1;mesh.faces[0].neighbor=0;
    mesh.faces[1].owner=0;mesh.faces[1].neighbor=2;
    mesh.faces[2].owner=0;mesh.faces[2].neighbor=3;
    Vector values(3);values<<0x1.0a886eef392b0p-41,0x1.222e5a01b2915p-41,-0x1.7a5eb12796654p-45;
    Vector delta(3);delta<<-0x1.cbb7c5d68e354p-97,0x1.38f2f01fedb29p-96,-0x1.eced30b34cd0cp-100;
    ConservativeFlux flux(values,true);
    for(int j=0;j<3;++j) {
        const auto before=flux.get(j);flux.add(j,{delta[j],0.});emit("captured",before,{delta[j],0.},flux.get(j));
        if(values[j]+delta[j]!=values[j] || flux.get(j).low==0.)throw std::runtime_error("Captured lost-update case was not reproduced");
    }
    const auto balance=flux.negativeDivergence(mesh);
    for(int c=0;c<4;++c)emit("balance",{double(c),0.},{0.,0.},balance.get(c));
    if(!(std::abs(balance.quotient(0,mesh.cells[0].volume))<1e-8))throw std::runtime_error("Captured cell still fails original projection bound");
    Eigen::Matrix<double,3,1> face;face<<1.25,-.5,3.;
    const auto weighted=flux.negativeWeightedDivergence(mesh,face,0);
    for(int c=0;c<4;++c)emit("weighted",{double(c),0.},{0.,0.},weighted.get(c));
    Eigen::SparseMatrix<double,Eigen::RowMajor> redistribution(4,4);
    std::vector<Eigen::Triplet<double>> entries{{1,0,.25},{2,0,.75},{1,1,1.},{2,2,1.},{3,3,1.}};
    redistribution.setFromTriplets(entries.begin(),entries.end());
    const auto redistributed=weighted.multiplied(redistribution);
    for(int c=0;c<4;++c)emit("redistributed",{double(c),0.},{0.,0.},redistributed.get(c));
    Vector areas=Vector::Constant(3,2.384185791015625e-10);
    for(int j=0;j<3;++j)emit("quotient",flux.get(j),{areas[j],0.},{flux.quotient(j,areas[j]),0.});
    ConservativeFlux copy=flux;flux.setDouble(0,0.);flux=copy;
    if(flux.high!=copy.high || flux.low!=copy.low)throw std::runtime_error("Flux copy discarded low parts");
    ConservativeFlux legacy(values,false);legacy.subtractProduct(Vector::Ones(3),-delta);
    if(legacy.high!=values || legacy.low.size()!=0)throw std::runtime_error("Legacy double path changed");
}
