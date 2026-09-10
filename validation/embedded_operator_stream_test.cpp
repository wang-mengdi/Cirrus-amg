// Read a serialized, previously emitted native cut mesh without allocating a
// second GPU grid. Link this same driver against old and new EmbeddedOperators
// and compare every compressed coefficient byte, in addition to its analytic
// and conservation checks. This driver does not solve a flow.
#include "EmbeddedOperators.h"
#include "QuadraticReconstruction.h"
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
template<class T> T read(std::istream& input) {
    T value{};input.read(reinterpret_cast<char*>(&value),sizeof(value));
    if(!input)throw std::runtime_error("Truncated native mesh fixture");
    return value;
}
template<class T> void write(std::ostream& out,const T* values,size_t count) {
    if(!count)return;
    out.write(reinterpret_cast<const char*>(values),sizeof(T)*count);
    if(!out)throw std::runtime_error("Coefficient dump failed");
}
simple::Vec3 vector(std::istream& input) {
    simple::Vec3 v;for(int d=0;d<3;++d)v[d]=read<double>(input);
    if(!v.allFinite())throw std::runtime_error("Nonfinite fixture vector");return v;
}
simple::Mesh meshFrom(const std::filesystem::path& path) {
    std::ifstream in(path,std::ios::binary);char magic[8]{};in.read(magic,8);
    if(std::string(magic,8)!="CIRRMESH")throw std::runtime_error("Wrong native mesh fixture");
    const auto nc=read<uint64_t>(in),nf=read<uint64_t>(in);
    if(!nc||!nf||nc>INT32_MAX||nf>INT32_MAX)throw std::runtime_error("Invalid fixture size");
    simple::Mesh mesh;mesh.extent=vector(in);mesh.periodicZ=read<uint8_t>(in)!=0;
    mesh.embedded=true;mesh.backend="saved-native-mesh";
    mesh.cells.resize(nc);mesh.faces.resize(nf);
    for(auto& cell:mesh.cells) {
        cell.center=vector(in);cell.h=read<double>(in);cell.volume=read<double>(in);
        cell.level=read<int32_t>(in);cell.cut=read<uint8_t>(in)!=0;
        for(auto& k:cell.key)k=read<int32_t>(in);
        if(!(cell.h>0&&cell.volume>0&&std::isfinite(cell.volume)))throw std::runtime_error("Invalid fixture cell");
    }
    for(auto& face:mesh.faces) {
        face.owner=read<int32_t>(in);face.neighbor=read<int32_t>(in);face.axis=read<int32_t>(in);
        face.sign=read<double>(in);face.area=read<double>(in);face.distance=read<double>(in);
        face.center=vector(in);face.delta=vector(in);face.ownerOffset=vector(in);
        face.neighborOffset=vector(in);face.embeddedNormal=vector(in);face.boundary=read<int32_t>(in);
        if(face.owner<0||face.owner>=int(nc)||face.neighbor>=int(nc)||!(face.area>0&&face.distance>0))
            throw std::runtime_error("Invalid fixture face");
        if(face.neighbor>=0&&mesh.cells[face.owner].level!=mesh.cells[face.neighbor].level)++mesh.coarseFineFaces;
    }
    if(in.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Trailing fixture bytes");
    return mesh;
}
void dump(const simple::EmbeddedOperators::Sparse& matrix,const std::filesystem::path& path) {
    if(!matrix.isCompressed())throw std::runtime_error("Uncompressed operator");
    const uint64_t shape[]={uint64_t(matrix.rows()),uint64_t(matrix.cols()),uint64_t(matrix.nonZeros())};
    std::ofstream out(path,std::ios::binary);write(out,shape,3);
    // Default-constructed optional operators have no allocated outer array.
    const int empty=0;
    write(out,matrix.outerIndexPtr()?matrix.outerIndexPtr():&empty,size_t(matrix.rows())+1);
    write(out,matrix.innerIndexPtr(),size_t(matrix.nonZeros()));
    write(out,matrix.valuePtr(),size_t(matrix.nonZeros()));
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=6)throw std::runtime_error("Usage: operator_test mesh.bin output convection explicit redistribution");
        const bool convection=std::stoi(argv[3])!=0,explicitUpdate=std::stoi(argv[4])!=0,redistribute=std::stoi(argv[5])!=0;
        const std::filesystem::path out=argv[2];std::filesystem::create_directories(out);
        const auto mesh=meshFrom(argv[1]);
        std::unique_ptr<simple::QuadraticReconstruction> quadratic;
        if(mesh.coarseFineFaces)quadratic=std::make_unique<simple::QuadraticReconstruction>(mesh);
        const simple::EmbeddedOperators op(mesh,quadratic.get(),convection,explicitUpdate,redistribute);
        op.check(mesh,(out/"checks.json").string());
        const auto save=[&](const auto& matrix,const char* name){dump(matrix,out/(std::string(name)+".bin"));};
        save(op.interpolation,"interpolation");save(op.faceGradient,"face_gradient");save(op.wallGradient,"wall_gradient");
        save(op.compactDiffusion,"compact_diffusion");save(op.deferredDiffusion,"deferred_diffusion");
        save(op.cartesianFaceInterpolation,"face_mix");save(op.redistribution,"redistribution");
        for(int d=0;d<3;++d)save(op.cellGradient[d],("cell_gradient_"+std::to_string(d)).c_str());
        std::ofstream weights(out/"boundary_weights.bin",std::ios::binary);
        write(weights,op.interpolationBoundaryWeight.data(),size_t(op.interpolationBoundaryWeight.size()));
        if(convection) {
            Eigen::VectorXd flux(mesh.faces.size());
            for(int j=0;j<flux.size();++j)flux[j]=mesh.faces[j].neighbor<0?0:std::sin(j*.731)*mesh.faces[j].area;
            const auto advection=op.upwindAdvection(mesh,flux,1.);
            save(advection.first,"advection_compact");save(advection.second,"advection_deferred");
        }
        std::cout<<"Checked native mesh: "<<mesh.cells.size()<<" cells, "<<mesh.faces.size()<<" faces, "
                 <<mesh.coarseFineFaces<<" coarse/fine faces\n";
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
