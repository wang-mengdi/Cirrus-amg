// Compare actual reconstruction coefficients, fields, interface evaluation and guards against the frozen predecessor.
#include "QuadraticReconstruction.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <algorithm>
#include <functional>
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
void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
void dumpDerivatives(const simple::QuadraticReconstruction::Derivatives& value,
                     const simple::QuadraticReconstruction& reconstruction,size_t cells,
                     const std::filesystem::path& path) {
    require(value.gradient.size()==cells && value.hessian.size()==cells,"Derivative dimensions differ");
    std::ofstream out(path,std::ios::binary);const uint64_t size=cells;write(out,&size,1);
    for(size_t i=0;i<cells;++i) {
        if(!reconstruction.active(int(i)))require(value.gradient[i].isZero(0) && value.hessian[i].isZero(0),"Inactive derivative is nonzero");
        write(out,value.gradient[i].data(),3);write(out,value.hessian[i].data(),9);
    }
}
}
int main(int argc,char** argv) {
    try {
        require(argc==3,"Usage: quadratic_test mesh.bin output");
        const auto mesh=meshFrom(argv[1]);const int nc=int(mesh.cells.size());
        const std::filesystem::path out=argv[2];std::filesystem::create_directories(out);
        const simple::QuadraticReconstruction reconstruction(mesh);
        std::vector<bool> expected(nc,false);
        for(const auto& f:mesh.faces)if(f.neighbor>=0 && mesh.cells[f.owner].level!=mesh.cells[f.neighbor].level)
            expected[f.owner]=expected[f.neighbor]=true;
        int active=0,firstActive=-1,firstInactive=-1;
        std::ofstream coefficients(out/"coefficients.bin",std::ios::binary);
        for(int cell=0;cell<nc;++cell) {
            require(reconstruction.active(cell)==expected[cell],"Active cell IDs differ");
            if(!expected[cell]) {if(firstInactive<0)firstInactive=cell;continue;}
            if(firstActive<0)firstActive=cell;++active;
            for(int d=0;d<9;++d) {
                const auto row=reconstruction.derivativeRow(cell,d);
                const uint64_t record[]={uint64_t(cell),uint64_t(d),row.size()};write(coefficients,record,3);
                for(const auto& term:row) {const int32_t col=term.first;write(coefficients,&col,1);write(coefficients,&term.second,1);}
            }
        }
        require(active==reconstruction.activeCount() && !reconstruction.active(-1) && !reconstruction.active(nc),"Active count or range differs");
        int guards=0;
        const auto rejects=[&](const std::function<void()>& f) {
            bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}catch(const std::runtime_error&){caught=true;}
            require(caught,"Invalid reconstruction input accepted");++guards;
        };
        rejects([&]{reconstruction.derivativeRow(-1,0);});rejects([&]{reconstruction.derivativeRow(nc,0);});
        if(firstInactive>=0)rejects([&]{reconstruction.derivativeRow(firstInactive,0);});
        rejects([&]{reconstruction.evaluate(Eigen::VectorXd::Zero(nc+1));});
        rejects([&]{auto v=Eigen::VectorXd::Zero(nc).eval();v[0]=std::numeric_limits<double>::quiet_NaN();reconstruction.evaluate(v);});
        if(firstActive>=0) {
            rejects([&]{reconstruction.derivativeRow(firstActive,-1);});rejects([&]{reconstruction.derivativeRow(firstActive,9);});
            rejects([&]{reconstruction.evaluateManufactured([](const simple::Vec3&){return std::numeric_limits<double>::infinity();});});
        }
        const double scale=mesh.extent[1];Eigen::VectorXd probe(nc);
        for(int i=0;i<nc;++i) {
            const auto& x=mesh.cells[i].center;
            probe[i]=std::sin(6.283185307179586*x[0]/mesh.extent[0])+.23*x[1]/scale+.4*std::cos(3.7*x[2]/scale);
        }
        const auto derivatives=reconstruction.evaluate(probe);dumpDerivatives(derivatives,reconstruction,nc,out/"probe_derivatives.bin");
        std::ofstream faces(out/"face_probe.bin",std::ios::binary);int faceCount=0;
        for(int j=0;j<int(mesh.faces.size());++j) {
            const auto& f=mesh.faces[j];if(f.neighbor<0||mesh.cells[f.owner].level==mesh.cells[f.neighbor].level)continue;
            const int32_t id=j;const double q=simple::quadraticFaceValue(mesh,f,probe,derivatives),g=simple::quadraticFaceGradient(mesh,f,probe,derivatives);
            write(faces,&id,1);write(faces,&q,1);write(faces,&g,1);++faceCount;
        }
        double errorGradient=0,errorHessian=0;
        const int powers[10][2]={{-1,-1},{0,-1},{1,-1},{2,-1},{0,0},{1,1},{2,2},{0,1},{0,2},{1,2}};
        for(int k=0;k<10;++k) {
            const int a=powers[k][0],b=powers[k][1];
            const auto value=reconstruction.evaluateManufactured([&](const simple::Vec3& x){return a<0?1.:b<0?x[a]/scale:x[a]*x[b]/(scale*scale);});
            dumpDerivatives(value,reconstruction,nc,out/("polynomial_"+std::to_string(k)+".bin"));
            for(int i=0;i<nc;++i)if(expected[i]) {
                simple::Vec3 g=simple::Vec3::Zero();Eigen::Matrix3d h=Eigen::Matrix3d::Zero();
                if(a>=0 && b<0)g[a]=1/scale;
                if(b>=0) {g[a]+=mesh.cells[i].center[b]/(scale*scale);g[b]+=mesh.cells[i].center[a]/(scale*scale);h(a,b)+=1/(scale*scale);h(b,a)+=1/(scale*scale);}
                errorGradient=std::max(errorGradient,(value.gradient[i]-g).cwiseAbs().maxCoeff()*mesh.cells[i].h);
                errorHessian=std::max(errorHessian,(value.hessian[i]-h).cwiseAbs().maxCoeff()*mesh.cells[i].h*mesh.cells[i].h);
            }
        }
        require(errorGradient<1e-10 && errorHessian<1e-10,"Manufactured polynomial derivative failed");
        nlohmann::json result={{"passed",true},{"cells",nc},{"active_cells",active},{"coarse_fine_faces_tested",faceCount},{"derivative_rows_checked",active*9},{"manufactured_polynomials",10},{"scaled_gradient_error",errorGradient},{"scaled_hessian_error",errorHessian},{"invalid_inputs_rejected",guards},{"minimum_stencil_size",reconstruction.minimumStencilSize()},{"maximum_stencil_size",reconstruction.maximumStencilSize()},{"maximum_condition_number",reconstruction.maximumConditionNumber()}};
        std::ofstream(out/"checks.json")<<result.dump(2)<<'\n';std::cout<<result.dump()<<'\n';
    }catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
