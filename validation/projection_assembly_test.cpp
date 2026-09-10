// Compare the production assembler with the pre-change raw-triplet algorithm
// on retained native meshes and actual embedded operator dumps.
#include "ProjectionDiagnostics.h"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

using namespace simple;
using namespace simple::projectionAssembly;
using Triplet=Eigen::Triplet<double>;
namespace {
void require(bool value,const std::string& message) {
    if(!value)throw std::runtime_error(message);
}
template<class A,class B> void identical(const A& a,const B& b,const std::string& name) {
    require(a.isCompressed()&&b.isCompressed(),name+": not compressed");
    require(a.rows()==b.rows()&&a.cols()==b.cols()&&a.nonZeros()==b.nonZeros(),name+": shape/support differs");
    require(!std::memcmp(a.outerIndexPtr(),b.outerIndexPtr(),(a.outerSize()+1)*sizeof(int)),name+": outer indices differ");
    if(a.nonZeros()) {
        require(!std::memcmp(a.innerIndexPtr(),b.innerIndexPtr(),a.nonZeros()*sizeof(int)),name+": inner indices differ");
        require(!std::memcmp(a.valuePtr(),b.valuePtr(),a.nonZeros()*sizeof(double)),name+": coefficient bits differ");
    }
    std::cout<<"Identical "<<name<<" rows="<<a.rows()<<" columns="<<a.cols()<<" entries="<<a.nonZeros()<<'\n';
}
void diagnostics(const Sparse& d,const Eigen::VectorXd& area,const Sparse& gradient,const Sparse& expected) {
    Sparse actual(expected.rows(),expected.cols());actual.reserve(expected.nonZeros());
    pressureColumns(d,area,gradient,[&](const Row& column) {
        actual.startVec(column.face);
        for(const auto& term:column.terms)actual.insertBack(term.first,column.face)=term.second;
    });
    actual.finalize();identical(expected,actual,"streamed_pressure_columns");
    const Eigen::VectorXd diagonal=expected.diagonal();
    for(int mode=0;mode<4;++mode) {
        Eigen::VectorXd probe;
        if(mode) {
            probe.resize(expected.cols());
            for(int i=0;i<probe.size();++i)probe[i]=mode==1?0.:mode==2?1.:.03*std::sin(.17*i)+.02*(i%13);
        }
        const auto now=pressureDiagnostics(d,area,gradient,probe);
        require(!std::memcmp(diagonal.data(),now.diagonal.data(),size_t(diagonal.size())*sizeof(double)),"Pressure diagonal bytes differ");
        if(diagonal.size()) {
            Eigen::Index a,b;diagonal.maxCoeff(&a);now.diagonal.maxCoeff(&b);require(a==b,"Pressure gauge differs");
        }
        if(mode) {
            const Eigen::VectorXd image=expected*probe;
            require(!std::memcmp(image.data(),now.image.data(),size_t(image.size())*sizeof(double)),"Pressure image bytes differ");
        } else require(now.image.size()==0,"Empty probe unexpectedly allocated");
    }
    std::cout<<"PASS diagnostic diagonal, gauge, zero, constant and nonconstant images\n";
}
struct Operators {Sparse divergence,gradient,correction;Rows owner,neighbor;};
Operators legacy(const Mesh& mesh,const Rows& gradient,const QuadraticReconstruction* quadratic) {
    const int nc=int(mesh.cells.size()),nf=int(mesh.faces.size());
    std::vector<Triplet> bt,nt,cf,taylor[2];
    for(int j=0;j<nf;++j) {
        const auto& f=mesh.faces[j];if(f.neighbor<0)continue;
        bt.emplace_back(f.owner,j,1);bt.emplace_back(f.neighbor,j,-1);
        nt.emplace_back(j,f.owner,-1/f.distance);nt.emplace_back(j,f.neighbor,1/f.distance);
        if(mesh.cells[f.owner].level==mesh.cells[f.neighbor].level)continue;
        for(Rows::InnerIterator it(gradient,j);it;++it)cf.emplace_back(j,it.col(),it.value());
        cf.emplace_back(j,f.owner,1/f.distance);cf.emplace_back(j,f.neighbor,-1/f.distance);
        for(int side=0;side<2;++side) {
            const int c=side?f.neighbor:f.owner;const Vec3 r=side?f.neighborOffset:f.ownerOffset;
            taylor[side].emplace_back(j,c,1);
            const double w[9]={r[0],r[1],r[2],.5*r[0]*r[0],.5*r[1]*r[1],.5*r[2]*r[2],r[0]*r[1],r[0]*r[2],r[1]*r[2]};
            for(int d=0;d<9;++d)for(const auto& term:quadratic->derivativeRow(c,d))
                taylor[side].emplace_back(j,term.first,w[d]*term.second);
        }
    }
    Operators out;
    out.divergence.resize(nc,nf);out.divergence.setFromTriplets(bt.begin(),bt.end());
    out.gradient.resize(nf,nc);out.gradient.setFromTriplets(nt.begin(),nt.end());
    out.correction.resize(nf,nc);out.correction.setFromTriplets(cf.begin(),cf.end());
    out.owner.resize(nf,nc);out.owner.setFromTriplets(taylor[0].begin(),taylor[0].end());
    out.neighbor.resize(nf,nc);out.neighbor.setFromTriplets(taylor[1].begin(),taylor[1].end());
    return out;
}
void compare(const Mesh& mesh,const Rows& gradient,const Rows& interpolation) {
    std::unique_ptr<QuadraticReconstruction> quadratic;
    if(mesh.coarseFineFaces)quadratic=std::make_unique<QuadraticReconstruction>(mesh);
    auto old=legacy(mesh,gradient,quadratic.get());Operators now;
    assemble(mesh,gradient,quadratic.get(),now.divergence,now.gradient,now.correction,now.owner,now.neighbor);
    identical(old.divergence,now.divergence,"divergence");
    identical(old.gradient,now.gradient,"compact_gradient");
    identical(old.correction,now.correction,"interface_correction");
    identical(old.owner,now.owner,"taylor_owner");identical(old.neighbor,now.neighbor,"taylor_neighbor");
    const Eigen::VectorXd constant=interpolation*Eigen::VectorXd::Ones(mesh.cells.size());
    Eigen::VectorXd area(mesh.faces.size());
    for(int j=0;j<area.size();++j) {
        area[j]=mesh.faces[j].area;
        if(mesh.faces[j].neighbor>=0&&std::abs(constant[j]-1)>1e-12)area[j]*=constant[j];
    }
    const Sparse po=-old.divergence*area.asDiagonal()*old.gradient;
    const Sparse pn=-now.divergence*area.asDiagonal()*now.gradient;identical(po,pn,"compact_pressure");
    const Sparse co=-old.divergence*area.asDiagonal()*old.correction;
    const Sparse cn=-now.divergence*area.asDiagonal()*now.correction;identical(co,cn,"deferred_pressure");
    diagnostics(now.divergence,area,now.gradient,pn);
    diagnostics(now.divergence,area,now.correction,cn);
    Eigen::VectorXd probe(mesh.cells.size());
    for(int i=0;i<probe.size();++i)probe[i]=.03*std::sin(.17*i)+.02*(i%13);
    const auto compact=pressureDiagnostics(now.divergence,area,now.gradient,probe);
    const auto deferred=pressureDiagnostics(now.divergence,area,now.correction,probe);
    const Eigen::VectorXd expected=(pn*probe+cn*probe)/1.7;
    const Eigen::VectorXd image=(compact.image+deferred.image)/1.7;
    require(!std::memcmp(expected.data(),image.data(),size_t(image.size())*sizeof(double)),"Combined pressure probe bytes differ");
    old.gradient+=old.correction;now.gradient+=now.correction;identical(old.gradient,now.gradient,"full_gradient");
}
template<class Callback> void csv(const std::filesystem::path& path,const std::string& header,Callback visit) {
    std::ifstream in(path);require(bool(in),"Cannot read "+path.string());std::string line;
    require(bool(std::getline(in,line)),"Empty CSV");if(!line.empty()&&line.back()=='\r')line.pop_back();
    require(line==header,"Unexpected header in "+path.string());
    while(std::getline(in,line)) {
        std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);visit(row);
        require(!row.fail(),"Malformed CSV row in "+path.string());row>>std::ws;
        require(row.eof(),"Extra CSV columns in "+path.string());
    }
    require(in.eof(),"CSV read failure");
}
Mesh loadMesh(const std::filesystem::path& path) {
    Mesh mesh;
    csv(path/"mesh_cells.csv","id,level,i,j,k,x,y,z,h,volume",[&](std::istream& row){
        int id;Cell c;row>>id>>c.level>>c.key[0]>>c.key[1]>>c.key[2]>>c.center[0]>>c.center[1]>>c.center[2]>>c.h>>c.volume;
        require(id==int(mesh.cells.size()),"Cell IDs are not ordered");mesh.cells.push_back(c);
    });
    csv(path/"mesh_faces.csv","id,owner,neighbor,axis,sign,boundary,area,distance,x,y,z,dx,dy,dz,owner_dx,owner_dy,owner_dz,neighbor_dx,neighbor_dy,neighbor_dz",[&](std::istream& row){
        int id;Face f;row>>id>>f.owner>>f.neighbor>>f.axis>>f.sign>>f.boundary>>f.area>>f.distance;
        for(auto* v:{&f.center,&f.delta,&f.ownerOffset,&f.neighborOffset})for(int d=0;d<3;++d)row>>(*v)[d];
        require(id==int(mesh.faces.size()),"Face IDs are not ordered");
        require(f.owner>=0&&f.owner<int(mesh.cells.size())&&f.neighbor<int(mesh.cells.size()),"Invalid owner/neighbor");
        if(f.neighbor>=0&&mesh.cells[f.owner].level!=mesh.cells[f.neighbor].level)++mesh.coarseFineFaces;
        mesh.faces.push_back(f);
    });
    return mesh;
}
Rows loadMatrix(const std::filesystem::path& path,const Mesh& mesh) {
    std::vector<Triplet> terms;
    csv(path,"row,column,value",[&](std::istream& row){
        int i,j;double value;row>>i>>j>>value;
        require(i>=0&&i<int(mesh.faces.size())&&j>=0&&j<int(mesh.cells.size()),"Invalid matrix index");
        terms.emplace_back(i,j,value);
    });
    Rows result(mesh.faces.size(),mesh.cells.size());result.setFromTriplets(terms.begin(),terms.end());return result;
}
void edgeCases() {
    // Detect reassociation, loss of explicit zero entries and negative zero.
    std::vector<Row> rows{{1,{}},{4,{}}};std::vector<Triplet> terms;
    auto add=[&](size_t r,int col,double value){rows[r].add(col,value);terms.emplace_back(rows[r].face,col,value);};
    add(0,3,1e16);add(0,3,1);add(0,3,-1e16);add(0,0,-0.);add(0,2,0.);
    add(1,2,-2);add(1,0,5);add(1,2,2);add(1,1,-0.);add(1,1,-0.);
    Rows old(7,5);old.setFromTriplets(terms.begin(),terms.end());
    const auto now=consumeRows(rows,7,5);identical(old,now,"ordered_duplicate_and_zero_rows");
    require(rows.empty(),"Assembly rows were retained");
    require(std::signbit(now.coeff(1,0)),"Negative zero was lost");
    Mesh mesh;mesh.cells.resize(3);
    Face f;f.owner=2;f.neighbor=0;f.distance=.25;f.area=.0625;mesh.faces.push_back(f);
    f.owner=1;f.neighbor=1;mesh.faces.push_back(f);
    f.neighbor=-1;mesh.faces.push_back(f);
    Rows gradient(3,3),interpolation(3,3);interpolation.setIdentity();compare(mesh,gradient,interpolation);
    mesh.faces.clear();Rows empty(0,3);compare(mesh,empty,empty);
    bool overflow=false;try {checkedCount(size_t(std::numeric_limits<int>::max())+1);}catch(const std::length_error&) {overflow=true;}
    require(overflow,"Sparse count overflow was accepted");
    int rejected=0;
    try {pressureDiagnostics(Sparse(2,3),Eigen::VectorXd::Zero(2),Sparse(3,2),Eigen::VectorXd());}
    catch(const std::invalid_argument&) {++rejected;}
    try {pressureDiagnostics(Sparse(2,3),Eigen::VectorXd::Zero(3),Sparse(2,2),Eigen::VectorXd());}
    catch(const std::invalid_argument&) {++rejected;}
    try {pressureDiagnostics(Sparse(2,3),Eigen::VectorXd::Zero(3),Sparse(3,1),Eigen::VectorXd());}
    catch(const std::invalid_argument&) {++rejected;}
    try {pressureDiagnostics(Sparse(2,3),Eigen::VectorXd::Zero(3),Sparse(3,2),Eigen::VectorXd::Zero(1));}
    catch(const std::invalid_argument&) {++rejected;}
    require(rejected==4,"Invalid pressure diagnostic dimensions were accepted");
}
}
int main(int argc,char** argv) {
    try {
        require(argc>=2,"Expected one or more retained operators directories");edgeCases();
        for(int i=1;i<argc;++i) {
            const std::filesystem::path path(argv[i]);const auto mesh=loadMesh(path);
            const auto gradient=loadMatrix(path/"face_gradient.csv",mesh);
            const auto interpolation=loadMatrix(path/"interpolation.csv",mesh);
            std::cout<<"Case "<<path.string()<<" cells="<<mesh.cells.size()<<" faces="<<mesh.faces.size()<<" interfaces="<<mesh.coarseFineFaces<<'\n';
            compare(mesh,gradient,interpolation);
        }
        std::cout<<"PASS: all stored indices and coefficient bits match the legacy assembler\n";
    }catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
