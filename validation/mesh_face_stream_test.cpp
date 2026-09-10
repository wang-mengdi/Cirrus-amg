#include "SimpleMesh.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>

using namespace simple;

namespace {
int commonChecks=0,streamChecks=0;
void check(bool value,const std::string& message) {
    if(!value)throw std::runtime_error(message);
}
Mesh cells(bool adaptive) {
    Mesh mesh;mesh.extent=Vec3::Ones();
    for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int k=0;k<4;++k) {
        const bool refined=adaptive&&i>=2;
        const int count=refined?2:1;
        for(int a=0;a<count;++a)for(int b=0;b<count;++b)for(int d=0;d<count;++d) {
            Cell cell;cell.level=refined?1:0;cell.h=refined?.125:.25;
            cell.key={count*i+a,count*j+b,count*k+d};
            for(int axis=0;axis<3;++axis)cell.center[axis]=(cell.key[axis]+.5)*cell.h;
            cell.volume=cell.h*cell.h*cell.h;mesh.cells.push_back(cell);
        }
    }
    std::sort(mesh.cells.begin(),mesh.cells.end(),[](const Cell& a,const Cell& b){
        return std::tie(a.level,a.key)<std::tie(b.level,b.key);
    });
    return mesh;
}
void sameFace(const Face& a,const Face& b) {
    check(a.owner==b.owner&&a.neighbor==b.neighbor&&a.axis==b.axis&&a.sign==b.sign&&
          a.area==b.area&&a.distance==b.distance&&a.boundary==b.boundary&&
          a.center==b.center&&a.delta==b.delta&&a.ownerOffset==b.ownerOffset&&
          a.neighborOffset==b.neighborOffset&&a.embeddedNormal==b.embeddedNormal,
          "Streamed face or face order changed");
}
void rejected(const std::string& name,const std::function<void()>& operation,const std::string& expected) {
    try {operation();}
    catch(const std::runtime_error& error) {
        check(std::string(error.what()).find(expected)!=std::string::npos,name+": wrong failure: "+error.what());
        std::cout<<"Rejected "<<name<<": "<<error.what()<<'\n';return;
    }
    throw std::runtime_error(name+": malformed geometry was accepted");
}
}

int main(int argc,char** argv) {
    try {
        check(argc==2,"Expected a fresh output directory");
        const std::filesystem::path out(argv[1]);
        check(!std::filesystem::exists(out),"Preserve previous mesh tests");
        std::filesystem::create_directories(out);
        for(bool adaptive:{false,true})for(bool px:{false,true})for(bool pz:{false,true}) {
            const auto name=std::string(adaptive?"adaptive":"uniform")+"_x"+std::to_string(px)+"_z"+std::to_string(pz);
            auto mesh=cells(adaptive);buildFacesAndValidate(mesh,px,pz);validateMesh(mesh);
            check((mesh.coarseFineFaces>0)==adaptive,"Expected actual coarse/fine interfaces");
            dumpMeshCsv(mesh,(out/name).string());++commonChecks;
#ifdef HAVE_FACE_VISITOR
            auto bare=cells(adaptive);std::size_t count=0;
            const int interfaces=visitFacesAndValidate(bare,px,pz,[&](const Face& face){
                check(bare.faces.empty(),"Streaming retained a background face array");
                sameFace(face,mesh.faces.at(count++));
            });
            check(count==mesh.faces.size()&&interfaces==mesh.coarseFineFaces,"Streamed topology count differs");
            ++streamChecks;
#endif
        }
        auto reference=cells(false);buildFacesAndValidate(reference,true,false);
        const auto internal=std::find_if(reference.faces.begin(),reference.faces.end(),[](const Face& f){return f.neighbor>=0;});
        check(internal!=reference.faces.end(),"Missing internal test face");
        const auto face=std::size_t(internal-reference.faces.begin());
        const auto malformed=[&](const std::string& name,const std::function<void(Mesh&)>& mutate,const std::string& expected){
            auto mesh=reference;mutate(mesh);rejected(name,[&]{validateMesh(mesh);},expected);++commonChecks;
        };
        malformed("owner",[&](Mesh& m){m.faces[face].owner=int(m.cells.size());},"invalid face owner");
        malformed("normal",[&](Mesh& m){m.faces[face].axis=3;},"invalid face normal");
        malformed("area",[&](Mesh& m){m.faces[face].area=0;},"invalid face geometry");
        malformed("distance",[&](Mesh& m){m.faces[face].distance*=2;},"distance differs");
        malformed("periodic offsets",[&](Mesh& m){m.faces[face].neighborOffset[1]+=.01;},"periodic-image offsets");
        malformed("2:1",[&](Mesh& m){m.cells[m.faces[face].neighbor].level=3;},"non-2:1 interface");
        malformed("coverage",[](Mesh& m){m.faces.pop_back();},"six-direction area coverage");
        malformed("Gauss volume",[&](Mesh& m){auto& f=m.faces[face];f.ownerOffset[f.axis]+=.01;f.neighborOffset[f.axis]+=.01;},"Gauss cell-volume closure");
        malformed("leaf volume",[](Mesh& m){m.cells[0].volume*=1.01;},"leaf volume differs");
        malformed("box fill",[](Mesh& m){m.extent[0]*=2;},"leaf volumes do not fill channel");
        malformed("interface count",[](Mesh& m){++m.coarseFineFaces;},"coarse/fine face count inconsistent");
        for(bool duplicate:{false,true}) {
            auto mesh=cells(false);
            if(duplicate)mesh.cells.push_back(mesh.cells[0]);else mesh.cells.pop_back();
            const std::string message=duplicate?"duplicate native leaf cell":"missing face neighbor";
            rejected(message,[&]{buildFacesAndValidate(mesh,true,false);},message);++commonChecks;
#ifdef HAVE_FACE_VISITOR
            rejected("streamed "+message,[&]{visitFacesAndValidate(mesh,true,false,[](const Face&){});},message);++streamChecks;
#endif
        }
#ifdef HAVE_FACE_VISITOR
        auto bad=cells(false);bad.cells[0].volume*=1.01;
        rejected("streamed discarded faces still validate",[&]{visitFacesAndValidate(bad,true,false,[](const Face&){});},"leaf volume differs");++streamChecks;
        rejected("empty visitor",[&]{visitFacesAndValidate(reference,true,false,{});},"missing face visitor");++streamChecks;
#endif
        std::ofstream report(out/"result.json");
        report<<"{\"passed\":true,\"common_checks\":"<<commonChecks<<",\"stream_checks\":"<<streamChecks<<"}\n";
        check(report.good(),"Failed test report write");return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
