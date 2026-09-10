// File-history algebra, integrity and storage tests. CFD acceptance is checked
// separately by full native trajectories and independent mass reconstruction.
#include "AndersonAcceleration.h"
#include <nlohmann/json.hpp>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using Vector=Eigen::VectorXd;
using Matrix=Eigen::MatrixXd;
using Json=nlohmann::json;
using AA=simple::AndersonAcceleration;
namespace fs=std::filesystem;
void require(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
template<class F> void rejects(F f,const char* message) {
    bool caught=false;try{f();}catch(const std::exception&){caught=true;}require(caught,message);
}
std::size_t files(const fs::path& path) {return std::distance(fs::directory_iterator(path),fs::directory_iterator());}

Json compareHistories(const fs::path& root) {
    Json cases=Json::array();
    for(const int block:{1,7,64,65536})for(const double scale:{1e-120,1.,1e120}) {
        const auto path=root/("pair_"+std::to_string(cases.size()));
        double candidateError=0.,coefficientError=0.;int accepted=0;
        {
            AA dense,disk;disk.useFileHistory(path,block);
            constexpr int n=1003;
            for(int step=0;step<16;++step) {
                Vector x(n),g(n);
                for(int i=0;i<n;++i) {
                    x[i]=scale*(.2*std::sin(.31*i+.17*step));
                    g[i]=x[i]+scale*(std::cos(.013*(i+1)*(step+2))+.02*step);
                }
                const Vector a=dense.update(x,g),b=disk.update(x,g);
                const auto& da=dense.diagnostics();const auto& db=disk.diagnostics();
                require(da.accepted==db.accepted&&da.proposed==db.proposed&&da.rank==db.rank&&
                    da.historySize==db.historySize&&da.reason==db.reason,"Dense/disk decisions differ");
                const double ce=((a-b)/scale).norm()/std::max(1.,(a/scale).norm());
                const double ge=std::abs(da.gammaNorm-db.gammaNorm)/std::max(1.,da.gammaNorm);
                candidateError=std::max(candidateError,ce);coefficientError=std::max(coefficientError,ge);
                require(ce<2e-10&&ge<2e-10,"Dense/disk numerical difference exceeds tolerance");
                require(disk.residentHistoryValues()==0,"Disk backend retained dense history");
                require(disk.storedHistoryValues()==dense.storedHistoryValues(),"History state was omitted");
                require(files(path)==std::min(step+1,6),"History eviction leaked files");
                accepted+=db.accepted;
            }
            require(accepted>=10,"Test did not exercise enough acceleration proposals");
            require(disk.storageStatistics().peakScratchValues<=std::max<Eigen::Index>(n,8*std::min(n,block)),"Unbounded block scratch");
            disk.reset();require(files(path)==0&&disk.storedHistoryValues()==0,"Reset did not clear history");
            disk.update(Vector::Zero(19),Vector::Ones(19));require(files(path)==1,"Reset was not reusable at a new size");
        }
        require(!fs::exists(path),"Destructor left owned history files");
        cases.push_back({{"block_values",block},{"scale",scale},{"candidate_relative_error",candidateError},
                         {"gamma_relative_error",coefficientError},{"accepted_proposals",accepted}});
    }
    return cases;
}

Json solveMap(const fs::path& root) {
    AA disk;disk.useFileHistory(root/"equation",7);
    constexpr int n=23;Matrix a=Matrix::Zero(n,n);
    for(int i=0;i<n;++i) {a(i,i)=2.1;if(i+1<n)a(i,i+1)=a(i+1,i)=-1.;}
    const Vector exact=Vector::LinSpaced(n,.3,1.7),rhs=a*exact;
    Vector x=Vector::Zero(n);double error=1.;int iteration=0;
    for(;iteration<5000;++iteration) {
        const Vector g=x+.24*(rhs-a*x);x=disk.update(x,g);
        error=(a*x-rhs).norm()/rhs.norm();if(error<1e-10)break;
    }
    require(error<1e-10&&(x-exact).norm()/exact.norm()<1e-8,"Disk accelerated true equation residual failed");
    disk.reset();Matrix b(2,5);b<<1.,2.,-.7,.3,1.1,-.2,.9,1.7,1.,-.4;
    Vector target(2);target<<.37,-.21;
    const Matrix inverse=(b*b.transpose()).inverse();
    const Matrix projector=Matrix::Identity(5,5)-b.transpose()*inverse*b;
    const Vector particular=b.transpose()*inverse*target,drive=Vector::LinSpaced(5,-.1,.4);
    x=particular;double constraint=0.;
    for(int i=0;i<30;++i) {
        const Vector g=particular+.83*projector*x+projector*drive;
        x=disk.update(x,g);constraint=std::max(constraint,(b*x-target).norm());
    }
    require(constraint<1e-12,"Disk candidate violated nonzero affine constraints");
    return {{"true_equation_relative_residual",error},{"iterations",iteration+1},{"affine_constraint_l2",constraint}};
}

Json integrity(const fs::path& root) {
    for(int mode=0;mode<3;++mode) {
        const auto path=root/("damage_"+std::to_string(mode));AA disk;disk.useFileHistory(path,7);
        Vector x=Vector::Zero(31),g=Vector::Ones(31);disk.update(x,g);
        const auto file=path/"entry_0.bin";
        if(mode==0)fs::resize_file(file,17);
        else {
            std::fstream f(file,std::ios::binary|std::ios::in|std::ios::out);
            const std::streamoff offset=(mode==1?0:31)*sizeof(double)+3;
            char byte;f.seekg(offset);f.read(&byte,1);require(bool(f),"Cannot prepare corruption test");
            byte^=1;f.seekp(offset);f.write(&byte,1);f.flush();require(bool(f),"Cannot mutate history test file");
        }
        g*=2.;rejects([&]{disk.update(x,g);},"Damaged history was used silently");
        disk.reset();require(files(path)==0,"Reset after failed IO leaked owned history");
        disk.update(x,g);require(files(path)==1,"Backend not reusable after IO failure/reset");
    }
    const auto existing=root/"existing";fs::create_directory(existing);
    const auto sentinel=existing/"sentinel.txt";{std::ofstream f(sentinel);f<<"keep";}
    {AA disk;rejects([&]{disk.useFileHistory(existing);},"Existing directory was adopted");}
    require(fs::file_size(sentinel)==4,"Existing content was changed");
    const auto foreign=root/"foreign";
    {AA disk;disk.useFileHistory(foreign);std::ofstream f(foreign/"keep.txt");f<<"keep";}
    require(fs::file_size(foreign/"keep.txt")==4,"Destructor removed unowned content");
    AA disk;disk.useFileHistory(root/"guards",7);
    Vector x=Vector::Zero(1),g=Vector::Ones(1);disk.update(x,g);g[0]+=1e-8;
    const Vector fallback=disk.update(x,g);
    require(!disk.accepted()&&disk.coefficientNorm()>1e6&&(fallback-g).norm()==0.,"Disk gamma safeguard failed");
    g[0]=std::numeric_limits<double>::quiet_NaN();rejects([&]{disk.update(x,g);},"Disk NaN guard failed");
    rejects([&]{disk.useFileHistory(root/"late");},"Late backend change accepted");
    disk.reset();x=Vector::Zero(2);g=Vector::Ones(2);disk.update(x,g);x.setOnes();g*=2.;
    require((disk.update(x,g)-g).norm()==0.&&!disk.accepted(),"Disk zero-rank fallback failed");
    disk.reset();disk.update(x,x);require(!disk.accepted(),"Exact fixed point accelerated");
    return {{"truncated_file_rejected",true},{"f_bit_corruption_rejected",true},{"g_bit_corruption_rejected",true},
            {"existing_and_foreign_files_preserved",true},{"reset_and_reuse",true},{"algebra_guards",true}};
}
} // namespace

int main(int argc,char** argv) {
    if(argc!=2) {std::cerr<<"Usage: anderson_file_test FRESH_OUTPUT_DIRECTORY\n";return 2;}
    const fs::path root(argv[1]);Json report{{"passed",false},{"scope","File-backed Anderson algebra and IO; separate CFD validation required"}};
    bool owned=false;
    try {
        require(fs::create_directories(root),"Test output directory must be fresh");
        owned=true;
        report["dense_comparison"]=compareHistories(root);
        report["equations"]=solveMap(root);report["integrity"]=integrity(root);report["passed"]=true;
    }catch(const std::exception& e) {report["error"]=e.what();}
    std::cout<<report.dump(2)<<'\n';
    // If creation failed, never overwrite a report in an existing directory.
    if(owned) {std::ofstream f(root/"result.json");f<<report.dump(2)<<'\n';if(!f) return 1;}
    return report["passed"].get<bool>()?0:1;
}
