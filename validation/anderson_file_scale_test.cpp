// Full-size history with a repeated eight-component oracle. This tests storage
// scaling without allocating the equivalent multi-gigabyte dense history.
#include "AndersonAcceleration.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc,char** argv) {
    using Vector=Eigen::VectorXd;namespace fs=std::filesystem;
    if(argc!=3)return 2;
    const fs::path root(argv[1]);const Eigen::Index n=std::stoll(argv[2]);
    nlohmann::json report{{"passed",false},{"state_values",n},{"scope","Complete file-history storage with eight-component repeated oracle; no physical CFD claim"}};
    bool owned=false;
    try {
        if(n<8||n%8||!fs::create_directories(root))throw std::runtime_error("Expected a fresh output and positive multiple of eight");
        owned=true;
        {
            simple::AndersonAcceleration oracle,disk;disk.useFileHistory(root/"history");
            Vector x(n),g(n),smallX(8),smallG(8);double error=0.,gammaError=0.;int accepted=0;
            for(int step=0;step<7;++step) {
                for(int i=0;i<8;++i) {
                    smallX[i]=.2*std::sin(.17*i+.31*step);
                    smallG[i]=smallX[i]+std::cos(.37*(i+1)*(step+1));
                }
                for(Eigen::Index i=0;i<n;++i){x[i]=smallX[i%8];g[i]=smallG[i%8];}
                const Vector expected=oracle.update(smallX,smallG),candidate=disk.update(x,g);
                if(oracle.accepted()!=disk.accepted()||oracle.diagnostics().rank!=disk.diagnostics().rank)
                    throw std::runtime_error("Full-size decisions differ from oracle");
                for(Eigen::Index i=0;i<n;++i)error=std::max(error,std::abs(candidate[i]-expected[i%8]));
                gammaError=std::max(gammaError,std::abs(oracle.coefficientNorm()-disk.coefficientNorm())/std::max(1.,oracle.coefficientNorm()));
                if(error>=1e-9||gammaError>=1e-9||disk.residentHistoryValues()!=0)
                    throw std::runtime_error("Full-size numerical or resident-history check failed");
                accepted+=disk.accepted();
                std::cout<<"completed_update="<<step+1<<" stored_values="<<disk.storedHistoryValues()<<std::endl;
            }
            const auto& stats=disk.storageStatistics();
            if(disk.storedHistoryValues()!=12*n||stats.peakScratchValues>std::max<Eigen::Index>(n,8*65536)||accepted!=6)
                throw std::runtime_error("Incomplete full history or unbounded scratch");
            std::uintmax_t bytes=0;for(const auto& file:fs::directory_iterator(root/"history"))bytes+=file.file_size();
            if(bytes!=12*std::uintmax_t(n)*sizeof(double))throw std::runtime_error("History disk length differs");
            report.update({{"history_depth",5},{"retained_pairs",6},{"complete_history_bytes",bytes},
                {"resident_history_values",disk.residentHistoryValues()},{"peak_scratch_values",stats.peakScratchValues},
                {"written_bytes",stats.writtenBytes},{"read_bytes",stats.readBytes},
                {"maximum_candidate_absolute_error",error},{"maximum_gamma_relative_error",gammaError}});
        }
        if(fs::exists(root/"history"))throw std::runtime_error("Scratch history not cleaned");
        report["passed"]=true;
    }catch(const std::exception& e){report["error"]=e.what();}
    std::cout<<report.dump(2)<<std::endl;
    if(owned){std::ofstream f(root/"result.json");f<<report.dump(2)<<'\n';if(!f)return 1;}
    return report["passed"].get<bool>()?0:1;
}
