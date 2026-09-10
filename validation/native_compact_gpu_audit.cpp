// Exact-geometry oracle for the native GPU compact operator. This executable
// does not run a flow solve or certify complete GPU AMG integration.
#include "NativeCompactGpu.h"
#include "EmbeddedOperators.h"
#include "QuadraticReconstruction.h"
#include "native_tile_metadata_audit.h"
#include "native_amg_topology_audit.h"
#include "native_host_storage_audit.h"
#include <Eigen/Sparse>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <random>

int main(int argc,char** argv) {
    try {
        if(argc!=3)throw std::runtime_error("Usage: native_compact_gpu_audit config.json fresh-output-directory");
        nlohmann::json config;std::ifstream(argv[1])>>config;
        const std::filesystem::path output=argv[2];
        if(std::filesystem::exists(output))throw std::runtime_error("Preserve previous audit; choose a fresh output directory");
        std::filesystem::create_directories(output);
        auto mesh=simple::makeEmbeddedOctree(config.at("embedded_geometry"),config.value("adaptive",false));
        if(config.value("audit_native_host_storage_only",false)) {
            const auto report=auditNativeHostStorage(mesh,output/"host_storage");
            std::ofstream(output/"host_storage_checks.json")<<report.dump(2)<<'\n';
            return report.at("passed").get<bool>()?0:2;
        }
        if(config.value("audit_offload_native_host_tiles",false))
            simple::offloadNativeHostTiles(mesh,(output/"native_host_backing").string());
        if(config.value("audit_amg_topology_only",false)||config.value("audit_amg_compact_only",false)) {
            const auto report=auditNativeAmgTopology(mesh,output,config.value("audit_complete_periodic_ghosts",false));
            std::ofstream(output/"topology_checks.json")<<report.dump(2)<<'\n';
            if(!report.at("passed").get<bool>())return 2;
            if(config.value("audit_amg_compact_only",false)) {
                const auto linear=auditNativeAmgCompactSolve(mesh,output);
                std::ofstream(output/"compact_checks.json")<<linear.dump(2)<<'\n';
            }
            return 0;
        }
        if(config.value("audit_metadata_restore",false)) {
            const auto metadata=auditNativeTileMetadata(mesh,output/"metadata",
                config.value("gpu_preconditioner",std::string("jacobi"))=="native_amg");
            std::ofstream(output/"metadata_checks.json")<<metadata.dump(2)<<'\n';
        }
        const int n=int(mesh.cells.size());
        using Sparse=Eigen::SparseMatrix<double>;using Vector=Eigen::VectorXd;using Triplet=Eigen::Triplet<double>;
        std::vector<Triplet> entries;Vector volume(n),walls=Vector::Zero(n),magnitude=Vector::Zero(n);
        for(int c=0;c<n;++c)volume[c]=mesh.cells[c].volume;
        const bool variable=config.value("audit_variable_coefficients",false);
        std::vector<double> factors;
        if(variable) {
            factors.resize(mesh.faces.size());
            for(size_t j=0;j<factors.size();++j)factors[j]=.8+.15*std::sin(6.283185307179586*mesh.faces[j].center[0]/mesh.extent[0]);
        }
        for(size_t j=0;j<mesh.faces.size();++j) {
            const auto& f=mesh.faces[j];const double weight=f.area/f.distance*(variable?factors[j]:1.);
            if(f.neighbor<0) {walls[f.owner]+=weight;continue;}
            entries.emplace_back(f.owner,f.owner,weight);entries.emplace_back(f.neighbor,f.neighbor,weight);
            entries.emplace_back(f.owner,f.neighbor,-weight);entries.emplace_back(f.neighbor,f.owner,-weight);
            magnitude[f.owner]+=2*weight;magnitude[f.neighbor]+=2*weight;
        }
        Sparse pressure(n,n);pressure.setFromTriplets(entries.begin(),entries.end());
        const double mu=config.value("rho",1.)*config.value("nu",.01);
        const double mass=config.value("rho",1.)/config.value("time_step",.001);
        Sparse diffusion=mu*pressure;
        for(int c=0;c<n;++c)diffusion.coeffRef(c,c)+=mass*volume[c]+mu*walls[c];
        simple::NativeCompactGpu gpu(mesh,factors);
        int gauge=0;magnitude.maxCoeff(&gauge);
        const bool nativeAmg=config.value("gpu_preconditioner",std::string("jacobi"))=="native_amg";
        if(nativeAmg)gpu.configureNativeAmg(mesh,gauge,mu,mass);
        const int krylovDimension=config.value("gpu_krylov_dimension",20);
        if(!nativeAmg&&krylovDimension!=20)throw std::runtime_error("Changing Krylov dimension requires native AMG");
        if(nativeAmg)gpu.setKrylovDimension(krylovDimension);
        const std::string orthogonalization=config.value("gpu_orthogonalization",std::string("mgs2"));
        if(orthogonalization!="mgs2"&&orthogonalization!="cgs2")throw std::runtime_error("Unknown GPU orthogonalization");
        if(orthogonalization=="cgs2")gpu.useBatchedCgs2();
        const bool meanZero=config.value("gpu_pressure_gauge",std::string("pin"))=="mean_zero";
        if(meanZero)gpu.useMeanZeroPressure();
        const bool fullPressure=config.value("gpu_pressure_operator",std::string("compact"))=="full";
        const bool fullViscosity=config.value("gpu_viscosity_operator",std::string("compact"))=="full";
        std::unique_ptr<simple::QuadraticReconstruction> quadratic;
        std::unique_ptr<simple::EmbeddedOperators> embedded;
        if(fullPressure||fullViscosity) {
            if(mesh.coarseFineFaces)quadratic=std::make_unique<simple::QuadraticReconstruction>(mesh);
            embedded=std::make_unique<simple::EmbeddedOperators>(mesh,quadratic.get(),false,false,false);
        }
        if(fullPressure) {
            if(!nativeAmg||!meanZero)throw std::runtime_error("Full-pressure audit requires native AMG and mean-zero pressure");
            const auto& op=*embedded;
            std::vector<simple::NativePressureFaceStencil> stencils;
            std::vector<Triplet> corrections;
            for(size_t j=0;j<mesh.faces.size();++j) {
                const auto& f=mesh.faces[j];
                if(f.neighbor<0||mesh.cells[f.owner].level==mesh.cells[f.neighbor].level)continue;
                const double area=f.area*(variable?factors[j]:1.),weight=area/f.distance;
                corrections.emplace_back(f.owner,f.owner,-weight);corrections.emplace_back(f.neighbor,f.neighbor,-weight);
                corrections.emplace_back(f.owner,f.neighbor,weight);corrections.emplace_back(f.neighbor,f.owner,weight);
                simple::NativePressureFaceStencil stencil{j,{}};
                for(simple::EmbeddedOperators::Sparse::InnerIterator it(op.faceGradient,int(j));it;++it) {
                    stencil.gradient.emplace_back(it.col(),it.value());
                    corrections.emplace_back(f.owner,it.col(),-area*it.value());
                    corrections.emplace_back(f.neighbor,it.col(),area*it.value());
                }
                stencils.push_back(std::move(stencil));
            }
            Sparse correction(n,n);correction.setFromTriplets(corrections.begin(),corrections.end());
            pressure+=correction;gpu.configurePressureInterfaces(mesh,stencils);
            if(gpu.fullPressureInterfaceFaces()!=size_t(mesh.coarseFineFaces))throw std::runtime_error("Full pressure interface coverage differs");
        }
        if(fullViscosity) {
            if(!nativeAmg)throw std::runtime_error("Full-viscosity audit requires native AMG");
            std::vector<simple::NativeViscosityFaceStencil> stencils;std::vector<Triplet> full;
            // Independently assemble ALL face gradients for the reference, not
            // just the custom-face subset uploaded to the native tile operator.
            for(size_t j=0;j<mesh.faces.size();++j) {
                const auto& f=mesh.faces[j];const double h=mesh.cells[f.owner].h;
                const bool custom=f.neighbor<0||mesh.cells[f.owner].level!=mesh.cells[f.neighbor].level||std::abs(f.area-h*h)>1e-12*h*h;
                const auto& gradient=f.neighbor<0?embedded->wallGradient:embedded->faceGradient;
                const double area=f.area*(variable?factors[j]:1.);
                simple::NativeViscosityFaceStencil stencil{j,{}};
                for(simple::EmbeddedOperators::Sparse::InnerIterator it(gradient,int(j));it;++it) {
                    full.emplace_back(f.owner,it.col(),-mu*area*it.value());
                    if(f.neighbor>=0)full.emplace_back(f.neighbor,it.col(),mu*area*it.value());
                    if(custom)stencil.gradient.emplace_back(it.col(),it.value());
                }
                if(custom)stencils.push_back(std::move(stencil));
            }
            for(int c=0;c<n;++c)full.emplace_back(c,c,mass*volume[c]);
            diffusion.setFromTriplets(full.begin(),full.end());
            gpu.configureViscosityFaces(mesh,stencils);
            if(gpu.fullViscosityFaces()!=stencils.size())throw std::runtime_error("Full viscosity coverage differs");
        }
        if(gpu.interfaceFaces()!=size_t(mesh.coarseFineFaces))throw std::runtime_error("Native interface coverage differs");
        bool passed=true;nlohmann::json tests=nlohmann::json::array();
        std::mt19937_64 random(7834621);std::uniform_real_distribution<double> uniform(-1.,1.);
        for(const std::string pattern: {"constant","periodic_smooth","random"}) {
            std::vector<double> values(n);
            for(int c=0;c<n;++c) {
                const auto& x=mesh.cells[c].center;
                values[c]=pattern=="constant"?1.:pattern=="random"?uniform(random):
                    std::sin(6.283185307179586*x[0]/mesh.extent[0])+.3*x[1]/mesh.extent[1]-.2*x[2]/mesh.extent[2];
            }
            const Vector x=Eigen::Map<Vector>(values.data(),n);
            for(bool viscosity:{false,true}) {
                const std::string kind=viscosity?"implicit_diffusion":"pressure";
                const auto actual=gpu.apply(values,viscosity?mu:1.,viscosity?mass:0.,viscosity);
                const Vector reference=viscosity?Vector(diffusion*x):Vector(pressure*x);
                const Vector result=Eigen::Map<const Vector>(actual.data(),n),error=result-reference;
                const double scale=viscosity?(mu*(magnitude+walls)+mass*volume).maxCoeff():magnitude.maxCoeff();
                const double relative=error.cwiseAbs().maxCoeff()/(scale*x.cwiseAbs().maxCoeff());
                const double volumeLinf=(error.array().abs()/volume.array()).maxCoeff();
                const bool nullspace=pattern=="constant"&&!viscosity;
                const double nullspaceError=nullspace?result.cwiseAbs().maxCoeff():0.;
                const bool ok=relative<1e-12&&(!nullspace||nullspaceError==0.);passed=passed&&ok;
                nlohmann::json test{{"pattern",pattern},{"operator",kind},{"passed",ok},
                    {"scaled_absolute_linf",relative},{"absolute_linf",error.cwiseAbs().maxCoeff()},
                    {"actual_cut_volume_error_linf",volumeLinf},
                    {"relative_l2",nullspace?nlohmann::json(nullptr):nlohmann::json(error.norm()/std::max(reference.norm(),1e-300))}};
                if(nullspace) {
                    test["gpu_constant_nullspace_linf"]=nullspaceError;
                    test["relative_l2_note"]="Exact result is zero; CPU sparse summation roundoff is not a meaningful relative-error denominator";
                }
                tests.push_back(test);
                std::ofstream dump(output/(pattern+"_"+kind+".csv"));
                dump<<std::setprecision(17)<<"id,input,cpu_explicit,gpu_matrix_free,difference\n";
                for(int c=0;c<n;++c)dump<<c<<','<<x[c]<<','<<reference[c]<<','<<result[c]<<','<<error[c]<<'\n';
                if(!dump.good())throw std::runtime_error("GPU audit dump failed");
            }
        }
        nlohmann::json linearTests=nlohmann::json::array();
        Vector known(n);for(int c=0;c<n;++c)known[c]=std::sin(6.283185307179586*mesh.cells[c].center[0]/mesh.extent[0])+.1*uniform(random);
        for(bool viscosity:{false,true}) {
            Vector exact=known;if(!viscosity)exact.array()-=meanZero?known.mean():known[gauge];
            Vector source=viscosity?Vector(diffusion*exact):Vector(pressure*exact);
            if(!viscosity&&!meanZero)source[gauge]=0.;
            const auto solved=gpu.solve(std::vector<double>(source.data(),source.data()+n),viscosity?mu:1.,viscosity?mass:0.,viscosity,viscosity?-1:gauge,1e-13);
            const Vector result=Eigen::Map<const Vector>(solved.values.data(),n);
            Vector tail=Vector::Zero(n);
            if(!solved.lowValues.empty())tail=Eigen::Map<const Vector>(solved.lowValues.data(),n);
            if(!viscosity&&meanZero&&solved.lowValues.size()!=size_t(n))throw std::runtime_error("Missing twofold pressure tail");
            Vector residual=source-(viscosity?Vector(diffusion*result):Vector(pressure*result+pressure*tail));if(!viscosity&&!meanZero)residual[gauge]=0.;
            const double relative=residual.norm()/source.norm(),error=((result-exact)+tail).norm()/exact.norm();
            const bool ok=relative<1e-12&&error<1e-8;passed=passed&&ok;
            linearTests.push_back({{"operator",viscosity?"implicit_diffusion":"pressure"},{"passed",ok},
                {"cpu_explicit_relative_residual",relative},{"known_solution_relative_l2",error},
                {"gpu_true_relative_residual",solved.relativeResidual},{"iterations",solved.iterations},
                {"residual_restarts",solved.restarts},{"seconds",solved.seconds}});
            linearTests.back()["orthogonalization_host_transfers"]=solved.orthogonalizationTransfers;
            linearTests.back()["compatibility_relative_l2"]=solved.compatibilityRelativeL2;
            linearTests.back()["original_rhs_checks"]=solved.originalRhsChecks;
            linearTests.back()["original_rhs_accepted"]=solved.originalRhsAccepted;
            linearTests.back()["solution_storage"]=solved.lowValues.empty()?"double":"twofold";
            if(config.value("audit_test_failed_solve_reuse",false)) {
                bool failedAsExpected=false;
                nlohmann::json failure;
                try {
                    gpu.solve(std::vector<double>(source.data(),source.data()+n),viscosity?mu:1.,viscosity?mass:0.,
                              viscosity,viscosity?-1:gauge,1e-13,1);
                } catch(const simple::NativeGpuConvergenceError& error) {
                    failedAsExpected=error.iterations==1&&error.viscosity==viscosity&&
                        error.tolerance==1e-13&&error.relativeResidual>error.tolerance;
                    failure={{"iterations",error.iterations},{"true_relative_residual",error.relativeResidual},
                             {"tolerance",error.tolerance},{"operator_matches",error.viscosity==viscosity}};
                }
                if(!failedAsExpected)throw std::runtime_error("Forced one-iteration solve did not give the expected typed failure");
                const auto recovered=gpu.solve(std::vector<double>(source.data(),source.data()+n),viscosity?mu:1.,viscosity?mass:0.,
                                                viscosity,viscosity?-1:gauge,1e-13);
                double difference=0;for(int c=0;c<n;++c)difference=std::max(difference,std::abs(recovered.values[c]-solved.values[c]));
                const bool reuseOk=difference<1e-12&&recovered.relativeResidual<=1e-13;
                failure["reuse_passed"]=reuseOk;failure["recovered_vs_original_linf"]=difference;
                failure["recovered_true_relative_residual"]=recovered.relativeResidual;
                linearTests.back()["forced_failure_and_reuse"]=failure;passed=passed&&reuseOk;
            }
            std::ofstream dump(output/(std::string("solve_")+(viscosity?"diffusion":"pressure")+".csv"));
            dump<<std::setprecision(17)<<"id,known,rhs,gpu_solution,cpu_residual,gpu_solution_low\n";
            for(int c=0;c<n;++c)dump<<c<<','<<exact[c]<<','<<source[c]<<','<<result[c]<<','<<residual[c]<<','<<tail[c]<<'\n';
            if(!dump.good())throw std::runtime_error("GPU solve audit dump failed");
        }
        nlohmann::json report{{"passed",passed},{"cells",n},{"coarse_fine_faces",mesh.coarseFineFaces},
            {"face_coefficients",variable?"0.8+0.15*sin(2*pi*x/extent_x), including coarse/fine and wall faces":"constant one"},
            {"device_operator_bytes",gpu.allocatedBytes()},{"tests",tests},{"linear_tests",linearTests},
            {"native_amg_levels",gpu.amgLevels()},{"linear_method",nativeAmg?"double_fgmres_native_float_amg":"double_pcg_jacobi"},
            {"gpu_orthogonalization",orthogonalization},
            {"gpu_krylov_dimension",nativeAmg?krylovDimension:0},
            {"pressure_gauge",meanZero?"mean_zero":"pin"},
            {"pressure_operator",fullPressure?"full":"compact"},
            {"full_pressure_interface_faces",gpu.fullPressureInterfaceFaces()},
            {"viscosity_operator",fullViscosity?"full":"compact"},{"full_viscosity_faces",gpu.fullViscosityFaces()},
            {"scope",fullViscosity?
                "Native HA full viscosity including cut and wall gradients, pressure mode recorded separately; Ax and manufactured solves versus independent all-face CPU assembly, not flow validation":fullPressure?
                "Native HA tile full pressure with coarse/fine face stencils and compact implicit viscosity; Ax and manufactured linear solves versus explicit CPU matrices, not full flow validation":
                "Native HA tile double-precision compact pressure and implicit-viscosity Ax and manufactured GPU linear solves versus explicit CPU matrices; not full flow validation"},
            {"gpu_storage",fullViscosity?"Native tile neighbor pointers with local coarse/fine, cut-face and wall stencils; no global CSR matrix":"Native tile neighbor pointers and double coefficient/field sidecars; explicit connectivity only for coarse/fine subfaces; no CSR matrix"},
            {"limitations",nativeAmg?"Native float FAS cycles are an approximate preconditioner; the double operator and true residual define the solution. Transport GPU integration and complete flow validation remain separate":"Full wall reconstruction and nonorthogonal deferred terms still require GPU integration; original float AMG is not selected; transfer timings are not solver performance evidence"}};
        std::ofstream(output/"audit.json")<<report.dump(2)<<'\n';
        std::cout<<report.dump(2)<<std::endl;return passed?0:1;
    } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;return 1;}
}
