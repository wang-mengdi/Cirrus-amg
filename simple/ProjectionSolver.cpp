#include "ProjectionSolver.h"
#include "AmgPressure.h"
#include "AndersonAcceleration.h"
#include "NativeCompactGpu.h"
#include "PressureRoundoffCycle.h"
#include "ConstructionMemory.h"
#include "ProjectionAssembly.h"
#include "ProjectionDiagnostics.h"
#include "ConservativeFlux.h"
#include "ProjectionFluxTrace.h"
#include <Eigen/SparseCholesky>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>

namespace simple {
namespace {
using Sparse=Eigen::SparseMatrix<double>;
using Rows=Eigen::SparseMatrix<double,Eigen::RowMajor>;
using Field=Eigen::Matrix<double,Eigen::Dynamic,3>;
using Vector=Eigen::VectorXd;
using Flux=ConservativeFlux;
using Triplet=Eigen::Triplet<double>;
void checked(std::ofstream& stream) {
    stream.flush();if(!stream.good())throw std::runtime_error("Projection output write failed");
}
struct Linear {
    Sparse matrix;
    Eigen::SimplicialLDLT<Sparse> direct;
    AmgPressure amg;
    bool useAmg=false;
    void compute(const Sparse& input,bool use,double tolerance) {
        matrix=input;useAmg=use;
        if(useAmg)amg.compute(matrix,tolerance);
        else {direct.compute(matrix);if(direct.info()!=Eigen::Success)throw std::runtime_error("Projection factorization failed");}
    }
    Vector solve(const Vector& rhs) {
        Vector answer=useAmg?amg.solve(rhs):Vector(direct.solve(rhs));
        if(!answer.allFinite())throw std::runtime_error("Nonfinite projection linear solution");
        return answer;
    }
};
}

struct ProjectionSolver::Impl {
    Mesh mesh;Options options;
    std::unique_ptr<EmbeddedOperators> op;
    int nc=0,nf=0,gauge=0;
    Vector volumes,areas,mass,faceForce,pressure,sideArea,materialWeight,pressureAreas;
    Flux flux;
    std::vector<int> materialFaces;
    std::vector<unsigned char> flagged;
    Field velocity,diffusionGuess;
    Sparse divergence,normalGradient,pressureMatrix,pressureDeferred,diffusionMatrix,diffusionDeferred;
    bool storedFullDiffusion=false;
    bool andersonFileHistory=false;
    Rows cfTaylorOwner,cfTaylorNeighbor;
    Linear pressureLinear,diffusionLinear;
    std::unique_ptr<NativeCompactGpu> gpu;
    std::ofstream gpuTrace;
    int gpuCalls=0;
    double volume=0,accelerationScale=1;
    int maximumPressurePasses=0;
    int roundoffPressureExits=0;
    int cyclePressureExits=0;
    bool cycleExitEnabled=true;
    int pressureCalls=0;
    std::ofstream pressureTrace;
    std::ofstream pressureUpdateTrace,pressureFloorFaces;
    std::ofstream pressureRoundoffTrace;
    std::ofstream pressureCycleTrace;
    int acceleratedSteps=0,rejectedAccelerations=0;
    int activeStep=0,activeIteration=0,activeTrial=0,linearFailures=0;
    const char* linearPhase="initialization";
    int restartStep=0;
    nlohmann::json restart;

    Impl(Mesh m,Options o):mesh(std::move(m)),options(std::move(o)) {
        if(const char* value=std::getenv("SIMPLE_ANDERSON_FILE_HISTORY")) {
            if(std::string(value)!="0"&&std::string(value)!="1")
                throw std::invalid_argument("SIMPLE_ANDERSON_FILE_HISTORY must be 0 or 1");
            andersonFileHistory=std::string(value)=="1";
        }
        constructionMemory("projection.begin");
        nc=int(mesh.cells.size());nf=int(mesh.faces.size());
        if(!mesh.embedded||!nc)throw std::runtime_error("Projection requires an embedded fluid mesh");
        if(const char* value=std::getenv("SIMPLE_NATIVE_HOST_TILES_FILE")) {
            if(std::string(value)!="0"&&std::string(value)!="1")
                throw std::invalid_argument("SIMPLE_NATIVE_HOST_TILES_FILE must be 0 or 1");
            if(std::string(value)=="1")offloadNativeHostTiles(mesh,(options.output/"native_host_backing").string());
        }
        std::unique_ptr<QuadraticReconstruction> quadratic;
        if(mesh.coarseFineFaces) {
            if(!options.quadraticInterfaces)throw std::runtime_error("Projection interfaces require quadratic reconstruction");
            quadratic=std::make_unique<QuadraticReconstruction>(mesh);
        }
        // The original Aphros Proj implicit-diffusion path does not redistribute
        // its deferred viscosity. Advection still uses the original R operator.
        constructionMemory("projection.quadratic_ready");
        op=std::make_unique<EmbeddedOperators>(mesh,quadratic.get(),false,false,false);
        constructionMemory("projection.embedded_ready");
        op->check(mesh,(options.output/"embedded_operator_checks.json").string());
        constructionMemory("projection.embedded_checked");
        volumes.resize(nc);areas.resize(nf);mass.resize(nc);faceForce=Vector::Zero(nf);
        materialWeight=Vector::Ones(nf);pressureAreas.resize(nf);
        for(int c=0;c<nc;++c) {
            volumes[c]=mesh.cells[c].volume;mass[c]=options.rho*volumes[c]/options.timeStep;
        }
        volume=volumes.sum();accelerationScale=std::max(options.force.norm(),1e-30);
        sideArea=Vector::Zero(nc*6);flagged.assign(nc*6,0);
        Vector materialProbe,materialPressureOperator;
        // Material diagnostics need the side areas, but the evolving flow
        // fields are not used until construction and GPU setup have finished.
        // Release the interpolation probe immediately after its last use.
        {
            const Vector constant=op->interpolation*Vector::Ones(nc);
            for(int j=0;j<nf;++j) {
                const auto& f=mesh.faces[j];areas[j]=f.area;pressureAreas[j]=f.area;
                if(f.neighbor<0)continue;
                if(f.sign!=1)throw std::runtime_error("Projection expects positive Cartesian shared-face orientation");
                if(!(constant[j]>0&&std::isfinite(constant[j])))throw std::runtime_error("Nonpositive projection material interpolation");
                // Original Aphros initializes excluded Cartesian samples to zero.
                // A few tiny open faces therefore have I(1)!=1 even with constant
                // cell material: mu_face=mu*I(1), rho_face=rho/I(1).
                // Canonicalize only the roundoff range accepted by the old guard.
                if(std::abs(constant[j]-1)>1e-12) {
                    materialWeight[j]=constant[j];materialFaces.push_back(j);
                    pressureAreas[j]*=materialWeight[j];
                }
                faceForce[j]=pressureAreas[j]*options.force[f.axis];
                const double hf=std::min(mesh.cells[f.owner].h,mesh.cells[f.neighbor].h);
                const bool cut=std::abs(f.area-hf*hf)>1e-12*hf*hf;
                const int ids[2]={6*f.owner+2*f.axis+1,6*f.neighbor+2*f.axis};
                for(int id:ids) {sideArea[id]+=f.area;flagged[id]|=static_cast<unsigned char>(cut);}
            }
        }
        constructionMemory("projection.material_factors_ready");
        {
            Sparse correction;
            projectionAssembly::assemble(mesh,op->faceGradient,quadratic.get(),divergence,normalGradient,
                                         correction,cfTaylorOwner,cfTaylorNeighbor);
            constructionMemory("projection.face_matrices_ready");
            if(!materialFaces.empty()) {
                materialProbe.resize(nc);
                for(int c=0;c<nc;++c) {const auto& x=mesh.cells[c].center;materialProbe[c]=.03*std::sin(6.283185307179586*x[0]/mesh.extent[0])+x[1]*x[1]+.2*x[2];}
            }
            if(options.linearBackend=="native_gpu") {
                // GPU projection uses face operators. Only the compact diagonal
                // and material probe need the cell pressure product on the host.
                // Keep compact and deferred products separate to preserve their
                // original accumulation and final addition order.
                auto compact=projectionAssembly::pressureDiagnostics(divergence,pressureAreas,normalGradient,materialProbe);
                compact.diagonal.maxCoeff(&gauge);
                Vector().swap(compact.diagonal);
                if(materialProbe.size()) {
                    const auto deferred=projectionAssembly::pressureDiagnostics(divergence,pressureAreas,correction,materialProbe);
                    materialPressureOperator=(compact.image+deferred.image)/options.rho;
                }
            } else {
                pressureMatrix=-divergence*pressureAreas.asDiagonal()*normalGradient;
                pressureDeferred=-divergence*pressureAreas.asDiagonal()*correction;
                pressureMatrix.diagonal().maxCoeff(&gauge);
            }
            normalGradient+=correction;
            constructionMemory(options.linearBackend=="native_gpu"?"projection.pressure_diagnostics_ready":"projection.pressure_matrices_ready");
        }
        constructionMemory("projection.assembly_workspace_released");
        checkInterfaces();
        // The operators and Taylor rows now own all reconstruction weights.
        quadratic.reset();
        constructionMemory("projection.quadratic_released");
        const bool useAmg=options.pressureSolver=="amg" || (options.pressureSolver=="auto"&&nc>50000);
        diffusionMatrix=options.rho*options.nu*op->compactDiffusion;
        diffusionDeferred=options.rho*options.nu*op->deferredDiffusion;
        // Embedded operator verification/dumps have already run. The time-step
        // equations use the scaled copies above; release the unscaled copies.
        EmbeddedOperators::Sparse().swap(op->compactDiffusion);
        EmbeddedOperators::Sparse().swap(op->deferredDiffusion);
        if(!materialFaces.empty()) {
            std::vector<Triplet> compactTerms,fullTerms;
            for(int j:materialFaces) {
                const auto& f=mesh.faces[j];const double areaChange=areas[j]*(materialWeight[j]-1),k=areaChange/f.distance;
                compactTerms.emplace_back(f.owner,f.owner,k);compactTerms.emplace_back(f.neighbor,f.neighbor,k);
                compactTerms.emplace_back(f.owner,f.neighbor,-k);compactTerms.emplace_back(f.neighbor,f.owner,-k);
                for(Rows::InnerIterator it(op->faceGradient,j);it;++it) {
                    fullTerms.emplace_back(f.owner,it.col(),-areaChange*it.value());
                    fullTerms.emplace_back(f.neighbor,it.col(),areaChange*it.value());
                }
            }
            Sparse compactChange(nc,nc),fullChange(nc,nc);
            compactChange.setFromTriplets(compactTerms.begin(),compactTerms.end());
            fullChange.setFromTriplets(fullTerms.begin(),fullTerms.end());
            diffusionMatrix+=options.rho*options.nu*compactChange;
            diffusionDeferred+=options.rho*options.nu*(fullChange-compactChange);
        }
        {
            Vector materialViscosityOperator;Field materialAcceleration;
            if(!materialFaces.empty()) {
                materialAcceleration=acceleration(materialProbe);
                if(options.linearBackend!="native_gpu")
                    materialPressureOperator=(pressureMatrix*materialProbe+pressureDeferred*materialProbe)/options.rho;
                materialViscosityOperator=diffusionMatrix*materialProbe+diffusionDeferred*materialProbe;
            }
            std::ofstream materialDump(options.output/"material_faces.csv"),materialCells(options.output/"material_cells.csv");
            materialDump<<std::setprecision(17)<<"id,x,y,z,axis,area,weight,mu,rho,compact_gradient,full_gradient,pressure_flux,viscosity_flux,face_acceleration\n";
            materialCells<<std::setprecision(17)<<"face_x,face_y,face_z,axis,side,x,y,z,accel_x,accel_y,accel_z,pressure_operator,viscosity_operator\n";
            for(int j:materialFaces) {
                const auto& f=mesh.faces[j];materialDump<<j;
                for(double v:f.center)materialDump<<','<<v;
                const double gc=(materialProbe[f.neighbor]-materialProbe[f.owner])/f.distance;
                double gf=0;for(Rows::InnerIterator it(op->faceGradient,j);it;++it)gf+=it.value()*materialProbe[it.col()];
                materialDump<<','<<f.axis<<','<<f.area<<','<<materialWeight[j]<<','<<options.rho*options.nu*materialWeight[j]<<','<<options.rho/materialWeight[j]
                            <<','<<gc<<','<<gf<<','<<pressureAreas[j]/options.rho*gc<<','<<options.rho*options.nu*pressureAreas[j]*gf
                            <<','<<materialWeight[j]*(options.force[f.axis]-gf/options.rho)<<'\n';
                for(int side=0;side<2;++side) {
                    const int c=side?f.neighbor:f.owner;const auto& x=mesh.cells[c].center;
                    materialCells<<f.center[0]<<','<<f.center[1]<<','<<f.center[2]<<','<<f.axis<<','<<side<<','<<x[0]<<','<<x[1]<<','<<x[2];
                    for(int d=0;d<3;++d)materialCells<<','<<materialAcceleration(c,d);
                    materialCells<<','<<materialPressureOperator[c]<<','<<materialViscosityOperator[c]<<'\n';
                }
            }
            checked(materialDump);checked(materialCells);
        }
        Vector().swap(materialProbe);Vector().swap(materialPressureOperator);
        constructionMemory("projection.material_ready");
        if(options.linearBackend=="native_gpu") {
            if(options.gpuViscosityOperator=="full") {
                // The implicit solve uses the GPU face operator. Host residual
                // checks only need this constant sum. The previous lazy sum
                // re-merged both sparse inputs while multiplying. Preserve Eigen's
                // coefficient addition, including explicit cancellation zeros.
                Sparse full=diffusionMatrix+diffusionDeferred;
                diffusionMatrix.swap(full);Sparse().swap(diffusionDeferred);
                storedFullDiffusion=true;
            }
            constructionMemory("projection.gpu_host_operators_ready");
            std::vector<double> factors;
            if(!materialFaces.empty())factors.assign(materialWeight.data(),materialWeight.data()+nf);
            gpu=std::make_unique<NativeCompactGpu>(mesh,factors);
            constructionMemory("projection.gpu_fields_ready");
            if(options.gpuPreconditioner=="native_amg")gpu->configureNativeAmg(mesh,gauge,options.rho*options.nu,options.rho/options.timeStep);
            constructionMemory("projection.gpu_amg_ready");
            if(options.gpuPreconditioner=="native_amg")gpu->setKrylovDimension(options.gpuKrylovDimension);
            if(options.gpuOrthogonalization=="cgs2")gpu->useBatchedCgs2();
            if(options.gpuPressureGauge=="mean_zero")gpu->useMeanZeroPressure();
            if(options.gpuPressureOperator=="full") {
                const Rows gradient=normalGradient;std::vector<NativePressureFaceStencil> stencils;
                for(int j=0;j<nf;++j) {
                    const auto& face=mesh.faces[j];
                    if(face.neighbor<0||mesh.cells[face.owner].level==mesh.cells[face.neighbor].level)continue;
                    NativePressureFaceStencil stencil{size_t(j),{}};
                    for(Rows::InnerIterator it(gradient,j);it;++it)stencil.gradient.emplace_back(it.col(),it.value());
                    stencils.push_back(std::move(stencil));
                }
                gpu->configurePressureInterfaces(mesh,stencils);
            }
            constructionMemory("projection.gpu_pressure_ready");
            if(options.gpuViscosityOperator=="full") {
                std::vector<NativeViscosityFaceStencil> stencils;
                for(int j=0;j<nf;++j) {
                    const auto& face=mesh.faces[j];const double h=mesh.cells[face.owner].h;
                    if(face.neighbor>=0&&mesh.cells[face.owner].level==mesh.cells[face.neighbor].level&&
                       std::abs(face.area-h*h)<=1e-12*h*h)continue;
                    NativeViscosityFaceStencil stencil{size_t(j),{}};
                    const auto& gradient=face.neighbor<0?op->wallGradient:op->faceGradient;
                    for(Rows::InnerIterator it(gradient,j);it;++it)stencil.gradient.emplace_back(it.col(),it.value());
                    stencils.push_back(std::move(stencil));
                }
                gpu->configureViscosityFaces(mesh,stencils);
            }
            constructionMemory("projection.gpu_viscosity_ready");
            gpuTrace.open(options.output/"gpu_linear.csv");
            gpuTrace<<std::setprecision(17)<<"call,operator,iterations,residual_restarts,true_relative_residual,seconds,compatibility_relative_l2,original_rhs_checks,original_rhs_accepted\n";checked(gpuTrace);
            pressureRoundoffTrace.open(options.output/"projection_roundoff_exits.csv");
            pressureRoundoffTrace<<std::setprecision(17)<<"call,time_step,pass,divergence_linf,flux_velocity_change_linf,flux_velocity_scale,impulse_change_linf,impulse_scale,epsilon\n";
            checked(pressureRoundoffTrace);
            if(options.gpuPressureGauge=="mean_zero") {
                pressureCycleTrace.open(options.output/"projection_cycle_exits.csv");
                pressureCycleTrace<<std::setprecision(17)<<"call,time_step,exit_pass,period,pass,divergence_linf,flux_velocity_change_linf,flux_velocity_scale,impulse_change_upper_linf,impulse_scale,exit_compensated_divergence_linf,epsilon\n";
                checked(pressureCycleTrace);
            }
        } else {
            // The GPU backend never uses pinned CPU systems or factorizes
            // these matrices. Construct them only for the CPU linear backend.
            std::vector<Triplet> pinned;
            for(int col=0;col<nc;++col)for(Sparse::InnerIterator it(pressureMatrix,col);it;++it)
                if(it.row()!=gauge&&it.col()!=gauge)pinned.emplace_back(it.row(),it.col(),it.value());
            pinned.emplace_back(gauge,gauge,1.);
            Sparse fixed(nc,nc);fixed.setFromTriplets(pinned.begin(),pinned.end());
            Sparse implicit=diffusionMatrix;
            for(int c=0;c<nc;++c)implicit.coeffRef(c,c)+=mass[c];
            pressureLinear.compute(fixed,useAmg,options.linearTolerance);
            diffusionLinear.compute(implicit,useAmg,options.linearTolerance);
        }
        // Allocate the same zero state before any restart or flow operation.
        // This also runs in operator-only mode: its final initialized state
        // retains all arrays required by an actual flow solve.
        velocity=Field::Zero(nc,3);diffusionGuess=velocity;
        pressure=Vector::Zero(nc);flux=makeFlux(Vector::Zero(nf));
        constructionMemory("projection.flow_fields_ready");
        if(std::getenv("SIMPLE_PROJECTION_TRACE")) {
            pressureTrace.open(options.output/"projection_pressure.csv");
            pressureTrace<<std::setprecision(17)<<"call,time_step,pass,linear_refinements,divergence_linf,pass_seconds\n";
            checked(pressureTrace);
            pressureUpdateTrace.open(options.output/"projection_updates.csv");
            pressureUpdateTrace<<std::setprecision(17)<<"call,time_step,pass,divergence_linf,compensated_divergence_linf,divergence_volume_l2,intended_flux_velocity_change_linf,flux_velocity_scale,intended_impulse_change_linf,impulse_range,changed_flux_faces,changed_pressure_cells,unrepresented_flux_updates\n";
            pressureFloorFaces.open(options.output/"projection_floor_faces.csv");
            pressureFloorFaces<<std::setprecision(17)<<"call,time_step,pass,cell,cell_volume,cell_defect,face,owner,neighbor,area,flux_before,intended_delta,flux_after,flux_before_low,flux_after_low\n";
            checked(pressureUpdateTrace);checked(pressureFloorFaces);
        }
        dumpMeshCsv(mesh,(options.output/"mesh").string());
        std::ofstream metadata(options.output/"projection_method.json");
        metadata<<nlohmann::json{{"fluid_solver","proj"},{"convection_scheme","bcg"},
            {"diffusion_redistribution",false},{"advection_redistribution",true},
            {"uniform_method","Original Aphros Proj ordering with implicit deferred diffusion and BCG"},
            {"interface_method","Shared subface fluxes, area-averaged side derivatives, quadratic Taylor face values and pressure-gradient correction"},
            {"interface_pressure_solve","Incremental conservative flux correction, preconditioned by the compact pressure matrix"},
            {"linear_backend",gpu?(gpu->amgLevels()?"native_gpu_fgmres_amg":"native_gpu_pcg_jacobi"):useAmg?"amg":"ldlt"},
            {"host_pressure_matrices",gpu?"not_assembled":"retained"},
            {"host_diffusion_storage",storedFullDiffusion?"full_sum":"compact_and_deferred"},
            {"gpu_orthogonalization",gpu?options.gpuOrthogonalization:"not_used"},
            {"gpu_krylov_dimension",gpu&&gpu->amgLevels()?options.gpuKrylovDimension:0},
            {"native_amg_levels",gpu?gpu->amgLevels():0},
            {"pressure_gauge",gpu?options.gpuPressureGauge:"pin"},
            {"pressure_iterate_storage",gpu&&options.gpuPressureGauge=="mean_zero"?"twofold":"double"},
            {"conservative_flux_storage",flux.compensated()?"twofold":"double"},
            {"anderson_history_storage",andersonFileHistory?"file_binary64_checked_blocks":"memory_binary64"},
            {"pressure_operator",gpu?options.gpuPressureOperator:"compact"},
            {"full_pressure_interface_faces",gpu?gpu->fullPressureInterfaceFaces():0},
            {"viscosity_operator",gpu?options.gpuViscosityOperator:"compact"},
            {"full_viscosity_faces",gpu?gpu->fullViscosityFaces():0},
            {"material_weighted_open_faces",materialFaces.size()},
            {"minimum_material_weight",materialWeight.minCoeff()},
            {"pressure_roundoff_exit",gpu?"Repeated divergence <= 1e-8 and both face-velocity and pressure-impulse corrections <= double epsilon times their physical scales; final continuity gates unchanged":"disabled"},
            {"pressure_roundoff_cycle_exit",gpu && options.gpuPressureGauge=="mean_zero"?
                "Two complete nonconstant periods of length 2..4; every correction <= half double epsilon times post-update physical scales, raw and final compensated divergence <= 1e-8; final continuity gates unchanged":"disabled"},
            {"gpu_integration_scope",gpu?(options.gpuViscosityOperator=="full"?
                "Full implicit viscosity including cut and wall faces on GPU; pressure mode recorded separately; CPU stencil construction, transport and diagnostics remain":options.gpuPressureOperator=="full"?
                "Full shared-face pressure operator and compact viscosity on GPU; CPU wall/deferred viscosity, transport and diagnostics remain":
                "Compact pressure and viscosity linear solves on GPU; explicit CPU operators remain for deferred terms, transport, diagnostics and cross-checks"):"CPU validation backend"}}.dump(2)<<'\n';checked(metadata);
        if(!options.restartCheckpoint.empty())loadRestart();
        constructionMemory("projection.ready");
    }
    nlohmann::json replayPressure(const std::filesystem::path& path,int repetitions) {
        if(!gpu || !gpu->amgLevels() || options.gpuPressureGauge!="mean_zero" ||
           options.gpuPressureOperator!="full" || restartStep || repetitions<1 || repetitions>16)
            throw std::runtime_error("Invalid pressure replay mode");
        const std::uint16_t endian=1;
        if(sizeof(double)!=8 || !std::numeric_limits<double>::is_iec559 ||
           *reinterpret_cast<const unsigned char*>(&endian)!=1 ||
           std::filesystem::file_size(path)!=size_t(nc)*sizeof(double))
            throw std::runtime_error("Pressure replay requires one little-endian float64 per mesh cell");
        std::vector<double> rhs(nc);
        std::ifstream input(path,std::ios::binary);
        input.read(reinterpret_cast<char*>(rhs.data()),std::streamsize(rhs.size()*sizeof(double)));
        if(!input || !std::all_of(rhs.begin(),rhs.end(),[](double v){return std::isfinite(v);}))
            throw std::runtime_error("Truncated or nonfinite pressure replay RHS");
        const auto writeVector=[&](const std::string& name,const double* data,size_t count) {
            std::ofstream stream(options.output/name,std::ios::binary);
            stream.write(reinterpret_cast<const char*>(data),std::streamsize(count*sizeof(double)));checked(stream);
        };
        writeVector("pressure_replay_rhs.bin",rhs.data(),rhs.size());
        gpu->dumpPressureFaces(mesh,options.output/"pressure_replay_operator.bin");
        const Vector source=Eigen::Map<const Vector>(rhs.data(),nc);
        if(!(source.norm()>0))throw std::runtime_error("Pressure replay requires a nonzero RHS");
        nlohmann::json report{{"scope","Diagnostic pressure solves only; no physical step or flow state advanced"},
            {"rhs_file",path.string()},{"cells",nc},{"faces",nf},{"pressure_amg_gauge",gauge},
            {"krylov_dimension",options.gpuKrylovDimension},{"orthogonalization",options.gpuOrthogonalization},
            {"linear_tolerance",options.linearTolerance},{"max_iterations",4000},
            {"material_weighted_open_faces",materialFaces.size()},{"minimum_material_weight",materialWeight.minCoeff()},
            {"full_pressure_interface_faces",gpu->fullPressureInterfaceFaces()},
            {"all_solve_checks_passed",true},{"trials",nlohmann::json::array()}};
        for(int trial=1;trial<=repetitions;++trial) {
            const auto begin=std::chrono::steady_clock::now();
            nlohmann::json row{{"trial",trial}};
            try {
                // Same scale, gauge, tolerance and original-RHS checks as linearSolve.
                const auto result=gpu->solve(rhs,1.,0.,false,gauge,options.linearTolerance);
                row.update({{"solver_accepted",true},{"iterations",result.iterations},{"residual_restarts",result.restarts},
                    {"true_relative_residual",result.relativeResidual},{"compatibility_relative_l2",result.compatibilityRelativeL2},
                    {"original_rhs_checks",result.originalRhsChecks},{"original_rhs_accepted",result.originalRhsAccepted},
                    {"solve_seconds",result.seconds}});
                const std::string prefix="pressure_replay_"+std::to_string(trial);
                writeVector(prefix+"_solution.bin",result.values.data(),result.values.size());
                auto actual=gpu->apply(result.values,1.,0.,false);
                Vector tail=Vector::Zero(nc);
                if(!result.lowValues.empty()) {
                    writeVector(prefix+"_solution_low.bin",result.lowValues.data(),result.lowValues.size());
                    const auto lowImage=gpu->apply(result.lowValues,1.,0.,false);
                    for(int c=0;c<nc;++c)actual[c]+=lowImage[c];
                    tail=Eigen::Map<const Vector>(result.lowValues.data(),nc);
                }
                row["solution_storage"]=result.lowValues.empty()?"double":"twofold";
                writeVector(prefix+"_gpu_ax.bin",actual.data(),actual.size());
                const Vector solution=Eigen::Map<const Vector>(result.values.data(),nc);
                const Vector gradient=normalGradient*solution+normalGradient*tail;
                const Vector cpu=-divergence*(pressureAreas.array()*gradient.array()).matrix();
                writeVector(prefix+"_cpu_ax.bin",cpu.data(),size_t(nc));
                // Independent readback diagnostics include unscale/rounding of the
                // returned doubles. They do not replace the solver's stopping rule.
                row["returned_solution_gpu_residual"]=(source-Eigen::Map<const Vector>(actual.data(),nc)).norm()/source.norm();
                row["returned_solution_cpu_residual"]=(source-cpu).norm()/source.norm();
                row["solve_checks_passed"]=result.originalRhsAccepted && result.relativeResidual<=options.linearTolerance;
            } catch(const NativeGpuConvergenceError& error) {
                row.update({{"solver_accepted",false},{"solve_checks_passed",false},{"iterations",error.iterations},
                    {"true_relative_residual",error.relativeResidual},{"tolerance",error.tolerance},
                    {"compatibility_relative_l2",error.compatibilityRelativeL2},{"message",error.what()}});
                const std::string prefix="pressure_replay_"+std::to_string(trial)+"_failed_scaled";
                writeVector(prefix+"_iterate.bin",error.scaledIterate.data(),error.scaledIterate.size());
                writeVector(prefix+"_ax.bin",error.scaledAx.data(),error.scaledAx.size());
                if(!error.scaledLowIterate.empty())writeVector(prefix+"_low.bin",error.scaledLowIterate.data(),error.scaledLowIterate.size());
                row["rhs_scale"]=error.rhsScale;
                row["failed_iterate_accepted"]=false;
            }
            row["elapsed_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
            row["gpu_allocated_bytes"]=gpu->allocatedBytes();
            report["all_solve_checks_passed"]=report.at("all_solve_checks_passed").get<bool>() && row.at("solve_checks_passed").get<bool>();
            report["trials"].push_back(row);
            std::ofstream metadata(options.output/"pressure_replay.json");metadata<<report.dump(2)<<'\n';checked(metadata);
            std::cout<<"Pressure replay "<<row.dump()<<std::endl;
        }
        return report;
    }
    void loadRestart() {
        const std::filesystem::path path=options.restartCheckpoint;
        std::ifstream header(path);if(!header)throw std::runtime_error("Cannot read restart checkpoint");
        header>>restart;
        restartStep=restart.at("physical_step").get<int>();
        const auto restartFormat=restart.at("format").get<std::string>();
        const bool twofoldRestart=restartFormat=="cirrus_projection_twofold_restart_v4";
        const bool chainRestart=restartFormat=="cirrus_projection_chain_restart_v3";
        const bool prefixRestart=restartFormat=="cirrus_projection_prefix_restart_v2" || chainRestart;
        if((restartFormat!="cirrus_projection_restart_v1" && !prefixRestart && !twofoldRestart) || restartStep<1 || restartStep>=options.timeSteps ||
           restart.at("cells").get<int>()!=nc || restart.at("faces").get<int>()!=nf ||
           restart.at("time_step").get<double>()!=options.timeStep ||
           restart.at("physical_time").get<double>()!=restartStep*options.timeStep ||
           restart.at("time_history").size()!=size_t(restartStep))
            throw std::runtime_error("Restart geometry size, physical time or complete prefix differs");
        nlohmann::json config;std::ifstream(options.output/"case.json")>>config;
        auto parent=restart.at("parent_config");
        for(const char* key:{"output","time_steps","output_stride","dump_iterations","restart_checkpoint"}) {
            parent.erase(key);config.erase(key);
        }
        if(prefixRestart) {
            // Both verified orthogonalizations solve the same implicit native
            // operator to the same true-residual tolerance. Record the choice
            // in each run, but permit changing it at a physical-step boundary.
            for(const auto* c:{&parent,&config}) {
                const auto mode=c->value("gpu_orthogonalization",std::string("mgs2"));
                if(mode!="mgs2" && mode!="cgs2")throw std::runtime_error("Unknown restart orthogonalization");
            }
            if(restart.at("parent_proof_kind")!=(chainRestart?"completed_physical_prefix_chain":"completed_physical_prefix"))
                throw std::runtime_error("Prefix restart requires a completed-prefix proof");
            parent.erase("gpu_orthogonalization");config.erase("gpu_orthogonalization");
        }
        if(parent!=config)throw std::runtime_error("Restart changes the physical or numerical configuration");
        if(twofoldRestart && (!flux.compensated() || restart.value("flux_storage",std::string())!="twofold"))
            throw std::runtime_error("Twofold restart requires both conservative face-flux parts");
        if(!twofoldRestart) {
            nlohmann::json method;std::ifstream input(std::filesystem::path(restart.at("parent_run").get<std::string>())/"projection_method.json");
            if(!input)throw std::runtime_error("Cannot verify parent flux storage");input>>method;
            if(method.value("conservative_flux_storage",std::string("double"))=="twofold")
                throw std::runtime_error("Legacy restart would discard conservative flux low parts; use the twofold checkpoint producer");
        }
        for(int d=0;d<3;++d)if(restart.at("extent").at(d).get<double>()!=mesh.extent[d])
            throw std::runtime_error("Restart domain extent differs");
        const std::uint16_t endian=1;
        if(sizeof(double)!=8 || !std::numeric_limits<double>::is_iec559 ||
           *reinterpret_cast<const unsigned char*>(&endian)!=1)
            throw std::runtime_error("Restart format requires little-endian IEEE float64");
        const auto statePath=path.parent_path()/restart.at("state_file").get<std::string>();
        const size_t count=7*size_t(nc)+size_t(nf)*(twofoldRestart?2:1);
        if(std::filesystem::file_size(statePath)!=count*sizeof(double))throw std::runtime_error("Wrong restart state length");
        std::vector<double> values(count);
        std::ifstream state(statePath,std::ios::binary);
        state.read(reinterpret_cast<char*>(values.data()),std::streamsize(count*sizeof(double)));
        if(!state || !std::all_of(values.begin(),values.end(),[](double value){return std::isfinite(value);}))
            throw std::runtime_error("Truncated or nonfinite restart state");
        for(int c=0;c<nc;++c) {
            for(int d=0;d<3;++d) {velocity(c,d)=values[7*size_t(c)+d];diffusionGuess(c,d)=values[7*size_t(c)+4+d];}
            pressure[c]=values[7*size_t(c)+3];
        }
        for(int f=0;f<nf;++f)flux.set(f,{values[7*size_t(nc)+f],twofoldRestart?values[7*size_t(nc)+nf+f]:0.});
        for(int f=0;f<nf;++f)if(mesh.faces[f].neighbor<0 && (flux.get(f).high!=0||flux.get(f).low!=0))
            throw std::runtime_error("Restart has nonzero stationary-wall flux");
        if(continuityResidual()>=1e-7)throw std::runtime_error("Restart state fails actual mass conservation");
        // Echo the exact unscaled restored values before any physical update.
        // The independent restart checker compares these bytes with the parent
        // final velocity, pressure, implicit predictor and conservative flux.
        std::ofstream echo(options.output/"restart_loaded.bin",std::ios::binary);
        for(int c=0;c<nc;++c) {
            double row[7]={velocity(c,0),velocity(c,1),velocity(c,2),pressure[c],
                           diffusionGuess(c,0),diffusionGuess(c,1),diffusionGuess(c,2)};
            echo.write(reinterpret_cast<const char*>(row),sizeof(row));
        }
        echo.write(reinterpret_cast<const char*>(flux.high.data()),std::streamsize(nf*sizeof(double)));
        if(twofoldRestart)echo.write(reinterpret_cast<const char*>(flux.low.data()),std::streamsize(nf*sizeof(double)));
        checked(echo);
        std::cout<<"Restored complete projection state after step "<<restartStep<<" at time "<<restartStep*options.timeStep<<std::endl;
    }
    Vector linearSolve(const Vector& source,bool viscosity,Vector* low=nullptr) {
        if(low)*low=Vector::Zero(source.size());
        if(!gpu)return viscosity?diffusionLinear.solve(source):pressureLinear.solve(source);
        NativeGpuSolveResult result;
        try {
            result=gpu->solve(std::vector<double>(source.data(),source.data()+source.size()),
                viscosity?options.rho*options.nu:1.,viscosity?options.rho/options.timeStep:0.,viscosity,
                viscosity?-1:gauge,options.linearTolerance);
        } catch(const NativeGpuConvergenceError& error) {
            const auto folder=options.output/("linear_failure_"+std::to_string(++linearFailures));
            if(!std::filesystem::create_directory(folder))throw std::runtime_error("Preserve previous linear failure dump");
            std::ofstream rhsFile(folder/"rhs.bin",std::ios::binary);
            rhsFile.write(reinterpret_cast<const char*>(source.data()),std::streamsize(source.size()*sizeof(double)));checked(rhsFile);
            for(const auto& item:{std::make_pair("failed_scaled_iterate.bin",&error.scaledIterate),
                                  std::make_pair("failed_scaled_ax.bin",&error.scaledAx),
                                  std::make_pair("failed_scaled_low.bin",&error.scaledLowIterate)}) {
                std::ofstream file(folder/item.first,std::ios::binary);
                file.write(reinterpret_cast<const char*>(item.second->data()),std::streamsize(item.second->size()*sizeof(double)));checked(file);
            }
            nlohmann::json info{{"physical_step",activeStep},{"iteration",activeIteration},
                {"phase",linearPhase},{"anderson_attempt",activeTrial},{"pressure_call",pressureCalls},
                {"successful_gpu_calls_before_failure",gpuCalls},{"operator",viscosity?"diffusion":"pressure"},
                {"true_relative_residual",error.relativeResidual},{"tolerance",error.tolerance},
                {"iterations",error.iterations},{"compatibility_relative_l2",error.compatibilityRelativeL2},
                {"rhs_scale",error.rhsScale},{"failed_iterate_accepted",false},
                {"rhs_format","native little-endian float64 in mesh cell ID order"},{"rhs_count",source.size()},
                {"message",error.what()},{"failed_solution_accepted",false}};
            if(options.steadyAndersonDepth) {info["steady_iteration"]=activeStep;info.erase("physical_step");}
            std::ofstream metadata(folder/"failure.json");metadata<<info.dump(2)<<'\n';checked(metadata);
            throw;
        }
        gpuTrace<<++gpuCalls<<','<<(viscosity?"diffusion":"pressure")<<','<<result.iterations<<','<<result.restarts<<','<<result.relativeResidual<<','<<result.seconds<<','<<result.compatibilityRelativeL2<<','<<result.originalRhsChecks<<','<<result.originalRhsAccepted<<'\n';checked(gpuTrace);
        if(!result.lowValues.empty()) {
            if(!low)throw std::runtime_error("Twofold pressure consumer must retain the low solution");
            *low=Eigen::Map<const Vector>(result.lowValues.data(),result.lowValues.size());
        }
        return Eigen::Map<const Vector>(result.values.data(),result.values.size());
    }
    void checkInterfaces() const {
        double maximum=0;int count=0;
        for(int j=0;j<nf;++j) {
            const auto& f=mesh.faces[j];if(f.neighbor<0||mesh.cells[f.owner].level==mesh.cells[f.neighbor].level)continue;
            ++count;const double h=mesh.cells[f.owner].h;
            for(const Rows* matrix:{&cfTaylorOwner,&cfTaylorNeighbor})for(int term=0;term<10;++term) {
                double value=0;
                for(Rows::InnerIterator it(*matrix,j);it;++it) {
                    Vec3 x=mesh.cells[it.col()].center-f.center;
                    x[0]-=std::round(x[0]/mesh.extent[0])*mesh.extent[0];x/=h;
                    const double monomials[10]={1,x[0],x[1],x[2],x[0]*x[0],x[1]*x[1],x[2]*x[2],x[0]*x[1],x[0]*x[2],x[1]*x[2]};
                    value+=it.value()*monomials[term];
                }
                maximum=std::max(maximum,std::abs(value-(term==0?1.:0.)));
            }
        }
        std::ofstream report(options.output/"projection_interface_checks.json");
        report<<nlohmann::json{{"passed",maximum<1e-10},{"coarse_fine_faces_tested",count},
            {"quadratic_upwind_face_value_max_error",maximum},
            {"scope","Actual assembled owner/neighbor interface Taylor rows on degree <=2 monomials; not flow/grid convergence"}}.dump(2)<<'\n';checked(report);
        if(maximum>=1e-10)throw std::runtime_error("Projection interface reconstruction check failed");
    }
    nlohmann::json replayProjection(const std::filesystem::path& path,int repetitions,double dt) {
        if(!gpu || !gpu->amgLevels() || options.gpuPressureGauge!="mean_zero" ||
           options.gpuPressureOperator!="full" || restartStep || repetitions<1 || repetitions>4)
            throw std::runtime_error("Invalid projection replay mode");
        if(std::filesystem::file_size(path)!=size_t(nf)*sizeof(double))
            throw std::runtime_error("Projection replay flux length differs from the native face count");
        Vector original(nf);
        std::ifstream input(path,std::ios::binary);
        input.read(reinterpret_cast<char*>(original.data()),std::streamsize(nf*sizeof(double)));
        if(!input || !original.allFinite())throw std::runtime_error("Invalid projection replay flux values");
        for(int f=0;f<nf;++f)if(mesh.faces[f].neighbor<0 && original[f]!=0)
            throw std::runtime_error("Projection replay has a nonzero stationary-wall flux");
        const double scale=(original.array().abs()/areas.array()).maxCoeff();
        if(!(scale>0))throw std::runtime_error("Projection replay requires a nonzero physical flux");
        linearPhase="projection_replay";
        nlohmann::json rows=nlohmann::json::array();
        bool passed=true;
        for(int repeat=1;repeat<=repetitions;++repeat)for(bool enabled:{false,true}) {
            cycleExitEnabled=enabled;
            Flux q=makeFlux(original);Vector p=Vector::Zero(nc);
            const int oldCalls=gpuCalls,oldCycles=cyclePressureExits,oldRoundoff=roundoffPressureExits;
            const auto begin=std::chrono::steady_clock::now();
            project(q,p,dt);
            const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
            const double raw=(pressureDefect(q).array().abs()/volumes.array()).maxCoeff();
            const double compensated=(pressureDefect(q).array().abs()/volumes.array()).maxCoeff();
            const double change=q.maximumVelocityDifference(makeFlux(original),areas);
            const std::string prefix="projection_replay_"+std::to_string(repeat)+(enabled?"_cycle":"_scalar");
            for(const auto& item:{std::make_pair("_flux.bin",&q.high),std::make_pair("_flux_low.bin",&q.low),std::make_pair("_pressure.bin",&p)}) {
                std::ofstream file(options.output/(prefix+item.first),std::ios::binary);
                file.write(reinterpret_cast<const char*>(item.second->data()),std::streamsize(item.second->size()*sizeof(double)));checked(file);
            }
            const bool accepted=q.allFinite() && p.allFinite() && std::isfinite(raw) && raw<=1e-8 &&
                std::isfinite(compensated) && compensated<=1e-8;
            passed=passed && accepted;
            rows.push_back({{"repeat",repeat},{"cycle_exit_enabled",enabled},{"pressure_call",pressureCalls},
                {"gpu_calls",gpuCalls-oldCalls},{"cycle_exits",cyclePressureExits-oldCycles},
                {"scalar_repeat_exits",roundoffPressureExits-oldRoundoff},{"seconds",seconds},
                {"divergence_linf",raw},{"compensated_divergence_linf",compensated},
                {"flux_velocity_change_linf",change},{"original_velocity_scale",scale},
                {"pressure_correction_linf",p.cwiseAbs().maxCoeff()},{"output_prefix",prefix},
                {"accepted",accepted}});
        }
        cycleExitEnabled=true;
        nlohmann::json report{{"scope","Paired operator-only projection of the same retained flux with zero initial pressure impulse; no physical time step or full-flow agreement claim"},
            {"all_projection_checks_passed",passed},{"cells",nc},{"faces",nf},{"time_step",dt},
            {"input_flux",path.string()},{"initial_divergence_linf",((divergence*original).array().abs()/volumes.array()).maxCoeff()},
            {"initial_compensated_divergence_linf",(pressureDefect(makeFlux(original)).array().abs()/volumes.array()).maxCoeff()},
            {"rows",rows}};
        std::ofstream result(options.output/"projection_replay.json");result<<report.dump(2)<<'\n';checked(result);
        std::cout<<report.dump(2)<<std::endl;
        return report;
    }
    nlohmann::json replayPredictor(const std::filesystem::path& path,int repetitions) {
        const bool twofold=std::filesystem::file_size(path)==(7*size_t(nc)+2*size_t(nf))*sizeof(double);
        const size_t count=7*size_t(nc)+size_t(nf)*(twofold?2:1);
        if(std::filesystem::file_size(path)!=count*sizeof(double))throw std::runtime_error("Wrong predictor state length");
        std::vector<double> values(count);
        std::ifstream input(path,std::ios::binary);
        input.read(reinterpret_cast<char*>(values.data()),std::streamsize(count*sizeof(double)));
        if(!input || !std::all_of(values.begin(),values.end(),[](double v){return std::isfinite(v);}))
            throw std::runtime_error("Invalid predictor state");
        Field u(nc,3);Vector p(nc),q(nf);
        for(int c=0;c<nc;++c) {
            for(int d=0;d<3;++d)u(c,d)=values[7*size_t(c)+d];
            p[c]=values[7*size_t(c)+3];
        }
        for(int f=0;f<nf;++f)q[f]=values[7*size_t(nc)+f];
        for(int f=0;f<nf;++f)if(mesh.faces[f].neighbor<0 && q[f]!=0)
            throw std::runtime_error("Predictor source has a nonzero wall flux");
        // Reproduce the first BCG predictor of the next physical step using
        // the same source, interpolation, and arithmetic as predictedFlux().
        // Keep the pre-projection flux so the two exit rules see identical data.
        Flux conservative=makeFlux(q);
        if(twofold) {
            if(!conservative.compensated())throw std::runtime_error("Twofold predictor state requires conservative flux storage");
            for(int f=0;f<nf;++f)conservative.low[f]=values[7*size_t(nc)+nf+f];
            for(int f=0;f<nf;++f)if(mesh.faces[f].neighbor<0&&conservative.low[f]!=0)
                throw std::runtime_error("Predictor source has a nonzero wall flux low part");
        }
        const Field source=acceleration(p),predicted=bcg(u,conservative,source);
        Vector before=Vector::Zero(nf);
        for(int f=0;f<nf;++f)if(mesh.faces[f].neighbor>=0)before[f]=areas[f]*predicted(f,mesh.faces[f].axis);
        const auto fluxPath=options.output/"predictor_before_projection.bin";
        std::ofstream output(fluxPath,std::ios::binary);
        output.write(reinterpret_cast<const char*>(before.data()),std::streamsize(nf*sizeof(double)));checked(output);output.close();
        auto report=replayProjection(fluxPath,repetitions,.5*options.timeStep);
        report["predictor_source_state"]=path.string();
        report["predictor_full_time_step"]=options.timeStep;
        report["scope"]="Paired pressure projection of the next-step BCG predictor reconstructed from a verified full physical checkpoint; identical pre-projection flux for both rules; no physical time step advanced";
        std::ofstream result(options.output/"projection_replay.json");result<<report.dump(2)<<'\n';checked(result);
        return report;
    }
    double norm(const Field& field) const {
        double sum=0;for(int c=0;c<nc;++c)sum+=volumes[c]*field.row(c).squaredNorm();
        return std::sqrt(sum/volume);
    }
    Field acceleration(const Vector& p) const {
        Field result(nc,3);
        for(int d=0;d<3;++d)result.col(d)=Vector::Constant(nc,options.force[d])-op->cellGradient[d]*p/options.rho;
        for(int j:materialFaces) {
            const auto& f=mesh.faces[j];double gradient=0;
            for(Rows::InnerIterator it(op->faceGradient,j);it;++it)gradient+=it.value()*p[it.col()];
            const double correction=(materialWeight[j]-1)*areas[j]*(options.force[f.axis]-gradient/options.rho);
            for(int c:{f.owner,f.neighbor})result(c,f.axis)+=correction/(sideArea[6*c+2*f.axis]+sideArea[6*c+2*f.axis+1]);
        }
        return result;
    }
    Vector velocityFlux(const Field& u) const {
        Vector result=Vector::Zero(nf);
        for(int d=0;d<3;++d) {
            const Vector face=op->interpolation*u.col(d);
            for(int j=0;j<nf;++j)if(mesh.faces[j].neighbor>=0&&mesh.faces[j].axis==d)result[j]=areas[j]*face[j];
        }
        return result;
    }
    Flux makeFlux(Vector values) const {return Flux(std::move(values),bool(gpu)&&options.gpuPressureGauge=="mean_zero");}
    Vector pressureDefect(const Flux& q) const {
        if(q.compensated())return q.negativeDivergence(mesh).rounded();
        if(!gpu||options.gpuPressureGauge!="mean_zero")return -divergence*q.high;
        // Fluxes can be many orders larger than their imbalance. Accumulate
        // shared faces with Neumaier compensation before casting the balance
        // to one double, preserving compatibility even for tiny corrections.
        // In particular, MSVC long double would not give extra precision.
        Vector sum=Vector::Zero(nc),error=Vector::Zero(nc);
        auto add=[&](int c,double value) {
            const double next=sum[c]+value;
            error[c]+=std::abs(sum[c])>=std::abs(value)?(sum[c]-next)+value:(value-next)+sum[c];
            sum[c]=next;
        };
        for(int j=0;j<nf;++j) {
            const auto& face=mesh.faces[j];if(face.neighbor<0)continue;
            add(face.owner,-q[j]);add(face.neighbor,q[j]);
        }
        return sum+error;
    }
    void tracePressureUpdate(const Flux& before,const Vector& impulseBefore,const Vector& correction,const Vector& correctionGradient,
            const Flux& q,const Vector& impulse,double dt,int pass,double residual,bool dumpFloor,const ProjectionFluxTrace* compact=nullptr) {
        if(!pressureUpdateTrace.is_open())return;
        // Read-only diagnostics: use the same intended face update, but measure
        // changes against the stored doubles. A small linear residual alone
        // does not show whether another pressure pass can change the flux.
        const Vector defect=pressureDefect(q);
        Eigen::Index worst=0;
        const double compensated=(defect.array().abs()/volumes.array()).maxCoeff(&worst);
        const double l2=std::sqrt((defect.array().square()/volumes.array()).sum()/volume);
        double velocityChange=0,velocityScale=0;
        int changedFlux=0,changedPressure=0,unrepresented=0;
        if(compact) {
            compact->verify(q,worst,defect[worst],compensated,l2);
            velocityChange=compact->velocityChange;velocityScale=compact->velocityScale;
            changedFlux=compact->changedFlux;unrepresented=compact->unrepresented;
            if(dumpFloor)for(const auto& row:compact->worstFaces) {
                const int j=row.face;const auto& f=mesh.faces[j];
                pressureFloorFaces<<pressureCalls<<','<<dt<<','<<pass<<','<<worst<<','<<volumes[worst]<<','
                    <<defect[worst]<<','<<j<<','<<f.owner<<','<<f.neighbor<<','<<areas[j]<<','
                    <<row.before.rounded()<<','<<row.delta<<','<<q[j]<<','<<row.before.low<<','<<q.get(j).low<<'\n';
            }
        } else {
        for(int j=0;j<nf;++j) {
            const auto& f=mesh.faces[j];if(f.neighbor<0)continue;
            const double delta=-(pressureAreas[j]*correctionGradient[j]);
            velocityChange=std::max(velocityChange,std::abs(delta)/areas[j]);
            velocityScale=std::max(velocityScale,std::abs(before[j])/areas[j]);
            const bool unchanged=q.get(j).high==before.get(j).high&&q.get(j).low==before.get(j).low;
            changedFlux+=!unchanged;
            unrepresented+=delta!=0 && unchanged;
            if(dumpFloor && (f.owner==worst || f.neighbor==worst))
                pressureFloorFaces<<pressureCalls<<','<<dt<<','<<pass<<','<<worst<<','<<volumes[worst]<<','
                    <<defect[worst]<<','<<j<<','<<f.owner<<','<<f.neighbor<<','<<areas[j]<<','
                    <<before[j]<<','<<delta<<','<<q[j]<<','<<before.get(j).low<<','<<q.get(j).low<<'\n';
        }
        }
        for(int c=0;c<nc;++c)changedPressure+=impulse[c]!=impulseBefore[c];
        pressureUpdateTrace<<pressureCalls<<','<<dt<<','<<pass<<','<<residual<<','<<compensated<<','<<l2<<','
            <<velocityChange<<','<<velocityScale<<','<<correction.cwiseAbs().maxCoeff()<<','
            <<impulseBefore.maxCoeff()-impulseBefore.minCoeff()<<','<<changedFlux<<','<<changedPressure<<','<<unrepresented<<'\n';
        checked(pressureUpdateTrace);if(dumpFloor)checked(pressureFloorFaces);
    }
    void project(Flux& q,Vector& p,double dt) {
        ++pressureCalls;
        if(mesh.coarseFineFaces||gpu) {
            // Correct the stored conservative flux incrementally. Replacing an
            // absolute pressure field and re-evaluating P*p stalls on roundoff
            // when divided by tiny cut volumes. This solves the same full
            // nonorthogonal operator while measuring the actual face balance.
            // The GPU path uses this on uniform cut grids too: CPU sparse Ax
            // and the GPU stencil sum round differently, so a CPU Ax defect
            // after a GPU solve is not a reliable physical continuity measure.
            Vector impulse=dt/options.rho*p;
            const double pin=impulse[gauge];impulse.array()-=pin;
            q.subtractProduct(pressureAreas,normalGradient*impulse);
            double residual=std::numeric_limits<double>::infinity();
            double previousResidual=-1;bool floorDumped=false;
            PressureRoundoffCycle cycle;
            int pass=0;
            for(pass=1;pass<=options.nonorthIterations;++pass) {
                const auto begin=std::chrono::steady_clock::now();
                Vector defect=pressureDefect(q);
                // A previous pressure estimate may already leave the actual
                // face flux converged. Do not solve a roundoff-only Neumann
                // RHS, whose tiny global imbalance has no compatible solution.
                // Use the same physical tolerance as after a correction below.
                if(gpu) {
                    residual=(defect.array().abs()/volumes.array()).maxCoeff();
                    if(residual<1e-10)break;
                }
                if(!gpu||options.gpuPressureGauge!="mean_zero")defect[gauge]=0;
                Vector correctionLow;
                const Vector correction=linearSolve(defect,false,&correctionLow);
                defect.resize(0);
                // Apply both parts to the face gradient before returning to the
                // stored physical flux. Never drop the sub-ulp linear correction.
                const Vector correctionGradient=normalGradient*correction+normalGradient*correctionLow;
                Flux before;Vector impulseBefore;
                ProjectionFluxTrace compactTrace;
                const bool compact=pressureUpdateTrace.is_open()&&q.compensated();
                if(pressureUpdateTrace.is_open()) {
                    if(compact)compactTrace.capture(mesh,q,areas,pressureAreas,correctionGradient,volumes,volume);
                    else before=q;
                    impulseBefore=impulse;
                }
                impulse+=correction;
                impulse+=correctionLow;
                const double correctionLowMaximum=correctionLow.cwiseAbs().maxCoeff();
                correctionLow.resize(0);
                constructionMemory("projection.face_update_workspace_ready");
                q.subtractProduct(pressureAreas,correctionGradient);
                residual=(pressureDefect(q).array().abs()/volumes.array()).maxCoeff();
                bool roundoffLimited=false;
                if(gpu && residual>=1e-10 && residual<=1e-8) {
                    const Vector delta=pressureAreas.array()*correctionGradient.array();
                    const double velocityChange=(delta.array().abs()/areas.array()).maxCoeff();
                    const double velocityScale=q.maximumVelocity(areas);
                    const double impulseChange=correction.cwiseAbs().maxCoeff();
                    // The driving pressure drop supplies a scale even for a
                    // roundoff-only repair whose initial impulse is zero.
                    const double impulseScale=std::max(impulse.maxCoeff()-impulse.minCoeff(),
                                                       dt*accelerationScale*mesh.extent[0]);
                    const double epsilon=std::numeric_limits<double>::epsilon();
                    roundoffLimited=pass>=3 && residual==previousResidual &&
                                    std::isfinite(velocityScale) && std::isfinite(velocityChange) &&
                                    std::isfinite(impulseScale) && std::isfinite(impulseChange) &&
                                    velocityScale>0 && velocityChange<=epsilon*velocityScale &&
                                    impulseChange<=epsilon*impulseScale;
                    if(roundoffLimited) {
                        // Retain the actual stored flux and residual. Do not
                        // zero, redistribute, or waive a physical mass defect.
                        // The existing 1e-8 pressure acceptance bound and the
                        // separate final 1e-7 relative continuity gate remain.
                        ++roundoffPressureExits;
                        pressureRoundoffTrace<<pressureCalls<<','<<dt<<','<<pass<<','<<residual<<','
                            <<velocityChange<<','<<velocityScale<<','<<impulseChange<<','<<impulseScale<<','<<epsilon<<'\n';
                        checked(pressureRoundoffTrace);
                    }
                    if(cycleExitEnabled && options.gpuPressureGauge=="mean_zero") {
                        // Bound both returned pressure parts for the new cycle
                        // rule. The old scalar-repeat predicate above is kept.
                        const double impulseUpper=std::nextafter(impulseChange+correctionLowMaximum,
                                                                 std::numeric_limits<double>::infinity());
                        const int period=cycle.observe({residual,velocityChange,velocityScale,impulseUpper,impulseScale});
                        if(period && !roundoffLimited) {
                            const double compensated=(pressureDefect(q).array().abs()/volumes.array()).maxCoeff();
                            if(std::isfinite(compensated) && compensated<=1e-8) {
                                roundoffLimited=true;++cyclePressureExits;
                                // Preserve the actual observations which caused
                                // this decision, including post-update scales.
                                const int first=cycle.size()-2*period;
                                for(int i=first;i<cycle.size();++i) {
                                    const auto& item=cycle.at(i);
                                    pressureCycleTrace<<pressureCalls<<','<<dt<<','<<pass<<','<<period<<','
                                        <<pass-(cycle.size()-1-i)<<','<<item.divergence<<','<<item.velocityChange<<','
                                        <<item.velocityScale<<','<<item.impulseChange<<','<<item.impulseScale<<','
                                        <<compensated<<','<<epsilon<<'\n';
                                }
                                checked(pressureCycleTrace);
                            }
                        }
                    }
                } else if(gpu && cycleExitEnabled && options.gpuPressureGauge=="mean_zero") {
                    // Out-of-bound or still-large defects break a consecutive
                    // small-update window, even if later scalar values repeat.
                    cycle.observe({residual,std::numeric_limits<double>::infinity(),0,0,0});
                }
                const bool dumpFloor=roundoffLimited || (!floorDumped && residual>=1e-10 && residual==previousResidual);
                tracePressureUpdate(before,impulseBefore,correction,correctionGradient,q,impulse,dt,pass,residual,dumpFloor,compact?&compactTrace:nullptr);
                if(dumpFloor)floorDumped=true;
                previousResidual=residual;
                if(pressureTrace.is_open()) {
                    pressureTrace<<pressureCalls<<','<<dt<<','<<pass<<",0,"<<residual<<','
                        <<std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()<<'\n';
                    checked(pressureTrace);
                }
                if(residual<1e-10 || roundoffLimited)break;
            }
            maximumPressurePasses=std::max(maximumPressurePasses,std::min(pass,options.nonorthIterations));
            if(!std::isfinite(residual)||residual>1e-8) {
                std::ostringstream message;message<<std::setprecision(17)
                    <<"Incremental projection did not converge in actual cut volumes: call="<<pressureCalls
                    <<", passes="<<options.nonorthIterations<<", divergence="<<residual;
                throw std::runtime_error(message.str());
            }
            p=options.rho/dt*impulse;p.array()-=p.dot(volumes)/volume;
            return;
        }
        const Vector rhs=-divergence*q.high;
        Vector impulse=dt/options.rho*p;
        const int passes=mesh.coarseFineFaces?options.nonorthIterations:1;
        double residual=std::numeric_limits<double>::infinity();int pass=0;
        for(pass=1;pass<=passes;++pass) {
            const auto begin=std::chrono::steady_clock::now();
            Vector effective=rhs-pressureDeferred*impulse;
            Vector pinned=effective;pinned[gauge]=0;
            impulse=linearSolve(pinned,false);
            int refine=0;
            for(;refine<4;++refine) {
                Vector defect=effective-pressureMatrix*impulse;
                if((defect.array().abs()/volumes.array()).maxCoeff()<1e-12)break;
                defect[gauge]=0;impulse+=linearSolve(defect,false);
            }
            residual=((rhs-pressureMatrix*impulse-pressureDeferred*impulse).array().abs()/volumes.array()).maxCoeff();
            if(pressureTrace.is_open()) {
                pressureTrace<<pressureCalls<<','<<dt<<','<<pass<<','<<refine<<','<<residual<<','
                    <<std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()<<'\n';
                checked(pressureTrace);
            }
            if(residual<1e-10)break;
        }
        maximumPressurePasses=std::max(maximumPressurePasses,std::min(pass,passes));
        if(!std::isfinite(residual)||residual>1e-8) {
            std::ostringstream message;message<<std::setprecision(17)
                <<"Projection pressure correction did not converge in actual cut volumes: call="<<pressureCalls
                <<", passes="<<passes<<", divergence="<<residual;
            throw std::runtime_error(message.str());
        }
        q.subtractProduct(pressureAreas,normalGradient*impulse);
        p=options.rho/dt*impulse;
        p.array()-=p.dot(volumes)/volume;
    }
    // BCG ordering follows Aphros (Copyright ETH Zurich, MIT;
    // see validation/aphros/LICENSE.aphros). Excluded faces remain unflagged.
    Field bcg(const Field& u,const Flux& q,const Field& source) const {
        // Flux is shared by all velocity components. Stream each component's
        // derivatives, and evaluate only the upwind Taylor row at an interface;
        // two dense nf-by-3 Taylor fields would otherwise be mostly unused.
        const bool traceMemory=std::getenv("SIMPLE_BCG_MEMORY")!=nullptr;
        if(traceMemory)constructionMemory("bcg.begin");
        Field transverseSpeed=Field::Zero(nc,3);
        {
            Vector sideFlux=Vector::Zero(nc*6);
            Vector sideLow;if(q.compensated())sideLow=Vector::Zero(nc*6);
            if(traceMemory)constructionMemory("bcg.side_flux_workspace");
            for(int j=0;j<nf;++j) {
                const auto& f=mesh.faces[j];if(f.neighbor<0)continue;
                for(int id:{6*f.owner+2*f.axis+1,6*f.neighbor+2*f.axis}) {
                    if(q.compensated()) {
                        const auto sum=FluxPair{sideFlux[id],sideLow[id]}+q.get(j);
                        sideFlux[id]=sum.high;sideLow[id]=sum.low;
                    } else sideFlux[id]+=q[j];
                }
            }
            // These speeds are independent of the predicted component and
            // upwind face. Consume both flux parts with the same arithmetic,
            // then release the side sums before allocating face gradients.
            for(int c=0;c<nc;++c)for(int axis=0;axis<3;++axis) {
                const int a=6*c+2*axis,b=a+1;
                if(flagged[a]||flagged[b])continue;
                const double area=sideArea[a]+sideArea[b];
                if(area>0)transverseSpeed(c,axis)=q.compensated()?((FluxPair{sideFlux[a],sideLow[a]}+FluxPair{sideFlux[b],sideLow[b]})/area).rounded():(sideFlux[a]+sideFlux[b])/area;
            }
        }
        if(traceMemory)constructionMemory("bcg.transverse_speed_ready");
        Field result=Field::Zero(nf,3);
        for(int d=0;d<3;++d) {
            const Vector gradient=op->faceGradient*u.col(d);
            Vector sideGradient=Vector::Zero(nc*6);
            if(traceMemory)constructionMemory("bcg.component_workspace");
            for(int j=0;j<nf;++j) {
                const auto& f=mesh.faces[j];if(f.neighbor<0)continue;
                for(int id:{6*f.owner+2*f.axis+1,6*f.neighbor+2*f.axis})sideGradient[id]+=areas[j]*gradient[j];
            }
            for(int id=0;id<nc*6;++id)if(sideArea[id]>0)sideGradient[id]/=sideArea[id];
            for(int j=0;j<nf;++j) {
                const auto& f=mesh.faces[j];if(f.neighbor<0)continue;
                const int sign=q[j]>0?1:-1,c=sign>0?f.owner:f.neighbor;
                const int fm=6*c+2*f.axis,fp=fm+1;
                const bool interface=mesh.cells[f.owner].level!=mesh.cells[f.neighbor].level;
                const double slope=(flagged[fm]||flagged[fp]||interface)?gradient[j]:.5*(sideGradient[fm]+sideGradient[fp]);
                const double weight=mesh.cells[f.neighbor].h/(mesh.cells[f.owner].h+mesh.cells[f.neighbor].h);
                double temporal=weight*source(f.owner,d)+(1-weight)*source(f.neighbor,d)-slope*q.quotient(j,areas[j]);
                for(int offset=1;offset<=2;++offset) {
                    const int axis=(f.axis+offset)%3,a=6*c+2*axis,b=a+1;
                    if(flagged[a]||flagged[b])continue;
                    const double area=sideArea[a]+sideArea[b];
                    if(!(area>0))throw std::runtime_error("BCG upwind cell has no transverse open face");
                    const double speed=transverseSpeed(c,axis);
                    temporal-=sideGradient[speed>0?a:b]*speed;
                }
                if(interface) {
                    const auto& taylor=sign>0?cfTaylorOwner:cfTaylorNeighbor;
                    // Use the same Eigen sparse product as the full field:
                    // its row reduction may use multiple accumulators.
                    const Eigen::Matrix<double,1,1> value=taylor.middleRows(j,1)*u.col(d);
                    result(j,d)=value[0];
                } else result(j,d)=u(c,d)+slope*(sign*.5*mesh.cells[c].h);
                result(j,d)+=.5*options.timeStep*temporal;
            }
        }
        return result;
    }
    Flux predictedFlux(const Field& u,const Flux& q,const Field& source) {
        Flux result;
        {
            const Field predicted=bcg(u,q,source);result=makeFlux(Vector::Zero(nf));
            for(int j=0;j<nf;++j)if(mesh.faces[j].neighbor>=0)result.setDouble(j,areas[j]*predicted(j,mesh.faces[j].axis));
        }
        if(std::getenv("SIMPLE_BCG_MEMORY"))constructionMemory("predictor.face_velocity_released");
        Vector p=Vector::Zero(nc);
        project(result,p,.5*options.timeStep);return result;
    }
    Field advectiveRate(const Field& u,const Flux& q,const Field& source) const {
        const Field face=bcg(u,q,source);
        if(q.compensated()) {
            Field result(nc,3);
            for(int d=0;d<3;++d) {
                const auto balance=q.negativeWeightedDivergence(mesh,face,d).multiplied(op->redistribution);
                for(int c=0;c<nc;++c)result(c,d)=balance.quotient(c,volumes[c]);
            }
            return result;
        }
        Field result=op->redistribution*(-divergence*(q.high.asDiagonal()*face));
        for(int c=0;c<nc;++c)result.row(c)/=volumes[c];
        return result;
    }
    double steadyResidual() {
        const Field source=acceleration(pressure);const Flux predicted=predictedFlux(velocity,flux,source);
        Field residual;
        if(storedFullDiffusion)residual=diffusionMatrix*velocity;
        else residual=(diffusionMatrix+diffusionDeferred)*velocity;
        for(int c=0;c<nc;++c)residual.row(c)/=options.rho*volumes[c];
        residual-=source+advectiveRate(velocity,predicted,source);
        return norm(residual)/accelerationScale;
    }
    double fixedPointResidual(const Field& old,const Flux& oldFlux,int step) {
        // Residual of all velocity/diffusion blocks at an inner fixed point,
        // retaining the old PHYSICAL time fields and the special first-step
        // initial-pressure map. This is also the acceptance test for AA trials.
        Field source;
        {
        Vector sourcePressure=pressure;
        if(step==1) {
            Flux initial=makeFlux(velocityFlux(velocity)+options.timeStep*faceForce);
            project(initial,sourcePressure,options.timeStep);
        }
        source=acceleration(sourcePressure);
        }
        double diffusionNorm;
        {
        const Flux predicted=predictedFlux(old,oldFlux,source);
        const Field intermediate=old+options.timeStep*(advectiveRate(old,predicted,source)+source);
        Field diffusion;
        if(storedFullDiffusion)diffusion=diffusionMatrix*diffusionGuess+mass.asDiagonal()*(diffusionGuess-intermediate);
        else diffusion=(diffusionMatrix+diffusionDeferred)*diffusionGuess+mass.asDiagonal()*(diffusionGuess-intermediate);
        for(int c=0;c<nc;++c)diffusion.row(c)/=options.rho*volumes[c];
        diffusionNorm=norm(diffusion);
        }
        const Field correction=(diffusionGuess-velocity)/options.timeStep+acceleration(pressure)-source;
        return std::hypot(diffusionNorm,norm(correction))/accelerationScale;
    }
    Vector packState() const {
        const double us=accelerationScale*mesh.extent[1]*mesh.extent[1]/options.nu;
        const double ps=options.rho*accelerationScale*mesh.extent[0];
        Vector state(7*nc+nf);
        for(int c=0;c<nc;++c) {
            for(int d=0;d<3;++d)state[7*c+d]=velocity(c,d)/us;
            state[7*c+3]=pressure[c]/ps;
            for(int d=0;d<3;++d)state[7*c+4+d]=diffusionGuess(c,d)/us;
        }
        for(int j=0;j<nf;++j)state[7*nc+j]=flux.quotient(j,areas[j]*us);
        return state;
    }
    void unpackState(const Vector& state) {
        const double us=accelerationScale*mesh.extent[1]*mesh.extent[1]/options.nu;
        const double ps=options.rho*accelerationScale*mesh.extent[0];
        for(int c=0;c<nc;++c) {
            for(int d=0;d<3;++d)velocity(c,d)=state[7*c+d]*us;
            pressure[c]=state[7*c+3]*ps;
            for(int d=0;d<3;++d)diffusionGuess(c,d)=state[7*c+4+d]*us;
        }
        // Anderson candidates are approximate double guesses. Raw outputs are
        // restored as complete Flux objects on rejection, never via this map.
        for(int j=0;j<nf;++j)flux.setDouble(j,state[7*nc+j]*areas[j]*us);
    }
    double continuityResidual() const {
        const double speed=std::sqrt(velocity.rowwise().squaredNorm().maxCoeff());
        return (pressureDefect(flux).array().abs()/volumes.array()).maxCoeff()/std::max(speed/mesh.extent[1],1e-30);
    }
    bool repairTrialMass(double& relativeChange) {
        // Affine combinations of conservative outputs conserve in exact
        // arithmetic. Tiny cut volumes can magnify the summation roundoff.
        // A small projection restores mass, with paired u/p updates preserving
        // the collocated face relation. Reject changes larger than roundoff.
        const Flux oldFlux=flux;Vector correction=Vector::Zero(nc);
        project(flux,correction,options.timeStep);
        Field delta(nc,3);
        for(int d=0;d<3;++d)delta.col(d)=-options.timeStep/options.rho*(op->cellGradient[d]*correction);
        const double us=std::max(velocity.cwiseAbs().maxCoeff(),1e-30);
        const double ps=std::max(pressure.cwiseAbs().maxCoeff(),options.rho*accelerationScale*mesh.extent[0]*1e-10);
        const double qs=std::max(oldFlux.maximumVelocity(areas),us);
        relativeChange=std::max({delta.cwiseAbs().maxCoeff()/us,correction.cwiseAbs().maxCoeff()/ps,
                                flux.maximumVelocityDifference(oldFlux,areas)/qs});
        velocity+=delta;pressure+=correction;
        return std::isfinite(relativeChange)&&relativeChange<1e-10;
    }
    void dump(const std::filesystem::path& folder,const Field& advected,const Field& diffused,
              const Field& source,const Flux& predicted) const {
        std::filesystem::create_directories(folder);
        std::ofstream cells(folder/"cells.csv"),faces(folder/"faces.csv");
        cells<<std::setprecision(17)<<"id,x,y,z,u,v,w,p,u_adv,v_adv,w_adv,u_diff,v_diff,w_diff,source_x,source_y,source_z\n";
        for(int c=0;c<nc;++c) {
            cells<<c;for(double v:mesh.cells[c].center)cells<<','<<v;
            for(int d=0;d<3;++d)cells<<','<<velocity(c,d);cells<<','<<pressure[c];
            for(const Field* field:{&advected,&diffused,&source})for(int d=0;d<3;++d)cells<<','<<(*field)(c,d);
            cells<<'\n';
        }
        faces<<std::setprecision(17)<<"id,owner,neighbor,axis,area,flux,predicted_flux";
        if(flux.compensated())faces<<",flux_low,predicted_flux_low";
        faces<<'\n';
        for(int j=0;j<nf;++j) {
            const auto& f=mesh.faces[j];faces<<j<<','<<f.owner<<','<<f.neighbor<<','<<f.axis<<','<<f.area<<','<<flux.high[j]<<','<<predicted.high[j];
            if(flux.compensated())faces<<','<<flux.low[j]<<','<<predicted.get(j).low;
            faces<<'\n';
        }
        checked(cells);checked(faces);
    }
    nlohmann::json writeStep(const std::filesystem::path& folder,const Field& old,const Flux& oldFlux,int step,int iterations,bool converged,
                             double change,double diffusionResidual,double continuity,bool writeFields) {
        const bool steadyMode=options.steadyAndersonDepth>0;
        const double steadyKnown=steadyMode?steadyResidual():-1;
        const double temporalKnown=steadyMode?norm((velocity-old)/options.timeStep)/accelerationScale:-1;
        const double stationaryComplete=steadyMode?fixedPointResidual(velocity,flux,2):-1;
        if(steadyMode&&converged&&steadyKnown<std::min(options.tolerance,1e-8)&&temporalKnown<1e-8&&
           stationaryComplete<std::min(options.tolerance,1e-8))writeFields=true;
        writeFields=writeFields||!converged;
        std::ofstream cells,walls,faceFile;
        if(writeFields) {
            cells.open(folder/"solution.csv");walls.open(folder/"walls.csv");faceFile.open(folder/"flux.csv");
            cells<<std::setprecision(17)<<"id,level,x,y,z,h,volume,u,v,w,p\n";
            walls<<std::setprecision(17)<<"face_id,owner,x,y,z,nx,ny,nz,area,du_dn,dv_dn,dw_dn,tau_x,tau_y,tau_z\n";
            faceFile<<std::setprecision(17)<<"id,flux";
            if(flux.compensated())faceFile<<",flux_low";
            faceFile<<'\n';
        }
        std::vector<Vec3> nativeVelocity(nc);std::vector<double> nativePressure(nc);
        int cuts=0;double speed=0;
        for(int c=0;c<nc;++c) {
            const auto& cell=mesh.cells[c];
            if(writeFields) {
                cells<<c<<','<<cell.level;for(double v:cell.center)cells<<','<<v;
                cells<<','<<cell.h<<','<<cell.volume;
                for(int d=0;d<3;++d)cells<<','<<velocity(c,d);cells<<','<<pressure[c]<<'\n';
            }
            nativeVelocity[c]=velocity.row(c).transpose();nativePressure[c]=pressure[c];
            cuts+=cell.cut;speed=std::max(speed,nativeVelocity[c].norm());
        }
        const Field derivative=op->wallGradient*velocity;
        double wallArea=0,shearSum=0,maxh=0;
        for(const auto& c:mesh.cells)maxh=std::max(maxh,c.h);
        std::map<double,std::pair<double,double>> sections;
        std::map<double,double> sectionLow;
        for(int j=0;j<nf;++j) {
            const auto& f=mesh.faces[j];if(writeFields) {
                faceFile<<j<<','<<flux.high[j];if(flux.compensated())faceFile<<','<<flux.low[j];faceFile<<'\n';
            }
            if(f.neighbor>=0) {
                if(f.axis==0&&std::abs(f.center[0]/maxh-std::round(f.center[0]/maxh))<1e-8) {
                    double x=std::round(f.center[0]/maxh)*maxh;if(x>=mesh.extent[0]-1e-12)x=0;
                    if(flux.compensated()) {
                        const auto sum=FluxPair{sections[x].first,sectionLow[x]}+flux.get(j);
                        sections[x].first=sum.high;sectionLow[x]=sum.low;
                    } else sections[x].first+=flux[j];
                    sections[x].second+=areas[j];
                }
                continue;
            }
            const Vec3 n=f.embeddedNormal,d=derivative.row(j).transpose();
            const Vec3 tau=options.rho*options.nu*(d-n*d.dot(n));
            if(writeFields) {
                walls<<j<<','<<f.owner;for(const Vec3& v:{f.center,n})for(int d=0;d<3;++d)walls<<','<<v[d];
                walls<<','<<f.area;for(const Vec3& v:{d,tau})for(int d=0;d<3;++d)walls<<','<<v[d];walls<<'\n';
            }
            wallArea+=areas[j];shearSum+=areas[j]*tau.norm();
        }
        if(writeFields) {checked(cells);checked(walls);checked(faceFile);}
        std::ofstream sectionFile(folder/"sections.csv");sectionFile<<std::setprecision(17)<<"x,volume_flux,area\n";
        double qmin=std::numeric_limits<double>::max(),qmax=-qmin,qsum=0;
        for(const auto& s:sections) {
            const double sectionFlux=flux.compensated()?FluxPair{s.second.first,sectionLow[s.first]}.rounded():s.second.first;
            sectionFile<<s.first<<','<<sectionFlux<<','<<s.second.second<<'\n';
            qmin=std::min(qmin,sectionFlux);qmax=std::max(qmax,sectionFlux);qsum+=sectionFlux;
        }
        checked(sectionFile);if(sections.empty())throw std::runtime_error("No complete projection pipe section");
        const double q=qsum/sections.size(),spread=(qmax-qmin)/std::max(std::abs(q),1e-30);
        if(!writeFields&&spread>=1e-8) {
            sectionFile.close();
            return writeStep(folder,old,oldFlux,step,iterations,false,change,diffusionResidual,continuity,true);
        }
        const double steady=steadyMode?steadyKnown:steadyResidual();
        const double temporal=steadyMode?temporalKnown:norm((velocity-old)/options.timeStep)/accelerationScale;
        converged=converged&&spread<1e-8;
        nlohmann::json report{{"converged",converged},{"iterations",iterations},{"fluid_solver","proj"},
            {"field_output_written",writeFields},
            {"conservative_flux_storage",flux.compensated()?"twofold":"double"},
            {"mesh_backend",mesh.backend},{"embedded",true},{"geometry_source",mesh.geometrySource},
            {"cells",nc},{"faces",nf},{"cut_cells",cuts},{"coarse_fine_faces",mesh.coarseFineFaces},
            {"rho",options.rho},{"nu",options.nu},{"convection",true},{"convection_scheme","bcg"},
            {"time_step",options.timeStep},{"temporal_acceleration_relative_l2",temporal},
            {"steady_momentum_relative_l2",steady},{"implicit_diffusion_relative_l2",diffusionResidual},
            {"continuity_relative_linf",continuity},{"velocity_change_absolute_linf",change},
            {"speed_max",speed},{"volume",volume},{"volume_flux",q},{"cross_section_flux_relative_spread",spread},
            {"wall_area",wallArea},{"mean_wall_shear_magnitude",shearSum/wallArea},{"maximum_pressure_passes",maximumPressurePasses},
            {"roundoff_pressure_exits",roundoffPressureExits},
            {"cycle_pressure_exits",cyclePressureExits},
            {"scope","Original projection split on uniform cut grid; quadratic conservative subface extension on octree interfaces; time/grid convergence must be checked separately"}};
        if(options.andersonDepth||steadyMode) {
            report["anderson_depth"]=options.andersonDepth;
            report["anderson_accepted_steps"]=acceleratedSteps;
            report["anderson_rejected_steps"]=rejectedAccelerations;
            report["complete_inner_fixed_point_residual"]=fixedPointResidual(old,oldFlux,step);
        }
        if(steadyMode) {
            report["fluid_solver"]="proj_steady";
            report["steady_iteration"]=step;
            report["steady_anderson_depth"]=options.steadyAndersonDepth;
            report["steady_map_velocity_defect_relative_l2"]=temporal;
            report["steady_complete_fixed_point_residual"]=stationaryComplete;
            report.erase("temporal_acceleration_relative_l2");
            report["scope"]="Steady fixed-point iteration of the unchanged BCG/implicit-diffusion projection map; iteration indices and pseudo time are not a physical trajectory";
        }
        std::ofstream metrics(folder/"metrics.json");metrics<<report.dump(2)<<'\n';checked(metrics);
        if(writeFields)for(const auto& name:{"mesh_cells.csv","mesh_faces.csv"}) {
            std::error_code error;const auto source=options.output/name,destination=folder/name;
            std::filesystem::create_hard_link(source,destination,error);
            if(error) {
                error.clear();std::filesystem::copy_file(source,destination,std::filesystem::copy_options::none,error);
                if(error)throw std::runtime_error("Cannot share immutable projection mesh output: "+error.message());
            }
        }
        // Synchronize the native physical channels on EVERY step, preserving
        // exactly the same device state even when serialization is omitted.
        exportNativeFields(mesh,nativeVelocity,nativePressure,(folder/"native_fields.bin").string(),bool(gpu),writeFields);
        return report;
    }
    nlohmann::json run() {
        const auto root=options.output;const double dt=options.timeStep;
        const bool steadyMode=options.steadyAndersonDepth>0;
        std::ofstream historyStorage;
        if(andersonFileHistory) {
            historyStorage.open(root/"anderson_storage.csv");
            historyStorage<<"kind,step,iteration,stored_history_values,resident_history_values,peak_scratch_values,written_bytes,read_bytes\n";
            checked(historyStorage);
        }
        const auto recordHistoryStorage=[&](const char* kind,int step,int iteration,const AndersonAcceleration& a) {
            if(!andersonFileHistory)return;
            const auto& s=a.storageStatistics();
            historyStorage<<kind<<','<<step<<','<<iteration<<','<<a.storedHistoryValues()<<','<<a.residentHistoryValues()<<','
                <<s.peakScratchValues<<','<<s.writtenBytes<<','<<s.readBytes<<'\n';checked(historyStorage);
        };
        nlohmann::json config;std::ifstream(root/"case.json")>>config;
        std::ofstream times(root/(steadyMode?"steady_history.csv":"time_history.csv"));
        times<<std::setprecision(17)<<(steadyMode?
            "iteration,pseudo_time,inner_iterations,steady_map_velocity_defect_relative_l2,steady_momentum_relative_l2,inner_converged\n":
            "step,time,inner_iterations,temporal_acceleration_relative_l2,steady_momentum_relative_l2,inner_converged\n");
        std::unique_ptr<AndersonAcceleration> steadyAccelerator;
        std::ofstream steadyAccelerationLog;
        std::ofstream steadyAccelerationFailures;
        if(steadyMode) {
            steadyAccelerator=std::make_unique<AndersonAcceleration>(options.steadyAndersonDepth);
            if(andersonFileHistory)steadyAccelerator->useFileHistory(root/"anderson_history"/"steady");
            steadyAccelerationLog.open(root/"steady_acceleration.csv");
            steadyAccelerationLog<<std::setprecision(17)<<"after_iteration,accepted,gamma_norm,backtracking_factor,raw_equation_residual,candidate_equation_residual,candidate_continuity,roundoff_repair_relative_change\n";
        }
        int completed=restartStep;bool steady=false;
        std::vector<int> fieldSteps;
        if(restartStep) {
            for(int i=0;i<restartStep;++i) {
                const auto& row=restart.at("time_history").at(i);
                if(row.at("step").get<int>()!=i+1 || !row.at("inner_converged").get<bool>() ||
                   row.at("time").get<double>()!=(i+1)*dt)
                    throw std::runtime_error("Invalid completed restart time history");
                times<<i+1<<','<<row.at("time").get<double>()<<','<<row.at("inner_iterations").get<int>()<<','
                     <<row.at("temporal_acceleration_relative_l2").get<double>()<<','
                     <<row.at("steady_momentum_relative_l2").get<double>()<<",true\n";
            }
            fieldSteps=restart.at("field_output_steps").get<std::vector<int>>();
            if(fieldSteps.empty() || fieldSteps.front()!=1 || fieldSteps.back()!=restartStep ||
               !std::is_sorted(fieldSteps.begin(),fieldSteps.end()) ||
               std::adjacent_find(fieldSteps.begin(),fieldSteps.end())!=fieldSteps.end())
                throw std::runtime_error("Invalid restart field schedule");
            checked(times);
        }
        std::filesystem::path finalPath;
        for(int step=restartStep+1;step<=options.timeSteps;++step) {
            activeStep=step;
            const bool fieldStep=step==1||step==options.timeSteps||step%options.outputStride==0;
            const Field old=velocity;const Flux oldFlux=flux;
            // The first map has a special pressure initialization. Start the
            // outer history at the second map, whose definition is stationary.
            Vector steadyInput;if(steadyMode&&step>1)steadyInput=packState();
            acceleratedSteps=0;rejectedAccelerations=0;
            std::ostringstream name;name<<(steadyMode?"iterate_":"step_")<<std::setw(4)<<std::setfill('0')<<step;
            const auto folder=root/name.str();finalPath=folder;std::filesystem::create_directories(folder);
            auto current=config;current["output"]=folder.string();current["physical_step"]=step;current["physical_time"]=step*dt;
            if(steadyMode) {
                current["fluid_solver"]="proj_steady";current["steady_iteration"]=step;current["pseudo_time"]=step*dt;
                current.erase("physical_step");current.erase("physical_time");
            }
            std::ofstream caseFile(folder/"case.json");caseFile<<current.dump(2)<<'\n';checked(caseFile);
            std::ofstream history(folder/"history.csv");history<<std::setprecision(17)<<"iteration,velocity_change_absolute_linf,implicit_diffusion_relative_l2,continuity_relative_linf\n";
            std::unique_ptr<AndersonAcceleration> accelerator;
            std::ofstream accelerationLog;
            std::ofstream accelerationFailures;
            if(options.andersonDepth) {
                accelerator=std::make_unique<AndersonAcceleration>(options.andersonDepth);
                if(andersonFileHistory)accelerator->useFileHistory(root/"anderson_history"/name.str());
                accelerationLog.open(folder/"acceleration.csv");
                accelerationLog<<std::setprecision(17)<<"after_iteration,accepted,gamma_norm,backtracking_factor,raw_fixed_point_residual,candidate_fixed_point_residual,candidate_continuity,roundoff_repair_relative_change\n";
            }
            bool converged=false;int iteration=0;double change=0,diffusionResidual=0,continuity=0;
            for(iteration=1;iteration<=options.maxIterations;++iteration) {
                activeIteration=iteration;activeTrial=0;linearPhase="initial_pressure";
                Vector iterationInput;if(accelerator)iterationInput=packState();
                double completeResidual=0;
                {
                Field previous=velocity;
                if(step==1) {Flux initial=makeFlux(velocityFlux(velocity)+dt*faceForce);project(initial,pressure,dt);}
                linearPhase="predictor";
                const Field source=acceleration(pressure);const Flux predicted=predictedFlux(old,oldFlux,source);
                const Field intermediate=old+dt*(advectiveRate(old,predicted,source)+source);
                Field diffused(nc,3);
                linearPhase="implicit_diffusion";
                {
                Field rhs=mass.asDiagonal()*intermediate;
                if(options.gpuViscosityOperator!="full")rhs-=diffusionDeferred*diffusionGuess;
                for(int d=0;d<3;++d)diffused.col(d)=linearSolve(rhs.col(d),true);
                }
                diffusionGuess=diffused;
                {
                const Field provisional=diffused-dt*source;
                flux=makeFlux(velocityFlux(provisional)+dt*faceForce);
                linearPhase="velocity_projection";project(flux,pressure,dt);velocity=provisional+dt*acceleration(pressure);
                }
                change=(velocity-previous).cwiseAbs().maxCoeff();
                previous.resize(0,3);
                if(!velocity.allFinite())throw std::runtime_error("Nonfinite projection velocity");
                {
                Field residual;
                if(storedFullDiffusion)residual=diffusionMatrix*diffused+mass.asDiagonal()*(diffused-intermediate);
                else residual=(diffusionMatrix+diffusionDeferred)*diffused+mass.asDiagonal()*(diffused-intermediate);
                for(int c=0;c<nc;++c)residual.row(c)/=options.rho*volumes[c];
                diffusionResidual=norm(residual)/accelerationScale;
                }
                const double speed=std::sqrt(velocity.rowwise().squaredNorm().maxCoeff());
                continuity=(pressureDefect(flux).array().abs()/volumes.array()).maxCoeff()/std::max(speed/mesh.extent[1],1e-30);
                history<<iteration<<','<<change<<','<<diffusionResidual<<','<<continuity<<'\n';
                converged=change<options.projectionIterationTolerance&&diffusionResidual<options.tolerance&&continuity<1e-7;
                linearPhase="raw_fixed_point";
                constructionMemory("iteration.raw_fixed_point_begin");
                completeResidual=(accelerator||steadyMode)?fixedPointResidual(old,oldFlux,step):0;
                if(accelerator||steadyMode)converged=converged&&completeResidual<options.tolerance;
                if(iteration==1||iteration%100==0||converged)
                    std::cout<<(steadyMode?"steady iteration=":"projection step=")<<step<<" iter="<<iteration<<" change="<<std::setprecision(12)<<change<<" diffusion="<<diffusionResidual<<" continuity="<<continuity<<std::endl;
                if((!converged&&iteration==options.maxIterations)||((fieldStep||(steadyMode&&converged))&&(converged||std::find(options.dumpIterations.begin(),options.dumpIterations.end(),iteration)!=options.dumpIterations.end())))
                    dump(folder/("iter_"+std::to_string(iteration)),intermediate,diffused,source,predicted);
                }
                constructionMemory("iteration.raw_temporaries_released");
                if(converged)break;
                // Only alter the next input. All acceptance checks and dumps
                // above describe a fresh, unaccelerated projection map output.
                if(accelerator&&iteration<options.maxIterations) {
                    const Vector output=packState();const Vector candidate=accelerator->update(iterationInput,output);
                    iterationInput.resize(0);
                    recordHistoryStorage("inner",step,iteration,*accelerator);
                    constructionMemory("iteration.anderson_history_updated");
                    bool accepted=false;double factor=1,trialResidual=completeResidual,trialContinuity=continuity,repairChange=0;
                    bool restoredExactly=false;
                    Field rawVelocity,rawDiffusion;Vector rawPressure;Flux rawFlux;
                    if(accelerator->accepted()) {
                        // pack/unpack scales physical values and can round them.
                        // Keep an exact fallback when a trial's linear solve fails.
                        rawVelocity=velocity;rawDiffusion=diffusionGuess;rawPressure=pressure;rawFlux=flux;
                        bool numericalFailure=false;
                        for(int attempt=0;attempt<4;++attempt,factor*=.5) {
                            activeTrial=attempt+1;
                            unpackState(output+factor*(candidate-output));
                            trialContinuity=continuityResidual();
                            repairChange=0;
                            if(!std::isfinite(trialContinuity))continue;
                            try {
                                if(trialContinuity>=1e-7) {
                                    linearPhase="anderson_mass_repair";
                                    if(!repairTrialMass(repairChange))continue;
                                    trialContinuity=continuityResidual();
                                    if(trialContinuity>=1e-7)continue;
                                }
                                linearPhase="anderson_fixed_point";
                                trialResidual=fixedPointResidual(old,oldFlux,step);
                            } catch(const NativeGpuConvergenceError& error) {
                                // Reject only this optional candidate. Main physical
                                // solves and the raw fixed-point check still propagate.
                                numericalFailure=true;trialResidual=std::numeric_limits<double>::infinity();
                                if(!accelerationFailures.is_open()) {
                                    accelerationFailures.open(folder/"acceleration_linear_failures.csv");
                                    accelerationFailures<<std::setprecision(17)<<"after_iteration,attempt,factor,phase,failure_dump,true_relative_residual,tolerance,linear_iterations,failed_solution_accepted\n";
                                }
                                accelerationFailures<<iteration<<','<<activeTrial<<','<<factor<<','<<linearPhase<<','<<linearFailures<<','
                                    <<error.relativeResidual<<','<<error.tolerance<<','<<error.iterations<<",false\n";checked(accelerationFailures);
                                continue;
                            }
                            if(std::isfinite(trialResidual)&&trialResidual<completeResidual) {accepted=true;break;}
                        }
                        if(!accepted&&numericalFailure) {
                            velocity=rawVelocity;diffusionGuess=rawDiffusion;pressure=rawPressure;flux=rawFlux;restoredExactly=true;
                        }
                    }
                    activeTrial=0;
                    if(accepted)++acceleratedSteps;
                    else {
                        if(!restoredExactly) {
                            if(!flux.compensated())unpackState(output);
                            else if(rawFlux.size()) {velocity=rawVelocity;diffusionGuess=rawDiffusion;pressure=rawPressure;flux=rawFlux;}
                        }
                        ++rejectedAccelerations;
                    }
                    accelerationLog<<iteration<<','<<accepted<<','<<accelerator->coefficientNorm()<<','<<(accepted?factor:0.)<<','
                        <<completeResidual<<','<<trialResidual<<','<<trialContinuity<<','<<repairChange<<'\n';checked(accelerationLog);
                }
            }
            checked(history);
            activeTrial=0;linearPhase="step_diagnostics";
            auto report=writeStep(folder,old,oldFlux,step,std::min(iteration,options.maxIterations),converged,change,diffusionResidual,continuity,fieldStep);
            if(report["field_output_written"].get<bool>())fieldSteps.push_back(step);
            const auto temporalKey=steadyMode?"steady_map_velocity_defect_relative_l2":"temporal_acceleration_relative_l2";
            times<<step<<','<<step*dt<<','<<report["iterations"]<<','<<report[temporalKey]<<','<<report["steady_momentum_relative_l2"]<<','<<report["converged"]<<'\n';checked(times);
            if(!report["converged"].get<bool>())break;
            ++completed;steady=report["steady_momentum_relative_l2"].get<double>()<std::min(options.tolerance,1e-8)&&report[temporalKey].get<double>()<1e-8;
            if(steadyMode) {
                steady=steady&&report["steady_complete_fixed_point_residual"].get<double>()<std::min(options.tolerance,1e-8);
                if(steady)break;
                if(step>1&&step<options.timeSteps) {
                    // Accept only a better stationary equation residual. The
                    // next iteration always executes the original map again;
                    // an extrapolated state itself is never the final solution.
                    accelerator.reset();
                    const Vector output=packState();const Vector candidate=steadyAccelerator->update(steadyInput,output);
                    steadyInput.resize(0);
                    recordHistoryStorage("steady",step,0,*steadyAccelerator);
                    const Field rawVelocity=velocity,rawDiffusion=diffusionGuess;
                    const Vector rawPressure=pressure;const Flux rawFlux=flux;
                    const double rawResidual=std::max(report["steady_momentum_relative_l2"].get<double>(),
                        report["steady_complete_fixed_point_residual"].get<double>());
                    bool accepted=false;double factor=1,trialResidual=rawResidual,trialContinuity=continuity,repairChange=0;
                    if(steadyAccelerator->accepted()) {
                        for(int attempt=0;attempt<4;++attempt,factor*=.5) {
                            activeTrial=attempt+1;unpackState(output+factor*(candidate-output));
                            trialContinuity=continuityResidual();repairChange=0;
                            if(!std::isfinite(trialContinuity))continue;
                            try {
                                if(trialContinuity>=1e-7) {
                                    linearPhase="steady_anderson_mass_repair";
                                    if(!repairTrialMass(repairChange))continue;
                                    trialContinuity=continuityResidual();
                                    if(trialContinuity>=1e-7)continue;
                                }
                                linearPhase="steady_anderson_equations";
                                const double momentumResidual=steadyResidual();
                                const double stationaryResidual=fixedPointResidual(velocity,flux,2);
                                // std::max can hide a NaN in its second operand.
                                // Both original equation blocks must be finite.
                                trialResidual=std::isfinite(momentumResidual)&&std::isfinite(stationaryResidual)?
                                    std::max(momentumResidual,stationaryResidual):std::numeric_limits<double>::infinity();
                            } catch(const NativeGpuConvergenceError& error) {
                                if(!steadyAccelerationFailures.is_open()) {
                                    steadyAccelerationFailures.open(root/"steady_acceleration_failures.csv");
                                    steadyAccelerationFailures<<std::setprecision(17)<<"after_iteration,attempt,factor,phase,failure_dump,true_relative_residual,tolerance,failed_solution_accepted\n";
                                }
                                steadyAccelerationFailures<<step<<','<<activeTrial<<','<<factor<<','<<linearPhase<<','<<linearFailures<<','
                                    <<error.relativeResidual<<','<<error.tolerance<<",false\n";checked(steadyAccelerationFailures);
                                trialResidual=std::numeric_limits<double>::infinity();continue;
                            }
                            if(std::isfinite(trialResidual)&&trialResidual<rawResidual) {accepted=true;break;}
                        }
                    }
                    activeTrial=0;
                    if(!accepted) {velocity=rawVelocity;diffusionGuess=rawDiffusion;pressure=rawPressure;flux=rawFlux;}
                    steadyAccelerationLog<<step<<','<<accepted<<','<<steadyAccelerator->coefficientNorm()<<','<<(accepted?factor:0.)<<','
                        <<rawResidual<<','<<trialResidual<<','<<trialContinuity<<','<<repairChange<<'\n';checked(steadyAccelerationLog);
                }
            }
        }
        const bool all=steadyMode?steady:completed==options.timeSteps;
        nlohmann::json summary{{"converged",all},{"steps_completed",completed},{"time_step",dt},
            {"output_stride",options.outputStride},{"field_output_steps",fieldSteps},
            {"fluid_solver","proj"},{"final_output",finalPath.string()},{"steady_converged",all&&steady},
            {"scope","BCG projection with implicit diffusion; inner convergence, time-step convergence and grid convergence are distinct checks"}};
        if(restartStep) {
            summary["restart_step"]=restartStep;
            summary["steps_computed_this_run"]=completed-restartStep;
            summary["restart_checkpoint"]=options.restartCheckpoint;
        }
        if(steadyMode) {
            summary["fluid_solver"]="proj_steady";summary["steady_anderson_depth"]=options.steadyAndersonDepth;
            summary["iterations_completed"]=completed;summary["maximum_steady_iterations"]=options.timeSteps;
            summary["field_output_iterations"]=fieldSteps;summary.erase("field_output_steps");
            summary.erase("steps_completed");summary["physical_trajectory_claimed"]=false;
            summary["scope"]="Steady Anderson acceleration between fully converged original projection maps; final fields are an unaccelerated map output with original momentum, fixed-point and mass gates";
        }
        std::ofstream out(root/(steadyMode?"steady_summary.json":"transient_summary.json"));out<<summary.dump(2)<<'\n';checked(out);return summary;
    }
};
ProjectionSolver::ProjectionSolver(Mesh mesh,Options options):impl_(new Impl(std::move(mesh),std::move(options))) {}
ProjectionSolver::~ProjectionSolver()=default;
nlohmann::json ProjectionSolver::run(){return impl_->run();}
nlohmann::json ProjectionSolver::replayPressure(const std::filesystem::path& rhs,int repetitions){return impl_->replayPressure(rhs,repetitions);}
nlohmann::json ProjectionSolver::replayProjection(const std::filesystem::path& flux,int repetitions){return impl_->replayProjection(flux,repetitions,impl_->options.timeStep);}
nlohmann::json ProjectionSolver::replayPredictor(const std::filesystem::path& state,int repetitions){return impl_->replayPredictor(state,repetitions);}
}
