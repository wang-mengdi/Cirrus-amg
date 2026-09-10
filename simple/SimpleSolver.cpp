#include "SimpleSolver.h"
#include "AndersonAcceleration.h"
#include "AmgPressure.h"
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <sstream>
#include <cstdlib>
#include <cstring>

namespace simple {
namespace {
constexpr double pi = 3.14159265358979323846;
using Triplet = Eigen::Triplet<double>;
template<class Sparse>
bool identicalMatrix(const Sparse& a,const Sparse& b) {
    if(a.rows()!=b.rows() || a.cols()!=b.cols() || a.nonZeros()!=b.nonZeros() ||
       !a.isCompressed() || !b.isCompressed())return false;
    return std::memcmp(a.valuePtr(),b.valuePtr(),a.nonZeros()*sizeof(typename Sparse::Scalar))==0 &&
        std::memcmp(a.innerIndexPtr(),b.innerIndexPtr(),a.nonZeros()*sizeof(typename Sparse::StorageIndex))==0 &&
        std::memcmp(a.outerIndexPtr(),b.outerIndexPtr(),(a.outerSize()+1)*sizeof(typename Sparse::StorageIndex))==0;
}
double axialDrive(const Mesh& mesh, const Options& options) {
    return options.force[0] + (options.periodicX ? 0. :
        (options.pressureIn - options.pressureOut) / (options.rho * mesh.extent[0]));
}
double exactVelocity(const Vec3& x, const Mesh& mesh, const Options& options) {
    const double h = mesh.extent[1], w = mesh.extent[2], g = axialDrive(mesh, options);
    double value = g * x[1] * (h - x[1]) / (2 * options.nu);
    if (!options.periodicZ) {
        double sum = 0.;
        const double distance = std::abs(x[2] - w / 2);
        for (int n = 1; n <= 511; n += 2) {
            const double k = n * pi / h;
            // Stable cosh(k*(z-W/2))/cosh(k*W/2), avoiding overflow.
            const double ratio = std::exp(k * (distance - w / 2)) *
                (1 + std::exp(-2 * k * distance)) / (1 + std::exp(-k * w));
            sum += std::sin(n * pi * x[1] / h) * ratio / (double(n) * n * n);
        }
        value -= 4 * g * h * h * sum / (options.nu * pi * pi * pi);
    }
    return value;
}
double exactMeanVelocity(const Mesh& mesh, const Options& options) {
    const double h = mesh.extent[1], w = mesh.extent[2];
    double correction = 1.;
    if (!options.periodicZ) {
        double sum = 0.;
        for (int n = 1; n <= 1023; n += 2)
            sum += std::tanh(n * pi * w / (2 * h)) / std::pow(double(n), 5);
        correction -= 192 * h * sum / (std::pow(pi, 5) * w);
    }
    return axialDrive(mesh, options) * h * h * correction / (12 * options.nu);
}
Eigen::VectorXd component(const std::vector<Vec3>& v, int d) {
    Eigen::VectorXd q(v.size());
    for (int i = 0; i < q.size(); ++i) q[i] = v[i][d];
    return q;
}
void writeMatrix(const std::filesystem::path& path, const Eigen::SparseMatrix<double>& a) {
    std::ofstream out(path);
    out << std::setprecision(17) << "row,column,value\n";
    for (int j = 0; j < a.outerSize(); ++j)
        for (Eigen::SparseMatrix<double>::InnerIterator it(a, j); it; ++it)
            out << it.row() << ',' << it.col() << ',' << it.value() << '\n';
    out.flush();
    if(!out.good())throw std::runtime_error("Sparse matrix dump failed: "+path.string());
}
}

Options Options::fromJson(const nlohmann::json& j) {
    Options o;
    o.ny = j.value("ny", o.ny);
    o.maxIterations = j.value("max_iterations", o.maxIterations);
    o.nonorthIterations = j.value("nonorth_iterations", o.nonorthIterations);
    o.andersonDepth = j.value("anderson_depth", o.andersonDepth);
    if(j.contains("steady_anderson_depth")) {
        const auto& depth=j.at("steady_anderson_depth");
        if(!depth.is_number_integer())throw std::runtime_error("Steady Anderson depth must be an integer");
        // Check the JSON integer before narrowing: values such as 2^32 must
        // not wrap to zero and silently disable the requested steady mode.
        if(depth<0||depth>10)throw std::runtime_error("Steady Anderson depth must be 0..10");
    }
    o.steadyAndersonDepth = j.value("steady_anderson_depth", o.steadyAndersonDepth);
    o.pressureSolver = j.value("pressure_solver", o.pressureSolver);
    o.linearBackend = j.value("linear_backend", o.linearBackend);
    o.gpuPreconditioner = j.value("gpu_preconditioner", o.gpuPreconditioner);
    o.gpuPressureGauge = j.value("gpu_pressure_gauge", o.gpuPressureGauge);
    o.gpuPressureOperator = j.value("gpu_pressure_operator", o.gpuPressureOperator);
    o.gpuViscosityOperator = j.value("gpu_viscosity_operator", o.gpuViscosityOperator);
    o.gpuOrthogonalization = j.value("gpu_orthogonalization", o.gpuOrthogonalization);
    if(j.contains("gpu_krylov_dimension")&&!j.at("gpu_krylov_dimension").is_number_integer())
        throw std::runtime_error("GPU Krylov dimension must be an integer");
    o.gpuKrylovDimension = j.value("gpu_krylov_dimension", o.gpuKrylovDimension);
    o.embeddedGeometry = j.value("embedded_geometry", o.embeddedGeometry);
    o.momentumMode = j.value("momentum_mode", o.momentumMode);
    o.fluidSolver = j.value("fluid_solver", o.fluidSolver);
    o.restartCheckpoint = j.value("restart_checkpoint", o.restartCheckpoint);
    o.projectionIterationTolerance=j.value("projection_iteration_tolerance",o.projectionIterationTolerance);
    o.adaptive = j.value("adaptive", o.adaptive);
    o.periodicX = j.value("periodic_x", o.periodicX);
    o.periodicZ = j.value("periodic_z", o.periodicZ);
    o.quadraticInterfaces = j.value("quadratic_interfaces", o.quadraticInterfaces);
    o.convection = j.value("convection", o.convection);
    o.fluxRelaxationMemory = j.value("flux_relaxation_memory", o.fluxRelaxationMemory);
    o.rho = j.value("rho", o.rho); o.nu = j.value("nu", o.nu);
    o.alphaU = j.value("alpha_u", o.alphaU); o.alphaP = j.value("alpha_p", o.alphaP);
    o.pressureIn = j.value("pressure_in", o.pressureIn);
    o.pressureOut = j.value("pressure_out", o.pressureOut);
    o.perturbation = j.value("initial_perturbation", o.perturbation);
    o.tolerance = j.value("tolerance", o.tolerance);
    o.linearTolerance = j.value("linear_tolerance", o.linearTolerance);
    o.timeStep=j.value("time_step",o.timeStep);o.timeSteps=j.value("time_steps",o.timeSteps);
    o.outputStride=j.value("output_stride",o.outputStride);
    if (!o.periodicX) o.force.setZero();
    if (j.contains("force")) for (int d = 0; d < 3; ++d) o.force[d] = j.at("force").at(d);
    if (j.contains("dump_iterations")) o.dumpIterations = j.at("dump_iterations").get<std::vector<int>>();
    o.output = j.value("output", o.output.string());
    if (!(o.rho > 0 && o.nu > 0 && o.alphaU > 0 && o.alphaU <= 1 &&
          o.alphaP > 0 && o.alphaP <= 1 && o.tolerance > 0 && o.linearTolerance > 0 &&
          o.maxIterations > 0 && o.nonorthIterations > 0 && o.force.allFinite() &&
          o.timeStep>=0 && std::isfinite(o.timeStep) && o.timeSteps>0))
        throw std::runtime_error("Invalid SIMPLE physical/iteration parameters");
    for (double value : {o.rho, o.nu, o.alphaU, o.alphaP, o.pressureIn, o.pressureOut,
                         o.perturbation, o.tolerance, o.linearTolerance})
        if (!std::isfinite(value)) throw std::runtime_error("Nonfinite SIMPLE parameter");
    if (o.andersonDepth < 0 || o.andersonDepth > 10 ||
        (o.andersonDepth && o.convection && o.embeddedGeometry.empty()))
        throw std::runtime_error("Anderson depth must be 0..10; convection acceleration requires embedded geometry");
    if (o.pressureSolver != "auto" && o.pressureSolver != "cg" && o.pressureSolver != "ldlt" && o.pressureSolver != "amg")
        throw std::runtime_error("Pressure solver must be auto, cg, ldlt, or amg");
    if (o.force[1] != 0. || o.force[2] != 0.)
        throw std::runtime_error("Channel benchmark currently supports axial forcing only");
    if((o.timeSteps>1 && o.timeStep==0) || (o.timeStep>0 && (o.embeddedGeometry.empty() || !o.convection)))
        throw std::runtime_error("Time stepping currently requires embedded convection and positive time_step");
    if(o.fluidSolver!="simple" && o.fluidSolver!="proj")throw std::runtime_error("Fluid solver must be simple or proj");
    if(o.steadyAndersonDepth&&(o.fluidSolver!="proj"||o.linearBackend!="native_gpu"||
       o.gpuPreconditioner!="native_amg"||o.gpuPressureOperator!="full"||o.gpuViscosityOperator!="full"||
       o.gpuPressureGauge!="mean_zero"||!o.restartCheckpoint.empty()||o.timeStep<=0||o.timeSteps<2||
       j.value("operator_only",false)||j.value("geometry_only",false)))
        throw std::runtime_error("Steady acceleration requires full mean-zero native AMG projection, a positive pseudo step, at least two iterations and no physical restart or probe");
    if(!o.restartCheckpoint.empty() && (o.fluidSolver!="proj" || o.linearBackend!="native_gpu" ||
       o.gpuPreconditioner!="native_amg" || o.gpuPressureOperator!="full" || o.gpuViscosityOperator!="full" ||
       !o.andersonDepth || j.value("operator_only",false) || j.value("geometry_only",false)))
        throw std::runtime_error("Restart requires a complete native AMG projection state with full pressure/viscosity and Anderson checks");
    if(o.outputStride<1||(o.outputStride!=1&&o.fluidSolver!="proj"))
        throw std::runtime_error("Output stride must be positive; sparse field output currently requires projection");
    if(o.linearBackend!="cpu" && o.linearBackend!="native_gpu")throw std::runtime_error("Linear backend must be cpu or native_gpu");
    if(o.linearBackend=="native_gpu"&&o.fluidSolver!="proj")throw std::runtime_error("Native GPU linear integration currently requires projection");
    if(o.gpuPreconditioner!="jacobi"&&o.gpuPreconditioner!="native_amg")throw std::runtime_error("GPU preconditioner must be jacobi or native_amg");
    if(o.gpuPreconditioner=="native_amg"&&o.linearBackend!="native_gpu")throw std::runtime_error("Native AMG requires the native GPU linear backend");
    if(o.gpuOrthogonalization!="mgs2"&&o.gpuOrthogonalization!="cgs2")throw std::runtime_error("GPU orthogonalization must be mgs2 or cgs2");
    if(o.gpuOrthogonalization=="cgs2"&&(o.linearBackend!="native_gpu"||o.gpuPreconditioner!="native_amg"))
        throw std::runtime_error("Batched CGS2 requires native AMG/FGMRES");
    if(o.gpuKrylovDimension<2||o.gpuKrylovDimension>64)
        throw std::runtime_error("GPU Krylov dimension must be between 2 and 64");
    if(o.gpuKrylovDimension!=20&&(o.linearBackend!="native_gpu"||o.gpuPreconditioner!="native_amg"))
        throw std::runtime_error("Changing Krylov dimension requires native AMG/FGMRES");
    if(o.gpuPressureGauge!="pin"&&o.gpuPressureGauge!="mean_zero")throw std::runtime_error("GPU pressure gauge must be pin or mean_zero");
    if(o.gpuPressureGauge=="mean_zero"&&o.gpuPreconditioner!="native_amg")throw std::runtime_error("Mean-zero GPU pressure requires native AMG");
    if(o.gpuPressureOperator!="compact"&&o.gpuPressureOperator!="full")throw std::runtime_error("GPU pressure operator must be compact or full");
    if(o.gpuPressureOperator=="full"&&(o.linearBackend!="native_gpu"||o.gpuPressureGauge!="mean_zero"))
        throw std::runtime_error("Full GPU pressure requires native AMG and mean-zero pressure");
    if(o.gpuViscosityOperator!="compact"&&o.gpuViscosityOperator!="full")throw std::runtime_error("GPU viscosity operator must be compact or full");
    if(o.gpuViscosityOperator=="full"&&(o.linearBackend!="native_gpu"||o.gpuPreconditioner!="native_amg"))
        throw std::runtime_error("Full GPU viscosity requires native AMG");
    if(o.fluidSolver=="proj" && o.pressureSolver=="cg")
        throw std::runtime_error("Projection supports pressure_solver auto, ldlt or amg");
    const std::string expectedScheme=o.fluidSolver=="proj"?"bcg":"fou";
    if(o.convection && j.value("convection_scheme",expectedScheme)!=expectedScheme)
        throw std::runtime_error("Convection scheme does not match the selected fluid solver");
    if(o.fluidSolver=="proj" && (o.embeddedGeometry.empty() || !o.convection || o.timeStep<=0 ||
       !o.periodicX || o.momentumMode!="imp" || o.fluxRelaxationMemory ||
       o.perturbation!=0 || !(o.projectionIterationTolerance>0 && o.projectionIterationTolerance<1)))
        throw std::runtime_error("Projection requires zero-start periodic embedded NS, implicit diffusion, and no SIMPLE flux memory");
    if(o.momentumMode!="imp" && o.momentumMode!="exp")
        throw std::runtime_error("Momentum mode must be imp or exp");
    if(o.momentumMode=="exp" && (o.embeddedGeometry.empty() || !o.convection || o.timeStep<=0))
        throw std::runtime_error("Explicit momentum updates require embedded Navier-Stokes with a positive time step");
    if(j.value("wall_reconstruction",std::string("linear"))!="linear")
        throw std::runtime_error("Only linear wall reconstruction is supported; the rejected quadratic5 experiment is archived in validation/twisted/experiments");
    return o;
}

Solver::Solver(Mesh mesh, Options options) : mesh_(std::move(mesh)), options_(std::move(options)) {
    const int n = int(mesh_.cells.size());
    if (!n) throw std::runtime_error("Empty SIMPLE mesh");
    if (options_.quadraticInterfaces && mesh_.coarseFineFaces)
        quadratic_ = std::make_unique<QuadraticReconstruction>(mesh_);
    if (mesh_.embedded) {
        embedded_=std::make_unique<EmbeddedOperators>(mesh_,quadratic_.get(),options_.convection,options_.momentumMode=="exp");
        embedded_->check(mesh_,(options_.output/"embedded_operator_checks.json").string());
        if(options_.convection && std::getenv("SIMPLE_ADV_DUMP"))
            writeMatrix(options_.output/"advection_face_interpolation.csv",Sparse(embedded_->cartesianFaceInterpolation));
        if(std::getenv("SIMPLE_EMBEDDED_OPERATOR_DUMP")) {
            const auto folder=options_.output/"operators";
            std::filesystem::create_directories(folder);
            nlohmann::json shapes;
            auto save=[&](const std::string& name,const EmbeddedOperators::Sparse& matrix) {
                writeMatrix(folder/(name+".csv"),Sparse(matrix));
                shapes[name]={{"rows",matrix.rows()},{"columns",matrix.cols()},{"nonzeros",matrix.nonZeros()}};
            };
            save("interpolation",embedded_->interpolation);
            save("face_gradient",embedded_->faceGradient);
            save("wall_gradient",embedded_->wallGradient);
            save("compact_diffusion",embedded_->compactDiffusion);
            save("deferred_diffusion",embedded_->deferredDiffusion);
            save("redistribution",embedded_->redistribution);
            if(options_.convection)save("advection_face_interpolation",embedded_->cartesianFaceInterpolation);
            for(int d=0;d<3;++d)save("cell_gradient_"+std::to_string(d),embedded_->cellGradient[d]);
            dumpMeshCsv(mesh_,(folder/"mesh").string());
            std::ofstream metadata(folder/"manifest.json");
            metadata<<nlohmann::json{{"scope","Read-only assembled embedded operators; not a solved flow"},
                {"momentum_mode",options_.momentumMode},{"shapes",shapes}}.dump(2)<<'\n';
            metadata.flush();if(!metadata.good())throw std::runtime_error("Operator dump metadata write failed");
        }
    }
    velocity_.assign(n, Vec3::Zero());
    pressure_ = Eigen::VectorXd::Zero(n);
    reciprocalDiagonal_ = Eigen::VectorXd::Zero(n);
    wallSecondCell_.assign(mesh_.faces.size(), -1);
    std::vector<std::vector<int>> adjacent(n);
    for (int j = 0; j < int(mesh_.faces.size()); ++j) {
        const auto& f = mesh_.faces[j];
        adjacent[f.owner].push_back(j);
        if (f.neighbor >= 0) adjacent[f.neighbor].push_back(j);
    }
    for (int j = 0; j < int(mesh_.faces.size()); ++j) {
        const auto& wall = mesh_.faces[j];
        if (wall.boundary != 1 || embedded_) continue;
        for (int k : adjacent[wall.owner]) {
            const auto& f = mesh_.faces[k];
            if (f.neighbor < 0 || f.axis != wall.axis) continue;
            const int other = f.owner == wall.owner ? f.neighbor : f.owner;
            const Vec3 delta = f.owner == wall.owner ? f.delta : -f.delta;
            if (delta.dot(faceNormal(wall)) < 0 &&
                std::abs(mesh_.cells[other].h - mesh_.cells[wall.owner].h) < 1e-14)
                wallSecondCell_[j] = other;
        }
        if (wallSecondCell_[j] < 0)
            throw std::runtime_error("Quadratic wall closure needs two aligned same-level inward cells");
    }
    const double drive = axialDrive(mesh_, options_);
    velocityScale_ = std::max(std::abs(exactVelocity(mesh_.extent * .5, mesh_, options_)), 1e-12);
    accelerationScale_ = std::max(std::abs(drive), 1e-12);
    for (int i = 0; i < n; ++i) {
        const auto& x = mesh_.cells[i].center;
        const double sx = std::sin(2 * pi * x[0] / mesh_.extent[0]);
        const double sy = std::sin(pi * x[1] / mesh_.extent[1]);
        // Absolute velocity amplitude [length/time], matching the independent
        // Aphros dump case. This deliberately has nonzero initial divergence.
        velocity_[i][0] = options_.perturbation * sx * sy;
        velocity_[i][1] = options_.perturbation * std::cos(2 * pi * x[0] / mesh_.extent[0]) *
            std::sin(2 * pi * x[1] / mesh_.extent[1]);
    }
    // Area-weighted least squares, including Dirichlet or zero-normal-gradient
    // boundary constraints. All fields use the same geometrical metric.
    inverseGradientMetric_.assign(n, Eigen::Matrix3d::Zero());
    if(!embedded_) for (const auto& f : mesh_.faces) {
        Vec3 delta = f.neighbor >= 0 ? f.delta : f.ownerOffset;
        const Eigen::Matrix3d m = delta * delta.transpose() / delta.squaredNorm();
        inverseGradientMetric_[f.owner] += m * f.area / std::pow(mesh_.cells[f.owner].h, 2);
        if (f.neighbor >= 0)
            inverseGradientMetric_[f.neighbor] += m * f.area / std::pow(mesh_.cells[f.neighbor].h, 2);
    }
    if(!embedded_) for (auto& m : inverseGradientMetric_) {
        if (!(m.determinant() > 1e-12)) throw std::runtime_error("Singular gradient reconstruction metric");
        m = m.inverse().eval();
    }
    flux_ = velocityFlux(velocity_);
    timeVelocity_=velocity_;
}

Vec3 Solver::faceNormal(const Face& f) const {
    return f.embeddedNormal.squaredNorm()>0 ? f.embeddedNormal : Vec3(Vec3::Unit(f.axis) * f.sign);
}
double Solver::weight(const Face& f) const {
    return f.neighbor < 0 ? 1. : mesh_.cells[f.neighbor].h /
        (mesh_.cells[f.owner].h + mesh_.cells[f.neighbor].h);
}
double Solver::pressureBoundary(const Face& f, bool correction) const {
    return correction ? 0. : (f.boundary == 2 ? options_.pressureIn : options_.pressureOut);
}

Solver::Grad Solver::gradient(const Eigen::VectorXd& q, int comp, bool correction) const {
    Grad b(q.size(), Vec3::Zero());
    if(embedded_) {
        for(int d=0;d<3;++d) {
            Eigen::VectorXd derivative=embedded_->cellGradient[d]*q;
            for(int i=0;i<q.size();++i)b[i][d]=derivative[i];
        }
        return b;
    }
    for (int j = 0; j < int(mesh_.faces.size()); ++j) {
        const auto& f = mesh_.faces[j];
        const int p = f.owner, n = f.neighbor;
        if (n >= 0) {
            const Vec3 v = f.delta * ((q[n] - q[p]) / f.delta.squaredNorm()) * f.area;
            b[p] += v / std::pow(mesh_.cells[p].h, 2);
            b[n] += v / std::pow(mesh_.cells[n].h, 2);
        } else {
            // The wall pressure gradient used by cell momentum is extrapolated
            // from interior cells, as in Aphros SIMPLE. This is distinct from
            // the impermeable wall face flux, whose pressure correction is zero.
            if (comp < 0 && f.boundary == 1) {
                const double outwardGradient = (q[p] - q[wallSecondCell_[j]]) / mesh_.cells[p].h;
                b[p] += faceNormal(f) * outwardGradient * f.area / std::pow(mesh_.cells[p].h, 2);
                continue;
            }
            // comp=-1: pressure; comp=0..2: velocity.
            bool dirichlet = comp < 0 ? f.boundary >= 2 : f.boundary == 1;
            if (dirichlet) {
                double qb = comp < 0 ? pressureBoundary(f, correction) : 0.;
                b[p] += f.ownerOffset * ((qb - q[p]) / f.ownerOffset.squaredNorm()) *
                    f.area / std::pow(mesh_.cells[p].h, 2);
            }
        }
    }
    for (int i = 0; i < q.size(); ++i) b[i] = inverseGradientMetric_[i] * b[i];
    return b;
}

double Solver::interpolate(const Face& f, const Eigen::VectorXd& q, const Grad& grad) const {
    if (f.neighbor < 0) return q[f.owner];
    const double w = weight(f);
    const Vec3 skew = w * f.ownerOffset + (1 - w) * f.neighborOffset;
    return w * q[f.owner] + (1 - w) * q[f.neighbor] +
        (w * grad[f.owner] + (1 - w) * grad[f.neighbor]).dot(skew);
}
double Solver::compactGradient(const Face& f, const Eigen::VectorXd& q,
                              const Grad& grad, bool correction,
                              const QuadraticReconstruction::Derivatives* quadratic) const {
    if (f.neighbor < 0) return f.boundary == 1 ? 0. :
        (pressureBoundary(f, correction) - q[f.owner]) / f.distance;
    if (quadratic && mesh_.cells[f.owner].level != mesh_.cells[f.neighbor].level)
        return quadraticFaceGradient(mesh_, f, q, *quadratic);
    const double w = weight(f);
    const Vec3 tangent = f.delta - f.distance * faceNormal(f);
    return (q[f.neighbor] - q[f.owner] -
        (w * grad[f.owner] + (1 - w) * grad[f.neighbor]).dot(tangent)) / f.distance;
}
Eigen::VectorXd Solver::divergence(const Eigen::VectorXd& flux) const {
    Eigen::VectorXd sum = Eigen::VectorXd::Zero(mesh_.cells.size());
    for (int j = 0; j < flux.size(); ++j) {
        const auto& f = mesh_.faces[j];
        sum[f.owner] += flux[j];
        if (f.neighbor >= 0) sum[f.neighbor] -= flux[j];
    }
    return sum;
}
Eigen::VectorXd Solver::velocityFlux(const std::vector<Vec3>& u) const {
    Eigen::VectorXd flux = Eigen::VectorXd::Zero(mesh_.faces.size());
    if(embedded_) {
        for(int d=0;d<3;++d) {
            Eigen::VectorXd values=embedded_->interpolation*component(u,d);
            for(int j=0;j<flux.size();++j) {
                const auto& f=mesh_.faces[j];
                if(f.neighbor>=0 && f.axis==d) flux[j]=f.area*f.sign*values[j];
            }
        }
        return flux;
    }
    for (int d = 0; d < 3; ++d) {
        const auto q = component(u, d); const auto g = gradient(q, d);
        QuadraticReconstruction::Derivatives qd;
        if (quadratic_) qd = quadratic_->evaluate(q);
        for (int j = 0; j < flux.size(); ++j) {
            const auto& f = mesh_.faces[j];
            if (f.axis == d && f.boundary != 1) {
                const bool useQuadratic = quadratic_ && f.neighbor >= 0 &&
                    mesh_.cells[f.owner].level != mesh_.cells[f.neighbor].level;
                flux[j] = f.area * f.sign * (useQuadratic ? quadraticFaceValue(mesh_, f, q, qd) : interpolate(f, q, g));
            }
        }
    }
    return flux;
}

Solver::Momentum Solver::momentum(const std::vector<Vec3>& u, const Eigen::VectorXd& flux,
                                  const Eigen::VectorXd& pressure, bool relax) const {
    const int nc = int(mesh_.cells.size());
    Momentum result;
    result.rhs = Eigen::MatrixXd::Zero(nc, 3);
    result.diagonal = Eigen::VectorXd::Zero(nc);
    if(embedded_) {
        const double mu=options_.rho*options_.nu;
        result.matrix=mu*embedded_->compactDiffusion;
        const auto gp=gradient(pressure,-1);
        for(int i=0;i<nc;++i)result.rhs.row(i)=(mesh_.cells[i].volume*(options_.rho*options_.force-gp[i])).transpose();
        for(int d=0;d<3;++d)result.rhs.col(d)-=mu*(embedded_->deferredDiffusion*component(u,d));
        if(options_.convection) {
            auto adv=embedded_->upwindAdvection(mesh_,flux,options_.rho);
            const Sparse compactAdvection=adv.first;
            result.matrix+=compactAdvection;
            for(int d=0;d<3;++d)result.rhs.col(d)-=adv.second*component(u,d);
        }
        if(options_.timeStep>0)for(int i=0;i<nc;++i) {
            const double mass=options_.rho*mesh_.cells[i].volume/options_.timeStep;
            result.matrix.coeffRef(i,i)+=mass;result.rhs.row(i)+=(mass*timeVelocity_[i]).transpose();
        }
        for(int i=0;i<nc;++i) {
            double diagonal=result.matrix.coeff(i,i);
            if(!(diagonal>0))throw std::runtime_error("Nonpositive embedded momentum diagonal");
            if(relax) {
                const double extra=diagonal*(1/options_.alphaU-1);
                result.matrix.coeffRef(i,i)+=extra;
                result.rhs.row(i)+=(extra*u[i]).transpose();
                diagonal/=options_.alphaU;
            }
            result.diagonal[i]=diagonal;
        }
        return result;
    }
    std::vector<Triplet> entries;
    entries.reserve(mesh_.faces.size() * 4 + nc);
    const auto gp = gradient(pressure, -1);
    std::array<Grad, 3> gu;
    std::array<QuadraticReconstruction::Derivatives, 3> qu;
    for (int d = 0; d < 3; ++d) {
        const auto q = component(u, d);
        gu[d] = gradient(q, d);
        if (quadratic_) qu[d] = quadratic_->evaluate(q);
    }
    for (int i = 0; i < nc; ++i)
        result.rhs.row(i) = (mesh_.cells[i].volume * (options_.rho * options_.force - gp[i])).transpose();
    for (int j = 0; j < flux.size(); ++j) {
        const auto& f = mesh_.faces[j];
        const int p = f.owner, n = f.neighbor;
        const double mu = options_.rho * options_.nu;
        if (n >= 0) {
            const double diff = mu * f.area / f.distance;
            result.diagonal[p] += diff; result.diagonal[n] += diff;
            entries.emplace_back(p, n, -diff); entries.emplace_back(n, p, -diff);
            const Vec3 tangent = f.delta - f.distance * faceNormal(f);
            const double w = weight(f);
            for (int d = 0; d < 3; ++d) {
                // Outward diffusive flux is -mu*A*grad_n(u).
                const double explicitFlux = quadratic_ && mesh_.cells[p].level != mesh_.cells[n].level ?
                    -mu * f.area * quadraticDeferredNormalGradient(f, qu[d]) :
                    diff * (w * gu[d][p] + (1 - w) * gu[d][n]).dot(tangent);
                result.rhs(p, d) -= explicitFlux; result.rhs(n, d) += explicitFlux;
            }
            if (options_.convection) {
                const double massFlux = options_.rho * flux[j];
                if (massFlux >= 0) {
                    result.diagonal[p] += massFlux; entries.emplace_back(n, p, -massFlux);
                } else {
                    entries.emplace_back(p, n, massFlux); result.diagonal[n] -= massFlux;
                }
            }
        } else if (f.boundary == 1) {
            result.diagonal[p] += mu * f.area / f.distance; // wall distance h/2
            // Match Aphros' quadratic one-sided no-slip derivative. Keep the
            // stable two-point implicit part and defer the remaining flux.
            const int second = wallSecondCell_[j];
            const double k = mu * f.area / mesh_.cells[p].h;
            for (int d = 0; d < 3; ++d)
                result.rhs(p, d) -= k * (u[p][d] - u[second][d] / 3.);
        } else if (options_.convection) {
            // Pressure openings: zero normal derivative of velocity. Valid for
            // the developed-channel benchmark; reverse-flow inflow is not modeled.
            result.diagonal[p] += options_.rho * flux[j];
        }
    }
    for (int i = 0; i < nc; ++i) {
        if (!(result.diagonal[i] > 0)) throw std::runtime_error("Nonpositive momentum diagonal");
        if (relax) {
            const double extra = result.diagonal[i] * (1 / options_.alphaU - 1);
            result.rhs.row(i) += (extra * u[i]).transpose();
            result.diagonal[i] /= options_.alphaU;
        }
        entries.emplace_back(i, i, result.diagonal[i]);
    }
    result.matrix.resize(nc, nc);
    result.matrix.setFromTriplets(entries.begin(), entries.end());
    return result;
}

double Solver::momentumResidual(const std::vector<Vec3>& u, const Eigen::VectorXd& flux,
                                const Eigen::VectorXd& pressure, double volume,bool includeTime) const {
    const auto actual = momentum(u, flux, pressure, false);
    double sum = 0.;
    for (int d = 0; d < 3; ++d) {
        Eigen::VectorXd r = actual.matrix * component(u, d) - actual.rhs.col(d);
        if(!includeTime && options_.timeStep>0)for(int i=0;i<int(u.size());++i)
            r[i]-=options_.rho*mesh_.cells[i].volume*(u[i][d]-timeVelocity_[i][d])/options_.timeStep;
        for (int i = 0; i < int(u.size()); ++i) sum += r[i] * r[i] /
            (options_.rho * options_.rho * mesh_.cells[i].volume);
    }
    return std::sqrt(sum / volume) / accelerationScale_;
}

void Solver::dump(int iteration, const Momentum* m, const std::vector<Vec3>* predictor,
                  const Eigen::VectorXd* predFlux, const Eigen::VectorXd* rhs,
                  const Eigen::VectorXd* pc, const Sparse* pm) const {
    const auto dir = options_.output / ("iter_" + std::to_string(iteration));
    std::filesystem::create_directories(dir);
    std::ofstream cells(dir / "cells.csv");
    cells << std::setprecision(17) << "id,level,x,y,z,h,volume,u,v,w,p,aP,rhs_u,rhs_v,rhs_w,u_star,v_star,w_star,pressure_rhs,pressure_correction\n";
    for (int i = 0; i < pressure_.size(); ++i) {
        const auto& c = mesh_.cells[i];
        cells << i << ',' << c.level;
        for (int d = 0; d < 3; ++d) cells << ',' << c.center[d];
        cells << ',' << c.h << ',' << c.volume;
        for (int d = 0; d < 3; ++d) cells << ',' << velocity_[i][d];
        cells << ',' << pressure_[i] << ',' << (m ? m->diagonal[i] : 0.);
        for (int d = 0; d < 3; ++d) cells << ',' << (m ? m->rhs(i, d) : 0.);
        for (int d = 0; d < 3; ++d) cells << ',' << (predictor ? (*predictor)[i][d] : velocity_[i][d]);
        cells << ',' << (rhs ? (*rhs)[i] : 0.) << ',' << (pc ? (*pc)[i] : 0.) << '\n';
    }
    std::ofstream faces(dir / "faces.csv");
    faces << std::setprecision(17) << "id,owner,neighbor,axis,sign,boundary,x,y,z,area,distance,flux,predicted_flux\n";
    for (int j = 0; j < flux_.size(); ++j) {
        const auto& f = mesh_.faces[j];
        faces << j << ',' << f.owner << ',' << f.neighbor << ',' << f.axis << ',' << f.sign << ',' << f.boundary;
        for (int d = 0; d < 3; ++d) faces << ',' << f.center[d];
        faces << ',' << f.area << ',' << f.distance << ',' << flux_[j] << ',' << (predFlux ? (*predFlux)[j] : flux_[j]) << '\n';
    }
    if (m) writeMatrix(dir / "momentum_matrix.csv", m->matrix);
    if (pm) writeMatrix(dir / "pressure_matrix.csv", *pm);
}

nlohmann::json Solver::run() {
    if(options_.timeStep==0)return runStep();
    const auto root=options_.output;
    nlohmann::json config;std::ifstream(root/"case.json")>>config;
    std::ofstream history(root/"time_history.csv");
    history<<std::setprecision(17)<<"step,time,inner_iterations,temporal_acceleration_relative_l2,steady_momentum_relative_l2,inner_converged\n";
    nlohmann::json result;int step=0,completed=0;
    for(step=1;step<=options_.timeSteps;++step) {
        timeVelocity_=velocity_;
        std::ostringstream name;name<<"step_"<<std::setw(4)<<std::setfill('0')<<step;
        options_.output=root/name.str();std::filesystem::create_directories(options_.output);
        auto current=config;current["output"]=options_.output.string();current["physical_step"]=step;
        current["physical_time"]=step*options_.timeStep;
        std::ofstream(options_.output/"case.json")<<current.dump(2)<<'\n';
        result=runStep();
        history<<step<<','<<step*options_.timeStep<<','<<result.at("iterations")<<','
               <<result.at("temporal_acceleration_relative_l2")<<','<<result.at("steady_momentum_relative_l2")<<','
               <<result.at("converged")<<'\n';history.flush();
        if(!result.at("converged").get<bool>())break;
        ++completed;
    }
    const std::string finalPath=options_.output.string();options_.output=root;
    const bool allSteps=completed==options_.timeSteps;
    const bool steady=allSteps && result.at("steady_momentum_relative_l2").get<double>()<std::min(options_.tolerance,1e-8)
        && result.at("temporal_acceleration_relative_l2").get<double>()<1e-8;
    nlohmann::json summary{{"converged",allSteps},{"steps_completed",completed},
        {"time_step",options_.timeStep},{"final_output",finalPath},
        {"steady_converged",steady},
        {"scope","Backward Euler physical time steps; inner convergence does not imply a steady state"}};
    std::ofstream(root/"transient_summary.json")<<summary.dump(2)<<'\n';
    return summary;
}

nlohmann::json Solver::runStep() {
    std::filesystem::create_directories(options_.output);
    dump(0);
    std::ofstream history(options_.output / "history.csv");
    history << std::setprecision(17) << "iteration,momentum_relative_l2,continuity_relative_linf,velocity_change_relative_linf,pressure_nonorth_defect_relative_linf,linear_momentum_residual,linear_pressure_residual,nonorth_face_defect_relative_linf,flux_change_relative_linf,pressure_change_relative_linf\n";
    const int nc = int(mesh_.cells.size()), nf = int(mesh_.faces.size());
    double volume = 0.; for (const auto& c : mesh_.cells) volume += c.volume;
    AndersonAcceleration accelerator(std::max(options_.andersonDepth, 1));
    int acceleratedSteps = 0, rejectedAccelerations = 0;
    // Very small cut volumes amplify roundoff in affine flux combinations.
    // A trial remains below a fraction of the SAME final physical tolerance;
    // final acceptance still requires a fresh, unaccelerated SIMPLE step.
    const double candidateContinuityLimit=mesh_.embedded?std::max(1e-10,.25*options_.tolerance):1e-10;
    std::ofstream accelerationLog;
    if (options_.andersonDepth) {
        accelerationLog.open(options_.output / "acceleration.csv");
        accelerationLog << std::setprecision(17) << "after_iteration,accepted,gamma_norm,backtracking_factor,candidate_momentum,candidate_continuity\n";
        std::ofstream(options_.output / "dump_semantics.json") << nlohmann::json{
            {"anderson_depth", options_.andersonDepth},
            {"iteration_dumps", "Raw SIMPLE output before optional acceleration; predecessor output may differ from the next accelerated input"},
            {"candidate_continuity_limit",candidateContinuityLimit},
            {"acceptance", "Candidate is next input only; true momentum decreases, continuity stays below max(candidate_continuity_limit,10*raw continuity); final acceptance is always an unaccelerated SIMPLE step"}}.dump(2);
    }
    const double pressureScale = options_.rho * accelerationScale_ * mesh_.extent[0];
    auto packState = [&]() {
        Eigen::VectorXd state(4 * nc + nf);
        for (int i = 0; i < nc; ++i) {
            for (int d = 0; d < 3; ++d) state[4 * i + d] = velocity_[i][d] / velocityScale_;
            state[4 * i + 3] = pressure_[i] / pressureScale;
        }
        for (int j = 0; j < nf; ++j) state[4 * nc + j] = flux_[j] / (mesh_.faces[j].area * velocityScale_);
        return state;
    };
    bool converged = false;
    double momResidual = 0., continuity = 0., change = 0., nonorthDefect = 0.;
    double faceDefect = 0., fluxChange = 0., pressureChange = 0.;
    Sparse cachedMomentumMatrix, cachedPressureMatrix;
    Eigen::BiCGSTAB<Sparse, Eigen::IncompleteLUT<double>> momentumSolve;
    momentumSolve.setTolerance(options_.linearTolerance);
    momentumSolve.setMaxIterations(2000);
    momentumSolve.preconditioner().setDroptol(1e-3);
    momentumSolve.preconditioner().setFillfactor(4);
    Eigen::ConjugateGradient<Sparse, Eigen::Lower | Eigen::Upper, Eigen::IncompleteCholesky<double>> pressureSolve;
    pressureSolve.setTolerance(options_.linearTolerance);
    pressureSolve.setMaxIterations(4000);
    const bool usePressureLdlt = options_.pressureSolver == "ldlt" ||
        (options_.pressureSolver == "auto" && !options_.convection && nc >= 100000);
    const bool usePressureAmg=options_.pressureSolver=="amg";
    AmgPressure pressureAmg;
    Eigen::SimplicialLDLT<Sparse> pressureDirect;
    int iteration = 0;
    for (iteration = 1; iteration <= options_.maxIterations; ++iteration) {
        Eigen::VectorXd iterationInput;
        if (options_.andersonDepth) iterationInput = packState();
        const auto oldVelocity = velocity_;
        const auto oldFlux = flux_;
        auto mom = momentum(velocity_, flux_, pressure_, true);
        if (iteration == 1 || (options_.convection &&
            (options_.momentumMode!="exp" || !identicalMatrix(cachedMomentumMatrix,mom.matrix)))) {
            cachedMomentumMatrix = mom.matrix;
            momentumSolve.compute(cachedMomentumMatrix);
        }
        if (momentumSolve.info() != Eigen::Success) throw std::runtime_error("Momentum factorization failed");
        std::vector<Vec3> predictor(nc);
        double linearMomentum = 0.;
        double linearScale = 0.;
        for (const auto& cell : mesh_.cells)
            linearScale += std::pow(options_.rho * cell.volume * accelerationScale_, 2);
        linearScale = std::sqrt(linearScale);
        for (int d = 0; d < 3; ++d) {
            Eigen::VectorXd sol;
            if (mom.rhs.col(d).norm() < options_.linearTolerance * linearScale)
                sol = Eigen::VectorXd::Zero(nc);
            else {
                // Normalize the RHS explicitly. Eigen 5.0's BiCGSTAB loop uses
                // an absolute stopping threshold although its API describes a
                // relative tolerance; integral FV RHS values shrink with h^3.
                // Unit-norm RHS gives the intended tolerance in either version.
                const double scale = mom.rhs.col(d).norm();
                const Eigen::VectorXd b = mom.rhs.col(d) / scale;
                const Eigen::VectorXd guess = component(velocity_, d) / scale;
                sol = momentumSolve.solveWithGuess(b, guess);
                sol *= scale;
            }
            // Check the original matrix, not the Krylov solver's recursively
            // updated residual (which can drift near convergence).
            for (int refinement = 0; refinement < 4; ++refinement) {
                const Eigen::VectorXd r = mom.rhs.col(d) - mom.matrix * sol;
                if (r.norm() <= options_.linearTolerance * std::max(mom.rhs.col(d).norm(), linearScale)) break;
                const double scale = r.norm();
                const Eigen::VectorXd normalized = r / scale;
                const Eigen::VectorXd correction = momentumSolve.solve(normalized);
                sol += correction * scale;
            }
            double error = (mom.matrix * sol - mom.rhs.col(d)).norm() / std::max(mom.rhs.col(d).norm(), linearScale);
            linearMomentum = std::max(linearMomentum, error);
            if (!sol.allFinite() || error > 10 * options_.linearTolerance)
                throw std::runtime_error("Momentum linear solve did not converge: component=" + std::to_string(d) +
                    " relative residual=" + std::to_string(error) + " rhs norm=" + std::to_string(mom.rhs.col(d).norm()));
            for (int i = 0; i < nc; ++i) predictor[i][d] = sol[i];
        }
        for (int i = 0; i < nc; ++i) reciprocalDiagonal_[i] = mesh_.cells[i].volume / mom.diagonal[i];
        const auto gp = gradient(pressure_, -1);
        QuadraticReconstruction::Derivatives qp;
        if (quadratic_) qp = quadratic_->evaluate(pressure_);
        Eigen::VectorXd predFlux = velocityFlux(predictor);
        const Eigen::VectorXd oldInterpolatedFlux = velocityFlux(oldVelocity);
        Eigen::VectorXd conductance = Eigen::VectorXd::Zero(nf);
        Eigen::VectorXd embeddedReciprocal;
        if(embedded_)embeddedReciprocal=embedded_->interpolation*reciprocalDiagonal_;
        std::vector<Triplet> pt;
        Eigen::VectorXd pd = Eigen::VectorXd::Zero(nc);
        bool hasPressureBoundary = false;
        for (int j = 0; j < nf; ++j) {
            const auto& f = mesh_.faces[j];
            if (f.boundary == 1) continue;
            const int p = f.owner, n = f.neighbor;
            const double w = weight(f);
            const double df = embedded_ && mesh_.cells[p].level==mesh_.cells[n].level ? embeddedReciprocal[j] :
                w * reciprocalDiagonal_[p] + (n >= 0 ? (1 - w) * reciprocalDiagonal_[n] : 0.);
            Vec3 wide = reciprocalDiagonal_[p] * (gp[p] - options_.rho * options_.force);
            if (n >= 0) wide = w * wide + (1 - w) * reciprocalDiagonal_[n] * (gp[n] - options_.rho * options_.force);
            // Aphros treats the prescribed force as a balanced face force.
            // At cut faces, bilinear df need not equal the adjacent-cell mean.
            predFlux[j] += f.area * (wide.dot(faceNormal(f)) + df *
                (options_.rho * options_.force.dot(faceNormal(f)) - compactGradient(f, pressure_, gp, false, quadratic_ ? &qp : nullptr)));
            if (options_.fluxRelaxationMemory)
                predFlux[j] += (1 - options_.alphaU) * (oldFlux[j] - oldInterpolatedFlux[j]);
            const double k = df * f.area / f.distance;
            conductance[j] = k;
            pd[p] += k;
            if (n >= 0) {
                pd[n] += k;
                // Remove BOTH reference row and column for symmetric gauge fixing.
                if (!options_.periodicX || (p != 0 && n != 0)) {
                    pt.emplace_back(p, n, -k); pt.emplace_back(n, p, -k);
                }
            } else hasPressureBoundary = true;
        }
        if (options_.periodicX == hasPressureBoundary) throw std::runtime_error("Pressure boundary/gauge mismatch");
        for (int i = 0; i < nc; ++i) pt.emplace_back(i, i, options_.periodicX && i == 0 ? 1. : pd[i]);
        Sparse pressureMatrix(nc, nc); pressureMatrix.setFromTriplets(pt.begin(), pt.end());
        if (iteration == 1 || (options_.convection &&
            (options_.momentumMode!="exp" || !identicalMatrix(cachedPressureMatrix,pressureMatrix)))) {
            cachedPressureMatrix = pressureMatrix;
            // Exp momentum has a constant time diagonal within a physical step.
            // Reuse its factorization only after exact coefficient/index checks;
            // implicit convection keeps its previous rebuild behavior.
            if (usePressureAmg) pressureAmg.compute(cachedPressureMatrix,options_.linearTolerance);
            else if (usePressureLdlt) pressureDirect.compute(cachedPressureMatrix);
            else pressureSolve.compute(cachedPressureMatrix);
        }
        if (!usePressureAmg && (usePressureLdlt ? pressureDirect.info() : pressureSolve.info()) != Eigen::Success)
            throw std::runtime_error("Pressure factorization failed");
        const auto predDiv = divergence(predFlux);
        Eigen::VectorXd pc = Eigen::VectorXd::Zero(nc), pressureRhs(nc), explicitFlux = Eigen::VectorXd::Zero(nf);
        auto pressureExplicit = [&](const Eigen::VectorXd& value) {
            const auto gc = gradient(value, -1, true);
            QuadraticReconstruction::Derivatives qc;
            if (quadratic_) qc = quadratic_->evaluate(value);
            Eigen::VectorXd e = Eigen::VectorXd::Zero(nf);
            for (int j = 0; j < nf; ++j) {
                const auto& f = mesh_.faces[j];
                if (f.neighbor < 0) continue;
                const double w = weight(f);
                e[j] = quadratic_ && mesh_.cells[f.owner].level != mesh_.cells[f.neighbor].level ?
                    -conductance[j] * f.distance * quadraticDeferredNormalGradient(f, qc) :
                    conductance[j] * (w * gc[f.owner] + (1 - w) * gc[f.neighbor]).dot(f.delta - f.distance * faceNormal(f));
            }
            return e;
        };
        double linearPressure = 0.;
        for (int pass = 0; pass < options_.nonorthIterations; ++pass) {
            explicitFlux = pressureExplicit(pc);
            pressureRhs = -predDiv - divergence(explicitFlux);
            // No solve is needed if the predictor already satisfies continuity
            // to the absolute physical tolerance, including the gauge cell.
            double rhsContinuity = 0.;
            for (int i = 0; i < nc; ++i) rhsContinuity = std::max(rhsContinuity,
                std::abs(pressureRhs[i]) / (mesh_.cells[i].volume * velocityScale_ / mesh_.extent[1]));
            if (pass == 0 && rhsContinuity < options_.linearTolerance) { pc.setZero(); break; }
            if (options_.periodicX) pressureRhs[0] = 0.;
            const double rhsScale = pressureRhs.norm();
            if (rhsScale == 0.) {
                // A later deferred-correction RHS can cancel exactly. Solve the
                // homogeneous system, then still check flux consistency below.
                pc.setZero();
            } else {
                const Eigen::VectorXd normalized = pressureRhs / rhsScale;
                if (usePressureAmg) pc = pressureAmg.solve(normalized);
                else if (usePressureLdlt) pc = pressureDirect.solve(normalized).eval();
                else {
                    const Eigen::VectorXd guess = pc / rhsScale;
                    pc = pressureSolve.solveWithGuess(normalized, guess).eval();
                }
                pc *= rhsScale;
            }
            linearPressure = (pressureMatrix * pc - pressureRhs).norm() / std::max(pressureRhs.norm(), 1e-30);
            if (!pc.allFinite() || linearPressure > 1e-7) throw std::runtime_error("Pressure linear solve did not converge");
            if (mesh_.coarseFineFaces == 0) break;
            const Eigen::VectorXd nextExplicit = pressureExplicit(pc);
            const auto consistency = divergence(nextExplicit - explicitFlux);
            double error = 0.;
            for (int i = 0; i < nc; ++i) error = std::max(error, std::abs(consistency[i]) /
                (mesh_.cells[i].volume * velocityScale_ / mesh_.extent[1]));
            for (int j = 0; j < nf; ++j) error = std::max(error, std::abs(nextExplicit[j] - explicitFlux[j]) /
                (mesh_.faces[j].area * velocityScale_));
            if (error < .05 * options_.tolerance) break;
        }
        // Apply the SAME deferred flux used to assemble the final pressure solve.
        // This guarantees discrete continuity; separately report its consistency defect.
        flux_ = predFlux + explicitFlux;
        const auto gc = gradient(pc, -1, true);
        const Eigen::VectorXd newExplicitFlux = pressureExplicit(pc);
        for (int j = 0; j < nf; ++j) {
            const auto& f = mesh_.faces[j];
            if (f.boundary == 1) continue;
            flux_[j] -= conductance[j] * ((f.neighbor >= 0 ? pc[f.neighbor] : 0.) - pc[f.owner]);
        }
        for (int i = 0; i < nc; ++i) velocity_[i] = predictor[i] - reciprocalDiagonal_[i] * gc[i];
        pressure_ += options_.alphaP * pc;
        if (options_.periodicX) {
            double mean = 0.; for (int i = 0; i < nc; ++i) mean += pressure_[i] * mesh_.cells[i].volume;
            pressure_.array() -= mean / volume;
        }
        const auto div = divergence(flux_);
        const auto defect = divergence(newExplicitFlux - explicitFlux);
        continuity = 0.; nonorthDefect = 0.; change = 0.;
        faceDefect = 0.; fluxChange = 0.;
        pressureChange = options_.alphaP * pc.cwiseAbs().maxCoeff() /
            (options_.rho * accelerationScale_ * mesh_.extent[0]);
        for (int j = 0; j < nf; ++j) {
            const double scale = mesh_.faces[j].area * velocityScale_;
            faceDefect = std::max(faceDefect, std::abs(newExplicitFlux[j] - explicitFlux[j]) / scale);
            fluxChange = std::max(fluxChange, std::abs(flux_[j] - oldFlux[j]) / scale);
        }
        for (int i = 0; i < nc; ++i) {
            const double scale = mesh_.cells[i].volume * velocityScale_ / mesh_.extent[1];
            continuity = std::max(continuity, std::abs(div[i]) / scale);
            nonorthDefect = std::max(nonorthDefect, std::abs(defect[i]) / scale);
            change = std::max(change, (velocity_[i] - oldVelocity[i]).norm() / velocityScale_);
        }
        momResidual = momentumResidual(velocity_, flux_, pressure_, volume);
        history << iteration << ',' << momResidual << ',' << continuity << ',' << change << ',' << nonorthDefect << ',' << linearMomentum << ',' << linearPressure
                << ',' << faceDefect << ',' << fluxChange << ',' << pressureChange << '\n';
        history.flush();
        if (iteration <= 5 || iteration % 25 == 0)
            std::cout << "SIMPLE " << iteration << " momentum=" << momResidual << " continuity=" << continuity << " change=" << change << " nonorth=" << nonorthDefect << std::endl;
        if (!std::isfinite(momResidual + continuity + change)) throw std::runtime_error("Nonfinite SIMPLE state");
        if (momResidual > 1e6 || change > 1e6)
            throw std::runtime_error("SIMPLE iteration diverged; use velocity/pressure under-relaxation");
        converged = momResidual < options_.tolerance && continuity < options_.tolerance && nonorthDefect < options_.tolerance &&
            faceDefect < options_.tolerance && change < options_.tolerance && fluxChange < options_.tolerance && pressureChange < options_.tolerance;
        if (converged || iteration == options_.maxIterations ||
            std::find(options_.dumpIterations.begin(), options_.dumpIterations.end(), iteration) != options_.dumpIterations.end())
            dump(iteration, &mom, &predictor, &predFlux, &pressureRhs, &pc, &pressureMatrix);
        if (converged) break;
        // Optional fixed-point acceleration changes only the NEXT input.
        // All convergence checks and dumps above refer to a complete SIMPLE step.
        // A linear combination of conservative output fluxes stays conservative,
        // but we check the actual mass and unrelaxed momentum residual explicitly.
        if (options_.andersonDepth && iteration < options_.maxIterations) {
            const Eigen::VectorXd output = packState();
            const Eigen::VectorXd candidate = accelerator.update(iterationInput, output);
            if (accelerator.accepted()) {
                std::vector<Vec3> trialU(nc);
                Eigen::VectorXd trialP(nc), trialF(nf);
                bool accepted = false;
                double trialMomentum = 0., trialContinuity = 0., factor = 1.;
                for (int attempt = 0; attempt < 4; ++attempt, factor *= .5) {
                    const Eigen::VectorXd state = output + factor * (candidate - output);
                    for (int i = 0; i < nc; ++i) {
                        for (int d = 0; d < 3; ++d) trialU[i][d] = state[4 * i + d] * velocityScale_;
                        trialP[i] = state[4 * i + 3] * pressureScale;
                    }
                    for (int j = 0; j < nf; ++j) trialF[j] = state[4 * nc + j] * mesh_.faces[j].area * velocityScale_;
                    const auto trialDiv = divergence(trialF);
                    trialContinuity = 0.;
                    for (int i = 0; i < nc; ++i) trialContinuity = std::max(trialContinuity,
                        std::abs(trialDiv[i]) / (mesh_.cells[i].volume * velocityScale_ / mesh_.extent[1]));
                    // Reassemble convection from this trial's own flux and
                    // velocity. History is local to runStep(), so it never
                    // mixes different previous-time fields across BE steps.
                    trialMomentum = momentumResidual(trialU, trialF, trialP, volume);
                    if (std::isfinite(trialMomentum + trialContinuity) && trialMomentum < momResidual &&
                        trialContinuity < std::max(candidateContinuityLimit, 10 * continuity)) {
                        accepted = true; break;
                    }
                }
                accelerationLog << iteration << ',' << accepted << ',' << accelerator.coefficientNorm() << ','
                    << (accepted ? factor : 0.) << ',' << trialMomentum << ',' << trialContinuity << '\n';
                accelerationLog.flush();
                if (accepted) {
                    velocity_ = std::move(trialU); pressure_ = std::move(trialP); flux_ = std::move(trialF);
                    ++acceleratedSteps;
                } else { ++rejectedAccelerations; accelerator.reset(); }
            }
        }
    }
    if(embedded_) {
        std::ofstream final(options_.output/"solution.csv");
        final<<std::setprecision(17)<<"id,level,x,y,z,h,volume,u,v,w,p\n";
        double speedMax=0,volumeMean=0;int cutCells=0;
        for(int i=0;i<nc;++i) {
            const auto& c=mesh_.cells[i];
            final<<i<<','<<c.level;
            for(int d=0;d<3;++d)final<<','<<c.center[d];
            final<<','<<c.h<<','<<c.volume;
            for(int d=0;d<3;++d)final<<','<<velocity_[i][d];
            final<<','<<pressure_[i]<<'\n';
            speedMax=std::max(speedMax,velocity_[i].norm());volumeMean+=velocity_[i][0]*c.volume;
            cutCells+=c.cut;
        }
        std::array<Eigen::VectorXd,3> dw;
        for(int d=0;d<3;++d)dw[d]=embedded_->wallGradient*component(velocity_,d);
        std::ofstream wall(options_.output/"walls.csv");
        wall<<std::setprecision(17)<<"face_id,owner,x,y,z,nx,ny,nz,area,du_dn,dv_dn,dw_dn,tau_x,tau_y,tau_z\n";
        std::map<double,std::pair<double,double>> sections;
        double wallArea=0,shearIntegral=0,maxh=0;
        for(const auto& c:mesh_.cells)maxh=std::max(maxh,c.h);
        for(int j=0;j<nf;++j) {
            const auto& f=mesh_.faces[j];
            if(f.neighbor>=0 && f.axis==0 && std::abs(f.center[0]/maxh-std::round(f.center[0]/maxh))<1e-8) {
                double x=std::round(f.center[0]/maxh)*maxh;
                if(x>=mesh_.extent[0]-1e-12)x=0.;
                sections[x].first+=f.sign*flux_[j];sections[x].second+=f.area;
            }
            if(f.boundary!=1)continue;
            Vec3 derivative(dw[0][j],dw[1][j],dw[2][j]);const Vec3 normal=faceNormal(f);
            Vec3 tau=options_.rho*options_.nu*(derivative-normal*derivative.dot(normal));
            wall<<j<<','<<f.owner;
            for(const Vec3& v:{f.center,normal})for(int d=0;d<3;++d)wall<<','<<v[d];
            wall<<','<<f.area;
            for(const Vec3& v:{derivative,tau})for(int d=0;d<3;++d)wall<<','<<v[d];
            wall<<'\n';wallArea+=f.area;shearIntegral+=tau.norm()*f.area;
        }
        std::ofstream section(options_.output/"sections.csv");section<<std::setprecision(17)<<"x,volume_flux,area\n";
        double qmin=std::numeric_limits<double>::max(),qmax=-qmin,qsum=0;
        for(auto s:sections) {
            section<<s.first<<','<<s.second.first<<','<<s.second.second<<'\n';
            qmin=std::min(qmin,s.second.first);qmax=std::max(qmax,s.second.first);qsum+=s.second.first;
        }
        if(sections.empty())throw std::runtime_error("No complete periodic sections");
        const double q=qsum/sections.size();
        double temporal=0;
        if(options_.timeStep>0)for(int i=0;i<nc;++i)
            temporal+=mesh_.cells[i].volume*((velocity_[i]-timeVelocity_[i])/options_.timeStep).squaredNorm();
        temporal=std::sqrt(temporal/volume)/accelerationScale_;
        const double steadyResidual=options_.timeStep>0?momentumResidual(velocity_,flux_,pressure_,volume,false):momResidual;
        nlohmann::json report{{"converged",converged},{"iterations",std::min(iteration,options_.maxIterations)},
            {"mesh_backend",mesh_.backend},{"embedded",true},{"geometry_source",mesh_.geometrySource},
            {"cells",nc},{"faces",nf},{"cut_cells",cutCells},{"coarse_fine_faces",mesh_.coarseFineFaces},
            {"rho",options_.rho},{"nu",options_.nu},{"convection",options_.convection},
            {"convection_scheme",options_.convection?"fou":"none"},{"time_step",options_.timeStep},
            {"temporal_acceleration_relative_l2",temporal},{"steady_momentum_relative_l2",steadyResidual},
            {"momentum_relative_l2",momResidual},{"continuity_relative_linf",continuity},
            {"velocity_change_relative_linf",change},{"pressure_change_relative_linf",pressureChange},
            {"flux_change_relative_linf",fluxChange},{"nonorth_defect_relative_linf",nonorthDefect},
            {"nonorth_face_defect_relative_linf",faceDefect},{"speed_max",speedMax},
            {"volume",volume},{"mean_axial_velocity",volumeMean/volume},{"volume_flux",q},
            {"cross_section_flux_relative_spread",(qmax-qmin)/std::max(std::abs(q),1e-30)},
            {"wall_area",wallArea},{"mean_wall_shear_magnitude",shearIntegral/wallArea},
            {"analytic_reference",nullptr},{"pressure_units","Pa; periodic perturbation"},
            {"velocity_units","m/s"},{"wall_shear_units","Pa; tangential fluid traction on the outward wall normal"}};
        std::ofstream(options_.output/"metrics.json")<<report.dump(2)<<'\n';
        dumpMeshCsv(mesh_,(options_.output/"mesh").string());
        std::vector<double> nativePressure(pressure_.data(),pressure_.data()+pressure_.size());
        exportNativeFields(mesh_,velocity_,nativePressure,(options_.output/"native_fields.bin").string());
        std::cout<<report.dump(2)<<std::endl;return report;
    }
    const double drive = axialDrive(mesh_, options_);
    double error2 = 0., exact2 = 0., errorMax = 0., meanU = 0., crossMax = 0., pressureError2 = 0., sampledExactMean = 0.;
    std::ofstream final(options_.output / "solution.csv");
    final << std::setprecision(17) << "id,level,x,y,z,h,volume,u,v,w,p,u_exact,p_exact\n";
    for (int i = 0; i < nc; ++i) {
        const auto& c = mesh_.cells[i];
        const double exact = exactVelocity(c.center, mesh_, options_);
        const double pe = options_.periodicX ? 0. : options_.pressureIn +
            (options_.pressureOut - options_.pressureIn) * c.center[0] / mesh_.extent[0];
        error2 += (velocity_[i] - Vec3(exact, 0, 0)).squaredNorm() * c.volume;
        exact2 += exact * exact * c.volume;
        sampledExactMean += exact * c.volume;
        errorMax = std::max(errorMax, std::abs(velocity_[i][0] - exact));
        meanU += velocity_[i][0] * c.volume;
        crossMax = std::max(crossMax, std::hypot(velocity_[i][1], velocity_[i][2]));
        pressureError2 += std::pow(pressure_[i] - pe, 2) * c.volume;
        final << i << ',' << c.level;
        for (int d = 0; d < 3; ++d) final << ',' << c.center[d];
        final << ',' << c.h << ',' << c.volume;
        for (int d = 0; d < 3; ++d) final << ',' << velocity_[i][d];
        final << ',' << pressure_[i] << ',' << exact << ',' << pe << '\n';
    }
    const double exactMean = exactMeanVelocity(mesh_, options_);
    // Flux is measured from the authority used by continuity, not from cell u.
    // Every axial plane in this channel refinement pattern spans the full duct.
    std::map<double, std::pair<double, double>> sections;
    double shearForce = 0., wallArea = 0.;
    for (int j = 0; j < nf; ++j) {
        const auto& f = mesh_.faces[j];
        if (f.axis == 0) {
            double x = f.center[0];
            if (options_.periodicX && std::abs(x - mesh_.extent[0]) < 1e-12) x = 0.;
            sections[x].first += f.sign * flux_[j];
            sections[x].second += f.area;
        }
        if (f.boundary == 1) {
            shearForce += options_.rho * options_.nu * f.area *
                (9 * velocity_[f.owner][0] - velocity_[wallSecondCell_[j]][0]) / (3 * mesh_.cells[f.owner].h);
            wallArea += f.area;
        }
    }
    const double exactFlow = exactMean * mesh_.extent[1] * mesh_.extent[2];
    double qMin = std::numeric_limits<double>::max(), qMax = -qMin, qSum = 0.;
    std::ofstream sectionFile(options_.output / "sections.csv");
    sectionFile << std::setprecision(17) << "x,volume_flux,area,exact_volume_flux\n";
    for (const auto& entry : sections) {
        const auto q = entry.second.first, area = entry.second.second;
        if (std::abs(area - mesh_.extent[1] * mesh_.extent[2]) > 1e-10 * area)
            throw std::runtime_error("Incomplete channel cross section in flux metric");
        qMin = std::min(qMin, q); qMax = std::max(qMax, q); qSum += q;
        sectionFile << entry.first << ',' << q << ',' << area << ',' << exactFlow << '\n';
    }
    const double qMean = qSum / sections.size();
    const double exactShear = options_.rho * drive * volume / wallArea;
    nlohmann::json report{
        {"converged", converged}, {"iterations", std::min(iteration, options_.maxIterations)},
        {"cells", nc}, {"faces", nf}, {"coarse_fine_faces", mesh_.coarseFineFaces}, {"mesh_backend", mesh_.backend},
        {"ny", options_.ny}, {"adaptive", options_.adaptive}, {"periodic_x", options_.periodicX},
        {"periodic_z", options_.periodicZ}, {"quadratic_interfaces", options_.quadraticInterfaces},
        {"anderson_depth", options_.andersonDepth}, {"anderson_accepted_steps", acceleratedSteps},
        {"anderson_rejected_steps", rejectedAccelerations}, {"convection", options_.convection},
        {"pressure_solver_requested", options_.pressureSolver}, {"pressure_linear_solver", usePressureLdlt ? "ldlt" : "cg"},
        {"rho", options_.rho}, {"nu", options_.nu}, {"alpha_u", options_.alphaU}, {"alpha_p", options_.alphaP},
        {"momentum_relative_l2", momResidual}, {"continuity_relative_linf", continuity},
        {"nonorth_face_defect_relative_linf", faceDefect}, {"flux_change_relative_linf", fluxChange},
        {"pressure_change_relative_linf", pressureChange},
        {"nonorth_defect_relative_linf", nonorthDefect}, {"velocity_relative_l2", std::sqrt(error2 / std::max(exact2, 1e-30))},
        {"velocity_absolute_linf", errorMax}, {"mean_velocity", meanU / volume}, {"mean_velocity_exact", exactMean},
        {"sampled_exact_mean_velocity", sampledExactMean / volume},
        {"mean_velocity_vs_sampled_exact_relative_error", std::abs(meanU - sampledExactMean) / std::max(std::abs(sampledExactMean), 1e-30)},
        {"volume_mean_velocity_relative_error", std::abs(meanU / volume - exactMean) / std::max(std::abs(exactMean), 1e-30)},
        {"volume_flux", qMean}, {"volume_flux_exact", exactFlow},
        {"flow_rate_relative_error", std::abs(qMean - exactFlow) / std::max(std::abs(exactFlow), 1e-30)},
        {"cross_section_flux_relative_spread", (qMax - qMin) / std::max(std::abs(exactFlow), 1e-30)},
        {"wall_shear_mean", shearForce / wallArea}, {"wall_shear_exact", exactShear},
        {"wall_shear_relative_error", std::abs(shearForce / wallArea - exactShear) / std::max(std::abs(exactShear), 1e-30)},
        {"cross_velocity_absolute_linf", crossMax}, {"pressure_absolute_l2", std::sqrt(pressureError2 / volume)},
        {"pressure_units", "Pa (periodic case stores pressure fluctuation; forcing is separate)"},
        {"flux_units", "volume/time; one conservative flux per subface"}
    };
    std::ofstream(options_.output / "metrics.json") << report.dump(2) << '\n';
    std::vector<double> nativePressure(pressure_.data(), pressure_.data() + pressure_.size());
    exportNativeFields(mesh_, velocity_, nativePressure, (options_.output / "native_fields.bin").string());
    std::cout << report.dump(2) << std::endl;
    return report;
}
}
