#pragma once
#include "SimpleMesh.h"
#include "QuadraticReconstruction.h"
#include "EmbeddedOperators.h"
#include <Eigen/Sparse>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <string>
#include <vector>

namespace simple {
struct Options {
    int ny = 8, maxIterations = 2000, nonorthIterations = 8;
    int andersonDepth = 0;
    int steadyAndersonDepth = 0;
    std::string pressureSolver = "auto";
    std::string linearBackend = "cpu";
    std::string gpuPreconditioner = "jacobi";
    std::string gpuPressureGauge = "pin";
    std::string gpuPressureOperator = "compact";
    std::string gpuViscosityOperator = "compact";
    std::string gpuOrthogonalization = "mgs2";
    int gpuKrylovDimension = 20;
    std::string embeddedGeometry;
    std::string momentumMode = "imp";
    std::string fluidSolver = "simple";
    std::string restartCheckpoint;
    double projectionIterationTolerance = 1e-11;
    bool adaptive = false, periodicX = true, convection = true;
    bool periodicZ = true, quadraticInterfaces = true;
    bool fluxRelaxationMemory = false;
    double rho = 1., nu = .01, alphaU = .7, alphaP = .3;
    double pressureIn = 1., pressureOut = 0., perturbation = 0.;
    double tolerance = 1e-8, linearTolerance = 1e-11;
    double timeStep = 0.;
    int timeSteps = 1;
    int outputStride = 1;
    Vec3 force = Vec3(1., 0., 0.);
    std::vector<int> dumpIterations{0, 1, 2, 5, 10};
    std::filesystem::path output = "output/simple_channel";
    static Options fromJson(const nlohmann::json& j);
};

class Solver {
public:
    Solver(Mesh mesh, Options options);
    nlohmann::json run();
    const Mesh& mesh() const { return mesh_; }
    const std::vector<Vec3>& velocity() const { return velocity_; }
    const Eigen::VectorXd& pressure() const { return pressure_; }
private:
    using Sparse = Eigen::SparseMatrix<double>;
    using Grad = std::vector<Vec3>;
    struct Momentum { Sparse matrix; Eigen::MatrixXd rhs; Eigen::VectorXd diagonal; };
    Mesh mesh_;
    Options options_;
    std::vector<Vec3> velocity_;
    std::vector<Vec3> timeVelocity_;
    Eigen::VectorXd pressure_, flux_, reciprocalDiagonal_;
    std::vector<Eigen::Matrix3d> inverseGradientMetric_;
    std::vector<int> wallSecondCell_;
    std::unique_ptr<QuadraticReconstruction> quadratic_;
    std::unique_ptr<EmbeddedOperators> embedded_;
    double velocityScale_, accelerationScale_;
    Grad gradient(const Eigen::VectorXd& q, int component, bool correction = false) const;
    double pressureBoundary(const Face& f, bool correction) const;
    double weight(const Face& f) const;
    Vec3 faceNormal(const Face& f) const;
    double interpolate(const Face& f, const Eigen::VectorXd& q, const Grad& grad) const;
    double compactGradient(const Face& f, const Eigen::VectorXd& q, const Grad& grad,
                           bool correction, const QuadraticReconstruction::Derivatives* quadratic = nullptr) const;
    Eigen::VectorXd divergence(const Eigen::VectorXd& flux) const;
    Momentum momentum(const std::vector<Vec3>& velocity, const Eigen::VectorXd& flux,
                      const Eigen::VectorXd& pressure, bool relax) const;
    double momentumResidual(const std::vector<Vec3>& velocity, const Eigen::VectorXd& flux,
                            const Eigen::VectorXd& pressure, double volume,bool includeTime=true) const;
    nlohmann::json runStep();
    Eigen::VectorXd velocityFlux(const std::vector<Vec3>& velocity) const;
    void dump(int iteration, const Momentum* momentum = nullptr,
              const std::vector<Vec3>* predictor = nullptr,
              const Eigen::VectorXd* predictedFlux = nullptr,
              const Eigen::VectorXd* pressureRhs = nullptr,
              const Eigen::VectorXd* correction = nullptr,
              const Sparse* pressureMatrix = nullptr) const;
};
}
