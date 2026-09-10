#pragma once
#include "SimpleMesh.h"
#include <Eigen/Sparse>
#include <array>

namespace simple {
class QuadraticReconstruction;
// Independent matrix implementation of the Aphros embedded-boundary operators.
// Benchmark unknowns are stored at original cube centers, including cut cells.
struct EmbeddedOperators {
    using Sparse=Eigen::SparseMatrix<double,Eigen::RowMajor>;
    Sparse interpolation, faceGradient, wallGradient;
    Sparse compactDiffusion, deferredDiffusion;
    Sparse cartesianFaceInterpolation, redistribution;
    Eigen::VectorXd interpolationBoundaryWeight;
    std::array<Sparse,3> cellGradient;
    bool explicitMomentum=false;
    explicit EmbeddedOperators(const Mesh& mesh,const QuadraticReconstruction* quadratic=nullptr,bool convection=false,
                               bool explicitMomentum=false,bool redistributeDiffusion=true);
    std::pair<Sparse,Sparse> upwindAdvection(const Mesh& mesh,const Eigen::VectorXd& flux,double rho) const;
    void check(const Mesh& mesh,const std::string& output) const;
};
}
