#pragma once
#include "SimpleSolver.h"
#include <memory>

namespace simple {
// Single-phase projection on the actual native leaf/control-volume topology.
// Uniform cut-grid ordering follows Aphros Proj; no flow-map update is used.
class ProjectionSolver {
public:
    ProjectionSolver(Mesh mesh,Options options);
    ~ProjectionSolver();
    nlohmann::json run();
    // Diagnostic only: reuse the actual projection operator on a retained RHS.
    // No velocity, pressure, flux or physical time is advanced.
    nlohmann::json replayPressure(const std::filesystem::path& rhs,int repetitions);
    // Compare both exit rules on copies of a retained conservative flux.
    // This operator-only diagnostic does not advance a physical time step.
    nlohmann::json replayProjection(const std::filesystem::path& flux,int repetitions);
    nlohmann::json replayPredictor(const std::filesystem::path& state,int repetitions);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
