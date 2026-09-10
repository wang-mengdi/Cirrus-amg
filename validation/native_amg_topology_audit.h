#pragma once
#include "SimpleMesh.h"
#include <filesystem>
#include <nlohmann/json.hpp>

nlohmann::json auditNativeAmgTopology(const simple::Mesh& mesh,
    const std::filesystem::path& output,bool completePeriodic=false);
nlohmann::json auditNativeAmgCompactSolve(const simple::Mesh& mesh,
    const std::filesystem::path& output);
