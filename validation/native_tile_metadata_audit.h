#pragma once
#include "SimpleMesh.h"
#include <filesystem>
#include <nlohmann/json.hpp>

nlohmann::json auditNativeTileMetadata(simple::Mesh& mesh,
    const std::filesystem::path& output,bool exerciseAmg);
