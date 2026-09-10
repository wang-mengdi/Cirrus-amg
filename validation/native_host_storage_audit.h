#pragma once
#include "SimpleMesh.h"
#include <filesystem>
#include <nlohmann/json.hpp>

// Consumes nativeStorage at completion to verify scratch ownership cleanup.
nlohmann::json auditNativeHostStorage(simple::Mesh& mesh,const std::filesystem::path& output);
