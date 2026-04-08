# pragma once

#include <filesystem>
#include "FluidParams.h"
namespace fs = std::filesystem;

void ExportSingleFileTransform(const FluidParams& mParams, const fs::path& out_file, T time);