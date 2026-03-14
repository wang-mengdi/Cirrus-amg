#include <filesystem>
namespace fs = std::filesystem;

void ExportMeshTransforms(const fs::path& json_config_path, const fs::path& out_dir);