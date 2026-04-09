#include <fstream>
#include <iomanip>
#include <cstdio>
#include "ExportXform.h"

void ExportSingleFileTransform(const FluidParams& mParams, const fs::path& out_file, T time)
{
    auto xform = FluidScene::meshToWorldTransform(mParams, time);
    Eigen::Matrix<T, 4, 4> M = xform.matrix();
    std::ofstream ofs(out_file);
    ASSERT(ofs.good(), "ExportSingleFileTransform: failed to open {}", out_file.string());
    ofs << std::setprecision(17);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            ofs << M(i, j);
            if (j < 3) ofs << " ";
        }
        ofs << "\n";
    }
    Info("Exported transform at time {}s to {}", time, out_file.string());
}

void ExportMeshTransforms(
    const FluidParams& mParams,
    const fs::path& out_dir,
    T fps,
    int n)
{
    ASSERT(fps > 0, "ExportMeshTransforms: fps must be positive, got {}", fps);
    ASSERT(n >= 0, "ExportMeshTransforms: n must be non-negative, got {}", n);

    if (!fs::exists(out_dir)) {
        fs::create_directories(out_dir);
    }
    ASSERT(fs::exists(out_dir) && fs::is_directory(out_dir),
        "ExportMeshTransforms: failed to create output directory {}", out_dir.string());

    for (int frame = 0; frame <= n; ++frame) {
        T current_time = static_cast<T>(frame) / fps;
		ExportSingleFileTransform(mParams, out_dir / fmt::format("xform_{:04d}.txt", frame), current_time);
    }

    Info("Exported transforms for frames 0..{} to {}", n, out_dir.string());
}

void ExportMeshTransforms(const fs::path& json_config_path, const fs::path& out_dir) {
    json j;
    std::ifstream json_input(json_config_path.string());
    ASSERT(json_input.is_open(), "Failed to open json file {}", json_config_path.string());
    json_input >> j;
    json_input.close();
    Info("ExportMeshTransforms parse json: \n{}", j.dump(2));

    int fps = Json::Value<int>(j["driver"], "fps", 200);
    int last_frame = Json::Value<int>(j["driver"], "last_frame", 0);

    FluidParams params(j["scene"]);
	ExportMeshTransforms(params, out_dir, fps, last_frame);
}