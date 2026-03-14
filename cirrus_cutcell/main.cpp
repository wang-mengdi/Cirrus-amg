#include "Common.h"
#include "FluidEulerInitializer.h"
#include <vector>
#include <fmt/core.h>
#include <random>
#include "ExportXform.h"


int main(int argc, char** argv) {
	ExportMeshTransforms(argv[1], "output/mesh_transforms");
	return 0;


	ASSERT(argc == 2, "Usage: [].exe path_to_json_config.json");
	Run_FluidEuler(argv[1]);
	return 0;
}
