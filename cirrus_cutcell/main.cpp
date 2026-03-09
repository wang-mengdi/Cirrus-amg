#include "Common.h"
#include "FluidEulerInitializer.h"
#include <vector>
#include <fmt/core.h>
#include "MeshSDFAccel.h"
#include <random>


int main(int argc, char** argv) {
	ASSERT(argc == 2, "Usage: [].exe path_to_json_config.json");
	Run_FluidEuler(argv[1]);
	return 0;
}
