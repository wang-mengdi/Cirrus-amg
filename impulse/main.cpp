#include "Common.h"
#include "FluidEulerInitializer.h"
#include <vector>
#include <fmt/core.h>
#include "MeshSDFAccel.h"
#include <random>


int main(int argc, char** argv) {
	static_assert(std::is_trivially_copyable_v<HATileInfo<Tile>>, "T must be trivially copyable");
	static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
	static_assert(std::is_trivially_copyable_v<Coord>, "Coord must be trivially copyable");
	static_assert(std::is_trivially_copyable_v<Tile>, "Tile must be trivially copyable for raw binary load");


	try {
		json j = {
			{
				"driver",
				{
					{"last_frame",10}
				}
			},
			{"scene",json::object()}
		};
		if (argc > 1) {
			fmt::print("Read json file {}\n", argv[1]);
			std::ifstream json_input(argv[1]);
			json_input >> j;
			json_input.close();
		}

		std::string simulator = Json::Value(j, "simulator", std::string("euler"));

		if (simulator == "euler") {
			Run_FluidEuler(j);
		}

	}
	catch (nlohmann::json::exception& e)
	{
		fmt::print("json exception {}\n", e.what());
	}

	return 0;
}
