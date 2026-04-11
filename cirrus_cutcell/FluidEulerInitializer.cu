#include "FluidEuler.h"
#include "FluidEulerInitializer.h"
#include "Driver.h"
#include "Json.h"

#include <fstream>
#include <iomanip>

namespace {

void InitializeAndRunFluidEuler(const fs::path& json_path, FluidEuler& fluid) {
	Info("Reading json file {}", json_path.string());
	json j;
	std::ifstream json_input(json_path.string());
	ASSERT(json_input.is_open(), "Failed to open json file {}", json_path.string());
	json_input >> j;
	json_input.close();
	Info("Driver::Initialize_And_Run parse json: \n{}", j.dump(2));

	DriverMetaData meta_data;
	meta_data.init(json_path, j.at("driver"));
	Info("Initialized driver metadata: \n{}", j.at("driver").dump(2));
	fluid.init(j.at("scene"), meta_data);
	fs::create_directories(meta_data.base_path);
	fs::path dump_file = meta_data.base_path / fs::path("config.json");
	std::ofstream dump_output(dump_file.string());
	dump_output << std::setw(4) << j;
	dump_output.close();
	DriverFunc::Advance(fluid, meta_data);
}

} // namespace

void Run_FluidEuler(const fs::path& json_path) {
	FluidEuler fluid;
	InitializeAndRunFluidEuler(json_path, fluid);
}
