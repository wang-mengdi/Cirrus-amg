#include "FluidEuler.h"
#include "FluidEulerInitializer.h"
#include "Driver.h"

//template<class T>
class FluidEulerInitializer {
public:

	void Apply(json& j_all, FluidEuler& fluid, DriverMetaData &metadata) {
		json& j = j_all.at("scene");
		fluid.init(j, metadata);
	}
};



void Run_FluidEuler(const fs::path& json_path) {
	FluidEuler fluid;
	FluidEulerInitializer scene;
	DriverFunc::Initialize_And_Run(json_path, scene, fluid);
}   