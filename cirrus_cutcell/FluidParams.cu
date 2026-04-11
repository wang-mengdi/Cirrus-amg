#include "FluidParams.h"

#include "Json.h"

FluidParams::FluidParams(json& j)
{

	std::string test = Json::Value<std::string>(j, "test", "meshmotion");
	if (test == "meshmotion") mTestCase = MESHMOTION;
	else if (test == "aircraft") mTestCase = AIRCRAFT;
	else if (test == "sphere_circling") mTestCase = SPHERECIRCLING;
	else if (test == "jassm") mTestCase = JASSM;
	else ASSERT(false, "invalid test {}", test);

	if (mTestCase == MESHMOTION || mTestCase == AIRCRAFT || mTestCase == SPHERECIRCLING || mTestCase == JASSM) {
		mIsPureNeumann = true;
		mesh_motion_inflow = Json::Value<T>(j, "inflow_velocity", 1.0);
	}

	int init_x = Json::Value<int>(j, "initial_grid_size_x", 1);
	int init_y = Json::Value<int>(j, "initial_grid_size_y", 1);
	int init_z = Json::Value<int>(j, "initial_grid_size_z", 1);
	mInitialGridSize = Coord(init_x, init_y, init_z);

	mFlowMapStride = Json::Value<int>(j, "flowmap_stride", 5);
	mCoarseLevel = Json::Value<int>(j, "coarse_level", 0);
	mFineLevel = Json::Value<int>(j, "fine_level", 6);
	mGravity = Vec(0, 0, 0);
	mGravity[2] = Json::Value<double>(j, "gravity", -9.8);
	mParticleLife = Json::Value<T>(j, "particle_life", FLT_MAX);

	mSampleNumPerCell = Json::Value<int>(j, "sample_num_per_cell", 8);
	mRelativeParticleSampleBandwidth = Json::Value<double>(j, "particle_sample_relative_bandwidth", 5.0);
	mRelativeRefineBandwidth = Json::Value<double>(j, "refine_relative_bandwidth", 10.0);

	mExtrapolationIters = Json::Value<int>(j, "extrapolation_iters", 5);
}
