#pragma once
#include <string>

namespace FlowMapTests {
	void TestParticleAdvectionWithNFMInterpolation(const int grid_case);

	void TestFlowMapAdvection(const std::string grid_name, const int min_level, const int max_level);
}