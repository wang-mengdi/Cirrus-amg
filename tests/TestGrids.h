#include <string>

enum class TestGrids {
	uniform8 = 0,
	uniform128,
	uniform256,
	uniform512,
	staircase12,
	staircase21,
	staircase34,
	twosources67,
	twosources_deform,
	centersource
};

std::string ToString(const TestGrids grid_name);