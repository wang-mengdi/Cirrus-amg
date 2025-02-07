#include <string>

enum class TestGrids {
	uniform128 = 0,
	uniform256,
	uniform512,
	staircase34,
	twosources67,
	twosources_deform,
	centersource
};

std::string ToString(const TestGrids grid_name);