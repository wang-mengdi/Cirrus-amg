#include <string>

enum class TestGrids {
	uniform8 = 0,
	uniform32,
	uniform128,
	uniform256,
	uniform512,
	staircase12,
	staircase21,
	staircase34,
	staircase43,
	twosources67,
	twosources_deform,
	centersource,
};

std::string ToString(const TestGrids grid_name);

enum class ConvergenceTestGridName {
	sphere_shell_05 = 0, // R=0.5, center=(0.5, 0.5, 0.5)
};

std::string ToString(const ConvergenceTestGridName grid_name);