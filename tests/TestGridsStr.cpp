#include <string>
#include <magic_enum.hpp>

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

std::string ToString(const TestGrids grid_name) {
	return std::string(magic_enum::enum_name(grid_name));
}
