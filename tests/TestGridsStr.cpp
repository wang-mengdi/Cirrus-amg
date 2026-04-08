#include "TestGridsStr.h"
#include <magic_enum.hpp>

std::string ToString(const TestGrids grid_name) {
	return std::string(magic_enum::enum_name(grid_name));
}
