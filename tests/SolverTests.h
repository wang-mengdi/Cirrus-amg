#include "Common.h"
#include <magic_enum.hpp>


namespace SolverTests {
	enum class TestGrids {
		uniform128 = 0,
		uniform256,
		uniform512,
		staircase34,
		twosources67,
		twosources_deform,
		centersource
	};

	void TestNeumannDirichletRecovery(const TestGrids grid_name, const std::string algorithm);

}