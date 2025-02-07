#include "Common.h"
#include "SolverTests.h"

int main(int argc, char** argv) {
	// Run all tests

	for (auto grid_name : { 
		SolverTests::TestGrids::uniform128, 
		SolverTests::TestGrids::staircase34,
		SolverTests::TestGrids::twosources67,
		}) {

		SolverTests::TestNeumannDirichletRecovery(grid_name, "cmg");
	}
	return 0;
}