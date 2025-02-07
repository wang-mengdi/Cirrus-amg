#include "Common.h"
#include "SolverTests.h"


int main(int argc, char** argv) {


	// Run all tests

	for (auto grid_name : { 
		TestGrids::uniform128, 
		TestGrids::staircase34,
		TestGrids::twosources67,
		}) {

		SolverTests::TestNeumannDirichletRecovery(grid_name, "cmg");
	}
	return 0;
}