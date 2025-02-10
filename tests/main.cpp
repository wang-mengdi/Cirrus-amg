#include "Common.h"
#include "SolverTests.h"


int main(int argc, char** argv) {


	// Run all tests

	for (auto grid_name : { 
		////TestGrids::uniform8,
		//TestGrids::uniform32,
		//TestGrids::uniform128, 
		////TestGrids::staircase12, 
		////TestGrids::staircase21, 
		////TestGrids::staircase34,
		//TestGrids::staircase43,
		TestGrids::twosources67,
		}) {

		//SolverTests::TestAMGLaplacianAndFluxConsistency(grid_name);
		//SolverTests::TestNeumannDirichletRecovery(grid_name, "cmg");
		SolverTests::TestNeumannDirichletRecovery(grid_name, "amg");
	}
	return 0;
}