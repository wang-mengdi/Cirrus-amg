#include "Common.h"
#include "SolverTests.h"


int main(int argc, char** argv) {


	// Run all tests

	//for (auto grid_name : { 
	//	////TestGrids::uniform8,
	//	//TestGrids::uniform32,
	//	TestGrids::uniform128,
	//	//TestGrids::uniform512, 
	//	////TestGrids::staircase12, 
	//	////TestGrids::staircase21, 
	//	////TestGrids::staircase34,
	//	TestGrids::staircase43,
	//	TestGrids::twosources67,
	//	}) {

	//	//SolverTests::TestAMGLaplacianAndFluxConsistency(grid_name);
	//	//SolverTests::TestNeumannDirichletRecovery(grid_name, "cmg");
	//	SolverTests::TestNeumannDirichletRecovery(grid_name, "amg");
	//}


	SolverTests::TestAnalyticalConvergence(ConvergenceTestGridName::sphere_shell_05, 6, "gerris_sin", "amg");

	////for (int max_level : {3, 4, 5, 6}) {
	//for (int max_level : {6}){

	//	SolverTests::TestAnalyticalConvergence(ConvergenceTestGridName::sphere_shell_05, max_level, "athena_sin", "cmg");
	//	SolverTests::TestAnalyticalConvergence(ConvergenceTestGridName::sphere_shell_05, max_level, "athena_sin", "amg");
	//}
	return 0;
}