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


	//{
	//	//cg iters - error to grdt plot (set convergence test inside)
	//	SolverTests::TestSolverErrorWithAllNeumannBC("uniform", 5, 5, "athena_sin", "amg");
	//	SolverTests::TestSolverErrorWithAllNeumannBC("sphere_shell_05", 5, 7, "athena_sin", "amg");
	//	SolverTests::TestSolverErrorWithAllNeumannBC("star_shell", 3, 5, "athena_sin", "amg");
	//	SolverTests::TestSolverErrorWithAllNeumannBC("star_shell", 4, 6, "athena_sin", "amg");
	//	SolverTests::TestSolverErrorWithAllNeumannBC("star_shell", 5, 7, "athena_sin", "amg");
	//}

	//SolverTests::TestSolverErrorWithAllNeumannBC("star_shell", 2, 4, "athena_sin", "amg");


	// (int min_level : {2, 3, 4, 5}) {
		//grid convergence of cg
		//SolverTests::TestSolverErrorWithAllNeumannBC("uniform", min_level, min_level, "athena_sin", "amg");
		//SolverTests::TestSolverErrorWithAllNeumannBC("sphere_shell_05", min_level, min_level + 2, "athena_sin", "amg");
		//SolverTests::TestSolverErrorWithAllNeumannBC("star_shell", min_level, min_level + 2, "athena_sin", "amg");

		//SolverTests::TestDiscretizedLaplacian("uniform", min_level, min_level, "athena_sin");

		//SolverTests::TestSolverErrorWithAllNeumannBC("uniform", min_level, min_level, "athena_sin", "cmg");

		//SolverTests::TestSolverErrorWithAllNeumannBC("uniform", min_level, min_level, "athena_sin", "amg");
		
	//}
	SolverTests::TestSolverErrorSolid("sphere", 2, 2);


	//for (int min_level : {4, 5}) {
	for (int min_level : {4}) {
	//for (int min_level : {1}){
		//SolverTests::TestDiscretizedLaplacian("sphere_solid_05", min_level, min_level + 2, "athena_sin");

		//SolverTests::TestSolverErrorWithAllNeumannBC("sphere_shell_05", min_level, min_level + 2, "athena_sin", "cmg");
		//SolverTests::TestSolverErrorWithAllNeumannBC("star_shell", min_level, min_level + 2, "athena_sin", "cmg");

		//SolverTests::TestSolverErrorWithAllNeumannBC("star_shell", min_level, min_level + 2, "athena_sin", "amg");
	}
	

	////for (int max_level : {3, 4, 5, 6}) {
	//for (int max_level : {6}){

	//	SolverTests::TestAnalyticalConvergence(ConvergenceTestGridName::sphere_shell_05, max_level, "athena_sin", "cmg");
	//	SolverTests::TestAnalyticalConvergence(ConvergenceTestGridName::sphere_shell_05, max_level, "athena_sin", "amg");
	//}
	return 0;
}