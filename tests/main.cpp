#include "Common.h"
#include "SolverTests.h"
#include "FlowMapTests.h"

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

	{
		//advection convergence test for SIG revision
		for (int i : {8, 0, 9, 7, 2}) {
			//TestNeumannBC(i);
			FlowMapTests::TestFlowMapAdvection(i);
			fmt::print("====================================\n");
		}
	}


	//{
	//	//added projection test for SIG revision
	//	for (int min_level : {2, 3, 4, 5}) {
	//		SolverTests::TestSolverErrorWithAllNeumannBC("center", 1.0, min_level, min_level + 2, "athena_sin", "gmg");
	//	}
	//}

	{
		//test 1

		////test 1, grid convergence and runtime part
		//for (int min_level : {2, 3, 4, 5}) {
		//	SolverTests::TestSolverErrorWithAllNeumannBC("uniform", 1.0, min_level, min_level, "athena_sin", "amg");
		//	SolverTests::TestSolverErrorWithAllNeumannBC("sphere_shell_05", 1.0, min_level, min_level + 2, "athena_sin", "amg");
		//	SolverTests::TestSolverErrorWithAllNeumannBC("star_shell", 1.0, min_level, min_level + 2, "athena_sin", "amg");

		//	//an additional test for sphere with solid inside
		//	//SolverTests::TestSolverErrorWithAllNeumannBC("sphere_solid_05", 1.0, min_level, min_level + 2, "athena_sin", "amg");
		//}

		////test 1, iters convergence part
		//for (int min_level : {2, 3, 4, 5}) {
		//	//the paper shows the "full weighted L2"
		//	SolverTests::TestSolutionItersErrorWithAllNeumannBC("uniform", 1.0, min_level, min_level, "athena_sin", "amg");
		//	SolverTests::TestSolutionItersErrorWithAllNeumannBC("sphere_shell_05", 1.0, min_level, min_level + 2, "athena_sin", "amg");
		//	SolverTests::TestSolutionItersErrorWithAllNeumannBC("star_shell", 1.0, min_level, min_level + 2, "athena_sin", "amg");
		//}


		//////cmg version of test 1 grid convergence and runtime part
		//for (int min_level : {2, 3, 4, 5}) {
		//	SolverTests::TestSolverErrorWithAllNeumannBC("uniform", 1.0, min_level, min_level, "athena_sin", "cmg");
		//	SolverTests::TestSolverErrorWithAllNeumannBC("sphere_shell_05", 1.0, min_level, min_level + 2, "athena_sin", "cmg");
		//	SolverTests::TestSolverErrorWithAllNeumannBC("star_shell", 1.0, min_level, min_level + 2, "athena_sin", "cmg");
		//}

		////cmg version of test 1 convergence part
		//for (int min_level : {2, 3, 4, 5}) {
		//	SolverTests::TestSolutionItersErrorWithAllNeumannBC("uniform", 1.0, min_level, min_level, "athena_sin", "cmg");
		//	SolverTests::TestSolutionItersErrorWithAllNeumannBC("sphere_shell_05", 1.0, min_level, min_level + 2, "athena_sin", "cmg");
		//	SolverTests::TestSolutionItersErrorWithAllNeumannBC("star_shell", 1.0, min_level, min_level + 2, "athena_sin", "cmg");
		//}

		//////gmg version of test 1 grid convergence and runtime part
		//for (int min_level : {2, 3, 4, 5}) {
		//	SolverTests::TestSolverErrorWithAllNeumannBC("uniform", 1.0, min_level, min_level, "athena_sin", "gmg");
		//	SolverTests::TestSolverErrorWithAllNeumannBC("sphere_shell_05", 1.0, min_level, min_level + 2, "athena_sin", "gmg");
		//	SolverTests::TestSolverErrorWithAllNeumannBC("star_shell", 1.0, min_level, min_level + 2, "athena_sin", "gmg");
		//}

		//////gmg version of test 1 grid convergence and runtime part
		//for (int min_level : {2, 3, 4, 5}) {
		//	//SolverTests::TestSolverErrorWithAllNeumannBC("uniform", 1.0, min_level, min_level, "athena_sin", "gmg_jacobi");
		//	//SolverTests::TestSolverErrorWithAllNeumannBC("sphere_shell_05", 1.0, min_level, min_level + 2, "athena_sin", "gmg");
		//	SolverTests::TestSolverErrorWithAllNeumannBC("star_shell", 1.0, min_level, min_level + 2, "athena_sin", "gmg_jacobi");
		//}

		////amg-vcycle version of test 1 grid convergence and runtime part
		//for (int min_level : {2, 3, 4, 5}) {
		//	SolverTests::TestSolverErrorWithAllNeumannBC("uniform", 1.0, min_level, min_level, "athena_sin", "amg_vcycle");
		//	SolverTests::TestSolverErrorWithAllNeumannBC("sphere_shell_05", 1.0, min_level, min_level + 2, "athena_sin", "amg_vcycle");
		//	SolverTests::TestSolverErrorWithAllNeumannBC("star_shell", 1.0, min_level, min_level + 2, "athena_sin", "amg_vcycle");
		//}
	}

	//SolverTests::TestSolverErrorSolid("sphere", 2, 4);
	//SolverTests::TestSolverErrorSolid("sphere", 3, 5);
	//SolverTests::TestSolverErrorSolid("sphere", 4, 6);
	//SolverTests::TestSolverErrorSolid("sphere", 5, 7);
	 
	//SolverTests::TestRecoveryNew("sphere", 2, 4);
	//SolverTests::TestRecoveryNew("sphere", 3, 5);
	//SolverTests::TestRecoveryNew("sphere", 4, 6);
	//SolverTests::TestRecoveryNew("sphere", 5, 7);


	//for (int min_level : {4, 5}) {
	//for (int min_level : {4}) {
	//for (int min_level : {1}){
		//SolverTests::TestDiscretizedLaplacian("sphere_solid_05", min_level, min_level + 2, "athena_sin");

		//SolverTests::TestSolverErrorWithAllNeumannBC("sphere_shell_05", min_level, min_level + 2, "athena_sin", "cmg");
		//SolverTests::TestSolverErrorWithAllNeumannBC("star_shell", min_level, min_level + 2, "athena_sin", "cmg");

		//SolverTests::TestSolverErrorWithAllNeumannBC("star_shell", min_level, min_level + 2, "athena_sin", "amg");

	//}
	

	////for (int max_level : {3, 4, 5, 6}) {
	//for (int max_level : {6}){

	//	SolverTests::TestAnalyticalConvergence(ConvergenceTestGridName::sphere_shell_05, max_level, "athena_sin", "cmg");
	//	SolverTests::TestAnalyticalConvergence(ConvergenceTestGridName::sphere_shell_05, max_level, "athena_sin", "amg");
	//}
	return 0;
}