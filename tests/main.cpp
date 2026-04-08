#include "Common.h"
#include "SolverTests.h"
#include "FlowMapTests.h"

int main(int argc, char** argv) {


	// Run all tests

	{
		//Accuracy of the Laplacian Operator
		for (int min_level : {2, 3, 4, 5}) {
			SolverTests::TestDiscretizedLaplacian("sphere_empty", min_level, min_level + 2, "exponential");
			SolverTests::TestDiscretizedLaplacian("star_empty", min_level, min_level + 2, "exponential");
			SolverTests::TestDiscretizedLaplacian("uniform", min_level, min_level, "exponential");
		}
	}

	{
		//Sinusoidal Poisson Solver Test
		 
		//converged error
		for (int min_level : {2, 3, 4, 5}) {
			//for (int min_level : {5}) {
			SolverTests::TestSolverErrorWithAllNeumannBC("uniform", 1.0, min_level, min_level, "athena_sin", "amg");
			SolverTests::TestSolverErrorWithAllNeumannBC("sphere_empty", 1.0, min_level, min_level + 2, "athena_sin", "amg");
			SolverTests::TestSolverErrorWithAllNeumannBC("star_empty", 1.0, min_level, min_level + 2, "athena_sin", "amg");
		}


		//per-iter errors
		for (int min_level : {2, 3, 4, 5}) {
			//the paper shows the "full weighted L2"
			SolverTests::TestSolutionItersErrorWithAllNeumannBC("uniform", 1.0, min_level, min_level, "athena_sin", "amg");
			SolverTests::TestSolutionItersErrorWithAllNeumannBC("sphere_empty", 1.0, min_level, min_level + 2, "athena_sin", "amg");
			SolverTests::TestSolutionItersErrorWithAllNeumannBC("star_empty", 1.0, min_level, min_level + 2, "athena_sin", "amg");
		}
	}

	{
		//Static Pressure Projection Test with Cut Cells

		SolverTests::TestSolverErrorSolid("sphere_solid", 2, 4);
		SolverTests::TestSolverErrorSolid("sphere_solid", 3, 5);
		SolverTests::TestSolverErrorSolid("sphere_solid", 4, 6);
		SolverTests::TestSolverErrorSolid("sphere_solid", 5, 7);

		SolverTests::TestSolverErrorSolid("star_solid", 2, 4);
		SolverTests::TestSolverErrorSolid("star_solid", 3, 5);
		SolverTests::TestSolverErrorSolid("star_solid", 4, 6);
		SolverTests::TestSolverErrorSolid("star_solid", 5, 7);
	}


	//{
	//	//cg iters - error to grdt plot (set convergence test inside)
	//	SolverTests::TestSolverErrorWithAllNeumannBC("uniform", 5, 5, "athena_sin", "amg");
	//	SolverTests::TestSolverErrorWithAllNeumannBC("sphere_empty", 5, 7, "athena_sin", "amg");
	//	SolverTests::TestSolverErrorWithAllNeumannBC("star_empty", 3, 5, "athena_sin", "amg");
	//	SolverTests::TestSolverErrorWithAllNeumannBC("star_empty", 4, 6, "athena_sin", "amg");
	//	SolverTests::TestSolverErrorWithAllNeumannBC("star_empty", 5, 7, "athena_sin", "amg");
	//}

	//SolverTests::TestSolverErrorWithAllNeumannBC("star_empty", 2, 4, "athena_sin", "amg");

	//{
	//	for (int min_level : {1, 2, 3, 4, 5}) {
	//		FlowMapTests::TestFlowMapAdvection("center_point", min_level, min_level + 2);
	//	}

	////	//for (int min_level : {1, 2, 3, 4, 5}) {
	////	//	FlowMapTests::TestFlowMapAdvection("uniform", min_level, min_level);
	////	//}
	//}




	//{
		//added projection test for SIG revision
		for (int min_level : {1, 2, 3, 4, 5}) {
		//for (int min_level : {5}){
			
			
			//SolverTests::TestSolverErrorWithAllNeumannBC("uniform", 1.0, min_level, min_level, "athena_sin", "gmg");
			//SolverTests::TestSolverErrorWithAllNeumannBC("center_point", 1.0, min_level, min_level + 2, "athena_sin", "gmg");
			//SolverTests::TestSolverErrorWithAllNeumannBC("sphere_empty", 1.0, min_level, min_level + 2, "athena_sin", "gmg");
			//SolverTests::TestSolverErrorWithAllNeumannBC("star_empty", 1.0, min_level, min_level + 2, "athena_sin", "gmg");

			
			//SolverTests::TestSolverErrorWithAllNeumannBC("center_point", 1.0, min_level, min_level + 2, "athena_sin", "cmg");
			//SolverTests::TestSolverErrorWithAllNeumannBC("sphere_empty", 1.0, min_level, min_level + 2, "athena_sin", "cmg");
			//SolverTests::TestSolverErrorWithAllNeumannBC("sphere_solid", 1.0, min_level, min_level + 2, "athena_sin", "cmg");
			//SolverTests::TestSolverErrorWithAllNeumannBC("star_empty", 1.0, min_level, min_level + 2, "athena_sin", "cmg");
			//SolverTests::TestSolverErrorWithAllNeumannBC("uniform", 1.0, min_level, min_level, "athena_sin", "cmg");

			//JCP analytical solution test
			//SolverTests::TestSolverErrorWithAllNeumannBC("uniform", 1.0, min_level, min_level, "athena_sin", "amg");
			//SolverTests::TestSolverErrorWithAllNeumannBC("sphere_empty", 1.0, min_level, min_level + 2, "athena_sin", "amg");
			//SolverTests::TestSolverErrorWithAllNeumannBC("star_empty", 1.0, min_level, min_level + 2, "athena_sin", "amg");

			//SolverTests::TestSolutionItersErrorWithAllNeumannBC("sphere_empty", 1.0, min_level, min_level + 2, "athena_sin", "gmg");
			//SolverTests::TestSolutionItersErrorWithAllNeumannBC("sphere_empty", 1.0, min_level, min_level + 2, "athena_sin", "amg");
			//SolverTests::TestSolutionItersErrorWithAllNeumannBC("star_empty", 1.0, min_level, min_level + 2, "athena_sin", "amg");

			//SolverTests::TestSolutionItersErrorWithAllNeumannBC("center_point", 1.0, min_level, min_level + 2, "athena_sin", "cmg");
			//SolverTests::TestSolutionItersErrorWithAllNeumannBC("sphere_empty", 1.0, min_level, min_level + 2, "athena_sin", "cmg");
			//SolverTests::TestSolutionItersErrorWithAllNeumannBC("uniform", 1.0, min_level, min_level + 2, "athena_sin", "cmg");
		}


	//}

	{
		//test 1

		//test 1, grid convergence and runtime part
		//for (int _ : {0, 1, 2, 3, 4}) {

		//for (int min_level : {2, 3, 4, 5}) {
		//	//for (int min_level : {5}) {
		//	SolverTests::TestSolverErrorWithAllNeumannBC("uniform", 1.0, min_level, min_level, "athena_sin", "amg");
		//	SolverTests::TestSolverErrorWithAllNeumannBC("sphere_empty", 1.0, min_level, min_level + 2, "athena_sin", "amg");
		//	SolverTests::TestSolverErrorWithAllNeumannBC("star_empty", 1.0, min_level, min_level + 2, "athena_sin", "amg");

		//	//an additional test for sphere with solid inside
		//	//SolverTests::TestSolverErrorWithAllNeumannBC("sphere_solid", 1.0, min_level, min_level + 2, "athena_sin", "amg");
		//}

		//}

		////test 1, iters convergence part
		//for (int min_level : {2, 3, 4, 5}) {
		//	//the paper shows the "full weighted L2"
		//	SolverTests::TestSolutionItersErrorWithAllNeumannBC("uniform", 1.0, min_level, min_level, "athena_sin", "amg");
		//	SolverTests::TestSolutionItersErrorWithAllNeumannBC("sphere_empty", 1.0, min_level, min_level + 2, "athena_sin", "amg");
		//	SolverTests::TestSolutionItersErrorWithAllNeumannBC("star_empty", 1.0, min_level, min_level + 2, "athena_sin", "amg");
		//}


		//////cmg version of test 1 grid convergence and runtime part
		//for (int min_level : {2, 3, 4, 5}) {
		//	SolverTests::TestSolverErrorWithAllNeumannBC("uniform", 1.0, min_level, min_level, "athena_sin", "cmg");
		//	SolverTests::TestSolverErrorWithAllNeumannBC("sphere_empty", 1.0, min_level, min_level + 2, "athena_sin", "cmg");
		//	SolverTests::TestSolverErrorWithAllNeumannBC("star_empty", 1.0, min_level, min_level + 2, "athena_sin", "cmg");
		//}

		////cmg version of test 1 convergence part
		//for (int min_level : {2, 3, 4, 5}) {
		//	SolverTests::TestSolutionItersErrorWithAllNeumannBC("uniform", 1.0, min_level, min_level, "athena_sin", "cmg");
		//	SolverTests::TestSolutionItersErrorWithAllNeumannBC("sphere_empty", 1.0, min_level, min_level + 2, "athena_sin", "cmg");
		//	SolverTests::TestSolutionItersErrorWithAllNeumannBC("star_empty", 1.0, min_level, min_level + 2, "athena_sin", "cmg");
		//}

		//////gmg version of test 1 grid convergence and runtime part
		//for (int min_level : {2, 3, 4, 5}) {
		//	SolverTests::TestSolverErrorWithAllNeumannBC("uniform", 1.0, min_level, min_level, "athena_sin", "gmg");
		//	SolverTests::TestSolverErrorWithAllNeumannBC("sphere_empty", 1.0, min_level, min_level + 2, "athena_sin", "gmg");
		//	SolverTests::TestSolverErrorWithAllNeumannBC("star_empty", 1.0, min_level, min_level + 2, "athena_sin", "gmg");
		//}

		//////gmg version of test 1 grid convergence and runtime part
		//for (int min_level : {2, 3, 4, 5}) {
		//	//SolverTests::TestSolverErrorWithAllNeumannBC("uniform", 1.0, min_level, min_level, "athena_sin", "gmg_jacobi");
		//	//SolverTests::TestSolverErrorWithAllNeumannBC("sphere_empty", 1.0, min_level, min_level + 2, "athena_sin", "gmg");
		//	SolverTests::TestSolverErrorWithAllNeumannBC("star_empty", 1.0, min_level, min_level + 2, "athena_sin", "gmg_jacobi");
		//}

		////amg-vcycle version of test 1 grid convergence and runtime part
		//for (int min_level : {2, 3, 4, 5}) {
		//	SolverTests::TestSolverErrorWithAllNeumannBC("uniform", 1.0, min_level, min_level, "athena_sin", "amg_vcycle");
		//	SolverTests::TestSolverErrorWithAllNeumannBC("sphere_empty", 1.0, min_level, min_level + 2, "athena_sin", "amg_vcycle");
		//	SolverTests::TestSolverErrorWithAllNeumannBC("star_empty", 1.0, min_level, min_level + 2, "athena_sin", "amg_vcycle");
		//}
	}


	 
	//SolverTests::TestRecoveryNew("sphere_solid", 2, 4);
	//SolverTests::TestRecoveryNew("sphere_solid", 3, 5);
	//SolverTests::TestRecoveryNew("sphere_solid", 4, 6);
	//SolverTests::TestRecoveryNew("sphere_solid", 5, 7);



	

	////for (int max_level : {3, 4, 5, 6}) {
	//for (int max_level : {6}){

	//	SolverTests::TestAnalyticalConvergence(ConvergenceTestGridName::sphere_empty, max_level, "athena_sin", "cmg");
	//	SolverTests::TestAnalyticalConvergence(ConvergenceTestGridName::sphere_empty, max_level, "athena_sin", "amg");
	//}
	return 0;
}