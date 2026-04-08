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
		for (int min_level : {3, 4, 5}) {
			SolverTests::TestSolutionItersErrorWithAllNeumannBC("uniform", 1.0, min_level, min_level, "athena_sin", "amg");
		}

		//per-iter errors
		for (int min_level : {2, 3, 4, 5}) {
			SolverTests::TestSolutionItersErrorWithAllNeumannBC("sphere_empty", 1.0, min_level, min_level + 2, "athena_sin", "amg");
		}

		//per-iter errors
		for (int min_level : {2, 3, 4, 5}) {
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
	}
	return 0;
}