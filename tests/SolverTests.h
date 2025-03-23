#include "TestGrids.h"


namespace SolverTests {

	void TestAMGLaplacianAndFluxConsistency(TestGrids grid_name);
	void TestNeumannDirichletRecovery(const TestGrids grid_name, const std::string algorithm);


	void TestDiscretizedLaplacian(const std::string grid_name, const int min_level, const int max_level, const std::string bc_name);
	void TestSolverErrorWithAllNeumannBC(const std::string grid_name, const int min_level, const int max_level, const std::string bc_name, const std::string algorithm);
	void TestSolverErrorSolid(const std::string grid_name, const int min_level, const int max_level);
	void TestRecoveryNew(const std::string grid_name, const int min_level, const int max_level);
}