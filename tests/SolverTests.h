#include "TestGrids.h"


namespace SolverTests {

	void TestAMGLaplacianAndFluxConsistency(TestGrids grid_name);
	void TestNeumannDirichletRecovery(const TestGrids grid_name, const std::string algorithm);

}