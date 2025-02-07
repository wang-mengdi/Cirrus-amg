#include "Common.h"
#include "SolverTests.h"

int main(int argc, char** argv) {
	// Run all tests

	for (int i : {0, 1, 2}) {

		//SolverTests::TestNeumannBC(i);
		SolverTests::TestAMGNeumannBC(i);
	}
	return 0;
}