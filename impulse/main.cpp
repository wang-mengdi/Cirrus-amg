#include "Common.h"
#include "FluidEulerInitializer.h"
#include <vector>
#include <fmt/core.h>
#include "MeshSDFAccel.h"

//#include <vtkAMRBox.h>

int main(int argc, char** argv) {
	// Load mesh
	Eigen::Matrix<float, -1, 3> V;
	Eigen::Matrix<int, -1, 3> F;
	if (!igl::readOBJ("C:\\Users\\mwang\\Downloads\\sphere.obj", V, F)) {
		std::cerr << "Failed to read sphere.obj\n";
		return 1;
	}

	// Build acceleration structure
	MeshSDFAccel accel;
	accel.build(V, F);

	// Generate query points from (0,0,-10) to (0,0,10)
	const int N = 21; // 21 samples including both ends
	Eigen::Matrix<float, -1, 3> P(N, 3);
	for (int i = 0; i < N; ++i) {
		float z = -10.0f + 20.0f * i / (N - 1);
		P.row(i) << 0.0f, 0.0f, z;
	}

	// Query SDF
	Eigen::VectorXf sdf = accel.querySDF(P);

	// Print results
	for (int i = 0; i < N; ++i) {
		std::cout << "Point (0,0," << P(i, 2) << ") -> SDF = " << sdf[i] << "\n";
	}
	return 0;


	try {
		json j = {
			{
				"driver",
				{
					{"last_frame",10}
				}
			},
			{"scene",json::object()}
		};
		if (argc > 1) {
			fmt::print("Read json file {}\n", argv[1]);
			std::ifstream json_input(argv[1]);
			json_input >> j;
			json_input.close();
		}

		std::string simulator = Json::Value(j, "simulator", std::string("euler"));

		if (simulator == "euler") {
			Run_FluidEuler(j);
		}

	}
	catch (nlohmann::json::exception& e)
	{
		fmt::print("json exception {}\n", e.what());
	}

	return 0;
}
