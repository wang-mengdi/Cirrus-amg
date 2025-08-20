#include "Common.h"
#include "FluidEulerInitializer.h"
#include <vector>
#include <fmt/core.h>
#include "MeshSDFAccel.h"
#include <random>

//#include <vtkAMRBox.h>

// Generate N random points inside the mesh axis-aligned bounding box.
inline Eigen::Matrix<MeshSDFAccel::Scalar, -1, 3>
SamplePointsInMeshBBox(const Eigen::Matrix<MeshSDFAccel::Scalar, -1, 3>& V, int N,
	unsigned seed = 12345)
{
	using Scalar = MeshSDFAccel::Scalar;
	Eigen::Matrix<Scalar, 1, 3> vmin = V.colwise().minCoeff();
	Eigen::Matrix<Scalar, 1, 3> vmax = V.colwise().maxCoeff();

	std::mt19937 rng(seed);
	std::uniform_real_distribution<Scalar> dx(vmin(0), vmax(0));
	std::uniform_real_distribution<Scalar> dy(vmin(1), vmax(1));
	std::uniform_real_distribution<Scalar> dz(vmin(2), vmax(2));

	Eigen::Matrix<Scalar, -1, 3> P(N, 3);
	for (int i = 0; i < N; ++i) {
		P(i, 0) = dx(rng);
		P(i, 1) = dy(rng);
		P(i, 2) = dz(rng);
	}
	return P;
}

// Load OBJ, build accel, sample N points in its bbox, compute SDF.
// Returns true on success. Outputs P (N x 3) and SDF (N x 1).
inline bool RandomSDFInBBox(const std::string& obj_path, int N,
	Eigen::Matrix<MeshSDFAccel::Scalar, -1, 3>& P_out,
	Eigen::Matrix<MeshSDFAccel::Scalar, -1, 1>& S_out)
{
	using Scalar = MeshSDFAccel::Scalar;

	// Read OBJ into double then cast to Scalar to match MeshSDFAccel
	Eigen::Matrix<double, -1, 3> Vd;
	Eigen::Matrix<int, -1, 3> F;
	if (!igl::readOBJ(obj_path, Vd, F)) {
		std::cerr << "Failed to read OBJ: " << obj_path << "\n";
		return false;
	}
	Eigen::Matrix<Scalar, -1, 3> V = Vd.cast<Scalar>();

	// Build accel once
	MeshSDFAccel accel;
	accel.build(V, F);

	// Sample random points in bbox
	P_out = SamplePointsInMeshBBox(V, N);

	// Compute SDF (TBB parallel inside)
	auto t0 = std::chrono::steady_clock::now();
	S_out = accel.querySDF(P_out);
	auto t1 = std::chrono::steady_clock::now();
	double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
	std::cout << "querySDF time: " << ms << " ms ("
		<< static_cast<double>(N) / (ms / 1000.0) << " pts/s)\n";

	// Optional: brief stats (do not print all 1e6 lines)
	const Scalar s_min = S_out.minCoeff();
	const Scalar s_max = S_out.maxCoeff();
	const Scalar s_mean = static_cast<Scalar>(S_out.template cast<double>().mean());
	const auto inside_count = (S_out.array() < Scalar(0)).count();
	std::cout << "SDF stats — N=" << N
		<< "  min=" << s_min
		<< "  max=" << s_max
		<< "  mean≈" << s_mean
		<< "  inside=" << inside_count << "\n";

	return true;
}

int main(int argc, char** argv) {
	Eigen::Matrix<float, -1, 3> P;
	Eigen::VectorXf S;
	RandomSDFInBBox("C:\\Users\\mwang\\Downloads\\j-50-j50-sac\\source\\Shenyang J-50 SM_export\\Shenyang J-50 SM_export.obj", 1'000'000, P, S);
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
