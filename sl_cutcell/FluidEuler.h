//////////////////////////////////////////////////////////////////////////
// Fluid Euler
// Copyright (c) (2022-), Bo Zhu, Mengdi Wang
// This file is part of MESO, whose distribution is governed by the LICENSE file.
//////////////////////////////////////////////////////////////////////////
#pragma once

#include "Simulator.h"
#include "FluidParams.h"
//#include "Random.h"
#include "GPUTimer.h"
#include "AMGSolver.h"

#include "PoissonIOFunc.h"
#include "FMParticles.h"
#include "MarkerParticles.h"
#include "PoissonGrid.h"


#include <polyscope/polyscope.h>
#include <polyscope/point_cloud.h>

#include <sys/types.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/sysinfo.h>
#endif

#define CUDA_CHECK(call) do {                                     \
    cudaError_t err__ = (call);                                   \
    if (err__ != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                 \
                __FILE__, __LINE__, cudaGetErrorString(err__));   \
        abort();                                                  \
    }                                                            \
} while (0)

#define CUDA_KERNEL_CHECK() do {                                  \
    CUDA_CHECK(cudaGetLastError());                               \
    CUDA_CHECK(cudaDeviceSynchronize());                          \
} while (0)


extern std::atomic<int> amg_solver_open_visualization; // DEBUG

void SanityCheckCoeffs(HADeviceGrid<Tile>&grid, uint8_t launch_types);

void SanityCheckTiles(HADeviceGrid<Tile>&grid);


double CellPointRMSNormOnHostTiles(
	const std::shared_ptr<HAHostTileHolder<Tile>>&holder_ptr,
	int channel,                 // scalar channel index
	int level,                   // -1 for all levels, otherwise specific level
	uint8_t tile_types,          // e.g., LEAF | GHOST | NONLEAF
	int norm_type                // 1 (L1 mean), 2 (RMS), -1 (Linf)
);

double NodePointRMSNormOnHostTiles(
	const std::shared_ptr<HAHostTileHolder<Tile>>&holder_ptr,
	int channel,                 // scalar node channel
	int level,                   // -1 for all levels
	uint8_t tile_types,          // LEAF | GHOST | NONLEAF
	int norm_type                // 1 (L1 mean), 2 (RMS), -1 (Linf)
);

__device__ Vec NFMErodedAdvectionPoint(const int axis, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const Coord& l_ijk);


void MarkOldParticlesAsInvalid(thrust::device_vector<Particle>& particles, const T current_time, const T particle_life);


__device__ Vec SemiLagrangianBackwardPosition(const HATileAccessor<Tile>&acc, const int fine_level, const int coarse_level, const Vec & pos, const T dt, const int u_channel);


thrust::device_vector<MarkerParticle> SampleMarkerParticlesOutsideMeshBand(const MeshSDFAccel & mesh_sdf, const Eigen::Transform<T, 3, Eigen::Affine>&mesh_to_world, T dx, T relative_bandwidth, T samples_per_tile, T birth_time, RandomGenerator & rng);

void ClearAllNeumannNeighborFaces(HADeviceGrid<Tile>& grid, const int u_channel);



class FluidEuler : public Simulator {
public:
	//using Tile = PoissonTile<T>;
	using Coord = nanovdb::Coord;

	int time_step_counter = 0;

	std::vector<std::shared_ptr<HADeviceGrid<Tile>>> grid_ptrs;
	std::vector<double> time_steps;

	thrust::device_vector<MarkerParticle> marker_particles_d;

	FluidParams mParams;
	std::shared_ptr<MeshSDFAccel> mMeshSDFAccel = nullptr;


	RandomGenerator mRamdonGenerator;

	double advance_time = 0;
	double particle_advection_time = 0;
	double reseeding_time = 0;
	double adaptive_time = 0;
	double nfm_advection_time = 0;
	double projection_time = 0;


	//void addSolidVelocityWithFractionsToFaces(HADeviceGrid<Tile>& grid, const double current_time, const double dt) {
	//	auto params = mParams;
	//	grid.launchVoxelFuncOnAllTiles(
	//		[params, current_time, dt] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
	//		for (int axis : {0, 1, 2}) {
	//			params.addSolidVelocityToFaceCenter(current_time, dt, acc, info, l_ijk, axis);
	//		}
	//	}, LEAF
	//	);
	//}

	void mixFluidAndSolidVelocityOnFaces(HADeviceGrid<Tile>& grid, const double current_time, const double dt, const int coeff_channel, const int u_fluid_channel, const int u_mix_channel);

	void extrapolateFluidVelocityForAdvection(HADeviceGrid<Tile>& grid, const int iteration_times, const int u_fluid_channel, const int coeff_channel);

	void iterativeNodeSDFAndRefineNarrowBand(HADeviceGrid<Tile>& grid, const T current_time, const T solid_relative_bandwidth, const T fluid_relative_bandwidth);

	void buildTypesAndAMGCoeffsFromNodeSDFs(HADeviceGrid<Tile>& grid, const T current_time) {
		//Info("building types and AMG coeffs at time {}", current_time);

		//prepare the Poisson system along with cell types
		if (mMeshSDFAccel != nullptr) {
			
			//auto xform = params.meshToWorldTransform(current_time);

			//CalculateSDFOnNodes(grid, BufChnls::sdf, *mMeshSDFAccel, LEAF | GHOST, xform);


			//set wall types
			auto params = mParams;
			grid.launchVoxelFuncOnAllTiles(
				[=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
				params.setWallCellType(current_time, acc, info, l_ijk);
			}, LEAF
			);
			//set other solid cells and build the whole system
			CreateAMGLaplacianSystemWithSolidCutOnNodeSDF(grid, BufChnls::sdf, ProjChnls::c0, 0.5);


		}
		else {
			ASSERT(false, "need a mesh");
		}
	}

	void init(json &j, DriverMetaData &metadata) {
		//fmt::print("current path: {}\n", fs::current_path().string());

		std::string mesh_file = Json::Value<std::string>(j, "mesh_file", "mesh.obj");
		if (mesh_file != "") {
			mMeshSDFAccel = std::make_shared<MeshSDFAccel>(mesh_file);
		}
		else {
			mMeshSDFAccel = nullptr;
		}

		mParams = FluidParams(j);
		ASSERT(mParams.mFlowMapStride == 1, "SL only supports 1-step advection");

		if (metadata.first_frame != 0) return;

		//level-resolution:
		//0:8, 1:16, 2:32, 3:64, 4:128, 5:256, 6:512, 7:1024
		double h = 1.0 / 8;
		auto grid_ptr = std::make_shared<HADeviceGrid<Tile> >(h, std::initializer_list<uint32_t>({ 16, 16, 16, 16, 16, 16, 20, 16, 16, 16 }));
		grid_ptrs.clear();
		grid_ptrs.push_back(grid_ptr);
		auto& grid = *grid_ptr;

		{
			//1*1*1
			grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
			grid.compressHost();
			grid.syncHostAndDevice();
			grid.spawnGhostTiles();
		}

		{
			//refine to initial level target defined by params
			auto params = mParams;
			grid.iterativeRefine([=]__device__(const HATileAccessor<Tile> &acc, HATileInfo<Tile> &info) ->int {
				return params.initialLevelTarget(acc, info);
			});
		}

		if(mMeshSDFAccel != nullptr)
		{
			//refine using mesh vertices
			auto h_acc = grid.deviceAccessor();
			marker_particles_d = SampleMarkerParticlesOutsideMeshBand(*mMeshSDFAccel, mParams.meshToWorldTransform(0.), h_acc.voxelSize(mParams.mFineLevel), mParams.mRelativeSampleBandwidth, mParams.mSampleNumPerTile, 0., mRamdonGenerator);
			RefineWithMarkerParticles(grid, marker_particles_d, mParams.mCoarseLevel, mParams.mFineLevel, BufChnls::counter, false);

		}

		T current_time = 0.0, dt = 1e-5;
		FillChannelsInGridWithValue(grid, std::numeric_limits<T>::quiet_NaN(), LEAF | NONLEAF | GHOST, {});
		iterativeNodeSDFAndRefineNarrowBand(grid, current_time, mParams.mRelativeSampleBandwidth, mParams.mRelativeSampleBandwidth);
		buildTypesAndAMGCoeffsFromNodeSDFs(grid, current_time);




		//the velocity here is the composed velocity, which is weighted fluid + solid
		//clear velocity variables to 0
		//initial velocity: 0
		FillChannelsInGridWithValue(grid, 0.0, LEAF | NONLEAF | GHOST, { BufChnls::u, BufChnls::u + 1, BufChnls::u + 2 });
		//add solid velocity to velocity variables
		//Info("metadata current time: {} dt: {} fps {}", metadata.current_time, metadata.dt, metadata.fps);
		// addSolidVelocityWithFractionsToFaces(grid, 0.0, 1e-3);//t=0, dt=1e-3
		//{
		//	//set initial velocity
		//	auto params = mParams;
		//	grid.launchVoxelFuncOnAllTiles(
		//		[params] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
		//		params.addInitialVelocityToFaceCenter(acc, info, l_ijk);
		//	}, LEAF
		//	);
		//}
		project(grid, current_time, dt);
		CalculateVelocityAndVorticityMagnitudeOnLeafCellCenters(grid, mParams.mFineLevel, mParams.mCoarseLevel, BufChnls::u, BufChnls::u_cell, BufChnls::vor);
		extrapolateFluidVelocityForAdvection(grid, mParams.mExtrapolationIters, BufChnls::u, ProjChnls::c0);

	}

	virtual double CFL_Time(const double cfl) {
		//Warn("entering CFL calc");

		auto& grid = *grid_ptrs.back();
		//return FLT_MAX;
		HATileAccessor<Tile> acc = grid.deviceAccessor();
		double dx = acc.voxelSize(acc.mMaxLevel);
		
		double umax = NormSync(grid, -1, BufChnls::u, false);
		double vmax = NormSync(grid, -1, BufChnls::u + 1, false);
		double wmax = NormSync(grid, -1, BufChnls::u + 2, false);
		double max_vel = std::max(umax, std::max(vmax, wmax));

		Info("Calc CFL umax = {}, vmax = {}, wmax = {}, max_vel = {} calc dx {} cfl {} max_vel {} dt {}", umax, vmax, wmax, max_vel, dx, cfl, max_vel, dx * cfl / max_vel);

		//Warn("calculated math dt {}", dx * cfl / max_vel);

		return dx * cfl / max_vel;
	}
	virtual void Output(DriverMetaData& metadata) {
		{
			//snapshot  thing
			if (metadata.Should_Snapshot()) {
				Save_Frame(metadata);
			}
		}

		//return;
		auto& grid = *grid_ptrs.back();

		{

			auto holder = grid.getHostTileHolderForLeafs();

			//metadata.Append_Output_Thread(std::make_shared<std::thread>(IOFunc::OutputTilesAsVTU, holder, metadata.base_path / fmt::format("tiles{:04d}.vtu", metadata.current_frame)));
			//IOFunc::OutputTilesAsVTU(holder, metadata.base_path / fmt::format("tiles{:04d}.vtu", metadata.current_frame));

			metadata.Append_Output_Thread(std::make_shared<std::thread>(IOFunc::OutputPoissonGridAsStructuredVTI, holder,
				std::vector<std::pair<int, std::string>>{ {-1, "type"}, { -2, "level" }, { BufChnls::vor, "vorticity" }},
				//std::vector<std::pair<int, std::string>>{ },
				std::vector<std::pair<int, std::string>>{ {BufChnls::u_cell, "cell_velocity"} },
				//std::vector<std::pair<int, std::string>>{ { -1, "type" }, { Tile::vor_channel, "vorticity" }, { Tile::dye_channel, "dye_density" } },
				//std::vector<std::pair<int, std::string>>{ {Tile::u_channel, "velocity"} },
				metadata.base_path / fmt::format("fluid{:04d}.vti", metadata.current_frame)));

		}

		{
			WriteStatToFile(metadata);
		}

			auto particles_h_ptr = std::make_shared<thrust::host_vector<MarkerParticle>>(marker_particles_d);
			metadata.Append_Output_Thread(std::make_shared<std::thread>(IOFunc::OutputMarkerParticleSystemAsVTU,
				particles_h_ptr, metadata.base_path / fmt::format("particles{:04d}.vtu", metadata.current_frame)
			));
		//}
	}


	//current_time and dt for calculating solid velocity and doing time interpolation
	void project(HADeviceGrid<Tile>& grid, const T current_time, const T dt);

	void adaptAndAdvect(DriverMetaData& metadata, std::vector<std::shared_ptr<HADeviceGrid<Tile>>> grid_ptrs);

	//(I - nu * dt * lap)u^{n+1} = u^* for implicit viscosity
	void applyViscosity(HADeviceGrid<Tile>& grid, const T dt, const T nu);

	void applyExternalForce(HADeviceGrid<Tile>& grid, const double dt) {
		const nanovdb::Vec3R gravity = mParams.mGravity;
		auto params = mParams;
		grid.launchVoxelFunc(
			[dt, gravity, params] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
			auto& tile = info.tile();
			//if (!tile.isInterior(l_ijk)) return;
			for (int axis : {0, 1, 2}) {
				auto pos = acc.faceCenter(axis, info, l_ijk);

				//auto Ta = InterpolateCellValue(acc, pos, Tile::Ta_channel, tmp_ta_node);
				//auto buoyancy = params.mBuoyancy * (Ta - params.T_ambient);

				nanovdb::Vec3R a = gravity;
				//a[1] += buoyancy;
				//tile(BufChnls::u + axis, l_ijk) += a[axis] * dt;

				auto fluid_ratio = -tile(ProjChnls::c0 + axis, l_ijk) / acc.voxelSize(info);
				tile(BufChnls::u + axis, l_ijk) += fluid_ratio * a[axis] * dt;
			}
		}, -1, LEAF, LAUNCH_SUBTREE
		);
	}





	virtual void Advance(DriverMetaData& metadata);

	void WriteStatToFile(DriverMetaData& metadata) {
		fs::path output_path = metadata.base_path / "logs";
		fs::create_directories(output_path);
		fs::path output_file = output_path / fmt::format("simulator_stat_{:04d}.txt", metadata.current_frame);
		std::ofstream out(output_file);
		fmt::print(out,
			"frame {}\n"
			"total particles {}\n"
			"total leaf cells {}\n"
			"reseeding time {} ms\n"
			"particle_advection_time {} ms\n"
			"adaptive_time {} ms\n"
			"projection_time {} ms\n"
			"nfm_advection_time {} ms\n"
			"advance_time {} ms\n",
			metadata.current_frame,
			marker_particles_d.size(),
			grid_ptrs.back()->numTotalLeafTiles() * Tile::SIZE,
			reseeding_time,
			particle_advection_time,
			adaptive_time,
			projection_time,
			nfm_advection_time,
			advance_time);
		out.close();
	}

	void PrintMemoryInfo(void) {
		double M = 1024 * 1024, G = 1024 * 1024 * 1024;
		double particle_num = marker_particles_d.size();
		double particle_capacity = marker_particles_d.capacity();
		double total_tile_num = 0;
		for (auto grid_ptr : grid_ptrs) {
			auto& grid = *grid_ptr;
			total_tile_num += grid.numTotalTiles();
		}
		double total_num_voxels = total_tile_num * Tile::SIZE;
		double memory_size_gb = (particle_num * sizeof(Particle) + total_tile_num * sizeof(Tile)) / G;
		double capacity_size_gb = (particle_capacity * sizeof(Particle) + total_tile_num * sizeof(Tile)) / G;
		
		Info("total {:.3f}({:.3f})M particles, {:.3f}M tiles and {:.3f}M voxels, memory size {:.3f}({:.3f})G", particle_num / M, particle_capacity / M, total_tile_num / M, total_num_voxels / M, memory_size_gb, capacity_size_gb);

		size_t free_mem, total_mem;

		// 查询设备的内存信息
		cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
		if (err != cudaSuccess) {
			std::cerr << "Failed to get CUDA memory info: " << cudaGetErrorString(err) << std::endl;
			return;
		}

		// 转换为GB
		double free_mem_gb = static_cast<double>(free_mem) / (1024 * 1024 * 1024);
		double total_mem_gb = static_cast<double>(total_mem) / (1024 * 1024 * 1024);
		double used_mem_gb = total_mem_gb - free_mem_gb;
		Info("Using CUDA Memory: total {:.3f}G, free {:.3f}G, used {:.3f}G, background {:.3f}({:.3f})G", total_mem_gb, free_mem_gb, used_mem_gb, used_mem_gb - memory_size_gb, used_mem_gb - capacity_size_gb);

		// 查询 CPU 内存信息
#ifdef _WIN32
		MEMORYSTATUSEX memInfo;
		memInfo.dwLength = sizeof(MEMORYSTATUSEX);
		if (GlobalMemoryStatusEx(&memInfo)) {
			double total_physical_gb = static_cast<double>(memInfo.ullTotalPhys) / G;
			double free_physical_gb = static_cast<double>(memInfo.ullAvailPhys) / G;
			double used_physical_gb = total_physical_gb - free_physical_gb;
			Info("Using CPU Memory: total {:.3f}G, free {:.3f}G, used {:.3f}G",
				total_physical_gb, free_physical_gb, used_physical_gb);
		}
		else {
			std::cerr << "Failed to get CPU memory info on Windows." << std::endl;
		}
#else
		struct sysinfo memInfo;
		if (sysinfo(&memInfo) == 0) {
			double total_physical_gb = static_cast<double>(memInfo.totalram) * memInfo.mem_unit / G;
			double free_physical_gb = static_cast<double>(memInfo.freeram) * memInfo.mem_unit / G;
			double used_physical_gb = total_physical_gb - free_physical_gb;
			Info("Using CPU Memory: total {:.3f}G, free {:.3f}G, used {:.3f}G",
				total_physical_gb, free_physical_gb, used_physical_gb);
		}
		else {
			std::cerr << "Failed to get CPU memory info on Linux." << std::endl;
		}
#endif
	}

	void Save_Frame(DriverMetaData& metadata) {
		namespace fs = std::filesystem;

		fs::path folder = metadata.Snapshot_Base_Path();
		fs::create_directories(folder);

		fs::path file = folder / fmt::format("{:04d}.bin", metadata.current_frame);

		std::ofstream os(file, std::ios::binary);
		ASSERT(os.good(), "Save_Frame: failed to open {}", file.string());

		// Header
		uint32_t magic = 0x31464546u; // 'FEF1' (any magic you like)
		uint32_t version = 1;
		IOFunc::WritePod(os, magic);
		IOFunc::WritePod(os, version);

		// 1) time_step_counter
		IOFunc::WritePod(os, time_step_counter);

		// 2) time_steps
		IOFunc::WriteVector(os, time_steps);

		// 3) grids
		uint64_t num_grids = (uint64_t)grid_ptrs.size();
		IOFunc::WritePod(os, num_grids);

		for (uint64_t i = 0; i < num_grids; ++i) {
			auto& g = *grid_ptrs[i];

			// Ensure compressed before dumping (your dumpBinaryBlob requires this)
			g.compressHost(false);

			// Dump to blob
			std::vector<uint8_t> blob = g.dumpBinaryBlob(LEAF | GHOST | NONLEAF);

			uint64_t blob_size = (uint64_t)blob.size();
			IOFunc::WritePod(os, blob_size);
			if (blob_size) os.write(reinterpret_cast<const char*>(blob.data()), (std::streamsize)blob_size);
		}

		// 4) marker particles
		static_assert(std::is_trivially_copyable_v<MarkerParticle>,
			"MarkerParticle must be trivially copyable for raw checkpoint");

		uint64_t n_particles = (uint64_t)marker_particles_d.size();
		IOFunc::WritePod(os, n_particles);

		if (n_particles) {
			thrust::host_vector<MarkerParticle> h(marker_particles_d);
			os.write(reinterpret_cast<const char*>(h.data()),
				(std::streamsize)(sizeof(MarkerParticle) * (size_t)n_particles));
		}

		ASSERT(os.good(), "Save_Frame: write failed {}", file.string());
		Info("Saved snapshot: {}", file.string());
	}

	void Load_Frame(DriverMetaData& metadata) {
		namespace fs = std::filesystem;

		fs::path folder = metadata.Snapshot_Base_Path();
		fs::path file = folder / fmt::format("{:04d}.bin", metadata.current_frame);

		std::ifstream is(file, std::ios::binary);
		ASSERT(is.good(), "Load_Frame: failed to open {}", file.string());

		uint32_t magic = 0, version = 0;
		IOFunc::ReadPod(is, magic);
		IOFunc::ReadPod(is, version);

		ASSERT(magic == 0x31464546u, "Load_Frame: bad magic in {}", file.string());
		ASSERT(version == 1, "Load_Frame: unsupported version {} in {}", version, file.string());

		// 1) time_step_counter
		IOFunc::ReadPod(is, time_step_counter);

		// 2) time_steps
		IOFunc::ReadVector(is, time_steps);

		// 3) grids
		uint64_t num_grids = 0;
		IOFunc::ReadPod(is, num_grids);

		grid_ptrs.clear();
		grid_ptrs.reserve((size_t)num_grids);

		for (uint64_t i = 0; i < num_grids; ++i) {
			uint64_t blob_size = 0;
			IOFunc::ReadPod(is, blob_size);
			ASSERT(blob_size > 0, "Load_Frame: grid blob size is 0 (i={})", (size_t)i);

			std::vector<uint8_t> blob((size_t)blob_size);
			is.read(reinterpret_cast<char*>(blob.data()), (std::streamsize)blob_size);
			ASSERT(is.good(), "Load_Frame: truncated blob (i={})", (size_t)i);

			auto grid_ptr = HADeviceGrid<Tile>::loadBinaryBlob(blob);
			grid_ptrs.push_back(std::move(grid_ptr));
		}

		// 4) marker particles
		static_assert(std::is_trivially_copyable_v<MarkerParticle>,
			"MarkerParticle must be trivially copyable for raw checkpoint");

		uint64_t n_particles = 0;
		IOFunc::ReadPod(is, n_particles);

		thrust::host_vector<MarkerParticle> h_particles;
		h_particles.resize((size_t)n_particles);

		if (n_particles) {
			is.read(reinterpret_cast<char*>(h_particles.data()),
				(std::streamsize)(sizeof(MarkerParticle) * (size_t)n_particles));
			ASSERT(is.good(), "Load_Frame: truncated particle data");
		}

		marker_particles_d = thrust::device_vector<MarkerParticle>(h_particles.begin(), h_particles.end());

		ASSERT(is.good(), "Load_Frame: read failed {}", file.string());
		Info("Loaded snapshot: {}", file.string());
	}
};