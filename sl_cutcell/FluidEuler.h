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

	void buildTypesAndAMGCoeffs(HADeviceGrid<Tile>& grid, const T current_time) {
		//Info("building types and AMG coeffs at time {}", current_time);

		//prepare the Poisson system along with cell types
		if (mMeshSDFAccel != nullptr) {
			auto params = mParams;
			auto xform = params.meshToWorldTransform(current_time);

			CalculateSDFOnNodes(grid, BufChnls::sdf, *mMeshSDFAccel, LEAF | GHOST, xform);


			//set wall types
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

		FillChannelsInGridWithValue(grid, std::numeric_limits<T>::quiet_NaN(), LEAF | NONLEAF | GHOST, {});

		buildTypesAndAMGCoeffs(grid, 0.);




		//the velocity here is the composed velocity, which is weighted fluid + solid
		//clear velocity variables to 0
		//initial velocity: 0
		FillChannelsInGridWithValue(grid, 0.0, LEAF | NONLEAF | GHOST, { BufChnls::u, BufChnls::u + 1, BufChnls::u + 2 });
		//add solid velocity to velocity variables
		//Info("metadata current time: {} dt: {} fps {}", metadata.current_time, metadata.dt, metadata.fps);
		addSolidVelocityWithFractionsToFaces(grid, 0.0, 1e-3);//t=0, dt=1e-3
		{
			//set initial velocity
			auto params = mParams;
			grid.launchVoxelFuncOnAllTiles(
				[params] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
				params.addInitialVelocityToFaceCenter(acc, info, l_ijk);
			}, LEAF
			);
		}
		project(grid);

		CalculateVelocityAndVorticityMagnitudeOnLeafCellCenters(grid, mParams.mFineLevel, mParams.mCoarseLevel, BufChnls::u, BufChnls::u_cell, BufChnls::vor);



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
				std::vector<std::pair<int, std::string>>{ {BufChnls::u, "velocity"} },
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

	void adaptAndAdvect(DriverMetaData& metadata, std::vector<std::shared_ptr<HADeviceGrid<Tile>>> grid_ptrs) {
		const double dt = metadata.dt;
		const double current_time = metadata.current_time;


		CPUTimer timer; timer.start();

		//1. advect particles with last_grid
		//2. refine grid with particles
		//3. calculate type, advect dye_density and velocity on grid

		//012: temporary node velocity
		//3: temporary node dye density

		//shared by two grids
		//int u_channel = AdvChnls::u;//6

		
		//saved intermediate velocities
		int n = grid_ptrs.size() - 1;
		//we only need to prepare the last grid at this time
		auto& last_grid = *grid_ptrs[n - 1];
		CheckCudaError("prepare last grid");


		AdvectMarkerParticlesRK4ForwardAndMarkInvalid(
			last_grid, mParams.mFineLevel, mParams.mCoarseLevel,
			BufChnls::u, dt, current_time - mParams.mParticleLife,
			marker_particles_d
		);
		EraseInvalidParticles(marker_particles_d);
		cudaDeviceSynchronize(); particle_advection_time = timer.stop("Advect particles"); timer.start();
		CheckCudaError("adv particle");

		auto h_acc = last_grid.deviceAccessor();
		auto new_particles_d = SampleMarkerParticlesOutsideMeshBand(*mMeshSDFAccel, mParams.meshToWorldTransform(current_time), h_acc.voxelSize(mParams.mFineLevel), mParams.mRelativeSampleBandwidth, mParams.mSampleNumPerTile, current_time, mRamdonGenerator);
		marker_particles_d.insert(marker_particles_d.end(), new_particles_d.begin(), new_particles_d.end());
		cudaDeviceSynchronize(); reseeding_time = timer.stop("reseeding and remove particles in solid"); timer.start();
		Info("total {:.5f}M particles, time step counter {}", marker_particles_d.size() / (1024 * 1024 + 0.f), time_step_counter);
		CheckCudaError("reseeding particles");



		auto& grid = *grid_ptrs[n];
		RefineWithMarkerParticles(grid, marker_particles_d, mParams.mCoarseLevel, mParams.mFineLevel, BufChnls::counter, false);
		CoarsenWithMarkerParticles(grid, marker_particles_d, mParams.mCoarseLevel, mParams.mFineLevel, BufChnls::counter, false);
		cudaDeviceSynchronize(); adaptive_time = timer.stop("adapt with particles"); timer.start();
		CheckCudaError("adapt with particles");
		
		buildTypesAndAMGCoeffs(grid, current_time);

		Info("time step counter: {}", time_step_counter);
		auto nfm_query_grid_ptr = grid_ptrs[n - 1 - (time_step_counter % mParams.mFlowMapStride)];

		//prepare pointers for previous grids
		thrust::host_vector<HATileAccessor<Tile>> accs_h;
		for (int i = 0; i < n; i++) accs_h.push_back(grid_ptrs[i]->deviceAccessor());
		thrust::device_vector<HATileAccessor<Tile>> accs_d = accs_h;
		auto accs_d_ptr = thrust::raw_pointer_cast(accs_d.data());
		thrust::device_vector<double> time_steps_d = time_steps;
		auto time_steps_d_ptr = thrust::raw_pointer_cast(time_steps_d.data());
		CheckCudaError("prepare pointers");
		Info("prepare pointers");

		FillChannelsInGridWithValue(grid, 0., LEAF | NONLEAF | GHOST, { BufChnls::u, BufChnls::u + 1, BufChnls::u + 2 });

		//add solid velocity to velocity field
		addSolidVelocityWithFractionsToFaces(grid, current_time, dt);

		//add advected NFM velocity to velocity field
		{
			auto last_acc = last_grid.deviceAccessor();
			auto nfm_query_acc = nfm_query_grid_ptr->deviceAccessor();
			auto params = mParams;
			
			int fine_level = mParams.mFineLevel;
			int coarse_level = mParams.mCoarseLevel;

			int back_traced_steps = time_step_counter % mParams.mFlowMapStride;
			int nfm_start_idx = n - back_traced_steps - 1;
			Info("nfm start idx: {}, back traced steps: {}, accs_d size {}", nfm_start_idx, back_traced_steps, accs_d.size());
			
			grid.launchVoxelFuncOnAllTiles(
				[=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
				auto& tile = info.tile();


				{
					//grid velocity advection
					for (int axis : {0, 1, 2}) {
						{
							Vec pos = acc.faceCenter(axis, info, l_ijk);
							Vec back_pos = SemiLagrangianBackwardPosition(acc, fine_level, coarse_level, pos, dt, BufChnls::u + axis);
							Vec back_vel;
							KernelIntpVelocityMAC2(acc, fine_level, coarse_level, back_pos, BufChnls::u, back_vel);
							auto fluid_ratio = -tile(ProjChnls::c0 + axis, l_ijk) / acc.voxelSize(info);
							tile(BufChnls::u + axis, l_ijk) += fluid_ratio * back_vel[axis];



							////Vec psi = acc.faceCenter(axis, info, l_ijk);
							//Vec psi = NFMErodedAdvectionPoint(axis, acc, info, l_ijk);
							//Eigen::Matrix3<T> matT;


							//NFMBackMarchPsiAndT(accs_d_ptr, fine_level, coarse_level, time_steps_d_ptr, BufChnls::u, nfm_start_idx, n, psi, matT);


							//Vec m0; Eigen::Matrix3<T> _T;
							//KernelIntpVelocityAndJacobianMAC2(acc, fine_level, coarse_level, psi, BufChnls::u, m0, _T);


							//{
							//	auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
							//	CUDA_ASSERT(isfinite(m0[0]), "level %d global %d %d %d axis %d m00 value %f", info.mLevel, g_ijk[0], g_ijk[1], g_ijk[2], axis, m0[0]);
							//	CUDA_ASSERT(isfinite(m0[1]), "level %d global %d %d %d axis %d m01 value %f", info.mLevel, g_ijk[0], g_ijk[1], g_ijk[2], axis, m0[1]);
							//	CUDA_ASSERT(isfinite(m0[2]), "level %d global %d %d %d axis %d m02 value %f", info.mLevel, g_ijk[0], g_ijk[1], g_ijk[2], axis, m0[2]);

							//	//ASSERT 3*3 values in matT are finite
							//	for(int i=0; i<3; i++) {
							//		for(int j=0; j<3; j++) {
							//			CUDA_ASSERT(isfinite(matT(i,j)), "level %d global %d %d %d axis %d matT value %f at %d %d", info.mLevel, g_ijk[0], g_ijk[1], g_ijk[2], axis, matT(i,j), i, j);
							//		}
							//	}
							//}

							//Vec m1 = MatrixTimesVec(matT.transpose(), m0);


							//auto fluid_ratio = -tile(ProjChnls::c0 + axis, l_ijk) / acc.voxelSize(info);
							//tile(BufChnls::u + axis, l_ijk) += fluid_ratio * m1[axis];


							//{
							//	auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
							//	CUDA_ASSERT(isfinite(m1[axis]), "level %d global %d %d %d axis %d m1 value %f", info.mLevel, g_ijk[0], g_ijk[1], g_ijk[2], axis, m1[axis]);
							//}
						}
					}
				}
			}, LEAF, 4
			);

			CheckCudaError("launch nfm");

		}

		cudaDeviceSynchronize(); nfm_advection_time = timer.stop("NFM advection"); timer.start();
		CheckCudaError("nfm advection");
	}

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





	virtual void Advance(DriverMetaData& metadata) {

		CPUTimer timer;
		timer.start();

		//Info("before advance"); PrintMemoryInfo();

		if (grid_ptrs.size() < mParams.mFlowMapStride + 1) {
			//create a new grid
			auto nxt_ptr = grid_ptrs.back()->deepCopy();
			grid_ptrs.push_back(nxt_ptr);
			time_steps.push_back(metadata.dt);
		}
		else {
			auto nxt_ptr = grid_ptrs[0];
			grid_ptrs.erase(grid_ptrs.begin());
			grid_ptrs.push_back(nxt_ptr);
			time_steps.erase(time_steps.begin());
			time_steps.push_back(metadata.dt);
		}

		auto& grid = *grid_ptrs.back();
		auto& last_grid = *grid_ptrs[grid_ptrs.size() - 2];
		double dt = metadata.dt;

		fmt::print("\n");
		Pass("Advance frame {} current time {} dt {} step counter {}", metadata.current_frame, metadata.current_time, dt, time_step_counter);
		ASSERT(dt > 0, "dt should be positive");


		adaptAndAdvect(metadata, grid_ptrs);


		applyExternalForce(grid, dt);



		if (time_step_counter == 65) {
			amg_solver_open_visualization = 1;
		}
		else {
			amg_solver_open_visualization = 0;
		}


		//projection
		project(grid);





		CalculateVelocityAndVorticityMagnitudeOnLeafCellCenters(grid, mParams.mFineLevel, mParams.mCoarseLevel, BufChnls::u, BufChnls::u_cell, BufChnls::vor);

		//{
		//	//show velocity on polyscope before proj
		//	polyscope::init();
		//	polyscope::removeAllStructures();
		//	auto holder = grid.getHostTileHolderForLeafs();
		//	IOFunc::AddPoissonGridCellCentersToPolyscopePointCloud(holder, { { -1,"type" }, { BufChnls::vor, "vorticity" }, {ProjChnls::x, "pressure"}, { ProjChnls::b, "divergence" } }, { {BufChnls::u, "velocity"} });
		//	//IOFunc::AddLeveledPoissonGridCellCentersToPolyscopePointCloud(holder, { { -1,"type" }, { BufChnls::vor, "vorticity" } }, { { BufChnls::u, "velocity" } });
		//	polyscope::show();
		//}


		CheckCudaError("Advance");
		time_step_counter++;


		PrintMemoryInfo();

		cudaDeviceSynchronize(); advance_time = timer.stop();
	}

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