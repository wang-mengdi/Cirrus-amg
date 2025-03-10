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

//#include "AMGSolver.h"
#include "PoissonIOFunc.h"
#include "FMParticles.h"


#include <polyscope/polyscope.h>
#include <polyscope/point_cloud.h>

#include <sys/types.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/sysinfo.h>
#endif

__device__ Vec NFMErodedAdvectionPoint(const int axis, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const Coord& l_ijk);

//set all face velocities that has a neumann neighbor to 0
void ClearAllNeumannNeighborFaces(HADeviceGrid<Tile>& grid, const int u_channel);

void MarkOldParticlesAsInvalid(thrust::device_vector<Particle>& particles, const T current_time, const T particle_life);


__device__ Vec SemiLagrangianBackwardPosition(const HATileAccessor<Tile>& acc, const Vec& pos, const T dt, const int u_channel, const int node_u_channel);

int LockedRefineWithNonBoundaryNeumannCellsOneStep(const T current_time, HADeviceGrid<Tile>& grid, const FluidParams params, const int tmp_channel, bool verbose);
void ReseedParticles(HADeviceGrid<Tile>& grid, const FluidParams& params, const int tmp_channel, const double current_time, const int num_particles_per_cell, thrust::device_vector<Particle>& particles);


class FluidEuler : public Simulator {
public:
	using Tile = PoissonTile<T>;
	using Coord = nanovdb::Coord;

	//projection:
	//0
	static constexpr int coeff_channel = 11;

	int mNumParticlesPerCell = 8;
	int time_step_counter = 0;

	std::shared_ptr<HADeviceGrid<Tile>> nfm_query_grid_ptr;
	std::vector<std::shared_ptr<HADeviceGrid<Tile>>> grid_ptrs;
	std::vector<double> time_steps;

	thrust::device_vector<Particle> particles;
	thrust::device_vector<ParticleRecord> records_d;
	thrust::device_vector<int> tile_prefix_sum_d;

	FluidParams mParams;
	std::shared_ptr<MaskGrid> mMaskGrid = nullptr;
	std::shared_ptr<SDFGrid> mSDFGrid = nullptr;

	std::shared_ptr<SDFGrid> mAnimationSDFGrid0 = nullptr;
	std::shared_ptr<SDFGrid> mAnimationSDFGrid1 = nullptr;
	std::shared_ptr<SDFGrid> mAnimationVelocityGrids[3];

	std::string animation_sdf_path = "";
	fs::path flamingo_data_file = fs::path("data") / "flamingo-sdf.bin";
	int loaded_animated_frame = -1;

	//int cell_center_vel_channel = 3;

	double advance_time = 0;
	double projection_time = 0;
	double reseeding_time = 0;
	double particle_advection_time = 0;
	double adaptive_time = 0;
	double p2g_time = 0;
	double nfm_advection_time = 0;

	void applyVelocityBC(HADeviceGrid<Tile>& grid, const double time) {
		//if (mParams.mTestCase == TVORTEX || mParams.mTestCase==FORCE_) 
		{
			ClearAllNeumannNeighborFaces(grid, AdvChnls::u);
			auto params = mParams;
			grid.launchVoxelFuncOnAllTiles(
				[params, time] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
				auto& tile = info.tile();
				//if (!tile.isInterior(l_ijk)) {
					params.setBoundaryCondition(acc, info, l_ijk, time);
				//}
			}, LEAF
			);
		}
	}

	std::tuple<int, T> getBatAnimationFrameAndFrace(const T time) {
		int bat_fps = 200;
		T bat_f = time * bat_fps;
		int bat_frame = (int)bat_f;
		T frac = bat_f - bat_frame;
		return std::make_tuple(bat_frame, frac);
	}
	fs::path getBatCyclicAnimationFile(const fs::path& animation_sdf_path, int bat_frame) {
		//int cyc_idx = bat_frame % 205 + 1;
		int cyc_idx = bat_frame;
		return animation_sdf_path / fmt::format("{:04d}.bin", cyc_idx);
	}
	fs::path getBatCyclicVelocityFile(const fs::path& animation_sdf_path, int bat_frame, int axis) {
		//int cyc_idx = bat_frame % 205 + 1;
		int cyc_idx = bat_frame;
		if (axis == 0) return animation_sdf_path / fmt::format("{:04d}_x.bin", cyc_idx);
		else if (axis == 1) return animation_sdf_path / fmt::format("{:04d}_y.bin", cyc_idx);
		else if (axis == 2) return animation_sdf_path / fmt::format("{:04d}_z.bin", cyc_idx);
		else return fs::path("");
	}

	fs::path getFlamingoAnimationFile(const fs::path& animation_sdf_path, int frame) {
		return animation_sdf_path / fmt::format("flamingo-flock{:04d}.bin", frame);
	}
	fs::path generateFlamingoFileWithPython(const fs::path& animation_sdf_path, const int frame, const T frac, const T isovalue) {
		// Conda environment name and Python script path
		std::string conda_env = "py311_env";
		std::string python_script = fmt::format(
			"C:\\Code\\HASimulator\\gen_single_flamingo.py  --mesh_path {} --frame {} --frac {} --scale 256 --isovalue {} --output_file {}",
			animation_sdf_path.string(),
			frame,
			frac,
			isovalue,
			flamingo_data_file.string()
		);
		
		// Command to activate conda environment and execute the Python script
		std::string command = fmt::format(
			"conda run -n py311_env python {}",
			python_script
		);


		Info("Executing Python script: {}", command);
		// Execute the command
		int ret_code = std::system(command.c_str());

		if (ret_code == 0) {
			fmt::print("Python script executed successfully.\n");
		}
		else {
			fmt::print("Failed to execute the Python script. Return code: {}\n", ret_code);
		}

		return flamingo_data_file;
	}

	void init(json &j) {
		//Info("before init"); PrintMemoryInfo();

		std::string sdf_grid_file = Json::Value<std::string>(j, "sdf_grid_file", "");
		if (sdf_grid_file != "") {
			float solid_isovalue = Json::Value<float>(j, "solid_isovalue", 0);
			float sdf_isovalue = Json::Value<float>(j, "sdf_isovalue", 0.025);
			mMaskGrid = std::make_shared<MaskGrid>(sdf_grid_file, solid_isovalue, sdf_isovalue);
		}
		else {
			mMaskGrid = nullptr;
		}
		std::string sdf_float_grid_file = Json::Value<std::string>(j, "sdf_float_grid_file", "");
		if (sdf_float_grid_file != "") {
			float solid_ext_isovalue = Json::Value<float>(j, "solid_ext_isovalue", 0);
			float sdf_ext_isovalue = Json::Value<float>(j, "sdf_ext_isovalue", 0.025);
			mSDFGrid = std::make_shared<SDFGrid>(sdf_float_grid_file, solid_ext_isovalue, sdf_ext_isovalue);
		}
		else {
			mSDFGrid = nullptr;
		}
		animation_sdf_path = Json::Value<std::string>(j, "animation_sdf_path", "");
		Info("animation_sdf_path {}", animation_sdf_path);

		
		
		float reserve_particles_m = Json::Value<float>(j, "reserve_particles_m", 1.0);
		particles.reserve((size_t)(reserve_particles_m * 1024 * 1024));
		mParams = FluidParams(j, mMaskGrid, mSDFGrid);

		//level-resolution:
		//0:8, 1:16, 2:32, 3:64, 4:128, 5:256, 6:512, 7:1024
		double h = 1.0 / 8;
		auto grid_ptr = std::make_shared<HADeviceGrid<Tile> >(h, std::initializer_list<uint32_t>({ 16, 16, 16, 16, 16, 16, 18, 16, 16, 16 }));
		grid_ptrs.clear();
		grid_ptrs.push_back(grid_ptr);
		auto& grid = *grid_ptr;

		{
			//1*1*1
			grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
		}
		grid.compressHost();
		grid.syncHostAndDevice();
		grid.spawnGhostTiles();

		{

			auto params = mParams;
			auto levelTarget = [=]__device__(const HATileAccessor<Tile> &acc, HATileInfo<Tile> &info) ->int {
				return params.initialLevelTarget(acc, info);
			};
			grid.iterativeRefine(levelTarget);

			while (true) {
				grid.launchVoxelFuncOnAllTiles(
					[params] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
					params.setInitialCondition(acc, info, l_ijk);
				}, -1, LEAF
				);
				int cnt1 = LockedRefineWithNonBoundaryNeumannCellsOneStep(0.0, grid, params, 0, false);
				if (cnt1 == 0) break;
			}

			applyVelocityBC(grid, 0.0);
			CalcCellTypesFromLeafs(grid);
		}
		CalculateVorticityMagnitudeOnLeafs(*grid_ptr, mParams.mFineLevel, mParams.mCoarseLevel, AdvChnls::u, OutputChnls::u_node, OutputChnls::vor);

	}

	virtual double CFL_Time(const double cfl) {
		auto& grid = *grid_ptrs.back();
		//return FLT_MAX;
		HATileAccessor<Tile> acc = grid.deviceAccessor();
		double dx = acc.voxelSize(acc.mMaxLevel);
		
		double umax = NormSync(grid, -1, AdvChnls::u, false);
		double vmax = NormSync(grid, -1, AdvChnls::u + 1, false);
		double wmax = NormSync(grid, -1, AdvChnls::u + 2, false);
		double max_vel = std::max(umax, std::max(vmax, wmax));
		return dx * cfl / max_vel;
	}
	virtual void Output(DriverMetaData& metadata) {
		//return;
		auto& grid = *grid_ptrs.back();
		//if (metadata.current_frame == 0) {
		//	//IOFunc::OutputCellAsPoints(grid, metadata.base_path / "boundary.vtu", true);
		//	//IOFunc::OutputCellAsPoints(grid, metadata.base_path / "grid.vtu", false);
		//	IOFunc::OutputTilesAsVTU(grid, metadata.base_path / "tiles.vtu");
		//}
		
		//{
		//	polyscope::init();
		//	auto holder = grid.getHostTileHolderForLeafs();
		//	IOFunc::AddParticleSystemToPolyscope(particles, "particles");
		//	//IOFunc::AddPoissonGridCellCentersToPolyscopePointCloud(holder, { { -1,"type" }, {Tile::dye_channel, "dye density"} }, { {Tile::u_channel,"vel"} });
		//	//IOFunc::AddPoissonGridFaceCentersToPolyscopePointCloud(holder, { {Tile::u_channel, "vel"} });
		//	polyscope::show();
		//}

		{

			auto holder = grid.getHostTileHolderForLeafs();

			//metadata.Append_Output_Thread(std::make_shared<std::thread>(IOFunc::OutputTilesAsVTU, holder, metadata.base_path / fmt::format("tiles{:04d}.vtu", metadata.current_frame)));
			//IOFunc::OutputTilesAsVTU(holder, metadata.base_path / fmt::format("tiles{:04d}.vtu", metadata.current_frame));

			metadata.Append_Output_Thread(std::make_shared<std::thread>(IOFunc::OutputPoissonGridAsStructuredVTI, holder,
				std::vector<std::pair<int, std::string>>{ {-1, "type"}, { -2, "level" }, { OutputChnls::vor, "vorticity" }},
				//std::vector<std::pair<int, std::string>>{ },
				std::vector<std::pair<int, std::string>>{ {OutputChnls::u_cell, "velocity"} },
				//std::vector<std::pair<int, std::string>>{ { -1, "type" }, { Tile::vor_channel, "vorticity" }, { Tile::dye_channel, "dye_density" } },
				//std::vector<std::pair<int, std::string>>{ {Tile::u_channel, "velocity"} },
				metadata.base_path / fmt::format("fluid{:04d}.vti", metadata.current_frame)));

		}

		{
			WriteStatToFile(metadata);
		}

		//{
		//	auto change_drive_to_d = [](const fs::path& original_path) -> fs::path {
		//		if (original_path.has_root_name()) {
		//			return fs::path("D:") / original_path.relative_path();
		//		}
		//		else {
		//			return original_path;
		//		}
		//		};
		//	auto base_path_d = change_drive_to_d(metadata.base_path);
		//	//auto base_path_d = metadata.base_path;
		//	//make directory
		//	fs::create_directories(base_path_d);

		//	auto particles_h_ptr = std::make_shared<thrust::host_vector<Particle>>(particles);
		//	metadata.Append_Output_Thread(std::make_shared<std::thread>(IOFunc::OutputParticleSystemAsVTU,
		//		particles_h_ptr, base_path_d / fmt::format("particles{:04d}.vtu", metadata.current_frame)
		//	));
		//}

		//{
		//	if (metadata.current_frame == 200) {
		//		auto all_holder = grid.getHostTileHolder(LEAF | NONLEAF | GHOST, -1);
		//		IOFunc::WriteHAHostTileHolderToFile(*all_holder, metadata.base_path / "fluid_end.bin");
		//	}
		//}
	}


	void project(HADeviceGrid<Tile>& grid, int u_channel, double current_time) {
		auto c0_channel = ProjChnls::c0;



		////GMG
		//{
		//	for (int axis : {0, 1, 2}) {
		//		PropagateToChildren(grid, u_channel + axis, u_channel + axis, -1, GHOST, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);
		//	}
		//	VelocityVolumeDivergenceOnLeafs(grid, u_channel, Tile::b_channel);
		//	Info("div pt l2: {}", NormSync(grid, 2, Tile::b_channel, false));
		//	CalculateNeighborTiles(grid);
		//	GMGSolver solver(1., 1.);
		//	//AMGSolver solver(Tile::c0_channel, 1., 1.);
		//	//solver.prepareTypesAndCoeffs(grid);

		//	CPUTimer timer;
		//	timer.start();
		//	//auto [iters, err] = ConjugateGradientSync(grid, false, 1000, 2, 10, 1e-6, false);
		//	auto [iters, err] = solver.solve(grid, false, 1000, 1e-6, 1, 10, 1, mParams.mIsPureNeumann);
		//	cudaDeviceSynchronize();
		//	double elapsed = timer.stop("MGPCG");
		//	double total_cells = grid.numTotalTiles() * Tile::SIZE;
		//	double cells_per_second = (total_cells + 0.0) / (elapsed / 1000.0);
		//	Info("Total {:.5}M cells, MGPCG speed {:.5} M cells /s at {} iters", total_cells / (1024.0 * 1024), cells_per_second / (1024.0 * 1024), iters);
		//	projection_time = elapsed;
		//	Info("pressure pt l2: {}", NormSync(grid, 2, Tile::x_channel, false));
		//	AddGradientToFaceCenters(grid, Tile::x_channel, u_channel);
		//	applyVelocityBC(grid, current_time);
		//}

		//AMG
		{
			CalculateNeighborTiles(grid);

			AMGVolumeWeightedDivergenceOnLeafs(grid, u_channel, ProjChnls::b);
			Info("div pt l2: {}", NormSync(grid, 2, ProjChnls::b, false));


			AMGSolver solver(c0_channel, 0.5, 1, 1);
			solver.prepareTypesAndCoeffs(grid);

			CPUTimer timer;
			timer.start();
			auto [iters, err] = solver.solve(grid, false, 1000, 1e-6, 2, 10, 1, mParams.mIsPureNeumann);
			cudaDeviceSynchronize();
			double elapsed = timer.stop("AMGPCG");
			double total_cells = grid.numTotalTiles() * Tile::SIZE;
			double cells_per_second = (total_cells + 0.0) / (elapsed / 1000.0);
			Info("Total {:.5}M cells, AMGPCG speed {:.5} M cells /s at {} iters", total_cells / (1024.0 * 1024), cells_per_second / (1024.0 * 1024), iters);
			projection_time = elapsed;

			Info("pressure pt l2: {}", NormSync(grid, 2, ProjChnls::x, false));

			AMGAddGradientToFace(grid, -1, LEAF, ProjChnls::x, c0_channel, AdvChnls::u);
			applyVelocityBC(grid, current_time);

			AMGVolumeWeightedDivergenceOnLeafs(grid, u_channel, ProjChnls::b);
			//for (int i : {0, 1, 2}) {
			//	AccumulateToParents(grid, u_channel + i, u_channel + i, -1, LEAF, LAUNCH_SUBTREE, INTERIOR | DIRICHLET, 1.0 / 4.0, true);
			//}
			Info("div pt linf: {}", NormSync(grid, -1, ProjChnls::b, false));
		}



	}

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
		int u_channel = AdvChnls::u;//6

		//last_grid:
		//012: node u
		//345: u copy
		//678: face u
		//9: node dye
		//10: voxel dye
		int last_tmp_channel = 3;
		int last_u_node_channel = 0;//on last_grid
		//int last_dye_node_channel = Tile::vor_channel;//on last_grid
		
		//next grid:
		//012: uw
		//3: particle counter
		//678: face u
		//10: voxel dye
		int next_uw_channel = 0;//on grid
		int next_counter_channel = 4;//on grid
		
		//saved intermediate velocities
		int n = grid_ptrs.size() - 1;
		//we only need to prepare the last grid at this time
		auto& last_grid = *grid_ptrs[n - 1];
		InterpolateVelocitiesAtAllTiles(last_grid, u_channel, last_u_node_channel);

		
		//cudaDeviceSynchronize(); timer.stop("Prepare last grid"); timer.start();
		CheckCudaError("prepare last grid");

		thrust::host_vector<HATileAccessor<Tile>> accs_h;
		for (int i = 0; i < n; i++) {
			accs_h.push_back(grid_ptrs[i]->deviceAccessor());
		}
		thrust::device_vector<HATileAccessor<Tile>> accs_d = accs_h;
		auto accs_d_ptr = thrust::raw_pointer_cast(accs_d.data());
		thrust::device_vector<double> time_steps_d = time_steps;
		auto time_steps_d_ptr = thrust::raw_pointer_cast(time_steps_d.data());

		MarkParticlesOutsideFluidRegionAsInvalid(particles, last_grid);
		MarkOldParticlesAsInvalid(particles, current_time, mParams.mParticleLife);
		EraseInvalidParticles(particles);

		ReseedParticles(last_grid, mParams, last_tmp_channel, current_time, mNumParticlesPerCell, particles);		

		cudaDeviceSynchronize(); reseeding_time = timer.stop("reseeding and remove particles in solid"); timer.start();
		Info("total {:.5f}M particles, time step counter {}", particles.size() / (1024 * 1024 + 0.f), time_step_counter);

		//reset impulse for all particles
		if (time_step_counter % mParams.mFlowMapStride == 0) {
			nfm_query_grid_ptr = grid_ptrs[n - 1];
			ResetParticleImpulse(last_grid, u_channel, last_u_node_channel, particles);

			cudaDeviceSynchronize(); timer.stop("reset all particles impulse"); timer.start();
		}
		else {
			int fine_level = mParams.mFineLevel;
			int coarse_level = mParams.mCoarseLevel;

			int back_traced_steps = time_step_counter % mParams.mFlowMapStride;
			int nfm_start_idx = n - back_traced_steps - 1;
			auto particles_d_ptr = thrust::raw_pointer_cast(particles.data());
			LaunchIndexFunc([=] __device__(int idx) {
				auto& particle = particles_d_ptr[idx];

				if (particle.start_time == current_time) {

					Vec psi = particle.pos;
					Vec m0; Eigen::Matrix3<T> matT;
					NFMBackQueryImpulseAndT(accs_d_ptr, fine_level, coarse_level, time_steps_d_ptr, u_channel, last_u_node_channel, nfm_start_idx, n - 1, psi, m0, matT);
					particle.impulse = m0;
					particle.matT = matT;
				}
			}, particles.size(), 128);

			cudaDeviceSynchronize(); timer.stop("reset newly sampled particles impulse"); timer.start();
		}


		HistogramSortParticlesAtGivenLevel(last_grid, mParams.mFineLevel, last_tmp_channel, particles, tile_prefix_sum_d, records_d);
		OptimizedAdvectParticlesAndSingleStepGradMRK4ForwardAtGivenLevel(last_grid, mParams.mFineLevel, u_channel, last_u_node_channel, dt, tile_prefix_sum_d, records_d);
		EraseInvalidParticles(particles);

		cudaDeviceSynchronize(); double adv_elapsed = timer.stop("Advect particles"); timer.start(); particle_advection_time = adv_elapsed;
		{
			double num_particles_M = particles.size() / (1024. * 1024.);
			Info("Particle advection time: {} ms, {}M particles, throughput {}M particles/s", adv_elapsed, num_particles_M, num_particles_M / adv_elapsed * 1000);
		}
		CheckCudaError("adv particle");


		auto& grid = *grid_ptrs[n];

		RefineWithParticles(grid, particles, mParams.mCoarseLevel, mParams.mFineLevel, next_counter_channel, false);

		//cudaDeviceSynchronize(); timer.stop("Refine with particles"); timer.start();


		CoarsenWithParticles(grid, particles, mParams.mCoarseLevel, mParams.mFineLevel, next_counter_channel, false);
		CheckCudaError("adapt with particles");


		cudaDeviceSynchronize(); adaptive_time = timer.stop("adapt with particles"); timer.start();


		//ParticleImpulseToGridMACIntp(grid, particles, u_channel, next_uw_channel);
		HistogramSortParticlesAtGivenLevel(grid, mParams.mFineLevel, next_counter_channel, particles, tile_prefix_sum_d, records_d);
		//OptimizedP2GTransferAtGivenLevel(grid, mParams.mFineLevel, u_channel, next_uw_channel, tile_prefix_sum_d, records_d);
		EraseInvalidParticles(particles);

		CheckCudaError("pfm p2g");
		//Info("max impulse after pfm: {}", VelocityLinf(grid, u_channel, -1, LEAF, LAUNCH_SUBTREE));


		cudaDeviceSynchronize(); p2g_time = timer.stop("P2G"); timer.start();
		{
			double num_particles_M = particles.size() / (1024. * 1024.);
			Info("P2G time: {} ms, {}M particles, throughput {}M particles/s", p2g_time, num_particles_M, num_particles_M / p2g_time * 1000);
		}

		//advect dye and NFM
		{
			auto last_acc = last_grid.deviceAccessor();
			auto nfm_query_acc = nfm_query_grid_ptr->deviceAccessor();
			auto params = mParams;
			
			int fine_level = mParams.mFineLevel;
			int coarse_level = mParams.mCoarseLevel;

			int back_traced_steps = time_step_counter % mParams.mFlowMapStride;
			int nfm_start_idx = n - back_traced_steps - 1;
			
			grid.launchVoxelFuncOnAllTiles(
				[=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
				auto& tile = info.tile();
				//if (!tile.isInterior(l_ijk)) return;

				//type
				int boundary_axis, boundary_off;
				tile.type(l_ijk) = params.cellType(current_time, acc, info, l_ijk, boundary_axis, boundary_off);



				{
					//grid velocity advection
					for (int axis : {0, 1, 2}) {
						if (tile(next_uw_channel + axis, l_ijk) < 1 - 1e-3)
						{
							//Vec psi = acc.faceCenter(axis, info, l_ijk);
							Vec psi = NFMErodedAdvectionPoint(axis, acc, info, l_ijk);
							Vec m0; Eigen::Matrix3<T> matT;

							//NFMBackQueryImpulseAndT(accs_d_ptr, info.mLevel, coarse_level, time_steps_d_ptr, u_channel, last_u_node_channel, nfm_start_idx, n, psi, m0, matT);
							//NFMBackQueryImpulseAndT(accs_d_ptr, fine_level, coarse_level, time_steps_d_ptr, u_channel, last_u_node_channel, nfm_start_idx, n, psi, m0, matT);

							NFMBackMarchPsiAndT(accs_d_ptr, fine_level, coarse_level, time_steps_d_ptr, u_channel, last_u_node_channel, nfm_start_idx, n, psi, matT);
							//m0 = InterpolateFaceValue(accs_d_ptr[nfm_start_idx], psi, u_channel, last_u_node_channel);
							m0 = InterpolateFaceValue(nfm_query_acc, psi, u_channel, last_u_node_channel);

							Vec m1 = MatrixTimesVec(matT.transpose(), m0);

							tile(AdvChnls::u + axis, l_ijk) = m1[axis];

							//if (m1[axis] > 1e5) {
							//	auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
							//	printf("g_ijk %d %d %d axis %d m1 %f\n", g_ijk[0], g_ijk[1], g_ijk[2], axis, m1[axis]);
							//}
						}
					}
				}
			}, LEAF, 4
			);
		}


		CalcCellTypesFromLeafs(grid);

		cudaDeviceSynchronize(); nfm_advection_time = timer.stop("NFM advection"); timer.start();
		CheckCudaError("nfm advection");

		//Info("max impulse after nfm: {}", VelocityLinf(grid, u_channel, -1, LEAF, LAUNCH_SUBTREE));
	}

	void applyExternalForce(HADeviceGrid<Tile>& grid, const double dt) {

		//int tmp_ta_node = 0;
		//CalcLeafNodeValuesFromCellCenters(grid, Tile::Ta_channel, tmp_ta_node);


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
				tile(AdvChnls::u + axis, l_ijk) += a[axis] * dt;
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


		Info("frame {} dt {}", metadata.current_frame, dt);

		adaptAndAdvect(metadata, grid_ptrs);


		applyExternalForce(grid, dt);


		applyVelocityBC(grid, metadata.current_time);


		//projection
		project(grid, AdvChnls::u, metadata.current_time);

		CalculateVelocityAndVorticityMagnitudeOnLeafFaceCenters(grid, mParams.mFineLevel, mParams.mCoarseLevel, AdvChnls::u, OutputChnls::u_node, OutputChnls::u_cell, OutputChnls::vor);


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
			"p2g_time {} ms\n"
			"projection_time {} ms\n"
			"nfm_advection_time {} ms\n"
			"advance_time {} ms\n",
			metadata.current_frame,
			particles.size(),
			grid_ptrs.back()->numTotalLeafTiles() * Tile::SIZE,
			reseeding_time,
			particle_advection_time,
			adaptive_time,
			p2g_time,
			projection_time,
			nfm_advection_time,
			advance_time);
		out.close();
	}

	void PrintMemoryInfo(void) {
		double M = 1024 * 1024, G = 1024 * 1024 * 1024;
		double particle_num = particles.size();
		double particle_capacity = particles.capacity();
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
};