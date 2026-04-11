//////////////////////////////////////////////////////////////////////////
// Fluid Euler
// Copyright (c) (2022-), Bo Zhu, Mengdi Wang
// This file is part of MESO, whose distribution is governed by the LICENSE file.
//////////////////////////////////////////////////////////////////////////
#pragma once

#include "Simulator.h"
#include "FluidParams.h"
#include "JsonFwd.h"
//#include "Random.h"
#include "FMParticles.h"
#include "MarkerParticles.h"
#include "PoissonGrid.h"

class SDFAccelBase;
class MeshSDFAccel;
class SphereSDFAccel;

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

	thrust::device_vector<Particle> pfm_particles_d;

	FluidParams mParams;
	std::shared_ptr<SDFAccelBase> mMeshSDFAccel = nullptr;


	RandomGenerator mRamdonGenerator;

	double total_advance_time = 0; // total advance time
	double adapt_and_advect_time = 0; // adapt + advection
	double grid_adaptation_time = 0; // only adaptation (including particle adaptation and sdf calc/adapt)
	double total_projection_time = 0; // total projection time (including building system and solving)
	double projection_solve_time = 0; // only the linear solver time in projection


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

	void buildTypesAndAMGCoeffsFromNodeSDFs(HADeviceGrid<Tile>& grid, const T current_time);

	void init(json &j, DriverMetaData &metadata);

	virtual double CFL_Time(const double cfl);
	virtual void Output(DriverMetaData& metadata);


	//current_time and dt for calculating solid velocity and doing time interpolation
	void project(HADeviceGrid<Tile>& grid, const T current_time, const T dt);

	void adaptAndAdvect(DriverMetaData& metadata, std::vector<std::shared_ptr<HADeviceGrid<Tile>>> grid_ptrs);

	//(I - nu * dt * lap)u^{n+1} = u^* for implicit viscosity
	void applyViscosity(HADeviceGrid<Tile>& grid, const T dt, const T nu);

	void applyExternalForce(HADeviceGrid<Tile>& grid, const double dt);





	virtual void Advance(DriverMetaData& metadata);

	void WriteStatToFile(DriverMetaData& metadata);
	void PrintMemoryInfo(void);
	void Save_Frame(DriverMetaData& metadata);
	void Load_Frame(DriverMetaData& metadata);
};
