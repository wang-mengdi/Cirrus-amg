#include "FluidEuler.h"
#include "Random.h"

#include <cub/cub.cuh>
#include <cub/block/block_reduce.cuh>
#include <thrust/execution_policy.h>

__device__ Vec NFMErodedAdvectionPoint(const int axis, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const Coord& l_ijk) {
	auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
	auto ng_ijk = g_ijk; ng_ijk[axis]--;
	HATileInfo<Tile> ninfo; Coord nl_ijk;
	acc.findVoxel(info.mLevel, ng_ijk, ninfo, nl_ijk);

	int effect_level = info.mLevel;
	Coord effect_g_ijk = g_ijk;

	if (!ninfo.empty() && (ninfo.mType & GHOST)) {
		effect_level = info.mLevel - 1;
		effect_g_ijk = acc.parentCoord(g_ijk);
	}
	return acc.faceCenterGlobal(axis, effect_level, effect_g_ijk);
}

size_t SmartResizeParticlesForInsert(thrust::device_vector<Particle>& particles, const size_t insert_num) {
	size_t old_capacity = particles.capacity();
	size_t new_capacity = particles.size() + insert_num;

	size_t diff;
	if (new_capacity > old_capacity) {
		diff = new_capacity - old_capacity;
		size_t alloc_capacity = old_capacity + diff * 2;
		particles.reserve(alloc_capacity);

		//Info("alloc capacity: {} particles capacity: {}", alloc_capacity, particles.capacity());
	}
	else {
		diff = 0;
	}

	return diff;
}

template<class FuncII>
__hostdev__ void IterateFaceNeighborCellTypes(const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const Coord& l_ijk, const int axis, FuncII f) {
	auto& tile = info.tile();
	uint8_t type0 = tile.type(l_ijk);

	auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
	auto ng_ijk = g_ijk; ng_ijk[axis]--;
	HATileInfo<Tile> ninfo; Coord nl_ijk;
	acc.findVoxel(info.mLevel, ng_ijk, ninfo, nl_ijk);

	if (!ninfo.empty()) {
		if (!ninfo.isLeaf()) {
			for (int offj : {0, 1}) {
				for (int offk : {0, 1}) {
					Coord child_offset = acc.rotateCoord(axis, Coord(1, offj, offk));
					Coord nc_ijk = acc.childCoord(ng_ijk, child_offset);
					HATileInfo<Tile> nc_info; Coord ncl_ijk;
					acc.findVoxel(info.mLevel + 1, nc_ijk, nc_info, ncl_ijk);
					if (!nc_info.empty()) {
						auto& nctile = nc_info.tile();
						uint8_t type1 = nctile.type(ncl_ijk);
						f(type0, type1);
					}
				}
			}
		}
		else {
			if (ninfo.isGhost()) {
				//it's coarser
				Coord np_ijk = acc.parentCoord(ng_ijk);
				acc.findVoxel(info.mLevel - 1, np_ijk, ninfo, nl_ijk);

			}
			auto& ntile = ninfo.tile();
			uint8_t type1 = ntile.type(nl_ijk);
			f(type0, type1);
		}
	}
}

void ClearAllNeumannNeighborFaces(HADeviceGrid<Tile>& grid, const int u_channel)
{
	grid.launchVoxelFunc(
		[=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
		for (int axis : {0, 1, 2}) {
			bool to_set = false;
			IterateFaceNeighborCellTypes(acc, info, l_ijk, axis, [&](const uint8_t type0, const uint8_t type1) {
				if ((type0 & NEUMANN) || (type1 & NEUMANN) || ((type0 & DIRICHLET) && (type1 & DIRICHLET))) {
					to_set = true;
				}
				});
			if (to_set) {
				info.tile()(u_channel + axis, l_ijk) = 0;
			}
		}
	}, -1, LEAF, LAUNCH_SUBTREE
	);
}

void MarkOldParticlesAsInvalid(thrust::device_vector<Particle>& particles, const T current_time, const T particle_life) {
	auto particles_ptr = thrust::raw_pointer_cast(particles.data());
	LaunchIndexFunc([=] __device__(int idx) {
		auto& p = particles_ptr[idx];

		if (current_time - p.start_time > particle_life) {
			p.pos = Vec(NODATA, NODATA, NODATA);
		}

	}, particles.size(), 128);
}

__global__ void CalculateReseedingNumbersOnLeafTiles128Kernel(const T current_time, HATileAccessor<Tile> acc, HATileInfo<Tile>* tile_infos, FluidParams params, const int tmp_channel, const int num_particles_per_cell, int* reseed_number_per_cell) {
	int bidx = blockIdx.x;
	auto& info = tile_infos[bidx];
	auto& tile = info.tile();

	if (!(info.mType & LEAF)) return;

	//int reseed_threshold = num_particles_per_cell / 2;
	int reseed_threshold = 1;

	for (int i = threadIdx.x; i < Tile::SIZE; i += blockDim.x) {
		auto l_ijk = acc.localOffsetToCoord(i);
		int reseed_num = 0;

		if (tile.type(l_ijk) == INTERIOR && params.isInParticleGenerationRegion(current_time, acc, info, l_ijk)) {
			int num_particles = tile(tmp_channel, l_ijk);
			if (num_particles < reseed_threshold) {
				reseed_num = num_particles_per_cell - num_particles;
			}
		}
		reseed_number_per_cell[bidx * Tile::SIZE + i] = reseed_num;
	}
}

void ReseedParticles(HADeviceGrid<Tile>& grid, const FluidParams& params, const int tmp_channel, const double current_time, const int num_particles_per_cell, thrust::device_vector<Particle>& particles) {
	//CPUTimer timer; timer.start();
	static uint64_t global_particle_counter = 0;

	static RandomGenerator rng;
	CountParticleNumberInLeafCells(grid, particles, tmp_channel);

	//cudaDeviceSynchronize(); timer.stop("count particles"); timer.start();

	//int reseed_threshold = num_particles_per_cell / 2;
	//int reseed_threshold = 

	static thrust::device_vector<int> cell_particle_num_d;
	if (cell_particle_num_d.size() < grid.dAllTiles.size() * Tile::SIZE) {
		cell_particle_num_d.resize(grid.dAllTiles.size() * Tile::SIZE);
	}
	//Assert(cell_particle_num_d.size() == grid.dAllTiles.size() * Tile::SIZE,"");

	thrust::fill(cell_particle_num_d.begin(), cell_particle_num_d.end(), 0);
	auto cell_particle_num_d_ptr = thrust::raw_pointer_cast(cell_particle_num_d.data());
	
	auto info_ptr = thrust::raw_pointer_cast(grid.dAllTiles.data());
	auto acc_d = grid.deviceAccessor();
	CalculateReseedingNumbersOnLeafTiles128Kernel << <grid.dAllTiles.size(), 128 >> > (current_time, acc_d, info_ptr, params, tmp_channel, num_particles_per_cell, cell_particle_num_d_ptr);

	//cudaDeviceSynchronize(); timer.stop("calc reseeding number of each voxel"); timer.start();

	static thrust::host_vector<int> cell_particle_num_h;
	cell_particle_num_h = cell_particle_num_d;
	//Info("cell_particle_num_h: {}", cell_particle_num_h);

	auto acc = grid.hostAccessor();
	thrust::host_vector<Particle> reseed_particles_h;

	for (int idx = 0; idx < cell_particle_num_h.size(); idx++) {
		if (cell_particle_num_h[idx] > 0) {
			int tile_idx = idx / Tile::SIZE;
			int l_idx = idx % Tile::SIZE;
			auto& info = grid.hAllTiles[tile_idx];
			auto l_ijk = acc.localOffsetToCoord(l_idx);

			//sample random particles
			auto bbox = acc.voxelBBox(info, l_ijk);
			auto minPoint = bbox.min();
			auto maxPoint = bbox.max();

			for (int i = 0; i < cell_particle_num_h[idx]; i++) {
				auto x = rng.uniform(minPoint[0], maxPoint[0]);
				auto y = rng.uniform(minPoint[1], maxPoint[1]);
				auto z = rng.uniform(minPoint[2], maxPoint[2]);

				Particle p;
				p.global_idx = global_particle_counter++;
				p.pos = Vec(x, y, z);
				p.impulse = Vec(0., 0., 0.);
				p.matT = Eigen::Matrix3<T>::Identity();
				p.start_time = current_time;
				reseed_particles_h.push_back(p);
			}
		}
	}

	int diff = SmartResizeParticlesForInsert(particles, reseed_particles_h.size());
	particles.insert(particles.end(), reseed_particles_h.begin(), reseed_particles_h.end());
}

// Kernel to mark the interested area based on min and max values in a tile
__global__ void LockedMarkInterestAreaMinAndMax128Kernel(HATileAccessor<Tile> acc, HATileInfo<Tile>* infos, const uint8_t tmp_channel, int subtree_level, uint8_t launch_types) {
	int bi = blockIdx.x;  // Block index
	int ti = threadIdx.x; // Thread index within the block

	const auto& info = infos[bi];

	// Early exit if the subtree type does not match the launch type
	if (!(info.subtreeType(subtree_level) & launch_types)) {
		if (ti == 0) {
			auto& tile = info.tile();
			tile.mIsInterestArea = false;
			tile.mIsLockedRefine = false;
		}
		return;
	}

	auto& tile = info.tile();
	auto dataAsFloat4 = reinterpret_cast<float4*>(tile.mData[tmp_channel]);
	float4 value = dataAsFloat4[ti];

	// Calculate thread-local min and max
	T thread_min = min(min(value.x, value.y), min(value.z, value.w));
	T thread_max = max(max(value.x, value.y), max(value.z, value.w));

	// Use CUB to perform block-wide reduction
	typedef cub::BlockReduce<T, 128> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage_min;
	__shared__ typename BlockReduce::TempStorage temp_storage_max;

	T block_min = BlockReduce(temp_storage_min).Reduce(thread_min, cub::Min());
	T block_max = BlockReduce(temp_storage_max).Reduce(thread_max, cub::Max());

	// The first thread writes the results to the tile metadata
	if (ti == 0) {
		if (block_min == 0 && block_max > 0) {
			tile.mIsInterestArea = true;
			//tile.mIsLockedRefine = true;
		}
		else {
			tile.mIsInterestArea = false;
			//tile.mIsLockedRefine = false;
		}

		// Debugging output
		//printf("Block %d: minValue = %f, maxValue = %f\n", bi, block_min, block_max);
	}
}

int LockedRefineWithNonBoundaryNeumannCellsOneStep(const T current_time, HADeviceGrid<Tile>& grid, const FluidParams params, const int tmp_channel, bool verbose) {
	int coarse_level = params.mCoarseLevel;
	int fine_level = params.mFineLevel;
	auto levelTarget = [coarse_level, fine_level]__device__(const HATileAccessor<Tile> &acc, const HATileInfo<Tile> &info) ->int {
		auto& tile = info.tile();
		if (tile.mIsInterestArea > 0) return fine_level;
		return coarse_level;
	};
	//mark non-boundary neumann cells
	grid.launchVoxelFuncOnAllTiles(
		[=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
		auto& tile = info.tile();
		tile(tmp_channel, l_ijk) = 0;
		if (tile.type(l_ijk) & NEUMANN) {
			int boundary_axis, boundary_off;
			if (params.cellType(current_time, acc, info, l_ijk, boundary_axis, boundary_off) == NEUMANN) {
				if (boundary_axis == -1) {
					tile(tmp_channel, l_ijk) = 1;
				}
			}
			//int boundary_axis, boundary_off;
			//if (!FluidParams::queryBoundaryDirection(acc, info, l_ijk, boundary_axis, boundary_off)) {
			//	tile(tmp_channel, l_ijk) = 1;
			//}
		}
	}, LEAF, 4
	);

	{
		//Info("all {} tiles", grid.dAllTiles.size());
		auto info_ptr = thrust::raw_pointer_cast(grid.dAllTiles.data());
		LockedMarkInterestAreaMinAndMax128Kernel << <grid.dAllTiles.size(), 128 >> > (grid.deviceAccessor(), info_ptr, tmp_channel, -1, LEAF);
	}

	//struct LevelTargetFunctor {
	//	int coarse_level;
	//	int fine_level;

	//	__hostdev__
	//	int operator()(const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info) const {
	//		const auto& tile = info.tile();
	//		return (tile.mIsInterestArea > 0) ? fine_level : coarse_level;
	//	}
	//};
	//LevelTargetFunctor level_target = { coarse_level, fine_level };

	//auto refine_cnts = RefineLeafsOneStep(grid, levelTarget, verbose);
	auto refine_cnts = grid.refineStep(levelTarget, verbose);
	//auto refine_cnts = grid.refineLeafsOneStep<LevelTargetFunctor>(level_target, verbose);
	grid.spawnGhostTiles(verbose);
	//SpawnGhostTiles(grid, verbose);
	if (verbose) Info("Refine {} tiles on each layer", refine_cnts);
	auto cnt = std::accumulate(refine_cnts.begin(), refine_cnts.end(), 0);

	//Info("locked refine with non-boundary neumann cells one step: {}", cnt);

	return cnt;
}