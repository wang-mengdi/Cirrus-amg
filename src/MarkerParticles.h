#pragma once

#include "PoissonTile.h"//for Vec, T etc
#include "Random.h"

class MarkerParticle {
public:
	Vec pos;
	T birth_time;
};

size_t SmartResizeParticlesForInsert(thrust::device_vector<MarkerParticle>& particles, const size_t insert_num);
void CountParticleNumberInLeafCells(HADeviceGrid<Tile>& grid, const thrust::device_vector<MarkerParticle>& particles, const int tmp_channel);
void CoarsenWithMarkerParticles(HADeviceGrid<Tile>& grid, const thrust::device_vector<MarkerParticle>& particles, const int coarse_levels, const int fine_levels, const int counter_channel, bool verbose);
void RefineWithMarkerParticles(HADeviceGrid<Tile>& grid, const thrust::device_vector<MarkerParticle>& particles, const int coarse_levels, const int fine_levels, const int counter_channel, bool verbose);
void AdvectMarkerParticlesRK4ForwardAndMarkInvalid(HADeviceGrid<Tile>& grid, const int fine_level, const int coarse_level, const int u_channel, const double dt, const T earliest_birth_time, thrust::device_vector<MarkerParticle>& particles_d);
void EraseInvalidParticles(thrust::device_vector<MarkerParticle>& particles);

//should count number of particles in each cell before calling this function
template<class FuncABC>
__global__ void CalculateReseedingNumbersOnLeafTiles128Kernel(HATileAccessor<Tile> acc, HATileInfo<Tile>* tile_infos, const int tmp_channel, FuncABC func_in_generation_region, const int reseed_threshold, const int num_particles_per_cell, int* reseed_number_per_cell) {
	int bidx = blockIdx.x;
	auto& info = tile_infos[bidx];
	auto& tile = info.tile();

	if (!(info.mType & LEAF)) return;

	for (int i = threadIdx.x; i < Tile::SIZE; i += blockDim.x) {
		auto l_ijk = acc.localOffsetToCoord(i);
		int reseed_num = 0;

		if (tile.type(l_ijk) == INTERIOR && func_in_generation_region(acc, info, l_ijk)) {
			int num_particles = tile(tmp_channel, l_ijk);
			if (num_particles <= reseed_threshold) {
				reseed_num = num_particles_per_cell - num_particles;
			}
		}
		reseed_number_per_cell[bidx * Tile::SIZE + i] = reseed_num;
	}
}

template<class FuncABC>
void ReseedMarkerParticles(HADeviceGrid<Tile>& grid, const int tmp_channel, FuncABC func_in_generation_region, const T birth_time, const int reseed_threshold, const int num_particles_per_cell, thrust::device_vector<int>& cell_particle_num_d, thrust::device_vector<MarkerParticle>& particles) {
	static RandomGenerator rng;
	CountParticleNumberInLeafCells(grid, particles, tmp_channel);

	if (cell_particle_num_d.size() < grid.dAllTiles.size() * Tile::SIZE) {
		cell_particle_num_d.resize(grid.dAllTiles.size() * Tile::SIZE);
	}
	thrust::fill(cell_particle_num_d.begin(), cell_particle_num_d.end(), 0);
	auto cell_particle_num_d_ptr = thrust::raw_pointer_cast(cell_particle_num_d.data());

	auto info_ptr = thrust::raw_pointer_cast(grid.dAllTiles.data());
	auto acc_d = grid.deviceAccessor();
	CalculateReseedingNumbersOnLeafTiles128Kernel << <grid.dAllTiles.size(), 128 >> > (acc_d, info_ptr, tmp_channel, func_in_generation_region, reseed_threshold, num_particles_per_cell, cell_particle_num_d_ptr);

	thrust::host_vector<int> cell_particle_num_h;
	cell_particle_num_h = cell_particle_num_d;

	auto acc = grid.hostAccessor();
	thrust::host_vector<MarkerParticle> reseed_particles_h;

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

				MarkerParticle p;
				
				p.pos = Vec(x, y, z);
				p.birth_time = birth_time;
				reseed_particles_h.push_back(p);
			}
		}
	}

	int diff = SmartResizeParticlesForInsert(particles, reseed_particles_h.size());
	particles.insert(particles.end(), reseed_particles_h.begin(), reseed_particles_h.end());
}
