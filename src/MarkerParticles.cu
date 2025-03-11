#include "MarkerParticles.h"
#include "FlowMap.h"

void CountParticleNumberInLeafCells(HADeviceGrid<Tile>& grid, const thrust::device_vector<MarkerParticle>& particles, const int tmp_channel) {
	grid.launchVoxelFuncOnAllTiles(
		[tmp_channel]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
		auto& tile = info.tile();
		tile(tmp_channel, l_ijk) = 0;
	}, LEAF, 4
	);
	//Info("reset done");
	auto particles_ptr = thrust::raw_pointer_cast(particles.data());
	auto acc = grid.deviceAccessor();
	LaunchIndexFunc([=] __device__(int idx) {
		auto& p = particles_ptr[idx];
		HATileInfo<Tile> info; Coord l_ijk; Vec frac;
		acc.findLeafVoxelAndFrac(p.pos, info, l_ijk, frac);
		int local_off = acc.localCoordToOffset(l_ijk);
		if (!info.empty()) {
			auto& tile = info.tile();
			atomicAdd(&tile.mData[tmp_channel][local_off], (T)1);
			//atomicAdd(&tile(tmp_channel, l_ijk), 1);
		}
	}, particles.size());
}

void CalcInterestAreaFlagsWithParticlesOnLeafs(const thrust::device_vector<MarkerParticle>& particles, HADeviceGrid<Tile>& grid, int tmp_channel) {
	CountParticleNumberInLeafCells(grid, particles, tmp_channel);

	auto info_ptr = thrust::raw_pointer_cast(grid.dAllTiles.data());
	MarkRegionOfInterestWithChannelMinAndMax128Kernel << <grid.dAllTiles.size(), 128 >> > (
		grid.deviceAccessor(), info_ptr, -1, LEAF,
		tmp_channel,
		[=]__device__(const T tile_min, const T tile_max) {
		return tile_max > 0;
	},
		false//do not calculate locked
		);
}

void CoarsenWithParticles(HADeviceGrid<Tile>& grid, const thrust::device_vector<MarkerParticle>& particles, const int coarse_levels, const int fine_levels, const int counter_channel, bool verbose) {
	auto levelTarget = [fine_levels, coarse_levels]__device__(const HATileAccessor<Tile> &acc, const HATileInfo<Tile> &info) ->int {
		auto& tile = info.tile();
		if (tile.mIsInterestArea || tile.mIsLockedRefine) return fine_levels;
		return coarse_levels;
	};

	while (true) {
		CalcInterestAreaFlagsWithParticlesOnLeafs(particles, grid, counter_channel);
		auto coarsen_cnts = grid.coarsenStep(levelTarget, verbose);
		if (verbose) Info("Deleted {} tiles on each layer", coarsen_cnts);
		auto cnt = std::accumulate(coarsen_cnts.begin(), coarsen_cnts.end(), 0);
		if (cnt == 0) break;
	}
	grid.spawnGhostTiles(verbose);
}

void RefineWithParticles(HADeviceGrid<Tile>& grid, const thrust::device_vector<MarkerParticle>& particles, const int coarse_levels, const int fine_levels, const int counter_channel, bool verbose) {
	auto levelTarget = [fine_levels, coarse_levels]__device__(const HATileAccessor<Tile> &acc, const HATileInfo<Tile> &info) ->int {
		auto& tile = info.tile();
		if (tile.mIsInterestArea || tile.mIsLockedRefine) return fine_levels;
		return coarse_levels;
	};
	while (true) {
		CalcInterestAreaFlagsWithParticlesOnLeafs(particles, grid, counter_channel);

		//polyscope::init();
		//IOFunc::AddTilesToPolyscopeVolumetricMesh(grid, LEAF, "leafs");
		//IOFunc::AddParticleSystemToPolyscope(particles, "particles");
		//polyscope::show();

		//auto refine_cnts = RefineLeafsOneStep(grid, levelTarget, verbose);
		auto refine_cnts = grid.refineStep(levelTarget, verbose);
		grid.spawnGhostTiles(verbose);
		if (verbose) Info("Refine {} tiles on each layer", refine_cnts);
		auto cnt = std::accumulate(refine_cnts.begin(), refine_cnts.end(), 0);

		if (cnt == 0) break;
	}
}


void AdvectMarkerParticlesRK4Forward(HADeviceGrid<Tile>& grid, const int fine_level, const int coarse_level, const int u_channel, const double dt, thrust::device_vector<MarkerParticle>& particles_d, const bool erase_invalid) {
	//advect particles
	auto particles_ptr = thrust::raw_pointer_cast(particles_d.data());
	auto acc = grid.deviceAccessor();
	LaunchIndexFunc([=] __device__(int idx) {
		auto& p = particles_ptr[idx];
		RK4ForwardPosition(acc, fine_level, coarse_level, dt, u_channel, p.pos);
	}, particles_d.size(), 512, 4);
}

