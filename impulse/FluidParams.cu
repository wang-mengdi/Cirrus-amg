#include "FluidParams.h"


__device__ uint8_t FluidParams::cellType(const T current_time, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk, int& boundary_axis, int& boundary_off) const {
	if (mTestCase == SMOKESPHERE) {
		//z+ air
		//other walls
		bool is_neumann = false;
		bool is_dirichlet = false;

		//int boundary_axis, boundary_off;
		if (queryEffectiveBoundaryDirection1(acc, mCoarseLevel, info, l_ijk, boundary_axis, boundary_off)) {
			is_neumann = true;
		}

		if (is_neumann) return NEUMANN;
		else if (is_dirichlet) return DIRICHLET;

		auto pos = acc.cellCenter(info, l_ijk);
		if ((pos - smokesphere_center).length() <= smokesphere_radius) {
			return NEUMANN;
		}
		return INTERIOR;
	}
	else {
		return DIRICHLET;
	}
}

__device__ void FluidParams::setInitialCondition(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) const {
	double current_time = 0.0;

	if (mTestCase == SMOKESPHERE) {
		Vec initial_vel = smokesphere_source;

		Tile& tile = info.tile();
		for (int axis : {0, 1, 2}) {
			tile(AdvChnls::u + axis, l_ijk) = initial_vel[axis];
		}
		int boundary_axis, boundary_off;
		tile.type(l_ijk) = cellType(current_time, acc, info, l_ijk, boundary_axis, boundary_off);
	}
}
