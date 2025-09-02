#include "FluidParams.h"

__hostdev__ bool QueryBoundaryDirectionN1P1(const HATileAccessor<Tile>& acc, const int level, const Coord& g_ijk, int& boundary_axis, int& boundary_off)
{
	boundary_axis = -1;
	boundary_off = 0;

	for (int axis : {0, 1, 2}) {
		for (int off : {-1, 1}) {
			auto ng_ijk = g_ijk; ng_ijk[axis] += off;
			HATileInfo<Tile> ninfo; Coord nl_ijk;
			acc.findVoxel(level, ng_ijk, ninfo, nl_ijk);
			if (ninfo.empty()) {
				boundary_axis = axis;
				boundary_off = off;
				return true;
			}

		}
	}
	return false;
}

//the actual level might be finer than the checking level
//it's for like building a wall with respect to the coarse level
__hostdev__ bool QueryBoundaryDirectionN1P1OnCoarseLevel(const HATileAccessor<Tile>& acc, int coarse_chk_level, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk, int& boundary_axis, int& boundary_off) {
	coarse_chk_level = min(coarse_chk_level, info.mLevel);
	int level_diff = info.mLevel - coarse_chk_level;

	auto g_ijk = acc.localToGlobalCoord(info, l_ijk);


	g_ijk = Coord(g_ijk[0] >> level_diff, g_ijk[1] >> level_diff, g_ijk[2] >> level_diff);

	//printf("level %d chk level %d diff %d g_ijk %d %d %d\n", info.mLevel, chk_level, level_diff, g_ijk[0], g_ijk[1], g_ijk[2]);
	return QueryBoundaryDirectionN1P1(acc, coarse_chk_level, g_ijk, boundary_axis, boundary_off);
}

FluidParams::FluidParams(json& j)
{

	std::string test = Json::Value<std::string>(j, "test", "meshmotion");
	if (test == "meshmotion") mTestCase = MESHMOTION;
	else Assert(false, "invalid test {}", test);

	if (mTestCase == MESHMOTION) {
		mIsPureNeumann = true;
		mesh_motion_inflow = Json::Value<T>(j, "inflow_velocity", 1.0);
	}

	//mFlowMapStride = Json::Value<int>(j, "flowmap_stride", 5);
	mCoarseLevel = Json::Value<int>(j, "coarse_level", 0);
	mFineLevel = Json::Value<int>(j, "fine_level", 6);
	//mRefineThreshold = Json::Value<T>(j, "refine_threshold", 200);
	//mGravity = Vec(0, 0, 0);
	//mGravity[2] = Json::Value<double>(j, "gravity", -9.8);
	//mParticleLife = Json::Value<T>(j, "particle_life", FLT_MAX);
}


__hostdev__ int FluidParams::initialLevelTarget(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info) const {
	return mCoarseLevel;
}

__device__ uint8_t FluidParams::wallCellType(const T current_time, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) const {
	if (mTestCase == MESHMOTION) {
		//z+ air
		//other walls
		bool is_neumann = false;
		bool is_dirichlet = false;
		int boundary_axis, boundary_off;

		//int boundary_axis, boundary_off;
		if (QueryBoundaryDirectionN1P1OnCoarseLevel(acc, mCoarseLevel, info, l_ijk, boundary_axis, boundary_off)) {
			is_neumann = true;
		}

		if (is_neumann) return NEUMANN;
		else if (is_dirichlet) return DIRICHLET;
		return INTERIOR;
	}
	else {
		return DIRICHLET;
	}
}

__device__ void FluidParams::setWallCellType(const T current_time, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) const
{
	auto& tile = info.tile();
	tile.type(l_ijk) = wallCellType(current_time, acc, info, l_ijk);
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

__device__ void FluidParams::setVelocityBoundaryCondition(const T current_time, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) const
{
	auto& tile = info.tile();
	if (mTestCase == MESHMOTION) {
		Vec vel(mesh_motion_inflow, 0, 0);

		for (int axis : {0, 1, 2}) {
			//first set all faces around NEUMANN to 0
			//there might be something wrong with a T-junction?
			{
				bool to_set = false;
				IterateFaceNeighborCellTypes(acc, info, l_ijk, axis, [&](const uint8_t type0, const uint8_t type1) {
					if ((type0 & NEUMANN) || (type1 & NEUMANN) || ((type0 & DIRICHLET) && (type1 & DIRICHLET))) {
						to_set = true;
					}
					});
				if (to_set) {
					info.tile()(AdvChnls::u + axis, l_ijk) = 0;
				}
			}

			//then set the inflow velocity
			{
				auto nl_ijk = l_ijk; nl_ijk[axis] -= 1;
				if ((wallCellType(current_time, acc, info, l_ijk) & NEUMANN) || (wallCellType(current_time, acc, info, nl_ijk) & NEUMANN)) {
					tile(AdvChnls::u + axis, l_ijk) = vel[axis];
				}
			}
		}

	}
}

__device__ void FluidParams::setInitialVelocity(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) const {
	double current_time = 0.0;

	if (mTestCase == MESHMOTION) {
		Vec initial_vel = Vec(mesh_motion_inflow, 0, 0);

		Tile& tile = info.tile();
		for (int axis : {0, 1, 2}) {
			tile(AdvChnls::u + axis, l_ijk) = initial_vel[axis];
		}
		//int boundary_axis, boundary_off;
		//tile.type(l_ijk) = cellType(current_time, acc, info, l_ijk, boundary_axis, boundary_off);
	}
}

