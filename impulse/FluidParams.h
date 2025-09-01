#pragma once

#include "GMGSolver.h"
#include "SDFGrid.h"


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace BufChnls {
	constexpr int u = 6;
	constexpr int u_node = 0;
	constexpr int tmp = 3;
}

namespace AdvChnls {
	constexpr int u = 6;
	constexpr int counter = 4;
}


namespace ProjChnls {
	constexpr int x = 0;
	constexpr int b = 1;
	constexpr int c0 = 11;
}


namespace OutputChnls {
	constexpr int u_node = 0;
	constexpr int u_cell = 3;
	constexpr int vor = 9;
}

////Channel allocations
//      Buffer		Advection		Projection		Output
// 0    node u						x				node u
// 1    node v						b/r				node v
// 2	node w						p				node w
// 3	tmp							Ap				cell u
// 4				counter			z				cell v
// 5												cell w
// 6	u			u				u				u
// 7	v			v				v				v
// 8	w			w				w				w
// 9												vor
//10
//11								c0
//12								c1
//13								c2
//14								c3


//TVORTEX: tornado-like vortex, reference: Physically-based Simulation of Tornadoes
//enum TestCase { KARMAN = 0, SMOKESPHERE };
enum TestCase { SMOKESPHERE = 0 };

class FluidParams {
public:
	//parameters read from json
	TestCase mTestCase;
	int mFlowMapStride;
	int mCoarseLevel;
	int mFineLevel;
	T mRefineThreshold;
	T mParticleLife;
	nanovdb::Vec3R mGravity;


	//MaskGridAccessor mMaskGridAccessor;
	//SDFGridAccessor mSDFGridAccessor;

	//SDFGridAccessor mSDFVelocityAccessors[3];

	////for karman
	//T karman_source = 1.0;

	////Vec smokesphere_source = Vec(1.0, 0., 0.);
	//Vec smokesphere_source = Vec(0., 0., 0.4);
	//Vec smokesphere_center = Vec(0.5, 0.5, 0.3);
	////Vec smokesphere_center = Vec(0.5, 0.5, 0.7);
	//T smokesphere_radius = 0.05;

	//T nasa_source = 1.0;

	//T prop_source = 1.0;
	//T f1_source = 1.0;
	//T lizard_source = +1.0;
	//T fish_source = 1.0;
	//T bat_source = 1.0;
	//T flamingo_source = 1.0;

	//parameters set by itself
	bool mIsPureNeumann = false;

	FluidParams() {}
	FluidParams(json& j, std::shared_ptr<MaskGrid> mask_grid_ptr, std::shared_ptr<SDFGrid> sdf_float_grid_ptr) {

		std::string test = Json::Value<std::string>(j, "test", "tvortex");
		if (test == "smokesphere") mTestCase = SMOKESPHERE;
		else Assert(false, "invalid test {}", test);

		if (mTestCase == SMOKESPHERE) {
			mIsPureNeumann = true;
		}

		mFlowMapStride = Json::Value<int>(j, "flowmap_stride", 5);
		mCoarseLevel = Json::Value<int>(j, "coarse_level", 0);
		mFineLevel = Json::Value<int>(j, "fine_level", 6);
		mRefineThreshold = Json::Value<T>(j, "refine_threshold", 200);
		mGravity = Vec(0, 0, 0);
		mGravity[2] = Json::Value<double>(j, "gravity", -9.8);
		mParticleLife = Json::Value<T>(j, "particle_life", FLT_MAX);
	}

	__device__ static bool queryBoundaryDirection1(const HATileAccessor<Tile>& acc, const int level, const Coord& g_ijk, int& boundary_axis, int& boundary_off) {
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

	__device__ static bool queryEffectiveBoundaryDirection1(const HATileAccessor<Tile>& acc, int chk_level, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk, int& boundary_axis, int& boundary_off) {
		chk_level = min(chk_level, info.mLevel);
		int level_diff = info.mLevel - chk_level;
		
		auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
		

		g_ijk = Coord(g_ijk[0] >> level_diff, g_ijk[1] >> level_diff, g_ijk[2] >> level_diff);

		//printf("level %d chk level %d diff %d g_ijk %d %d %d\n", info.mLevel, chk_level, level_diff, g_ijk[0], g_ijk[1], g_ijk[2]);
		return queryBoundaryDirection1(acc, chk_level, g_ijk, boundary_axis, boundary_off);
	}

	__hostdev__ int initialLevelTarget(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info) const {
		return mCoarseLevel;
	}

	__device__ uint8_t cellType(const T current_time, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk, int& boundary_axis, int& boundary_off) const;




	__device__ void setBoundaryCondition(const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk, const double current_time) const {
		auto& tile = info.tile();
		if (mTestCase == SMOKESPHERE) {
			Vec vel = smokesphere_source;
			if (tile.type(l_ijk) & NEUMANN) {
				int boundary_axis, boundary_sgn;
				if (queryEffectiveBoundaryDirection1(acc, mCoarseLevel, info, l_ijk, boundary_axis, boundary_sgn)) {
					for (int axis : {0, 1, 2}) {
						tile(AdvChnls::u + axis, l_ijk) = vel[axis];
					}
				}
			}
			else if (tile.type(l_ijk) & INTERIOR) {
				for (int axis : {0, 1, 2}) {
					auto nl_ijk = l_ijk; nl_ijk[axis] -= 1;
					int boundary_axis, boundary_sgn;
					if (queryEffectiveBoundaryDirection1(acc, mCoarseLevel, info, nl_ijk, boundary_axis, boundary_sgn)) {
						if (boundary_sgn == -1) {
							tile(AdvChnls::u + axis, l_ijk) = vel[axis];
						}
					}
				}
			}
		}
	}


	//set type, velocity, smoke
	__device__ void setInitialCondition(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk)const;
};