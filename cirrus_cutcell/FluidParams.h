#pragma once

#include "JsonFwd.h"
#include "GMGSolver.h"
//#include "SDFGrid.h"
#include "MeshCutCell.h"


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace BufChnls {
	constexpr int u = 6;
	//constexpr int u_node = 0;
	constexpr int u_cell = 0;
	constexpr int tmp = 3;
	constexpr int counter = 4;
	constexpr int vor = 9;
	constexpr int sdf = 10;
}

namespace AdvChnls {
	constexpr int u_weight = 0;
}


namespace ProjChnls {
	constexpr int x = 0;
	constexpr int b = 1;
	constexpr int u_mix = 3;
	constexpr int c0 = 11;
}

namespace ViscChnls {
	constexpr int pressure_type = 5;
	constexpr int u_cell_sdf = 9;
}

//namespace OutputChnls {
//	//constexpr int u_node = 0;
//	//constexpr int u_cell = 3;
//	constexpr int u = 6;
//	constexpr int vor = 9;
//}

////Channel allocations
//      Buffer						Projection				Advection
// 0    cell u 						x						weight u
// 1    cell v						b/r						weight v
// 2    cell w						p						weight w
// 3	tmp							Ap/mix u				tmp
// 4	counter						z/mix v		
// 5								mix w					
// 6	u							u						u
// 7	v							v						v
// 8	w							w						w
// 9	vor													
//10    sdf													sdf
//11								c0
//12								c1
//13								c2
//14								c3


//TVORTEX: tornado-like vortex, reference: Physically-based Simulation of Tornadoes
//enum TestCase { KARMAN = 0, SMOKESPHERE };
enum TestCase { MESHMOTION = 0, AIRCRAFT, SPHERECIRCLING, JASSM };

class FluidParams {
public:
	TestCase mTestCase;
	Coord mInitialGridSize;

	int mFlowMapStride;
	int mCoarseLevel;
	int mFineLevel;
	T mParticleLife;
	nanovdb::Vec3R mGravity;
	T mesh_motion_inflow = 0.0;

	int mSampleNumPerCell;
	T mRelativeParticleSampleBandwidth;
	T mRelativeRefineBandwidth;

	int mExtrapolationIters;

	bool mIsPureNeumann = false;

	FluidParams() {}
	FluidParams(json& j);
};

namespace FluidScene {
	__hostdev__ int initialLevelTarget(const FluidParams& params, const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info);
	__hostdev__ uint8_t wallCellType(const FluidParams& params, const T current_time, const HATileAccessor<Tile>& acc, const int level, const Coord& g_ijk);
	__hostdev__ uint8_t wallCellType(const FluidParams& params, const T current_time, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk);
	__hostdev__ void setWallCellType(const FluidParams& params, const T current_time, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk);
	__hostdev__ Eigen::Transform<T, 3, Eigen::Affine> meshToWorldTransform(const FluidParams& params, const T current_time);
	__device__ T solidFaceCenterVelocity(const FluidParams& params, const T current_time, const T dt, const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk, const int axis);
}
