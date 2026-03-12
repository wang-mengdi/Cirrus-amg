#pragma once

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
enum TestCase { MESHMOTION = 0, AIRCRAFT, SPHERECIRCLING };

class FluidParams {
public:
	//parameters read from json
	TestCase mTestCase;
	Coord mInitialGridSize;//computational grid size at level 0, like (1,1,1) or (1,1,2)

	int mFlowMapStride;
	int mCoarseLevel;
	int mFineLevel;
	//T mRefineThreshold;
	T mParticleLife;
	nanovdb::Vec3R mGravity;
	T mesh_motion_inflow = 0.0;

	//int mSampleNumPerTile;//number of sampled points per finest tile
	int mSampleNumPerCell;
	T mRelativeParticleSampleBandwidth;//for particle generation, k*dx, where dx is the finest level
	T mRelativeRefineBandwidth;//for grid refinement, k*dx, where dx is the finest level

	int mExtrapolationIters;

	//parameters set by itself
	bool mIsPureNeumann = false;

	FluidParams() {}
	FluidParams(json& j);

	//initialization
	__hostdev__ int initialLevelTarget(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info) const;
	//__device__ void addInitialVelocityToFaceCenter(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) const;
	//set type, velocity, smoke
	//__hostdev__ void setInitialVelocity(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk)const;
	//includes the outer walls of the computational field, but not including the movable mesh inside
	//it will return a cell type in the computational grid only considering wall, fluid, air
	__hostdev__ uint8_t wallCellType(const T current_time, const HATileAccessor<Tile>& acc, const int level, const Coord& g_ijk)const;
	__hostdev__ uint8_t wallCellType(const T current_time, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) const;
	__hostdev__ void setWallCellType(const T current_time, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) const;

	__hostdev__ Eigen::Transform<T, 3, Eigen::Affine> meshToWorldTransform(const T current_time) const;

	__device__ T solidFaceCenterVelocity(const T current_time, const T dt, const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk, const int axis)const;
};