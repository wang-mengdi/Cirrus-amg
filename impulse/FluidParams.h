#pragma once

#include "GMGSolver.h"
//#include "SDFGrid.h"
#include "MeshCutCell.h"


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace BufChnls {
	constexpr int u = 6;
	constexpr int u_node = 0;
	constexpr int u_cell = 3;
	constexpr int tmp = 3;
	constexpr int counter = 4;
	constexpr int vor = 9;
	constexpr int sdf = 10;
}

//namespace AdvChnls {
//	constexpr int u = 6;
//	constexpr int counter = 4;
//}


namespace ProjChnls {
	constexpr int x = 0;
	constexpr int b = 1;
	constexpr int c0 = 11;
}

//namespace OutputChnls {
//	//constexpr int u_node = 0;
//	//constexpr int u_cell = 3;
//	constexpr int u = 6;
//	constexpr int vor = 9;
//}

////Channel allocations
//      Buffer						Projection		
// 0    node u						x				
// 1    node v						b/r				
// 2	node w						p				
// 3	tmp/cell u					Ap				
// 4	counter/cell v				z				
// 5	cell w											
// 6	u							u				
// 7	v							v				
// 8	w							w				
// 9	vor											
//10    sdf
//11								c0
//12								c1
//13								c2
//14								c3


//TVORTEX: tornado-like vortex, reference: Physically-based Simulation of Tornadoes
//enum TestCase { KARMAN = 0, SMOKESPHERE };
enum TestCase { MESHMOTION = 0 };

class FluidParams {
public:
	//parameters read from json
	TestCase mTestCase;
	int mFlowMapStride;
	int mCoarseLevel;
	int mFineLevel;
	//T mRefineThreshold;
	T mParticleLife;
	nanovdb::Vec3R mGravity;
	T mesh_motion_inflow = 0.0;

	int mSampleParticleCount;
	T mRelativeSampleRadius;//k*dx, where dx is the finest level

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
	FluidParams(json& j);

	//initialization
	__hostdev__ int initialLevelTarget(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info) const;
	//set type, velocity, smoke
	//__hostdev__ void setInitialVelocity(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk)const;
	//includes the outer walls of the computational field, but not including the movable mesh inside
	//it will return a cell type in the computational grid only considering wall, fluid, air
	__hostdev__ uint8_t wallCellType(const T current_time, const HATileAccessor<Tile>& acc, const int level, const Coord& g_ijk)const;
	__hostdev__ uint8_t wallCellType(const T current_time, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) const;
	__hostdev__ void setWallCellType(const T current_time, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) const;

	//require coeffs(fluid ratios) are precomputed
	//__hostdev__ void setVelocityBoundaryCondition(const T current_time, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) const;


	//runtime functions
	//__hostdev__ bool isInParticleGenerationRegion(const T current_time, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const Coord& l_ijk)const {
	//	return false;
	//}
	//__device__ uint8_t cellType(const T current_time, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk, int& boundary_axis, int& boundary_off) const {
	//	return DIRICHLET;
	//}

	__hostdev__ Eigen::Transform<T, 3, Eigen::Affine> meshToWorldTransform(const T current_time) const;

	__device__ void addSolidVelocityToFaceCenter(const T current_time, const T dt, const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk, const int axis)const;
};