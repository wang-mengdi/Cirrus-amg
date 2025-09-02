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
	constexpr int tmp = 3;
	constexpr int sdf = 10;
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
	//int mFlowMapStride;
	int mCoarseLevel;
	int mFineLevel;
	//T mRefineThreshold;
	//T mParticleLife;
	//nanovdb::Vec3R mGravity;
	T mesh_motion_inflow = 1.0;

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

	__hostdev__ int initialLevelTarget(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info) const;
	//includes the outer walls of the computational field, but not including the movable mesh inside
	__device__ uint8_t wallCellType(const T current_time, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) const;
	__device__ void setWallCellType(const T current_time, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) const;

	__device__ void setVelocityBoundaryCondition(const T current_time, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) const;


	//set type, velocity, smoke
	__device__ void setInitialVelocity(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk)const;
};