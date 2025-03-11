#pragma once

#include "GMGSolver.h"
#include "SDFGrid.h"


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace BufChnls {
	constexpr int u = 6;
	constexpr int u_node = 0;
}

namespace AdvChnls {
	constexpr int u = 6;
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
// 0    							x				node u
// 1    							b/r				node v
// 2								p				node w
// 3								Ap				cell u
// 4								z				cell v
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
enum TestCase { KARMAN = 0, SMOKESPHERE };

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
	MaskGridAccessor mMaskGridAccessor;
	SDFGridAccessor mSDFGridAccessor;

	SDFGridAccessor mSDFVelocityAccessors[3];

	//for karman
	T karman_source = 1.0;

	//Vec smokesphere_source = Vec(1.0, 0., 0.);
	Vec smokesphere_source = Vec(0., 0., 0.4);
	Vec smokesphere_center = Vec(0.5, 0.5, 0.3);
	//Vec smokesphere_center = Vec(0.5, 0.5, 0.7);
	T smokesphere_radius = 0.05;

	T nasa_source = 1.0;

	T prop_source = 1.0;
	T f1_source = 1.0;
	T lizard_source = +1.0;
	T fish_source = 1.0;
	T bat_source = 1.0;
	T flamingo_source = 1.0;

	//parameters set by itself
	bool mIsPureNeumann = false;

	FluidParams() {}
	FluidParams(json& j, std::shared_ptr<MaskGrid> mask_grid_ptr, std::shared_ptr<SDFGrid> sdf_float_grid_ptr) {

		std::string test = Json::Value<std::string>(j, "test", "tvortex");
		if (test == "karman") mTestCase = KARMAN;
		else if (test == "smokesphere") mTestCase = SMOKESPHERE;
		else Assert(false, "invalid test {}", test);

		if (mTestCase == KARMAN || mTestCase == SMOKESPHERE) {
			mIsPureNeumann = true;
		}

		mFlowMapStride = Json::Value<int>(j, "flowmap_stride", 5);
		mCoarseLevel = Json::Value<int>(j, "coarse_level", 0);
		mFineLevel = Json::Value<int>(j, "fine_level", 6);
		mRefineThreshold = Json::Value<T>(j, "refine_threshold", 200);
		mGravity = Vec(0, 0, 0);
		mGravity[2] = Json::Value<double>(j, "gravity", -9.8);
		mParticleLife = Json::Value<T>(j, "particle_life", FLT_MAX);

		if (mask_grid_ptr) {
			mMaskGridAccessor = mask_grid_ptr->GetDeviceAccessor();
		}
		if (sdf_float_grid_ptr) {
			mSDFGridAccessor = sdf_float_grid_ptr->GetDeviceAccessor();
		}
	}

	__device__ static bool queryBoundaryDirection(const HATileAccessor<Tile>& acc, const int level, const Coord& g_ijk, int& boundary_axis, int& boundary_off) {
		boundary_axis = -1;
		boundary_off = 0;

		for (int axis : {0, 1, 2}) {
			for (int off : {-1, 1, 2}) {
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

	__device__ static bool queryBoundaryDirection(const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk, int& boundary_axis, int& boundary_off) {
		auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
		return queryBoundaryDirection(acc, info.mLevel, g_ijk, boundary_axis, boundary_off);
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

	__hostdev__ bool isInParticleGenerationRegion(const T current_time, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const Coord& l_ijk)const {
		if (mTestCase == KARMAN) {
			auto pos = acc.cellCenter(info, l_ijk);
			


			//return true;
			const Vec center(0.25, 0.5, 0.5);
			T radius = 0.175;

			//return (0.5-radius <= pos[2] && pos[2] <= 0.5+radius);

			//radius = 0.35;


			if (0.25 <= pos[1] && pos[1] <= 0.75) {
				pos[1] = center[1];
				if ((pos - center).length() <= radius) {
					return true;
				}
			}
			return false;
		}
		else if (mTestCase == SMOKESPHERE) {
			auto pos = acc.cellCenter(info, l_ijk);
			T gen_radius = smokesphere_radius + 0.025;
			if ((pos - smokesphere_center).length() <= gen_radius) {
				return true;
			}
			return false;
		}
		return false;
	}

	__device__ uint8_t cellType(const T current_time, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk, int& boundary_axis, int& boundary_off) const;


	//Physically-based Simulation of Tornadoes, Xiangyang Ding
	__device__ Vec boundaryVelocityDing(const Vec& pos, const int axis, const int sgn) const {
		if (axis == 2) {//bottom/top boundary
			if (sgn == -1) {
				//zero for bottom
				return Vec(0, 0, 0);
			}
			else {
				T r = sqrt(0.05);
				auto x = pos[0];
				auto y = pos[1];

				auto rx2 = ((x - 0.5) / r) * ((x - 0.5) / r);
				auto ry2 = ((y - 0.5) / r) * ((y - 0.5) / r);
				T w = 2 * exp(-(rx2 + ry2));
				return Vec(0, 0, w);
			}
		}
		else {
			auto z = pos[2];
			//auto shear_scale = 2 * (1.0 - z);//bottom magnitude 2, top 0
			//auto shear_scale = (z >= 0.5) ? 1.0 : z * 2;
			auto shear_scale = 2;
			Vec v = Vec(1, 1, 0) * shear_scale;
			if ((axis == 0 && sgn == 1) || (axis == 1 && sgn == 1)) v[0] *= -1;
			if ((axis == 0 && sgn == -1) || (axis == 1 && sgn == 1)) v[1] *= -1;
			return v;
		}
	}

	__device__ Vec boundaryVelocityForceVortex(const Vec& pos, const int axis, const int sgn)const {
		//return boundaryVelocityDing(pos, axis, sgn);
		if (axis == 2) {//bottom/top boundary
			return Vec(0, 0, 0);
		}
		else {
			auto z = pos[2];
			//auto shear_scale = 2 * (1.0 - z) * 2;//bottom magnitude 2, top 0
			//auto shear_scale = 2 * z * 2;//bottom magnitude 0, top 2
			auto shear_scale = 2 * 1 * 2;
			Vec r = pos - Vec(0.5, 0.5, 0);
			r[2] = 0;
			Vec omega(0, 0, shear_scale);
			return omega.cross(r);
		}
	}

	//boundary axis, sgn indicates which face it is on relative to the whole computational domain
	//axis indicates the component to be set
	__device__ void setSlipBoundary(const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk, const int boundary_axis, const int boundary_off, const int axis) const {
		auto& tile = info.tile();
		//suppose boundary_axis=1, sgn=-1, l_ijk=(i,0,k), l1_ijk=(i,1,k)
		Coord l1_ijk = l_ijk; l1_ijk[boundary_axis] -= boundary_off;
		if (axis != boundary_axis) {
			tile(AdvChnls::u + axis, l_ijk) = tile(AdvChnls::u + axis, l1_ijk);
		}
		//otherwise there is a velocity node right on the face, should be either inflow/outflow or DIRICHLET
	}
	//non-slip with given velocity vector
	//vel_func(pos, boundary_axis, boundary_sgn)
	template<class FuncVII>
	__device__ void setFlowBoundary(const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk, const int boundary_axis, const int boundary_off, const FuncVII& vel_func, const int axis) const {
		auto& tile = info.tile();
		//suppose boundary_axis=1, sgn=-1, l_ijk=(i,0,k), l1_ijk=(i,1,k)
		//Coord l1_ijk = l_ijk; l1_ijk[boundary_axis] -= 1 * boundary_sgn;

		if (axis == boundary_axis) {
			if (boundary_off == -1) {
				//set face centers 0,1

				//we will not cross the tile so it's ok to use local coord
				auto l1_ijk = l_ijk; l1_ijk[boundary_axis] += 1;
				auto pos = acc.faceCenter(axis, info, l1_ijk);
				auto vel = vel_func(pos, boundary_axis, boundary_off);
				tile(AdvChnls::u + axis, l1_ijk) = vel[axis];

				Coord l2_ijk = l_ijk; l2_ijk[boundary_axis] += 2;
				tile(AdvChnls::u + axis, l_ijk) = 2 * vel[axis] - tile(AdvChnls::u + axis, l2_ijk);
			}
			else if (boundary_off == 1) {
				//auto pos = acc.faceCenter(axis, info, l_ijk);
				//auto vel = vel_func(pos, boundary_axis, boundary_sgn);
				//tile(AdvChnls::u + axis, l_ijk) = vel[axis];
			}
			else if (boundary_off == 2) {
				//set cells n-1, n-2
				//this cell is n-2

				auto pos = acc.faceCenter(axis, info, l_ijk);
				auto vel = vel_func(pos, boundary_axis, boundary_off);
				tile(AdvChnls::u + axis, l_ijk) = vel[axis];

				//cell n-3
				Coord l3_ijk = l_ijk; l3_ijk[boundary_axis] -= 1;
				Coord l1_ijk = l_ijk; l1_ijk[boundary_axis] += 1;
				tile(AdvChnls::u + axis, l1_ijk) = 2 * vel[axis] - tile(AdvChnls::u + axis, l3_ijk);
			}
		}
		else {
			Coord l1_ijk = l_ijk; l1_ijk[boundary_axis] -= boundary_off;
			auto pos0 = acc.faceCenter(axis, info, l_ijk);
			auto pos1 = acc.faceCenter(axis, info, l1_ijk);
			auto pos = (pos0 + pos1) * 0.5;
			auto vel = vel_func(pos, boundary_axis, boundary_off);
			tile(AdvChnls::u + axis, l_ijk) = 2 * vel[axis] - tile(AdvChnls::u + axis, l1_ijk);
		}
	}



	//__device__ Vec freeVortexLambOseenVelocity()

	__device__ void setBoundaryCondition(const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk, const double current_time) const {
		auto& tile = info.tile();
		if (mTestCase == KARMAN) {
			auto source_vel = karman_source;
			auto inflow_vel_func = [=]__device__(const Vec & pos, const int axis, const int sgn) {
				return Vec(source_vel, 0, 0);
			};
			int boundary_axis, boundary_sgn;
			if (queryBoundaryDirection(acc, info, l_ijk, boundary_axis, boundary_sgn)) {
				for (int axis : {0, 1, 2}) {
					if (axis != boundary_axis) {
						setSlipBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, axis);
					}
					else if (tile.type(l_ijk) == NEUMANN) {
						setFlowBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, inflow_vel_func, axis);
					}
				}
			}
		}
		else if (mTestCase == SMOKESPHERE) {
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

	//gamma: vortex strength
	//radius: radius of the vortex ring
	//delta: thickness of the vortex ring
	//center: center of the vortex ring
	//unit_x: unit vector of vortex in the x direction
	//unit_y: unit vector of vortex in the y direction
	//num_samples: number of sample points on the vortex ring
	//reference: https://github.com/zjw49246/particle-flow-maps/blob/main/3D/init_conditions.py
	__device__ void addVortexRingInitialVelocityAndSmoke(const Vec& pos, double gamma, float radius, float delta, const Vec& center, const Vec& unit_x, const Vec& unit_y, int num_samples, Vec& velocity, T& smoke) const {


		// Curve length per sample point
		float curve_length = (2 * M_PI * radius) / num_samples;

		// Loop through each sample point on the vortex ring
		for (int l = 0; l < num_samples; ++l) {
			float theta = l / float(num_samples) * 2 * M_PI;
			Vec p_sampled = radius * (cos(theta) * unit_x + sin(theta) * unit_y) + center;

			Vec p_diff = pos - p_sampled;
			float r = p_diff.length();

			Vec w_vector = gamma * (-sin(theta) * unit_x + cos(theta) * unit_y);
			float decay_factor = exp(-pow(r / delta, 3));

			// Biot-Savart law contribution
			velocity += curve_length * (-1 / (4 * M_PI * r * r * r)) * (1 - decay_factor) * p_diff.cross(w_vector);

			// Smoke density update based on decay factor
			smoke += (curve_length * decay_factor);
		}
	}

	//set type, velocity, smoke
	__device__ void setInitialCondition(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk)const;
};