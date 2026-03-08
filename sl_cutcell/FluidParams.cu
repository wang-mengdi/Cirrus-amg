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
__hostdev__ bool QueryBoundaryDirectionN1P1OnCoarseLevel(const HATileAccessor<Tile>& acc, int coarse_chk_level, int fine_cell_level, Coord g_ijk, int& boundary_axis, int& boundary_off) {
	CUDA_ASSERT(coarse_chk_level <= fine_cell_level, "QueryBoundaryDirectionN1P1OnCoarseLevel error");
	//coarse_chk_level = min(coarse_chk_level, info.mLevel);
	int level_diff = fine_cell_level - coarse_chk_level;


	g_ijk = Coord(g_ijk[0] >> level_diff, g_ijk[1] >> level_diff, g_ijk[2] >> level_diff);

	//printf("level %d chk level %d diff %d g_ijk %d %d %d\n", info.mLevel, chk_level, level_diff, g_ijk[0], g_ijk[1], g_ijk[2]);
	return QueryBoundaryDirectionN1P1(acc, coarse_chk_level, g_ijk, boundary_axis, boundary_off);
}

//the actual level might be finer than the checking level
//it's for like building a wall with respect to the coarse level
__hostdev__ bool QueryBoundaryDirectionN1P1OnCoarseLevel(const HATileAccessor<Tile>& acc, int coarse_chk_level, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk, int& boundary_axis, int& boundary_off) {
	auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
	return QueryBoundaryDirectionN1P1OnCoarseLevel(acc, coarse_chk_level, info.mLevel, g_ijk, boundary_axis, boundary_off);
}

FluidParams::FluidParams(json& j)
{

	std::string test = Json::Value<std::string>(j, "test", "meshmotion");
	if (test == "meshmotion") mTestCase = MESHMOTION;
	else ASSERT(false, "invalid test {}", test);

	if (mTestCase == MESHMOTION) {
		mIsPureNeumann = true;
		mesh_motion_inflow = Json::Value<T>(j, "inflow_velocity", 1.0);
	}

	mFlowMapStride = Json::Value<int>(j, "flowmap_stride", 5);
	mCoarseLevel = Json::Value<int>(j, "coarse_level", 0);
	mFineLevel = Json::Value<int>(j, "fine_level", 6);
	mGravity = Vec(0, 0, 0);
	mGravity[2] = Json::Value<double>(j, "gravity", -9.8);
	mParticleLife = Json::Value<T>(j, "particle_life", FLT_MAX);

	mSampleNumPerTile = Json::Value<int>(j, "sample_num_per_tile", 128);
	mRelativeSampleBandwidth = Json::Value<double>(j, "relative_sample_bandwidth", 5.0);

	mExtrapolationIters = Json::Value<int>(j, "extrapolation_iters", 5);
}


__hostdev__ int FluidParams::initialLevelTarget(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info) const {
	return mCoarseLevel;
}

//__device__ void FluidParams::addInitialVelocityToFaceCenter(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) const {
//
//	if (mTestCase == MESHMOTION) {
//
//
//		Vec initial_vel = Vec(0, 0, mesh_motion_inflow);
//
//		Tile& tile = info.tile();
//		for (int axis : {0, 1, 2}) {
//			cuda_vec4_t<T> sdfs = FaceCornerSDFs(BufChnls::sdf, acc, info, l_ijk, axis);
//
//			auto alpha = FaceFluidRatio(sdfs);
//
//			tile(BufChnls::u + axis, l_ijk) = alpha * initial_vel[axis];
//		}
//		//int boundary_axis, boundary_off;
//		//tile.type(l_ijk) = cellType(current_time, acc, info, l_ijk, boundary_axis, boundary_off);
//	}
//}


__hostdev__ uint8_t FluidParams::wallCellType(const T current_time, const HATileAccessor<Tile>& acc, const int level, const Coord& g_ijk) const
{
	if (mTestCase == MESHMOTION) {
		//uses 1 layer of solid walls on the coarse level

		//z+ air
		//other walls
		bool is_neumann = false;
		bool is_dirichlet = false;
		int boundary_axis, boundary_off;

		//int boundary_axis, boundary_off;
		if (QueryBoundaryDirectionN1P1OnCoarseLevel(acc, mCoarseLevel, level, g_ijk, boundary_axis, boundary_off)) {
			////z+ dirichlet and other neumann
			//if (boundary_axis == 2 && boundary_off == 1) {
			//	is_dirichlet = true;
			//}
			//else is_neumann = true;

			//pure neumann
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

__hostdev__ uint8_t FluidParams::wallCellType(const T current_time, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) const {
	auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
	return wallCellType(current_time, acc, info.mLevel, g_ijk);
}



__hostdev__ void FluidParams::setWallCellType(const T current_time, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) const
{
	auto& tile = info.tile();
	tile.type(l_ijk) = wallCellType(current_time, acc, info, l_ijk);
}

__hostdev__ Eigen::Transform<T, 3, Eigen::Affine> FluidParams::meshToWorldTransform(const T current_time) const
{
	if (mTestCase == MESHMOTION) {
		// Clamp time parameter to [0, 1]
		T t = current_time;

		// Linear interpolation of center position over time:
		// t = 0 -> (0.5, 0.5, 0.8)
		// t = 1 -> (0.5, 0.5, 0.3)
		double jitter = 2.9e-4;//to avoid the mesh sdf exactly on the grid points, which can cause issues for some of the boundary condition implementations
		const T x = 0.5 + jitter;
		const T y = 0.5 + jitter;
		const T z0 = 0.8 + jitter;// 0.8;
		const T z1 = 0.3 + jitter;// 0.3;
		const T z = (1 - t) * z0 + t * z1;

		Eigen::Transform<T, 3, Eigen::Affine> transform =
			Eigen::Transform<T, 3, Eigen::Affine>::Identity();

		// Set translation so that the mesh center moves along the z-axis
		transform.translation() = Eigen::Matrix<T, 3, 1>(x, y, z);

		return transform;
	}
	else {
		CUDA_ASSERT(false, "meshToWorldTransform not implemented for test case %d", int(mTestCase));
	}
}

__device__ T FluidParams::solidFaceCenterVelocity(const T current_time, const T dt, const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk, const int axis) const
{
	if (mTestCase == MESHMOTION) {
		Vec wall_vel(0, 0, mesh_motion_inflow);
		auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
		auto ng_ijk = g_ijk; ng_ijk[axis]--;
		if (wallCellType(current_time, acc, info.mLevel, g_ijk) == NEUMANN || wallCellType(current_time, acc, info.mLevel, ng_ijk) == NEUMANN) {
			return wall_vel[axis];
		}
		else {
			//mesh rigid velocity
			auto pos0 = acc.faceCenter(axis, info, l_ijk);

			auto T0 = meshToWorldTransform(current_time);
			auto T1 = meshToWorldTransform(current_time + dt);

			Eigen::Matrix<T, 3, 1> p0(pos0[0], pos0[1], pos0[2]);
			Eigen::Matrix<T, 3, 1> p1 = T1 * (T0.inverse() * p0);

			nanovdb::Vec3<T> pos1(p1[0], p1[1], p1[2]);
			auto rigid_vel = (pos1 - pos0) / dt;

			return rigid_vel[axis];
		}
	}
	else {
		CUDA_ASSERT(false);
		return 0;
	}
}

//__device__ void FluidParams::addSolidVelocityToFaceCenter(const T current_time, const T dt, const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk, const int axis) const
//{
//	if (mTestCase == MESHMOTION) {
//		Vec wall_vel(0, 0, mesh_motion_inflow);
//		T solid_ratio = 0, solid_vel = 0;
//
//		auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
//		auto ng_ijk = g_ijk; ng_ijk[axis]--;
//		if (wallCellType(current_time, acc, info.mLevel, g_ijk) == NEUMANN || wallCellType(current_time, acc, info.mLevel, ng_ijk) == NEUMANN) {
//			//it's a wall boundary NEUMANN cell
//			solid_ratio = 1;
//			solid_vel = wall_vel[axis];
//		}
//		else {
//			//it either intersects with the mesh sdf or not
//			cuda_vec4_t<T> sdfs = FaceCornerSDFs(BufChnls::sdf, acc, info, l_ijk, axis);
//			auto h = acc.voxelSize(info.mLevel);
//			if (FaceSDFAllOutside<T>(sdfs, h * SDF_REL_EPS)) {//does not intersect, no solid part
//				solid_ratio = 0;
//				solid_vel = 0;
//			}
//			else {
//				auto pos0 = acc.faceCenter(axis, info, l_ijk);
//
//				auto T0 = meshToWorldTransform(current_time);
//				auto T1 = meshToWorldTransform(current_time + dt);
//
//				Eigen::Matrix<T, 3, 1> p0(pos0[0], pos0[1], pos0[2]);
//				Eigen::Matrix<T, 3, 1> p1 = T1 * (T0.inverse() * p0);
//
//				nanovdb::Vec3<T> pos1(p1[0], p1[1], p1[2]);
//				auto rigid_vel = (pos1 - pos0) / dt;
//
//
//				solid_ratio = 1 - FaceFluidRatio(sdfs);
//				solid_vel = rigid_vel[axis];
//			}
//		}
//
//		auto& tile = info.tile();
//		tile(BufChnls::u + axis, l_ijk) += solid_ratio * solid_vel;
//	}
//	else {
//		CUDA_ASSERT(false, "solidVelocityAtFaceCenter not implemented for test case %d", int(mTestCase));
//	}
//}