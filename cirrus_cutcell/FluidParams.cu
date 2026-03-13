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
	else if (test == "aircraft") mTestCase = AIRCRAFT;
	else if (test == "sphere_circling") mTestCase = SPHERECIRCLING;
	else if (test == "jassm") mTestCase = JASSM;
	else ASSERT(false, "invalid test {}", test);

	if (mTestCase == MESHMOTION || mTestCase == AIRCRAFT || mTestCase == SPHERECIRCLING || mTestCase == JASSM) {
		mIsPureNeumann = true;
		mesh_motion_inflow = Json::Value<T>(j, "inflow_velocity", 1.0);
	}

	int init_x = Json::Value<int>(j, "initial_grid_size_x", 1);
	int init_y = Json::Value<int>(j, "initial_grid_size_y", 1);
	int init_z = Json::Value<int>(j, "initial_grid_size_z", 1);
	mInitialGridSize = Coord(init_x, init_y, init_z);

	mFlowMapStride = Json::Value<int>(j, "flowmap_stride", 5);
	mCoarseLevel = Json::Value<int>(j, "coarse_level", 0);
	mFineLevel = Json::Value<int>(j, "fine_level", 6);
	mGravity = Vec(0, 0, 0);
	mGravity[2] = Json::Value<double>(j, "gravity", -9.8);
	mParticleLife = Json::Value<T>(j, "particle_life", FLT_MAX);

	//mSampleNumPerTile = Json::Value<int>(j, "sample_num_per_tile", 128);
	mSampleNumPerCell = Json::Value<int>(j, "sample_num_per_cell", 8);
	mRelativeParticleSampleBandwidth = Json::Value<double>(j, "particle_sample_relative_bandwidth", 5.0);
	mRelativeRefineBandwidth = Json::Value<double>(j, "refine_relative_bandwidth", 10.0);

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
	if (mTestCase == MESHMOTION || mTestCase == AIRCRAFT || mTestCase == SPHERECIRCLING || mTestCase == JASSM) {
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
		const T z0 = 0.3 + jitter;// 0.8;
		const T z1 = 0.3 + jitter;// 0.3;
		const T z = (1 - t) * z0 + t * z1;

		Eigen::Transform<T, 3, Eigen::Affine> transform =
			Eigen::Transform<T, 3, Eigen::Affine>::Identity();

		// Set translation so that the mesh center moves along the z-axis
		transform.translation() = Eigen::Matrix<T, 3, 1>(x, y, z);

		return transform;
	}
	else if (mTestCase == AIRCRAFT) {
		using Vec3 = Eigen::Matrix<T, 3, 1>;
		using AngleAxisT = Eigen::AngleAxis<T>;
		using TransformT = Eigen::Transform<T, 3, Eigen::Affine>;

		T t = current_time;

		// -----------------------------
		// Parameters
		// -----------------------------

		const T x = T(0.5);
		const T y = T(0.5);

		// center translation: z from 1.3 -> 0.7
		const T z0 = T(1.3);
		const T z1 = T(0.7);

		// clamp motion time to [0, 2]
		const T t_clamped = std::max(T(0), std::min(t, T(2)));

		// translation progress in [0, 1]
		const T trans_s = t_clamped / T(2);
		const T z = (T(1) - trans_s) * z0 + trans_s * z1;

		// initial angle of attack in degrees
		const T alpha_deg = -10;   // example: 10 degrees. The sign is flipped, -10 means nose heading up 10 deg.
		const T alpha = alpha_deg * T(M_PI) / T(180);

		// barrel roll: 180 degrees in 2 seconds
		const T omega_deg = T(180);
		const T roll = (t_clamped / T(2)) * (omega_deg * T(M_PI) / T(180));

		TransformT transform = TransformT::Identity();

		// Initial angle of attack:
		// rotate around body right axis (+x)
		Eigen::Matrix<T, 3, 3> R_attack =
			AngleAxisT(alpha, Vec3::UnitX()).toRotationMatrix();

		// Barrel roll around body forward axis.
		// Initially forward is -z, so use local -z axis.
		// Compose on the left so it acts after attack-angle orientation.
		Eigen::Matrix<T, 3, 3> R_roll =
			AngleAxisT(roll, -Vec3::UnitZ()).toRotationMatrix();

		transform.linear() = R_roll * R_attack;
		transform.translation() = Vec3(x, y, z);

		return transform;
	}
	if (mTestCase == SPHERECIRCLING) {
		// Motion duration: 2 seconds
		// z: 1.7 -> 0.3
		// x-y: circle around (0.5, 0.5) with radius 0.15, one full revolution in 2s

		T t = current_time;
		if (t < (T)0) t = (T)0;
		if (t > (T)2) t = (T)2;

		const T center_x = (T)0.5;
		const T center_y = (T)0.5;
		const T orbit_r = (T)0.2;

		const T z0 = (T)1.7;
		const T z1 = (T)0.3;
		const T s = t / (T)2.0;

		const T theta = (T)(2.0 * M_PI) * s;

		const T x = center_x + orbit_r * std::cos(theta);
		const T y = center_y + orbit_r * std::sin(theta);
		const T z = ((T)1 - s) * z0 + s * z1;

		Eigen::Transform<T, 3, Eigen::Affine> transform =
			Eigen::Transform<T, 3, Eigen::Affine>::Identity();

		transform.translation() = Eigen::Matrix<T, 3, 1>(x, y, z);

		return transform;
	}
	else if (mTestCase == JASSM) {
		using Vec3 = Eigen::Matrix<T, 3, 1>;
		using Mat3 = Eigen::Matrix<T, 3, 3>;
		using AngleAxisT = Eigen::AngleAxis<T>;
		using TransformT = Eigen::Transform<T, 3, Eigen::Affine>;

		T t = current_time;
		if (t < (T)0) t = (T)0;
		if (t > (T)2) t = (T)2;

		// -----------------------------
		// Trajectory: one circle in x-y and descend in z
		// -----------------------------
		const T center_x = (T)0.5;
		const T center_y = (T)0.5;
		const T orbit_r = (T)0.25;

		const T z0 = (T)1.7;
		const T z1 = (T)0.3;

		const T s = t / (T)2.0;
		const T theta = (T)(2.0 * M_PI) * s;

		const T x = center_x + orbit_r * std::cos(theta);
		const T y = center_y + orbit_r * std::sin(theta);
		const T z = ((T)1 - s) * z0 + s * z1;

		Vec3 pos(x, y, z);

		// -----------------------------
		// Model local frame:
		//   local -Z = forward
		//   local +X = up
		//   local +Y = right
		//
		// Desired initial world frame:
		//   forward -> world -Z
		//   up      -> world +Y
		//   right   -> world +X
		// -----------------------------
		Vec3 forward_base((T)0, (T)0, (T)-1);
		Vec3 up_base((T)1, (T)0, (T)0);
		Vec3 right_base((T)0, (T)-1, (T)0);

		// -----------------------------
		// True barrel roll: 360 deg in 2 seconds
		// t = 1 -> 180 deg
		// -----------------------------
		const T roll = s * (T)(2.0 * M_PI);
		Mat3 R_roll = AngleAxisT(roll, forward_base).toRotationMatrix();

		Vec3 up_rolled = (R_roll * up_base).normalized();
		Vec3 right_rolled = (R_roll * right_base).normalized();
		Vec3 forward_rolled = forward_base;

		// -----------------------------
		// Optional angle of attack
		// Positive alpha pitches nose upward in the plane spanned by forward/up
		// -----------------------------
		const T alpha_deg = (T)10;   // set to 0 if you want no attack angle
		const T alpha = alpha_deg * (T)M_PI / (T)180.0;

		Mat3 R_attack = AngleAxisT(-alpha, right_rolled).toRotationMatrix();

		Vec3 forward_final = (R_attack * forward_rolled).normalized();
		Vec3 up_final = (R_attack * up_rolled).normalized();

		// Rebuild right to avoid tiny drift, keeping a right-handed frame
		Vec3 right_final = forward_final.cross(up_final).normalized();
		up_final = right_final.cross(forward_final).normalized();

		// -----------------------------
		// Map model local axes to world axes
		// local +X -> up
		// local +Y -> right
		// local +Z -> backward = -forward
		// -----------------------------
		Mat3 R;
		R.col(0) = up_final;
		R.col(1) = -right_final;
		R.col(2) = -forward_final;

		TransformT transform = TransformT::Identity();
		transform.linear() = R;
		transform.translation() = pos;
		return transform;
	}
	else {
		CUDA_ASSERT(false, "meshToWorldTransform not implemented for test case %d", int(mTestCase));
	}
}

__device__ T FluidParams::solidFaceCenterVelocity(const T current_time, const T dt, const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk, const int axis) const
{
	if (mTestCase == MESHMOTION || mTestCase == AIRCRAFT || mTestCase == SPHERECIRCLING || mTestCase == JASSM) {
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