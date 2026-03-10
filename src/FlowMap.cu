#include "FlowMap.h"
#include "PoissonIOFunc.h"
#include <polyscope/polyscope.h>

//quadratic kernel used by MPM and PFM
//does not guard 1.5
__hostdev__ T QuadraticKernelFast(const T x) {
	T abs_x = fabsf(x);
	T ret = abs_x < 0.5f ? (0.75f - abs_x * abs_x) : (0.5f * (1.5f - abs_x) * (1.5f - abs_x));
	return ret;
}
__hostdev__ T QuadraticKernel(const T x) {
	//float abs_x = fabsf(x);
	//float ret = abs_x < 0.5f ? (0.75f - abs_x * abs_x) : (0.5f * (1.5f - abs_x) * (1.5f - abs_x));
	//return ret;

	T abs_x = fabsf(x);
	if (abs_x < 0.5f) return 3.0f / 4.f - x * x;
	else if (abs_x < 1.5f) return 0.5f * (1.5f - abs_x) * (1.5f - abs_x);
	return 0;
}
__hostdev__ T QuadraticKernelDerivativeFast(const T x) {
	T abs_x = fabsf(x);
	T ret = abs_x < 0.5f ? (-2.0f * x) : (x - copysignf(1.5f, x));
	return ret;
}
__hostdev__ T QuadraticKernelDerivative(T x) {
	//float abs_x = fabsf(x);
	//float ret = abs_x < 0.5f ? (-2.0f * x) : (x - 1.5f * copysignf(1.0f, x));
	//return ret;

	T abs_x = fabsf(x);
	if (0.f <= abs_x && abs_x < 0.5f) return -2.f * x;
	if (0.5f <= abs_x && abs_x < 1.5f) return (x - copysignf(1.5f, x));
	return 0;
}
//__hostdev__ T QuadraticKernel(const Vec& r) {
//	return QuadraticKernel(r[0]) * QuadraticKernel(r[1]) * QuadraticKernel(r[2]);
//}
//__hostdev__ Vec QuadraticKernelGradient(const Vec& r) {
//	auto xp = r[0];
//	auto yp = r[1];
//	auto zp = r[2];
//	return Vec(
//		QuadraticKernelDerivative(xp) * QuadraticKernel(yp) * QuadraticKernel(zp),
//		QuadraticKernel(xp) * QuadraticKernelDerivative(yp) * QuadraticKernel(zp),
//		QuadraticKernel(xp) * QuadraticKernel(yp) * QuadraticKernelDerivative(zp)
//	);
//}
//__hostdev__ Vec QuadraticKernelGradient(const Vec& w, const Vec& dw) {
//	return Vec(
//		dw[0] * w[1] * w[2],
//		w[0] * dw[1] * w[2],
//		w[0] * w[1] * dw[2]
//	);
//}

__hostdev__ T QuadraticKernel(const Vec& r, T h) {
	return QuadraticKernel(r[0] / h) * QuadraticKernel(r[1] / h) * QuadraticKernel(r[2] / h);
}
__hostdev__ Vec QuadraticKernelGradient(Vec r, T h) {
	auto xp = r[0] / h;
	auto yp = r[1] / h;
	auto zp = r[2] / h;
	return Vec(
		QuadraticKernelDerivative(xp) * QuadraticKernel(yp) * QuadraticKernel(zp) / h,
		QuadraticKernel(xp) * QuadraticKernelDerivative(yp) * QuadraticKernel(zp) / h,
		QuadraticKernel(xp) * QuadraticKernel(yp) * QuadraticKernelDerivative(zp) / h
	);
}

//using quadratic kernel and calculate directly on MAC grid
//use face center values
//here info, l_ijk indicates which voxel 
__device__ bool KernelIntpVelocityComponentMAC2(const HATileAccessor<Tile>& acc, const int level, const Vec& pos, const int axis, const int u_channel, T& u_i) {
	//we're actually interpoating on the lattice grid of velocity
	T h = acc.voxelSize(level);
	//our kernel is truncated at 1.5
	//therefore, one point has 3*3*3 non-zero lattice neighbors
	Vec test_pos = pos - Vec(0.5 * h, 0.5 * h, 0.5 * h);
	for (int ii : {0, 1, 2}) {
		if (ii != axis) test_pos[ii] -= 0.5 * h;
	}
	//HATileInfo<Tile> info; Coord l_ijk; Vec frac;
	//acc.findLeafVoxelAndFrac(test_pos, info, l_ijk, frac);
	//auto g_ijk = acc.localToGlobalCoord(info, l_ijk);

	Coord g_ijk; Vec frac;
	acc.worldToVoxelAndFraction(level, test_pos, g_ijk, frac);

	//for example, if axis==0, then test_pos=pos-Vec(0.5h, h, h)
	//relative to the cell min, facex should be (0,0.5h,0.5h)
	//that means, the relative position of pos to the face center should be frac+(0.5,0.5,0.5)
	frac += Vec(0.5, 0.5, 0.5);

	u_i = 0;

	for (int offi = 0; offi < 3; offi++) {
		T wi = QuadraticKernel(frac[0] - offi);
		T dwi = QuadraticKernelDerivative(frac[0] - offi) / h;
		for (int offj = 0; offj < 3; offj++) {
			T wj = QuadraticKernel(frac[1] - offj);
			T dwj = QuadraticKernelDerivative(frac[1] - offj) / h;
			for (int offk = 0; offk < 3; offk++) {
				T wk = QuadraticKernel(frac[2] - offk);
				T dwk = QuadraticKernelDerivative(frac[2] - offk) / h;

				auto ng_ijk = g_ijk + Coord(offi, offj, offk);
				HATileInfo<Tile> ninfo; Coord nl_ijk; Vec n_frac;
				acc.findVoxel(level, ng_ijk, ninfo, nl_ijk);
				if (!ninfo.empty() && !(ninfo.mType & GHOST)) {
					auto& ntile = ninfo.tile();
					auto n_u_i = ntile(u_channel + axis, nl_ijk);

					T w = wi * wj * wk;
					Vec dw = Vec(dwi * wj * wk, wi * dwj * wk, wi * wj * dwk);

					u_i += w * n_u_i;
				}
				else {
					return false;
				}
			}
		}
	}

	return true;
}

//using quadratic kernel and calculate directly on MAC grid
//use face center values
//here info, l_ijk indicates which voxel 
__device__ bool KernelIntpVelocityComponentAndGradientMAC2(const HATileAccessor<Tile>& acc, const int level, const Vec& pos, const int axis, const int u_channel, T& u_i, Vec& gradu_i) {
	//we're actually interpoating on the lattice grid of velocity
	T h = acc.voxelSize(level);
	//our kernel is truncated at 1.5
	//therefore, one point has 3*3*3 non-zero lattice neighbors
	Vec test_pos = pos - Vec(0.5 * h, 0.5 * h, 0.5 * h);
	for (int ii : {0, 1, 2}) {
		if (ii != axis) test_pos[ii] -= 0.5 * h;
	}
	//HATileInfo<Tile> info; Coord l_ijk; Vec frac;
	//acc.findLeafVoxelAndFrac(test_pos, info, l_ijk, frac);
	//auto g_ijk = acc.localToGlobalCoord(info, l_ijk);

	Coord g_ijk; Vec frac;
	acc.worldToVoxelAndFraction(level, test_pos, g_ijk, frac);

	//for example, if axis==0, then test_pos=pos-Vec(0.5h, h, h)
	//relative to the cell min, facex should be (0,0.5h,0.5h)
	//that means, the relative position of pos to the face center should be frac+(0.5,0.5,0.5)
	frac += Vec(0.5, 0.5, 0.5);

	u_i = 0;
	gradu_i = Vec(0, 0, 0);
	//bool is_valid = true;

	for (int offi = 0; offi < 3; offi++) {
		T wi = QuadraticKernel(frac[0] - offi);
		T dwi = QuadraticKernelDerivative(frac[0] - offi) / h;
		for (int offj = 0; offj < 3; offj++) {
			T wj = QuadraticKernel(frac[1] - offj);
			T dwj = QuadraticKernelDerivative(frac[1] - offj) / h;
			for (int offk = 0; offk < 3; offk++) {
				T wk = QuadraticKernel(frac[2] - offk);
				T dwk = QuadraticKernelDerivative(frac[2] - offk) / h;

				auto ng_ijk = g_ijk + Coord(offi, offj, offk);
				HATileInfo<Tile> ninfo; Coord nl_ijk; Vec n_frac;
				acc.findVoxel(level, ng_ijk, ninfo, nl_ijk);
				if (!ninfo.empty() && !(ninfo.mType & GHOST)) {
					auto& ntile = ninfo.tile();
					auto n_u_i = ntile(u_channel + axis, nl_ijk);


					//{
					//	auto ng_ijk = acc.localToGlobalCoord(ninfo, nl_ijk);
					//	printf("axis %d accessing level %d ng_ijk %d %d %d n_u_i %f\n", axis, level, ng_ijk[0], ng_ijk[1], ng_ijk[2], n_u_i);
					//}


					T w = wi * wj * wk;
					Vec dw = Vec(dwi * wj * wk, wi * dwj * wk, wi * wj * dwk);

					u_i += w * n_u_i;
					gradu_i += dw * n_u_i;
				}
				else {
					return false;
				}
			}
		}
	}

	return true;
}

__device__ bool KernelIntpVelocityMAC2AtGivenLevel(const HATileAccessor<Tile>& acc, const int level, const Vec& pos, const int u_channel, Vec& vel) {
	for (int axis : {0, 1, 2}) {
		if (!KernelIntpVelocityComponentMAC2(acc, level, pos, axis, u_channel, vel[axis])) return false;
	}
	return true;
}

__device__ bool KernelIntpVelocityAndJacobianMAC2AtGivenLevel(const HATileAccessor<Tile>& acc, const int level, const Vec& pos, const int u_channel, Vec& vel, Eigen::Matrix3<T>& jacobian) {
	//set zero
	//vel = Vec(0, 0, 0);
	//jacobian.setZero();

	for (int axis : {0, 1, 2}) {
		Vec gradu_i;
		if (!KernelIntpVelocityComponentAndGradientMAC2(acc, level, pos, axis, u_channel, vel[axis], gradu_i)) return false;
		for (int t : {0, 1, 2}) {
			jacobian(axis, t) = gradu_i[t];
		}
	}

	//printf("flowmap intp pos %f %f %f level %d vel %f %f %f\n", pos[0], pos[1], pos[2], level, vel[0], vel[1], vel[2]);

	return true;
}

__device__ bool KernelIntpVelocityMAC2(const HATileAccessor<Tile>& acc, const int fine_level, const int coarse_level, const Vec& pos, const int u_channel, Vec& vel) {
	for (int i = fine_level; i >= coarse_level; i--) {
		if (KernelIntpVelocityMAC2AtGivenLevel(acc, i, pos, u_channel, vel)) {
			return true;
		}
	}

	vel = Vec(0, 0, 0);
	return false;
}

__device__ bool KernelIntpVelocityAndJacobianMAC2(const HATileAccessor<Tile>& acc, const int fine_level, const int coarse_level, const Vec& pos, const int u_channel, Vec& vel, Eigen::Matrix3<T>& jacobian) {
	for (int i = fine_level; i >= coarse_level; i--) {
		//printf("flowmap intp pos %f %f %f level %d\n", pos[0], pos[1], pos[2], i);

		if (KernelIntpVelocityAndJacobianMAC2AtGivenLevel(acc, i, pos, u_channel, vel, jacobian)) {


			{
				for (int ii = 0; ii < 3; ii++) {
					CUDA_ASSERT(isfinite(vel[ii]), "vel[%d]=%f at i=%d", ii, vel[ii], i);
				}

				// iterate over 3*3 and ASSERT matT elements are finite
				for (int ii = 0; ii < 3; ii++) {
					for (int jj = 0; jj < 3; jj++) {
						CUDA_ASSERT(isfinite(jacobian(ii, jj)), "jacobian(%d,%d)=%f at i=%d", ii, jj, jacobian(ii, jj), i);
					}
				}
			}

			return true;
		}
	}

	vel = Vec(0, 0, 0);
	#pragma unroll
	for (int k = 0; k < 9; k++) // 3 by 3
		jacobian.data()[k] = T(0);
	return false;
}

////dual operation of KernelIntpVelocityComponentAndGradientMAC2
//__device__ void KernelScatterVelocityComponentMAC2(const HATileAccessor<Tile>& acc, const int level, const int axis, const int u_channel, const int uw_channel, const Vec& pos, const Vec& vel, const Eigen::Matrix3<T>& gradu) {
//	T h = acc.voxelSize(level);
//	Vec test_pos = pos - Vec(0.5 * h, 0.5 * h, 0.5 * h);
//	for (int ii : {0, 1, 2}) {
//		if (ii != axis) test_pos[ii] -= 0.5 * h;
//	}
//
//	HATileInfo<Tile> info; Coord l_ijk; Vec frac;
//	acc.findLeafVoxelAndFrac(test_pos, info, l_ijk, frac);
//
//	auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
//
//	for (int offi = 0; offi < 3; offi++) {
//		for (int offj = 0; offj < 3; offj++) {
//			for (int offk = 0; offk < 3; offk++) {
//				auto ng_ijk = g_ijk + Coord(offi, offj, offk);
//
//				HATileInfo<Tile> ninfo;	Coord nl_ijk;
//				acc.findVoxel(level, ng_ijk, ninfo, nl_ijk);
//
//				if (!ninfo.empty()) {
//					auto& ntile = ninfo.tile();
//
//					// relative vector between face center and the particle
//					Vec fpos = acc.faceCenter(axis, ninfo, nl_ijk);
//					Vec dpos = pos - fpos;
//
//					auto u = vel + MatrixTimesVec(gradu, fpos - pos);
//
//					T weight = QuadraticKernel(dpos, h);
//
//					atomicAdd(&(ntile(u_channel + axis, nl_ijk)), weight * u[axis]);
//					atomicAdd(&(ntile(uw_channel + axis, nl_ijk)), weight);
//				}
//			}
//		}
//	}
//}

__device__ void KernelScatterVelocityComponentMAC2(
	const HATileAccessor<Tile>& acc, const int level, const int axis, const int u_channel, const int uw_channel,
	const Vec& pos, const Vec& vel, const Eigen::Matrix3<T>& gradu) {

	T h = acc.voxelSize(level);

	// Calculate test position adjusted for face-center alignment on the specified axis
	Vec test_pos = pos - Vec(0.5 * h, 0.5 * h, 0.5 * h);
	for (int ii : {0, 1, 2}) {
		if (ii != axis) test_pos[ii] -= 0.5 * h;
	}

	// Convert world position to voxel coordinates and fractional part
	Coord g_ijk; Vec frac;
	acc.worldToVoxelAndFraction(level, test_pos, g_ijk, frac);

	// Adjust fractional coordinates for face alignment
	frac += Vec(0.5, 0.5, 0.5);

	// Iterate over the 3x3x3 neighborhood
	for (int offi = 0; offi < 3; ++offi) {
		T wi = QuadraticKernel(frac[0] - offi);
		for (int offj = 0; offj < 3; ++offj) {
			T wj = QuadraticKernel(frac[1] - offj);
			for (int offk = 0; offk < 3; ++offk) {
				T wk = QuadraticKernel(frac[2] - offk);

				// Compute total weight
				T weight = wi * wj * wk;

				// Global voxel coordinate
				auto ng_ijk = g_ijk + Coord(offi, offj, offk);

				// Find tile and local voxel index
				HATileInfo<Tile> ninfo; Coord nl_ijk;
				acc.findVoxel(level, ng_ijk, ninfo, nl_ijk);

				if (!ninfo.empty()) {
					auto& ntile = ninfo.tile();

					// Compute relative position for velocity and weight scattering
					Vec fpos = acc.faceCenter(axis, ninfo, nl_ijk);
					//Vec dpos = pos - fpos;

					// Compute velocity at the face
					auto u = vel + MatrixTimesVec(gradu, fpos - pos);

					// Scatter velocity and weight
					atomicAdd(&(ntile(u_channel + axis, nl_ijk)), weight * u[axis]);
					atomicAdd(&(ntile(uw_channel + axis, nl_ijk)), weight);

					//{
					//	//if (axis == 2 && ng_ijk == Coord(1,0,0))
					//	{
					//		printf("scatter axis %d level %d pos %f %f %f fpos %f %f %f vel %f %f %f u %f w %f g_ijk %d %d %d to ng_ijk %d %d %d\n", axis, level, pos[0], pos[1], pos[2], fpos[0], fpos[1], fpos[2], vel[0], vel[1], vel[2], u[axis], weight, g_ijk[0], g_ijk[1], g_ijk[2], ng_ijk[0], ng_ijk[1], ng_ijk[2]);
					//	}
					//}

					//if (u[axis] > 1e5) {
					//	printf("scatter axis %d level %d pos %f %f %f fpos %f %f %f dpos %f %f %f vel %f %f %f u %f gradm norm %f\n", axis, level, pos[0], pos[1], pos[2], fpos[0], fpos[1], fpos[2], dpos[0], dpos[1], dpos[2], vel[0], vel[1], vel[2], u[axis], gradu.norm());
					//}
				}
			}
		}
	}
}


__device__ void KernelScatterVelocityMAC2(const HATileAccessor<Tile>& acc, const int u_channel, const int uw_channel, const Vec& pos, const Vec& vec, const Eigen::Matrix3<T>& gradv) {
	//find the voxel that contains the position
	HATileInfo<Tile> info; Coord l_ijk; Vec frac;
	acc.findLeafVoxelAndFrac(pos, info, l_ijk, frac);

	if (info.empty()) return;

	for (int axis : {0, 1, 2}) {
		KernelScatterVelocityComponentMAC2(acc, info.mLevel, axis, u_channel, uw_channel, pos, vec, gradv);
	}
}

//__device__ void VelocityJacobian(const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const Coord& l_ijk, const Vec& pos, const int node_u_channel, const T h, Eigen::Matrix3<T>& jacobian) {
//	auto& tile = info.tile();
//	jacobian.setZero();
//	for (int offi : {0, 1}) {
//		for (int offj : {0, 1}) {
//			for (int offk : {0, 1}) {
//				Coord r_ijk = l_ijk + Coord(offi, offj, offk);
//				for (int axis : {0, 1, 2}) {
//					Vec rpos = acc.cellCorner(info, r_ijk);
//					Vec dpos = pos - rpos;
//					Vec dw = QuadraticKernelGradient(dpos, h);
//					for (int t : {0, 1, 2}) {
//						jacobian(axis, t) += dw[t] * tile.node(node_u_channel + axis, r_ijk);
//					}
//				}
//			}
//		}
//	}
//}

//__device__ void VelocityAndJacobian(const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const Coord& l_ijk, const Vec& pos, const int node_u_channel, const T h, Vec& vel, Eigen::Matrix3<T>& jacobian) {
//	auto& tile = info.tile();
//	vel = Vec(0, 0, 0);
//	jacobian.setZero();
//	for (int offi : {0, 1}) {
//		for (int offj : {0, 1}) {
//			for (int offk : {0, 1}) {
//				Coord r_ijk = l_ijk + Coord(offi, offj, offk);
//				for (int axis : {0, 1, 2}) {
//					Vec rpos = acc.cellCorner(info, r_ijk);
//					Vec dpos = pos - rpos;
//
//					vel[axis] += QuadraticKernel(dpos, h) * tile.node(node_u_channel + axis, r_ijk);
//
//
//					Vec dw = QuadraticKernelGradient(dpos, h);
//					for (int t : {0, 1, 2}) {
//						jacobian(axis, t) += dw[t] * tile.node(node_u_channel + axis, r_ijk);
//					}
//				}
//			}
//		}
//	}
//}
//
//
//
//
//__device__ Eigen::Matrix3<T> VelocityJacobian(const HATileAccessor<Tile>& acc, const Vec& pos, const int node_u_channel) {
//	Eigen::Matrix3<T> jacobian;
//	jacobian.setZero();
//	HATileInfo<Tile> info; Coord l_ijk; Vec frac;
//	acc.findLeafVoxelAndFrac(pos, info, l_ijk, frac);
//	if (!info.empty()) {
//		VelocityJacobian(acc, info, l_ijk, pos, node_u_channel, acc.voxelSize(info), jacobian);
//	}
//	return jacobian;
//}
//
//__device__ void VelocityAndJacobian(const HATileAccessor<Tile>& acc, const Vec& pos, const int node_u_channel, Vec& vel, Eigen::Matrix3<T>& jacobian) {
//	vel = Vec(0, 0, 0);
//	jacobian.setZero();
//	HATileInfo<Tile> info; Coord l_ijk; Vec frac;
//	acc.findLeafVoxelAndFrac(pos, info, l_ijk, frac);
//	if (!info.empty()) {
//		VelocityAndJacobian(acc, info, l_ijk, pos, node_u_channel, acc.voxelSize(info), vel, jacobian);
//	}
//}

__hostdev__ Vec MatrixTimesVec(const Eigen::Matrix3<T>& A, const Vec& b) {
	return Vec(
		A(0, 0) * b[0] + A(0, 1) * b[1] + A(0, 2) * b[2],
		A(1, 0) * b[0] + A(1, 1) * b[1] + A(1, 2) * b[2],
		A(2, 0) * b[0] + A(2, 1) * b[1] + A(2, 2) * b[2]
	);
}

//void InterpolateFaceVelocitiesAtAllTiles(HADeviceGrid<Tile>& grid, const int face_u_channel, const int leaf_node_u_channel)
//{
//
//	CalcLeafNodeValuesFromFaceCenters(grid, face_u_channel, leaf_node_u_channel);
//
//	grid.launchVoxelFuncOnAllTiles(
//		[=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
//		auto& tile = info.tile();
//		for (int axis : {0, 1, 2}) {
//			auto fpos = acc.faceCenter(axis, info, l_ijk);
//			auto vel = InterpolateFaceValue(acc, fpos, face_u_channel, leaf_node_u_channel);
//			tile(face_u_channel + axis, l_ijk) = vel[axis];
//		}
//	}, NONLEAF | GHOST, 4
//	);
//}

__device__ bool RK4ForwardPosition(const HATileAccessor<Tile>& acc, const int fine_level, const int coarse_level, const T dt, const int u_channel, Vec& phi) {
	Vec u1;
	if (!KernelIntpVelocityMAC2(acc, fine_level, coarse_level, phi, u_channel, u1)) return false;
	Vec phi1 = phi + 0.5 * dt * u1;

	Vec u2;
	if (!KernelIntpVelocityMAC2(acc, fine_level, coarse_level, phi1, u_channel, u2)) return false;
	Vec phi2 = phi + 0.5 * dt * u2;

	Vec u3;
	if (!KernelIntpVelocityMAC2(acc, fine_level, coarse_level, phi2, u_channel, u3)) return false;
	Vec phi3 = phi + dt * u3;

	Vec u4;
	if (!KernelIntpVelocityMAC2(acc, fine_level, coarse_level, phi3, u_channel, u4)) return false;

	phi = phi + dt / 6.0 * (u1 + 2 * u2 + 2 * u3 + u4);

	return true;
}

//__device__ void RK2ForwardPositionAndF(const HATileAccessor<Tile>& acc, const int fine_level, const int coarse_level, const T dt, const int u_channel, const int node_u_channel, Vec& pos, Eigen::Matrix3<T>& F) {
//	//Vec vel1 = InterpolateFaceValue(acc, pos, u_channel, node_u_channel);
//	Vec u1; Eigen::Matrix3<T> gradu1;
//	//VelocityAndJacobian(acc, pos, node_u_channel, u1, gradu1);
//	KernelIntpVelocityAndJacobianMAC2(acc, fine_level, coarse_level, pos, u_channel, u1, gradu1);
//
//	Vec pos1 = pos + u1 * 0.5 * dt;
//	//Vec vel2 = InterpolateFaceValue(acc, pos1, u_channel, node_u_channel);
//	//auto gradu2 = VelocityJacobian(acc, pos1, node_u_channel);
//
//	Vec u2; Eigen::Matrix3<T> gradu2;
//	//VelocityAndJacobian(acc, pos1, node_u_channel, u2, gradu2);
//	KernelIntpVelocityAndJacobianMAC2(acc, fine_level, coarse_level, pos1, u_channel, u2, gradu2);
//	auto dFdt2 = gradu2 * F;
//	pos = pos + dt * u2;
//	F = F + dt * dFdt2;
//}

__device__ bool RK4ForwardPositionAndF(const HATileAccessor<Tile>& acc, const int fine_level, const int coarse_level, const T dt, const int u_channel, Vec& phi, Eigen::Matrix3<T>& F) {
	bool success = true;

	Vec u1; Eigen::Matrix3<T> gradu1;
	//VelocityAndJacobian(acc, phi, node_u_channel, u1, gradu1);
	bool ok1 = KernelIntpVelocityAndJacobianMAC2(acc, fine_level, coarse_level, phi, u_channel, u1, gradu1);
	success = success && ok1;

	//printf("advect rk4 forward phi and F with phi=%f %f %f u1=%f %f %f\n", phi[0], phi[1], phi[2], u1[0], u1[1], u1[2]);



	Eigen::Matrix3<T> dFdt1 = gradu1 * F;
	Vec phi1 = phi + 0.5 * dt * u1;
	Eigen::Matrix3<T> F1 = F + 0.5 * dt * dFdt1;
	
	{
		for (int ii = 0; ii < 3; ii++) {
			CUDA_ASSERT(isfinite(u1[ii]), "u1[%d]=%f", ii, (double)u1[ii]);
			for (int jj = 0; jj < 3; jj++) {
				CUDA_ASSERT(isfinite(gradu1(ii, jj)), "gradu1(%d,%d)=%f", ii, jj, (double)gradu1(ii, jj));
				CUDA_ASSERT(isfinite(dFdt1(ii, jj)), "dFdt1(%d,%d)=%f", ii, jj, (double)dFdt1(ii, jj));
				CUDA_ASSERT(isfinite(F1(ii, jj)), "F1(%d,%d)=%f", ii, jj, (double)F1(ii, jj));
			}
		}
	}

	Vec u2; Eigen::Matrix3<T> gradu2;
	//VelocityAndJacobian(acc, phi1, node_u_channel, u2, gradu2);
	bool ok2 = KernelIntpVelocityAndJacobianMAC2(acc, fine_level, coarse_level, phi1, u_channel, u2, gradu2);
	success = success && ok2;

	//printf("advect rk4 forward phi and F with phi=%f %f %f u1=%f %f %f\n", phi[0], phi[1], phi[2], u1[0], u1[1], u1[2]);

	Eigen::Matrix3<T> dFdt2 = gradu2 * F1;
	Vec phi2 = phi + 0.5 * dt * u2;
	Eigen::Matrix3<T> F2 = F + 0.5 * dt * dFdt2;

	{
		for (int ii = 0; ii < 3; ii++) {
			CUDA_ASSERT(isfinite(u2[ii]), "u2[%d]=%f", ii, (double)u2[ii]);
			for (int jj = 0; jj < 3; jj++) {
				CUDA_ASSERT(isfinite(gradu2(ii, jj)), "gradu2(%d,%d)=%f", ii, jj, (double)gradu2(ii, jj));
				CUDA_ASSERT(isfinite(dFdt2(ii, jj)), "dFdt2(%d,%d)=%f", ii, jj, (double)dFdt2(ii, jj));
				CUDA_ASSERT(isfinite(F2(ii, jj)), "F2(%d,%d)=%f", ii, jj, (double)F2(ii, jj));
			}
		}
	}

	Vec u3; Eigen::Matrix3<T> gradu3;
	//VelocityAndJacobian(acc, phi2, node_u_channel, u3, gradu3);
	bool ok3 = KernelIntpVelocityAndJacobianMAC2(acc, fine_level, coarse_level, phi2, u_channel, u3, gradu3);
	success = success && ok3;

	//printf("advect rk4 forward phi and F with phi=%f %f %f u1=%f %f %f\n", phi[0], phi[1], phi[2], u1[0], u1[1], u1[2]);

	Eigen::Matrix3<T> dFdt3 = gradu3 * F2;
	Vec phi3 = phi + dt * u3;
	Eigen::Matrix3<T> F3 = F + dt * dFdt3;

	{
		for (int ii = 0; ii < 3; ii++) {
			CUDA_ASSERT(isfinite(u3[ii]), "u3[%d]=%f", ii, (double)u3[ii]);
			for (int jj = 0; jj < 3; jj++) {
				CUDA_ASSERT(isfinite(gradu3(ii, jj)), "gradu3(%d,%d)=%f", ii, jj, (double)gradu3(ii, jj));
				CUDA_ASSERT(isfinite(dFdt3(ii, jj)), "dFdt3(%d,%d)=%f", ii, jj, (double)dFdt3(ii, jj));
				CUDA_ASSERT(isfinite(F3(ii, jj)), "F3(%d,%d)=%f", ii, jj, (double)F3(ii, jj));
			}
		}
	}

	Vec u4; Eigen::Matrix3<T> gradu4;
	//VelocityAndJacobian(acc, phi3, node_u_channel, u4, gradu4);
	bool ok4 = KernelIntpVelocityAndJacobianMAC2(acc, fine_level, coarse_level, phi3, u_channel, u4, gradu4);
	success = success && ok4;

	//printf("advect rk4 forward phi and F with phi=%f %f %f u1=%f %f %f\n", phi[0], phi[1], phi[2], u1[0], u1[1], u1[2]);

	Eigen::Matrix3<T> dFdt4 = gradu4 * F3;
	phi = phi + dt / 6.0 * (u1 + 2 * u2 + 2 * u3 + u4);
	F = F + dt / 6.0 * (dFdt1 + 2 * dFdt2 + 2 * dFdt3 + dFdt4);

	{
		for (int ii = 0; ii < 3; ii++) {
			CUDA_ASSERT(isfinite(u4[ii]), "u4[%d]=%f", ii, (double)u4[ii]);
			for (int jj = 0; jj < 3; jj++) {
				CUDA_ASSERT(isfinite(gradu4(ii, jj)), "gradu4(%d,%d)=%f", ii, jj, (double)gradu4(ii, jj));
				CUDA_ASSERT(isfinite(dFdt4(ii, jj)), "dFdt4(%d,%d)=%f", ii, jj, (double)dFdt4(ii, jj));
				CUDA_ASSERT(isfinite(F(ii, jj)), "F(%d,%d)=%f", ii, jj, (double)F(ii, jj));
			}
		}
	}

	return success;
}

//__device__ bool RK4ForwardPositionAndFAtGivenLevel(const HATileAccessor<Tile>& acc, const int level, const T dt, const int u_channel, const int node_u_channel, Vec& phi, Eigen::Matrix3<T>& F, const T eps) {
//	Vec u1; Eigen::Matrix3<T> gradu1;
//	if (!KernelIntpVelocityAndJacobianMAC2AtGivenLevel(acc, level, phi, u_channel, u1, gradu1, eps)) return false;
//
//	Eigen::Matrix3<T> dFdt1 = gradu1 * F;
//	Vec phi1 = phi + 0.5 * dt * u1;
//	Eigen::Matrix3<T> F1 = F + 0.5 * dt * dFdt1;
//
//	Vec u2; Eigen::Matrix3<T> gradu2;
//	if (!KernelIntpVelocityAndJacobianMAC2AtGivenLevel(acc, level, phi1, u_channel, u2, gradu2, eps)) return false;
//
//	Eigen::Matrix3<T> dFdt2 = gradu2 * F1;
//	Vec phi2 = phi + 0.5 * dt * u2;
//	Eigen::Matrix3<T> F2 = F + 0.5 * dt * dFdt2;
//
//	Vec u3; Eigen::Matrix3<T> gradu3;
//	if (!KernelIntpVelocityAndJacobianMAC2AtGivenLevel(acc, level, phi2, u_channel, u3, gradu3, eps)) return false;
//
//	Eigen::Matrix3<T> dFdt3 = gradu3 * F2;
//	Vec phi3 = phi + dt * u3;
//	Eigen::Matrix3<T> F3 = F + dt * dFdt3;
//
//	Vec u4; Eigen::Matrix3<T> gradu4;
//	if (!KernelIntpVelocityAndJacobianMAC2AtGivenLevel(acc, level, phi3, u_channel, u4, gradu4, eps)) return false;
//
//	Eigen::Matrix3<T> dFdt4 = gradu4 * F3;
//	phi = phi + dt / 6.0 * (u1 + 2 * u2 + 2 * u3 + u4);
//	F = F + dt / 6.0 * (dFdt1 + 2 * dFdt2 + 2 * dFdt3 + dFdt4);
//
//	return true;
//}

//__device__ void RK2ForwardPositionAndT(const HATileAccessor<Tile>& acc, const int fine_level, const int coarse_level, const T dt, const int u_channel, const int node_u_channel, Vec& pos, Eigen::Matrix3<T>& matT) {
//	//Vec vel1 = InterpolateFaceValue(acc, pos, u_channel, node_u_channel);
//	Vec u1; Eigen::Matrix3<T> gradu1;
//	//VelocityAndJacobian(acc, pos, node_u_channel, u1, gradu1);
//	KernelIntpVelocityAndJacobianMAC2(acc, fine_level, coarse_level, pos, u_channel, u1, gradu1);
//
//	Vec pos1 = pos + u1 * 0.5 * dt;
//	//Vec vel2 = InterpolateFaceValue(acc, pos1, u_channel, node_u_channel);
//	//auto gradu2 = VelocityJacobian(acc, pos1, node_u_channel);
//
//	Vec u2; Eigen::Matrix3<T> gradu2;
//	//VelocityAndJacobian(acc, pos1, node_u_channel, u2, gradu2);
//	KernelIntpVelocityAndJacobianMAC2(acc, fine_level, coarse_level, pos1, u_channel, u2, gradu2);
//	auto dTdt2 = -matT * gradu2;
//	pos = pos + dt * u2;
//	matT = matT + dt * dTdt2;
//}

__device__ bool RK4ForwardPositionAndT(const HATileAccessor<Tile>& acc, const int fine_level, const int coarse_level, const T dt, const int u_channel, Vec& phi, Eigen::Matrix3<T>& matT) {
	bool success = true;

	Vec u1; Eigen::Matrix3<T> gradu1;
	//VelocityAndJacobian(acc, phi, node_u_channel, u1, gradu1);
	success = success && KernelIntpVelocityAndJacobianMAC2(acc, fine_level, coarse_level, phi, u_channel, u1, gradu1);

	//printf("rk4 intp phi %f %f %f u1 %f %f %f\n", phi[0], phi[1], phi[2], u1[0], u1[1], u1[2]);

	//printf("advect rk4 forward phi and T with phi=%f %f %f u1=%f %f %f\n", phi[0], phi[1], phi[2], u1[0], u1[1], u1[2]);
		
	Eigen::Matrix3<T> dTdt1 = -matT * gradu1;
	Vec phi1 = phi + 0.5 * dt * u1;
	Eigen::Matrix3<T> T1 = matT + 0.5 * dt * dTdt1;

	//printf("advect rk4 forward phi1 and T1 with phi1=%f %f %f\n", phi1[0], phi1[1], phi1[2]);


	Vec u2; Eigen::Matrix3<T> gradu2;
	//VelocityAndJacobian(acc, phi1, node_u_channel, u2, gradu2);
	success = success && KernelIntpVelocityAndJacobianMAC2(acc, fine_level, coarse_level, phi1, u_channel, u2, gradu2);


	Eigen::Matrix3<T> dTdt2 = -T1 * gradu2;
	Vec phi2 = phi + 0.5 * dt * u2;
	Eigen::Matrix3<T> T2 = matT + 0.5 * dt * dTdt2;

	//printf("advect rk4 forward phi2 and T2 with phi2=%f %f %f\n", phi2[0], phi2[1], phi2[2]);


	Vec u3; Eigen::Matrix3<T> gradu3;
	//VelocityAndJacobian(acc, phi2, node_u_channel, u3, gradu3);
	success = success && KernelIntpVelocityAndJacobianMAC2(acc, fine_level, coarse_level, phi2, u_channel, u3, gradu3);


	Eigen::Matrix3<T> dTdt3 = -T2 * gradu3;
	Vec phi3 = phi + dt * u3;
	Eigen::Matrix3<T> T3 = matT + dt * dTdt3;

	//printf("advect rk4 forward phi3 and T3 with phi3=%f %f %f\n", phi3[0], phi3[1], phi3[2]);

	Vec u4; Eigen::Matrix3<T> gradu4;
	//VelocityAndJacobian(acc, phi3, node_u_channel, u4, gradu4);
	success = success && KernelIntpVelocityAndJacobianMAC2(acc, fine_level, coarse_level, phi3, u_channel, u4, gradu4);

	Eigen::Matrix3<T> dTdt4 = -T3 * gradu4;
	phi = phi + dt / 6.0 * (u1 + 2 * u2 + 2 * u3 + u4);
	matT = matT + dt / 6.0 * (dTdt1 + 2 * dTdt2 + 2 * dTdt3 + dTdt4);

	return success;
}


__device__ bool RK4ForwardPositionAndTAtGivenLevel(const HATileAccessor<Tile>& acc, const int level, const T dt, const int u_channel, Vec& phi, Eigen::Matrix3<T>& matT) {
	Vec u1; Eigen::Matrix3<T> gradu1;
	if (!KernelIntpVelocityAndJacobianMAC2AtGivenLevel(acc, level, phi, u_channel, u1, gradu1)) return false;

	Eigen::Matrix3<T> dTdt1 = -matT * gradu1;
	Vec phi1 = phi + 0.5 * dt * u1;
	Eigen::Matrix3<T> T1 = matT + 0.5 * dt * dTdt1;

	Vec u2; Eigen::Matrix3<T> gradu2;
	if (!KernelIntpVelocityAndJacobianMAC2AtGivenLevel(acc, level, phi1, u_channel, u2, gradu2)) return false;


	Eigen::Matrix3<T> dTdt2 = -T1 * gradu2;
	Vec phi2 = phi + 0.5 * dt * u2;
	Eigen::Matrix3<T> T2 = matT + 0.5 * dt * dTdt2;

	Vec u3; Eigen::Matrix3<T> gradu3;
	if (!KernelIntpVelocityAndJacobianMAC2AtGivenLevel(acc, level, phi2, u_channel, u3, gradu3)) return false;


	Eigen::Matrix3<T> dTdt3 = -T2 * gradu3;
	Vec phi3 = phi + dt * u3;
	Eigen::Matrix3<T> T3 = matT + dt * dTdt3;

	Vec u4; Eigen::Matrix3<T> gradu4;
	if (!KernelIntpVelocityAndJacobianMAC2AtGivenLevel(acc, level, phi3, u_channel, u4, gradu4)) return false;

	Eigen::Matrix3<T> dTdt4 = -T3 * gradu4;
	phi = phi + dt / 6.0 * (u1 + 2 * u2 + 2 * u3 + u4);
	matT = matT + dt / 6.0 * (dTdt1 + 2 * dTdt2 + 2 * dTdt3 + dTdt4);

	return true;
}

__device__ void NFMBackMarchPsiAndT(const HATileAccessor<Tile>* accs_d_ptr, const int fine_level, const int coarse_level, const double* time_steps_d_ptr, const int u_channel, const int start_step, const int end_step, Vec& psi, Eigen::Matrix3<T>& matT) {
	CUDA_ASSERT(end_step > start_step, "end_step %d should be greater than start_step %d", end_step, start_step);
	matT = Eigen::Matrix3<T>::Identity();
	for (int i = end_step - 1; i >= start_step; i--) {
		const auto& acc = accs_d_ptr[i];
		RK4ForwardPositionAndF(acc, fine_level, coarse_level, -time_steps_d_ptr[i], u_channel, psi, matT);

		// iterate over 3*3 and ASSERT matT elements are finite
		for (int ii = 0; ii < 3; ii++) {
			for (int jj = 0; jj < 3; jj++) {
				CUDA_ASSERT(isfinite(matT(ii, jj)), "matT(%d,%d)=%f at i=%d where endstep %d startstep %d", ii, jj, matT(ii, jj), i, end_step, start_step);
			}
		}

	}
}

//__device__ void NFMBackQueryImpulseAndT(const HATileAccessor<Tile>* accs_d_ptr, const int fine_level, const int coarse_level, const double* time_steps_d_ptr, const int u_channel, const int node_u_channel, const int start_step, const int end_step, Vec& psi, Vec& impulse, Eigen::Matrix3<T>& matT) {
//	matT = Eigen::Matrix3<T>::Identity();
//	for (int i = end_step - 1; i >= start_step; i--) {
//		const auto& acc = accs_d_ptr[i];
//		RK4ForwardPositionAndF(acc, fine_level, coarse_level, -time_steps_d_ptr[i], u_channel, node_u_channel, psi, matT);
//	}
//
//	impulse = InterpolateFaceValue(accs_d_ptr[start_step], psi, u_channel, node_u_channel);
//}

//void CalculateVorticityMagnitudeOnLeafs(HADeviceGrid<Tile>& grid, const int fine_level, const int coarse_level, const int u_channel, const int tmp_u_node_channel, const int vor_channel) {
//	////prepare u node
//	//for (int axis : {0, 1, 2}) {
//	//    PropagateToChildren(grid, Tile::u_channel + axis, Tile::u_channel + axis, -1, GHOST, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);
//	//}
//	//CalcLeafNodeValuesFromFaceCenters(grid, Tile::u_channel, tmp_u_node_channel);
//	InterpolateFaceVelocitiesAtAllTiles(grid, u_channel, tmp_u_node_channel);
//
//	grid.launchVoxelFuncOnAllTiles(
//		[=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
//		auto& tile = info.tile();
//		auto h = acc.voxelSize(info);
//		//if (!tile.isInterior(l_ijk)) {
//		//	tile(vor_channel, l_ijk) = 0;
//		//	return;
//		//}
//		// 
//		//auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
//		//if (!(g_ijk == Coord(127,130,89) && info.mLevel == 5)) return;
//
//		auto pos = acc.cellCenter(info, l_ijk);
//		//J[i][j] = du[i]/dx[j]
//		Vec vel;
//		Eigen::Matrix3<T> jacobian;
//		KernelIntpVelocityAndJacobianMAC2(acc, fine_level, coarse_level, pos, u_channel, vel, jacobian);
//		//VelocityJacobian(acc, info, l_ijk, pos, tmp_u_node_channel, h, jacobian);
//		//auto jacobian = VelocityJacobian(acc, pos, tmp_u_node_channel, h);
//
//		Vec omega(
//			jacobian(2, 1) - jacobian(1, 2),
//			jacobian(0, 2) - jacobian(2, 0),
//			jacobian(1, 0) - jacobian(0, 1)
//		);
//
//		tile(vor_channel, l_ijk) = omega.length();
//
//		//printf("vorticity at level %d coord %d %d %d %f %f %f %f\n", info.mLevel, g_ijk[0], g_ijk[1], g_ijk[2], pos[0], pos[1], pos[2], tile(vor_channel, l_ijk));
//	}, LEAF, 8
//	);
//}

void CalculateVelocityAndVorticityMagnitudeOnLeafCellCenters(HADeviceGrid<Tile>& grid, const int fine_level, const int coarse_level, const int face_u_channel, const int cell_u_channel, const int vor_channel) {
	////prepare u node
	//for (int axis : {0, 1, 2}) {
	//    PropagateToChildren(grid, Tile::u_channel + axis, Tile::u_channel + axis, -1, GHOST, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);
	//}
	//CalcLeafNodeValuesFromFaceCenters(grid, Tile::u_channel, tmp_u_node_channel);
	//InterpolateFaceVelocitiesAtAllTiles(grid, face_u_channel, node_u_channel);
	 
	// acquire interpolated, conservative velocities at NONLEAF cells
	AccumulateFacesFromLeafsToAllNonLeafs(grid, face_u_channel, 1. / 4, false, INTERIOR | DIRICHLET | NEUMANN);

	//Warn("CalculateVelocityAndVorticityMagnitudeOnLeafCellCenters after InterpolateFaceVelocitiesAtAllTiles");
	//{
	//	SanityCheckChannelCellValues(grid, face_u_channel, LEAF | NONLEAF | GHOST);
	//	SanityCheckChannelCellValues(grid, face_u_channel + 1, LEAF | NONLEAF | GHOST);
	//	SanityCheckChannelCellValues(grid, face_u_channel + 2, LEAF | NONLEAF | GHOST);


	//	SanityCheckChannelNodeValues(grid, node_u_channel);
	//	SanityCheckChannelNodeValues(grid, node_u_channel + 1);
	//	SanityCheckChannelNodeValues(grid, node_u_channel + 2);
	//}

	//{
	//	//show velocity on polyscope before proj
	//	polyscope::init();
	//	auto holder = grid.getHostTileHolderForLeafs();
	//	IOFunc::AddPoissonGridCellCentersToPolyscopePointCloud(holder,
	//		{ { -1,"type" } },
	//		{ {face_u_channel, "face u"}, {node_u_channel, "node u"} });
	//	//IOFunc::AddLeveledPoissonGridCellCentersToPolyscopePointCloud(holder, { { -1,"type" }, { BufChnls::vor, "vorticity" } }, { { BufChnls::u, "velocity" } });
	//	polyscope::show();
	//}

	grid.launchVoxelFuncOnAllTiles(
		[=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
		auto& tile = info.tile();
		auto h = acc.voxelSize(info);
		//if (!tile.isInterior(l_ijk)) {
		//	tile(vor_channel, l_ijk) = 0;
		//	return;
		//}
		// 
		////{

		//	//DBG
		//	auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
		//	//if (!((g_ijk == Coord(2, 26, 36) || g_ijk == Coord(3, 26, 36)) && info.mLevel == 3)) {
		//	if (!((g_ijk == Coord(2, 26, 36)) && info.mLevel == 3)) {
		//		tile(vor_channel, l_ijk) = 0;
		//		return;
		//	}
		////}

		auto pos = acc.cellCenter(info, l_ijk);
		//J[i][j] = du[i]/dx[j]
		Vec vel;
		Eigen::Matrix3<T> jacobian;
		KernelIntpVelocityAndJacobianMAC2(acc, info.mLevel, coarse_level, pos, face_u_channel, vel, jacobian);
		//VelocityJacobian(acc, info, l_ijk, pos, tmp_u_node_channel, h, jacobian);
		//auto jacobian = VelocityJacobian(acc, pos, tmp_u_node_channel, h);

		Vec omega(
			jacobian(2, 1) - jacobian(1, 2),
			jacobian(0, 2) - jacobian(2, 0),
			jacobian(1, 0) - jacobian(0, 1)
		);

		tile(vor_channel, l_ijk) = (tile.type(l_ijk) & INTERIOR) ? omega.length() : 0.;

			////DBG
			//printf("vorticity at level %d coord %d %d %d %f %f %f %f\n", info.mLevel, g_ijk[0], g_ijk[1], g_ijk[2], pos[0], pos[1], pos[2], omega.length());

		{
			//vel = InterpolateFaceValue(acc, pos, face_u_channel, node_u_channel);
			//KernelIntpVelocityAndJacobianMAC2

			{
				auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
				CUDA_ASSERT(isfinite(vel[0]), "cell vel[0] is %f at level %d coord %d %d %d", (double)vel[0], info.mLevel, g_ijk[0], g_ijk[1], g_ijk[2]);
				CUDA_ASSERT(isfinite(vel[1]), "cell vel[1] is %f at level %d coord %d %d %d", (double)vel[1], info.mLevel, g_ijk[0], g_ijk[1], g_ijk[2]);
				CUDA_ASSERT(isfinite(vel[2]), "cell vel[2] is %f at level %d coord %d %d %d", (double)vel[2], info.mLevel, g_ijk[0], g_ijk[1], g_ijk[2]);
			}

			tile(cell_u_channel, l_ijk) = vel.length();
			for (int axis : {0, 1, 2}) {
				tile(cell_u_channel + axis, l_ijk) = vel[axis];
			}
		}

		//printf("vorticity at level %d coord %d %d %d %f %f %f %f\n", info.mLevel, g_ijk[0], g_ijk[1], g_ijk[2], pos[0], pos[1], pos[2], tile(vor_channel, l_ijk));
	}, LEAF, 4
	);

	//Warn("CalculateVelocityAndVorticityMagnitudeOnLeafCellCenters done, manually exiting...");
	//exit(0);
}
