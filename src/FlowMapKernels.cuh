#pragma once

#include "PoissonGrid.h"
#include <Eigen/Core>

__hostdev__ inline Vec MatrixTimesVec(const Eigen::Matrix3<T>& A, const Vec& b) {
	return Vec(
		A(0, 0) * b[0] + A(0, 1) * b[1] + A(0, 2) * b[2],
		A(1, 0) * b[0] + A(1, 1) * b[1] + A(1, 2) * b[2],
		A(2, 0) * b[0] + A(2, 1) * b[1] + A(2, 2) * b[2]
	);
}

__hostdev__ inline T QuadraticKernelFast(const T x) {
	T abs_x = fabsf(x);
	T ret = abs_x < 0.5f ? (0.75f - abs_x * abs_x) : (0.5f * (1.5f - abs_x) * (1.5f - abs_x));
	return ret;
}
__hostdev__ inline T QuadraticKernel(const T x) {
	T abs_x = fabsf(x);
	if (abs_x < 0.5f) return 3.0f / 4.f - x * x;
	else if (abs_x < 1.5f) return 0.5f * (1.5f - abs_x) * (1.5f - abs_x);
	return 0;
}
__hostdev__ inline T QuadraticKernelDerivativeFast(const T x) {
	T abs_x = fabsf(x);
	T ret = abs_x < 0.5f ? (-2.0f * x) : (x - copysignf(1.5f, x));
	return ret;
}
__hostdev__ inline T QuadraticKernelDerivative(T x) {
	T abs_x = fabsf(x);
	if (0.f <= abs_x && abs_x < 0.5f) return -2.f * x;
	if (0.5f <= abs_x && abs_x < 1.5f) return (x - copysignf(1.5f, x));
	return 0;
}

__hostdev__ inline T QuadraticKernel(const Vec& r, T h) {
	return QuadraticKernel(r[0] / h) * QuadraticKernel(r[1] / h) * QuadraticKernel(r[2] / h);
}
__hostdev__ inline Vec QuadraticKernelGradient(Vec r, T h) {
	auto xp = r[0] / h;
	auto yp = r[1] / h;
	auto zp = r[2] / h;
	return Vec(
		QuadraticKernelDerivative(xp) * QuadraticKernel(yp) * QuadraticKernel(zp) / h,
		QuadraticKernel(xp) * QuadraticKernelDerivative(yp) * QuadraticKernel(zp) / h,
		QuadraticKernel(xp) * QuadraticKernel(yp) * QuadraticKernelDerivative(zp) / h
	);
}

static __device__ bool KernelIntpVelocityComponentMAC2(const HATileAccessor<Tile>& acc, const int level, const Vec& pos, const int axis, const int u_channel, T& u_i) {
	T h = acc.voxelSize(level);
	Vec test_pos = pos - Vec(0.5 * h, 0.5 * h, 0.5 * h);
	for (int ii : {0, 1, 2}) {
		if (ii != axis) test_pos[ii] -= 0.5 * h;
	}

	Coord g_ijk; Vec frac;
	acc.worldToVoxelAndFraction(level, test_pos, g_ijk, frac);
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
				if (!ninfo.empty()) {
					auto& ntile = ninfo.tile();
					auto n_u_i = ntile(u_channel + axis, nl_ijk);
					if (n_u_i == NODATA) return false;

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

static __device__ bool KernelIntpVelocityComponentAndGradientMAC2(const HATileAccessor<Tile>& acc, const int level, const Vec& pos, const int axis, const int u_channel, T& u_i, Vec& gradu_i) {
	T h = acc.voxelSize(level);
	Vec test_pos = pos - Vec(0.5 * h, 0.5 * h, 0.5 * h);
	for (int ii : {0, 1, 2}) {
		if (ii != axis) test_pos[ii] -= 0.5 * h;
	}

	Coord g_ijk; Vec frac;
	acc.worldToVoxelAndFraction(level, test_pos, g_ijk, frac);
	frac += Vec(0.5, 0.5, 0.5);

	u_i = 0;
	gradu_i = Vec(0, 0, 0);

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
				if (!ninfo.empty()) {
					auto& ntile = ninfo.tile();
					auto n_u_i = ntile(u_channel + axis, nl_ijk);
					if (n_u_i == NODATA) return false;

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

static __device__ bool KernelIntpVelocityMAC2AtGivenLevel(const HATileAccessor<Tile>& acc, const int level, const Vec& pos, const int u_channel, Vec& vel) {
	for (int axis : {0, 1, 2}) {
		if (!KernelIntpVelocityComponentMAC2(acc, level, pos, axis, u_channel, vel[axis])) return false;
	}
	return true;
}

static __device__ bool KernelIntpVelocityAndJacobianMAC2AtGivenLevel(const HATileAccessor<Tile>& acc, const int level, const Vec& pos, const int u_channel, Vec& vel, Eigen::Matrix3<T>& jacobian) {
	for (int axis : {0, 1, 2}) {
		Vec gradu_i;
		if (!KernelIntpVelocityComponentAndGradientMAC2(acc, level, pos, axis, u_channel, vel[axis], gradu_i)) return false;
		for (int t : {0, 1, 2}) {
			jacobian(axis, t) = gradu_i[t];
		}
	}

	return true;
}

static __device__ bool KernelIntpVelocityMAC2(const HATileAccessor<Tile>& acc, const int fine_level, const int coarse_level, const Vec& pos, const int u_channel, Vec& vel) {
	for (int i = fine_level; i >= coarse_level; i--) {
		if (KernelIntpVelocityMAC2AtGivenLevel(acc, i, pos, u_channel, vel)) {
			return true;
		}
	}

	vel = Vec(0, 0, 0);
	return false;
}

static __device__ bool KernelIntpVelocityAndJacobianMAC2(const HATileAccessor<Tile>& acc, const int fine_level, const int coarse_level, const Vec& pos, const int u_channel, Vec& vel, Eigen::Matrix3<T>& jacobian) {
	for (int i = fine_level; i >= coarse_level; i--) {
		if (KernelIntpVelocityAndJacobianMAC2AtGivenLevel(acc, i, pos, u_channel, vel, jacobian)) {
			{
				for (int ii = 0; ii < 3; ii++) {
					CUDA_ASSERT(isfinite(vel[ii]), "vel[%d]=%f at i=%d", ii, vel[ii], i);
				}
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
	for (int k = 0; k < 9; k++)
		jacobian.data()[k] = T(0);
	return false;
}

static __device__ void KernelScatterVelocityComponentMAC2(
	const HATileAccessor<Tile>& acc, const int level, const int axis, const int u_channel, const int uw_channel,
	const Vec& pos, const Vec& vel, const Eigen::Matrix3<T>& gradu) {

	T h = acc.voxelSize(level);
	Vec test_pos = pos - Vec(0.5 * h, 0.5 * h, 0.5 * h);
	for (int ii : {0, 1, 2}) {
		if (ii != axis) test_pos[ii] -= 0.5 * h;
	}

	Coord g_ijk; Vec frac;
	acc.worldToVoxelAndFraction(level, test_pos, g_ijk, frac);
	frac += Vec(0.5, 0.5, 0.5);

	for (int offi = 0; offi < 3; ++offi) {
		T wi = QuadraticKernel(frac[0] - offi);
		for (int offj = 0; offj < 3; ++offj) {
			T wj = QuadraticKernel(frac[1] - offj);
			for (int offk = 0; offk < 3; ++offk) {
				T wk = QuadraticKernel(frac[2] - offk);
				T weight = wi * wj * wk;
				auto ng_ijk = g_ijk + Coord(offi, offj, offk);
				HATileInfo<Tile> ninfo; Coord nl_ijk;
				acc.findVoxel(level, ng_ijk, ninfo, nl_ijk);
				if (!ninfo.empty()) {
					auto& ntile = ninfo.tile();
					Vec fpos = acc.faceCenter(axis, ninfo, nl_ijk);
					auto u = vel + MatrixTimesVec(gradu, fpos - pos);
					atomicAdd(&(ntile(u_channel + axis, nl_ijk)), weight * u[axis]);
					atomicAdd(&(ntile(uw_channel + axis, nl_ijk)), weight);
				}
			}
		}
	}
}

static __device__ void KernelScatterVelocityMAC2(const HATileAccessor<Tile>& acc, const int u_channel, const int uw_channel, const Vec& pos, const Vec& vec, const Eigen::Matrix3<T>& gradv) {
	HATileInfo<Tile> info; Coord l_ijk; Vec frac;
	acc.findLeafVoxelAndFrac(pos, info, l_ijk, frac);
	if (info.empty()) return;
	for (int axis : {0, 1, 2}) {
		KernelScatterVelocityComponentMAC2(acc, info.mLevel, axis, u_channel, uw_channel, pos, vec, gradv);
	}
}

static __device__ bool RK4ForwardPosition(const HATileAccessor<Tile>& acc, const int fine_level, const int coarse_level, const T dt, const int u_channel, Vec& phi) {
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

static __device__ bool RK4ForwardPositionAndF(const HATileAccessor<Tile>& acc, const int fine_level, const int coarse_level, const T dt, const int u_channel, Vec& phi, Eigen::Matrix3<T>& F) {
	bool success = true;

	Vec u1; Eigen::Matrix3<T> gradu1;
	bool ok1 = KernelIntpVelocityAndJacobianMAC2(acc, fine_level, coarse_level, phi, u_channel, u1, gradu1);
	success = success && ok1;

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
	bool ok2 = KernelIntpVelocityAndJacobianMAC2(acc, fine_level, coarse_level, phi1, u_channel, u2, gradu2);
	success = success && ok2;

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
	bool ok3 = KernelIntpVelocityAndJacobianMAC2(acc, fine_level, coarse_level, phi2, u_channel, u3, gradu3);
	success = success && ok3;

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
	bool ok4 = KernelIntpVelocityAndJacobianMAC2(acc, fine_level, coarse_level, phi3, u_channel, u4, gradu4);
	success = success && ok4;

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

static __device__ bool RK4ForwardPositionAndT(const HATileAccessor<Tile>& acc, const int fine_level, const int coarse_level, const T dt, const int u_channel, Vec& phi, Eigen::Matrix3<T>& matT) {
	bool success = true;

	Vec u1; Eigen::Matrix3<T> gradu1;
	success = success && KernelIntpVelocityAndJacobianMAC2(acc, fine_level, coarse_level, phi, u_channel, u1, gradu1);

	Eigen::Matrix3<T> dTdt1 = -matT * gradu1;
	Vec phi1 = phi + 0.5 * dt * u1;
	Eigen::Matrix3<T> T1 = matT + 0.5 * dt * dTdt1;

	Vec u2; Eigen::Matrix3<T> gradu2;
	success = success && KernelIntpVelocityAndJacobianMAC2(acc, fine_level, coarse_level, phi1, u_channel, u2, gradu2);

	Eigen::Matrix3<T> dTdt2 = -T1 * gradu2;
	Vec phi2 = phi + 0.5 * dt * u2;
	Eigen::Matrix3<T> T2 = matT + 0.5 * dt * dTdt2;

	Vec u3; Eigen::Matrix3<T> gradu3;
	success = success && KernelIntpVelocityAndJacobianMAC2(acc, fine_level, coarse_level, phi2, u_channel, u3, gradu3);

	Eigen::Matrix3<T> dTdt3 = -T2 * gradu3;
	Vec phi3 = phi + dt * u3;
	Eigen::Matrix3<T> T3 = matT + dt * dTdt3;

	Vec u4; Eigen::Matrix3<T> gradu4;
	success = success && KernelIntpVelocityAndJacobianMAC2(acc, fine_level, coarse_level, phi3, u_channel, u4, gradu4);

	Eigen::Matrix3<T> dTdt4 = -T3 * gradu4;
	phi = phi + dt / 6.0 * (u1 + 2 * u2 + 2 * u3 + u4);
	matT = matT + dt / 6.0 * (dTdt1 + 2 * dTdt2 + 2 * dTdt3 + dTdt4);

	return success;
}

static __device__ bool RK4ForwardPositionAndTAtGivenLevel(const HATileAccessor<Tile>& acc, const int level, const T dt, const int u_channel, Vec& phi, Eigen::Matrix3<T>& matT) {
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

static __device__ void NFMBackMarchPsiAndT(const HATileAccessor<Tile>* accs_d_ptr, const int fine_level, const int coarse_level, const double* time_steps_d_ptr, const int u_channel, const int start_step, const int end_step, Vec& psi, Eigen::Matrix3<T>& matT) {
	CUDA_ASSERT(end_step > start_step, "end_step %d should be greater than start_step %d", end_step, start_step);
	matT = Eigen::Matrix3<T>::Identity();
	for (int i = end_step - 1; i >= start_step; i--) {
		const auto& acc = accs_d_ptr[i];
		RK4ForwardPositionAndF(acc, fine_level, coarse_level, -time_steps_d_ptr[i], u_channel, psi, matT);

		for (int ii = 0; ii < 3; ii++) {
			for (int jj = 0; jj < 3; jj++) {
				CUDA_ASSERT(isfinite(matT(ii, jj)), "matT(%d,%d)=%f at i=%d where endstep %d startstep %d", ii, jj, matT(ii, jj), i, end_step, start_step);
			}
		}
	}
}
