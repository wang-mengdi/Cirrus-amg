#include "FlowMap.h"
#include "FlowMapKernels.cuh"
#include "PoissonIOFunc.h"
#include <polyscope/polyscope.h>

void CalculateVelocityAndVorticityMagnitudeOnLeafCellCenters(HADeviceGrid<Tile>& grid, const int fine_level, const int coarse_level, const int face_u_channel, const int cell_u_channel, const int vor_channel) {
	AccumulateFacesFromLeafsToAllNonLeafs(grid, face_u_channel, 1. / 4, false, INTERIOR | DIRICHLET | NEUMANN);

	grid.launchVoxelFuncOnAllTiles(
		[=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
		auto& tile = info.tile();
		for (int axis : {0, 1, 2}) {
			auto fpos = acc.faceCenter(axis, info, l_ijk);
			Vec vel;
			KernelIntpVelocityMAC2(acc, info.mLevel - 1, coarse_level, fpos, face_u_channel, vel);
			tile(face_u_channel + axis, l_ijk) = vel[axis];
		}
	}, GHOST, 4
	);

	grid.launchVoxelFuncOnAllTiles(
		[=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
		auto& tile = info.tile();
		auto h = acc.voxelSize(info);

		auto pos = acc.cellCenter(info, l_ijk);
		Vec vel;
		Eigen::Matrix3<T> jacobian;
		KernelIntpVelocityAndJacobianMAC2(acc, info.mLevel, coarse_level, pos, face_u_channel, vel, jacobian);

		Vec omega(
			jacobian(2, 1) - jacobian(1, 2),
			jacobian(0, 2) - jacobian(2, 0),
			jacobian(1, 0) - jacobian(0, 1)
		);

		tile(vor_channel, l_ijk) = (tile.type(l_ijk) & INTERIOR) ? omega.length() : 0.;

		{
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
	}, LEAF, 4
	);
}
