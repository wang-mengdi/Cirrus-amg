#pragma once

#include "MeshSDFAccel.h"
#include "PoissonGrid.h"

void CalculateSDFOnNodes(HADeviceGrid<Tile>& grid, int node_sdf_channel, const MeshSDFAccel& mesh_sdf, const uint8_t launch_types, const Eigen::Transform<T, 3, Eigen::Affine>& xform);
void CreateAMGLaplacianSystemWithSolidCutOnNodeSDF(HADeviceGrid<Tile>& grid, const int node_sdf_channel, const int coeff_channel, const T R_matrix_coeff);
__device__ cuda_vec4_t<T> FaceCornerSDFs(const int node_sdf_channel, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const Coord& l_ijk, const int axis);
template<typename T> __device__  __forceinline__ bool AllNonNegative(const cuda_vec4_t<T>& v)
{
    return v.x >= 0 && v.y >= 0 && v.z >= 0 && v.w >= 0;
}
__hostdev__ T FaceFluidRatio(const cuda_vec4_t<T>& corner_phis);
