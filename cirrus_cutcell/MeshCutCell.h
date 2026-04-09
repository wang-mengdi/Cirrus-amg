#pragma once

#include <Eigen/Geometry>
#include "PoissonGrid.h"

class SDFAccelBase;

constexpr T SDF_REL_EPS = 1e-3;

void CalculateTSDFOnGivenTiles(HADeviceGrid<Tile>& grid, const thrust::host_vector<HATileInfo<Tile>>& tile_infos, int node_sdf_channel, const SDFAccelBase& mesh_sdf, const Eigen::Transform<T, 3, Eigen::Affine>& xform, T truncation);
void CalculateTSDFOnNodes(HADeviceGrid<Tile>& grid, int node_sdf_channel, const SDFAccelBase& mesh_sdf, const uint8_t launch_types, const Eigen::Transform<T, 3, Eigen::Affine>& xform, const T truncation);
void CreateAMGLaplacianSystemWithSolidCutOnNodeSDF(HADeviceGrid<Tile>& grid, const int node_sdf_channel, const int coeff_channel, const T R_matrix_coeff);
__device__ cuda_vec4_t<T> FaceCornerSDFs(const int node_sdf_channel, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const Coord& l_ijk, const int axis);
__hostdev__ int CellCornerSDFInsideCount(const Tile& tile, const int node_sdf_channel, const Coord& l_ijk, T isovalue);
template<typename T> __device__  __forceinline__ bool FaceSDFAllOutside(const cuda_vec4_t<T>& v, const T isovalue)
{
    return v.x >= isovalue && v.y >= isovalue && v.z >= isovalue && v.w >= isovalue;
}
__hostdev__ T FaceFluidRatio(const cuda_vec4_t<T>& corner_phis);
