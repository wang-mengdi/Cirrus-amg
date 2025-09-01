#pragma once

#include "MeshSDFAccel.h"
#include "PoissonGrid.h"

void CalculateSDFOnNodes(HADeviceGrid<Tile>& grid, int node_sdf_channel, const MeshSDFAccel& mesh_sdf, const uint8_t launch_types, const Eigen::Transform<T, 3, Eigen::Affine>& xform);
void CreateAMGLaplacianSystemWithSolidCutOnNodeSDF(HADeviceGrid<Tile>& grid, const int node_sdf_channel, const int coeff_channel, const T R_matrix_coeff);