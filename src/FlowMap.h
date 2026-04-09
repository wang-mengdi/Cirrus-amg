#pragma once

#include "PoissonGrid.h"
#include <Eigen/Core>

void CalculateVelocityAndVorticityMagnitudeOnLeafCellCenters(HADeviceGrid<Tile>& grid, const int fine_level, const int coarse_level, const int face_u_channel, const int cell_u_channel, const int vor_channel);
