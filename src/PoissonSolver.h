#pragma once

#include "PoissonGrid.h"
#include <Eigen/Dense>

//coarsening for Multigrid
//calculate NONLEAF and GHOST cell types from 
void CalcCellTypesFromLeafs(HADeviceGrid<Tile>& grid);

void ReCenterLeafCells(HADeviceGrid<Tile>& grid, const int channel, DeviceReducer<double>& cnt_reducer, double* d_mean, double* d_count);