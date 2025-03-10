#pragma once

#include "PoissonTile.h"//for Vec, T etc

class MarkerParticle {
public:
	Vec pos;
};

void AdvectMarkerParticlesRK4Forward(HADeviceGrid<Tile>& grid, const int fine_level, const int coarse_level, const int u_channel, const double dt, thrust::device_vector<MarkerParticle>& particles_d, const bool erase_invalid);