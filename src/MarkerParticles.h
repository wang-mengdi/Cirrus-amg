#pragma once

#include "PoissonTile.h"//for Vec, T etc

class MarkerParticle {
public:
	Vec pos;
	T birth_time;
};

void AdvectMarkerParticlesRK4ForwardAndMarkInvalid(HADeviceGrid<Tile>& grid, const int fine_level, const int coarse_level, const int u_channel, const double dt, const T earliest_birth_time, thrust::device_vector<MarkerParticle>& particles_d);
void EraseInvalidParticles(thrust::device_vector<MarkerParticle>& particles);