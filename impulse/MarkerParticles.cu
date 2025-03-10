#include "MarkerParticles.h"
#include "FlowMap.h"

void AdvectMarkerParticlesRK4Forward(HADeviceGrid<Tile>& grid, const int fine_level, const int coarse_level, const int u_channel, const double dt, thrust::device_vector<MarkerParticle>& particles_d, const bool erase_invalid) {
	//advect particles
	auto particles_ptr = thrust::raw_pointer_cast(particles_d.data());
	auto acc = grid.deviceAccessor();
	LaunchIndexFunc([=] __device__(int idx) {
		auto& p = particles_ptr[idx];
		RK4ForwardPosition(acc, fine_level, coarse_level, dt, u_channel, p.pos);
	}, particles_d.size(), 512, 4);
}