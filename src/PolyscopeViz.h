#pragma once

#include "PoissonGrid.h"
#include "FMParticles.h"
#include "MarkerParticles.h"

namespace PolyscopeWrapper {
    void init();
    void show();
    void removeAllStructures();
}

namespace IOFunc {
    void AddMarkerParticlesToPolyscope(thrust::device_vector<MarkerParticle> particles_d, std::string name);
    void AddParticlesToPolyscope(thrust::device_vector<Particle> particles_d, std::string name);

    void AddTilesToPolyscopeVolumetricMesh(HADeviceGrid<Tile>& grid, const uint8_t types, std::string name);
    void AddLeveledTilesToPolyscopeVolumetricMesh(HADeviceGrid<Tile>& grid, const uint8_t types, const std::string& base_name);
    void AddPoissonGridCellCentersToPolyscopePointCloud(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const std::vector<std::pair<int, std::string>> scalar_channels, std::vector<std::pair<int, std::string>> vec_channels, const double invalid_value = 1e15);
    void AddLeveledPoissonGridCellCentersToPolyscopePointCloud(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const std::vector<std::pair<int, std::string>> scalar_channels, std::vector<std::pair<int, std::string>> vec_channels, int level = -1, const double invalid_value = FLT_MAX);
    void AddPoissonGridFaceCentersToPolyscopePointCloud(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, std::vector<std::pair<int, std::string>> vec_channels, const double invalid_value = FLT_MAX);
    void AddPoissonGridNodesToPolyscope(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const std::vector<std::pair<int, std::string>> scalar_channels, std::vector<std::pair<int, std::string>> vec_channels, const double invalid_value = FLT_MAX);
}
