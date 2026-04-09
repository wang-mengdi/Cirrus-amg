#include "PolyscopeViz.h"

#include <polyscope/polyscope.h>
#include <polyscope/point_cloud.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/volume_mesh.h>

namespace PolyscopeWrapper {
    void init() {
        polyscope::init();
    }

    void show() {
        polyscope::show();
    }

    void removeAllStructures() {
        polyscope::removeAllStructures();
    }
}

namespace IOFunc {

    void AddMarkerParticlesToPolyscope(thrust::device_vector<MarkerParticle> particles_d, std::string name) {
		thrust::host_vector<MarkerParticle> particles = particles_d;

        std::vector<Vec> positions;

        positions.reserve(particles.size());

        for (int i = 0; i < particles.size(); i++) {
            auto p = particles[i].pos;
            positions.emplace_back(p[0], p[1], p[2]);
        }

        polyscope::PointCloud* psCloud = polyscope::registerPointCloud(name, positions);
    }

    void AddParticlesToPolyscope(thrust::device_vector<Particle> particles_d, std::string name) {
        thrust::host_vector<Particle> particles = particles_d;

        std::vector<Vec> positions;
        std::vector<Vec> impulses;

        positions.reserve(particles.size());
        impulses.reserve(particles.size());

        for (int i = 0; i < particles.size(); i++) {
            auto p = particles[i].pos;
            auto v = particles[i].impulse;
            positions.emplace_back(p[0], p[1], p[2]);
            impulses.emplace_back(v[0], v[1], v[2]);
        }

        polyscope::PointCloud* psCloud = polyscope::registerPointCloud(name, positions);
        psCloud->addVectorQuantity("impulse", impulses);
    }

    void AddTilesToPolyscopeVolumetricMesh(HADeviceGrid<Tile>& grid, const uint8_t types, std::string name) {
        using Coord = typename Tile::CoordType;

        int hex_offs[8][3] = {
            {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
            {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
        };

        auto acc = grid.hostAccessor();

        std::vector<Vec> vertices;
        std::vector<std::array<size_t, 8>> hexCells;

        struct CoordKey {
            int x, y, z;
            bool operator==(const CoordKey& other) const {
                return x == other.x && y == other.y && z == other.z;
            }
        };

        struct CoordKeyHash {
            size_t operator()(const CoordKey& k) const {
                size_t h1 = std::hash<int>{}(k.x);
                size_t h2 = std::hash<int>{}(k.y);
                size_t h3 = std::hash<int>{}(k.z);
                return h1 ^ (h2 << 1) ^ (h3 << 2);
            }
        };

        std::unordered_map<CoordKey, size_t, CoordKeyHash> vertexMap;

        for (int level = 0; level < grid.mNumLevels; ++level) {
            for (int i = 0; i < grid.hNumTiles[level]; ++i) {
                auto& info = grid.hTileArrays[level][i];
                if (!(info.mType & types)) continue;

                auto bbox = acc.tileBBox(info);
                std::array<size_t, 8> hexCell;

                for (int s = 0; s < 8; ++s) {
                    Coord off(hex_offs[s][0], hex_offs[s][1], hex_offs[s][2]);

                    auto p = bbox.min() + Vec(off[0], off[1], off[2]) * bbox.dim();
                    Vec vertex(p[0], p[1], p[2]);

                    auto c = info.mTileCoord + off;
                    CoordKey key{ (int)c[0], (int)c[1], (int)c[2] };

                    auto it = vertexMap.find(key);
                    if (it != vertexMap.end()) {
                        hexCell[s] = it->second;
                    }
                    else {
                        size_t id = vertices.size();
                        vertices.push_back(vertex);
                        vertexMap[key] = id;
                        hexCell[s] = id;
                    }
                }

                hexCells.push_back(hexCell);
            }
        }

        if (hexCells.empty()) return;

        polyscope::registerVolumeMesh(name, vertices, hexCells);
    }

    void AddLeveledTilesToPolyscopeVolumetricMesh(HADeviceGrid<Tile>& grid, const uint8_t types, const std::string& base_name) {
        using Coord = typename Tile::CoordType;

        int hex_offs[8][3] = {
            {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
            {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
        };

        auto acc = grid.hostAccessor();

        for (int level = 0; level < grid.mNumLevels; ++level) {
            std::vector<Vec> vertices;
            std::vector<std::array<size_t, 8>> hexCells;
            std::vector<int> cellLevels;
            std::vector<Coord> cellCoords;
            std::vector<int> cellInterestFlags;

            struct CoordKey {
                int x, y, z;
                bool operator==(const CoordKey& other) const {
                    return x == other.x && y == other.y && z == other.z;
                }
            };

            struct CoordKeyHash {
                size_t operator()(const CoordKey& k) const {
                    size_t h1 = std::hash<int>{}(k.x);
                    size_t h2 = std::hash<int>{}(k.y);
                    size_t h3 = std::hash<int>{}(k.z);
                    return h1 ^ (h2 << 1) ^ (h3 << 2);
                }
            };

            std::unordered_map<CoordKey, size_t, CoordKeyHash> vertexMap;

            for (int i = 0; i < grid.hNumTiles[level]; ++i) {
                auto& info = grid.hTileArrays[level][i];
                if (!(info.mType & types)) continue;

                auto bbox = acc.tileBBox(info);
                std::array<size_t, 8> hexCell;

                for (int s = 0; s < 8; ++s) {
                    Coord off(hex_offs[s][0], hex_offs[s][1], hex_offs[s][2]);

                    auto p = bbox.min() + Vec(off[0], off[1], off[2]) * bbox.dim();
                    Vec vertex(p[0], p[1], p[2]);

                    auto c = info.mTileCoord + off;
                    CoordKey key{ (int)c[0], (int)c[1], (int)c[2] };

                    auto it = vertexMap.find(key);
                    if (it != vertexMap.end()) {
                        hexCell[s] = it->second;
                    }
                    else {
                        size_t id = vertices.size();
                        vertices.push_back(vertex);
                        vertexMap[key] = id;
                        hexCell[s] = id;
                    }
                }

                hexCells.push_back(hexCell);
                cellLevels.push_back(level);
                cellCoords.push_back(info.mTileCoord);

                auto tile = info.getTile(DEVICE);
                cellInterestFlags.push_back(tile.mIsInterestArea);
            }

            if (hexCells.empty()) continue;

            std::string name = fmt::format("{}_level_{}", base_name, level);
            auto mesh = polyscope::registerVolumeMesh(name, vertices, hexCells);
            mesh->addCellScalarQuantity("Level", cellLevels);
            mesh->addCellVectorQuantity("Tile Coord", cellCoords);
            mesh->addCellScalarQuantity("Interest Flag", cellInterestFlags);
            mesh->setTransparency(0.2);
        }
    }

    void AddLeveledPoissonGridCellCentersToPolyscopePointCloud(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const std::vector<std::pair<int, std::string>> scalar_channels, std::vector<std::pair<int, std::string>> vec_channels, int level, const double invalid_value) {
        auto& holder = *holder_ptr;
        using Coord = typename Tile::CoordType;

        auto acc = holder.coordAccessor();
        auto add_data = [&](const int level, const uint8_t tile_types, const std::string name) {
            std::vector<Vec> points;
            std::vector<int> levels;
            std::vector<int> ttypes;
            std::vector<Coord> ijk_values;

            std::vector<std::vector<float>> scalar_data(scalar_channels.size());
            std::vector<std::vector<Vec>> vec_data(vec_channels.size());
			std::vector<std::vector<float>> vec_length(vec_channels.size());

            for (auto& info : holder.mHostLevels[level]) {
                if (!(info.mType & tile_types)) continue;
                auto& tile = info.tile();

                for (int i = 0; i < Tile::DIM; i++) {
                    for (int j = 0; j < Tile::DIM; j++) {
                        for (int k = 0; k < Tile::DIM; k++) {
                            Coord l_ijk(i, j, k);

                            auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
                            auto pos = acc.cellCenter(info, l_ijk);

                            ijk_values.push_back(g_ijk);

                            points.push_back(pos);
                            levels.push_back(level);
                            ttypes.push_back(info.mType);

                            for (int s = 0; s < scalar_channels.size(); s++) {
                                int channel = scalar_channels[s].first;
                                T f = (channel == -1) ? tile.type(l_ijk) : tile(scalar_channels[s].first, l_ijk);
                                if (!isfinite(f) || fabs(f) > invalid_value) f = invalid_value;
                                scalar_data[s].push_back(f);
                            }

                            for (int t = 0; t < vec_channels.size(); t++) {
                                int u_channel = vec_channels[t].first;
                                auto u = tile(u_channel, l_ijk);
                                auto v = tile(u_channel + 1, l_ijk);
                                auto w = tile(u_channel + 2, l_ijk);
                                vec_data[t].push_back({ u, v, w });

								auto len = std::sqrt(u * u + v * v + w * w);
                                if (!isfinite(len) || len > invalid_value) {
                                    Info("Non-finite or large vector length at level {}, g_ijk {}: ({}, {}, {})", level, g_ijk, u, v, w);
                                    len = invalid_value;
                                }
								vec_length[t].push_back(len);

                                if (len > 100) {
									Info("Large vector magnitude at level {}, g_ijk {}: ({}, {}, {}), len {}", level, g_ijk, u, v, w, len);
                                }
                            }
                        }
                    }
                }
            }

            if (points.empty()) return;
                
            auto ps = polyscope::registerPointCloud(name, points);
            ps->addScalarQuantity("Level", levels);
            ps->addVectorQuantity("IJK", ijk_values);
            ps->addScalarQuantity("Tile Type", ttypes);

            for (int s = 0; s < scalar_channels.size(); s++) {
                ps->addScalarQuantity(scalar_channels[s].second, scalar_data[s]);
            }

            for (int v = 0; v < vec_channels.size(); v++) {
                ps->addVectorQuantity(vec_channels[v].second, vec_data[v]);
				ps->addScalarQuantity(vec_channels[v].second + "_len", vec_length[v]);
            }
            };

        int beg = 0, end = holder.mMaxLevel;
		if (level != -1) beg = end = level;
        for (int i = beg; i <= end; i++) {
            add_data(i, LEAF, fmt::format("Level{}LEAF", i));
            add_data(i, GHOST, fmt::format("Level{}GHOST", i));
            add_data(i, NONLEAF, fmt::format("Level{}NONLEAF", i));
        }
    }

    void AddPoissonGridCellCentersToPolyscopePointCloud(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const std::vector<std::pair<int, std::string>> scalar_channels, std::vector<std::pair<int, std::string>> vec_channels, const double invalid_value) {
        auto& holder = *holder_ptr;
        using Coord = typename Tile::CoordType;

        std::vector<Vec> points;
        std::vector<int> levels;
        std::vector<Coord> ijk_values;

        std::vector<std::vector<float>> scalar_data(scalar_channels.size());
        std::vector<std::vector<Vec>> vec_data(vec_channels.size());
        std::vector<std::vector<float>> vec_length(vec_channels.size());

        auto acc = holder.coordAccessor();
        for (int level = 0; level <= holder.mMaxLevel; level++) {
            for (auto& info : holder.mHostLevels[level]) {
                if (info.isLeaf()) {
                    auto& tile = info.tile();

                    for (int i = 0; i < Tile::DIM; i++) {
                        for (int j = 0; j < Tile::DIM; j++) {
                            for (int k = 0; k < Tile::DIM; k++) {
                                Coord l_ijk(i, j, k);
                                auto pos = acc.cellCenter(info, l_ijk);
                                points.push_back(pos);
                                levels.push_back(level);

								auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
                                ijk_values.push_back(g_ijk);

                                for (int s = 0; s < scalar_channels.size(); s++) {
                                    int channel = scalar_channels[s].first;
                                    T f = (channel == -1) ? tile.type(l_ijk) : tile(scalar_channels[s].first, l_ijk);
                                    if (isnan(f) || isinf(f)) f = invalid_value;
                                    scalar_data[s].push_back(f);
                                }

                                for (int t = 0; t < vec_channels.size(); t++) {
                                    int u_channel = vec_channels[t].first;
                                    auto u = tile(u_channel, l_ijk);
                                    auto v = tile(u_channel + 1, l_ijk);
                                    auto w = tile(u_channel + 2, l_ijk);
                                    vec_data[t].push_back({ u, v, w });

									auto len = std::sqrt(u * u + v * v + w * w);
                                    if (!isfinite(len)) {
                                        Info("Non-finite vector length at level {}, g_ijk {}: ({}, {}, {})", level, g_ijk, u, v, w);
                                        len = invalid_value;
                                    }
                                    vec_length[t].push_back(len);

                                    if (len > 10) {
										Warn("Large vector magnitude at level {}, g_ijk {}: ({}, {}, {})", level, g_ijk, u, v, w);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        polyscope::registerPointCloud("Poisson Grid", points);
        polyscope::getPointCloud("Poisson Grid")->addScalarQuantity("Level", levels);
        polyscope::getPointCloud("Poisson Grid")->addVectorQuantity("IJK", ijk_values);

        for (int s = 0; s < scalar_channels.size(); s++) {
            polyscope::getPointCloud("Poisson Grid")->addScalarQuantity(scalar_channels[s].second, scalar_data[s]);
        }

        for (int v = 0; v < vec_channels.size(); v++) {
            polyscope::getPointCloud("Poisson Grid")->addVectorQuantity(vec_channels[v].second, vec_data[v]);
            polyscope::getPointCloud("Poisson Grid")->addScalarQuantity(vec_channels[v].second + "_len", vec_length[v]);
        }
    }

    void AddPoissonGridFaceCentersToPolyscopePointCloud(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, std::vector<std::pair<int, std::string>> vec_channels, const double invalid_value) {
        auto& holder = *holder_ptr;
        using Coord = typename Tile::CoordType;

        std::vector<Vec> points[3];
        std::vector<int> levels[3];
        std::vector<Coord> ijk_values[3];

        std::vector<std::vector<float>> face_center_data[3];
        for (int axis : {0, 1, 2}) {
            face_center_data[axis].resize(vec_channels.size());
        }

        auto acc = holder.coordAccessor();
        holder.iterateLeafCells([&](const HATileInfo<Tile>& info, const Coord& l_ijk) {
            auto& tile = info.tile();

            for (int axis : {0, 1, 2}) {
                auto pos = acc.faceCenter(axis, info, l_ijk);
                points[axis].push_back(pos);
                levels[axis].push_back(info.mLevel);

                auto g_ijk = holder.coordAccessor().localToGlobalCoord(info, l_ijk);
                ijk_values[axis].push_back(g_ijk);

                for (int t = 0; t < vec_channels.size(); t++) {
                    int u_channel = vec_channels[t].first;
                    face_center_data[axis][t].push_back(tile(u_channel + axis, l_ijk));
                }
            }
            });

        for (int axis : {0, 1, 2}) {
			auto pc = polyscope::registerPointCloud(fmt::format("Poisson Grid Face Centers Axis {}", axis), points[axis]);
			pc->addScalarQuantity("Level", levels[axis]);
			pc->addVectorQuantity("IJK", ijk_values[axis]);
			for (int v = 0; v < vec_channels.size(); v++) {
				pc->addScalarQuantity(vec_channels[v].second, face_center_data[axis][v]);
			}
        }
    }

    void AddPoissonGridNodesToPolyscope(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const std::vector<std::pair<int, std::string>> scalar_channels, std::vector<std::pair<int, std::string>> vec_channels, const double invalid_value) {
        auto& holder = *holder_ptr;
        using Coord = typename Tile::CoordType;

        auto acc = holder.coordAccessor();

        for (int level = 0; level <= holder.mMaxLevel; level++) {
            std::vector<Vec> points;
            std::vector<Coord> tile_ijks, local_ijks;

            std::vector<std::vector<float>> scalar_data(scalar_channels.size());
            std::vector<std::vector<Vec>> vec_data(vec_channels.size());

            std::vector<Vec> tile_vertices;
            std::vector<std::array<size_t, 4>> tile_faces;

            for (auto& info : holder.mHostLevels[level]) {
                if (info.isLeaf()) {
                    auto& tile = info.tile();

                    for (int i = 0; i <= Tile::DIM; i++) {
                        for (int j = 0; j <= Tile::DIM; j++) {
                            for (int k = 0; k <= Tile::DIM; k++) {
                                Coord r_ijk(i, j, k);
                                auto pos = acc.cellCorner(info, r_ijk);

                                {
                                    auto ref_point = acc.cellCenterGlobal(3, Coord(60, 23, 99));
                                    if ((pos - ref_point).length() > 0.03) continue;
                                }


                                points.push_back(pos);

                                tile_ijks.push_back(info.mTileCoord);
                                local_ijks.push_back(r_ijk);

                                for (int s = 0; s < scalar_channels.size(); s++) {
                                    int channel = scalar_channels[s].first;
                                    T f = tile.node(scalar_channels[s].first, r_ijk);
                                    if (isnan(f) || isinf(f)) f = invalid_value;
                                    scalar_data[s].push_back(f);
                                }

                                for (int t = 0; t < vec_channels.size(); t++) {
                                    int u_channel = vec_channels[t].first;
                                    auto u = tile.node(u_channel, r_ijk);
                                    auto v = tile.node(u_channel + 1, r_ijk);
                                    auto w = tile.node(u_channel + 2, r_ijk);
                                    vec_data[t].push_back({ u, v, w });
                                }
                            }
                        }
                    }

                    std::vector<Vec> cube_vertices;
                    for (int dx = 0; dx <= 1; ++dx) {
                        for (int dy = 0; dy <= 1; ++dy) {
                            for (int dz = 0; dz <= 1; ++dz) {
                                Coord l_ijk(dx * Tile::DIM, dy * Tile::DIM, dz * Tile::DIM);
                                cube_vertices.push_back(acc.cellCorner(info, l_ijk));
                            }
                        }
                    }

                    auto base_idx = tile_vertices.size();
                    tile_vertices.insert(tile_vertices.end(), cube_vertices.begin(), cube_vertices.end());

                    std::array<size_t, 4> faces[6] = {
                        {base_idx + 0, base_idx + 1, base_idx + 3, base_idx + 2},
                        {base_idx + 4, base_idx + 5, base_idx + 7, base_idx + 6},
                        {base_idx + 0, base_idx + 1, base_idx + 5, base_idx + 4},
                        {base_idx + 2, base_idx + 3, base_idx + 7, base_idx + 6},
                        {base_idx + 0, base_idx + 2, base_idx + 6, base_idx + 4},
                        {base_idx + 1, base_idx + 3, base_idx + 7, base_idx + 5}
                    };

                    tile_faces.insert(tile_faces.end(), std::begin(faces), std::end(faces));

                }
            }

            if (points.empty()) continue;

            auto level_cloud_name = fmt::format("Level {} Nodes", level);
            auto pc = polyscope::registerPointCloud(level_cloud_name, points);
            pc->addVectorQuantity("b_ijk", tile_ijks);
			pc->addVectorQuantity("r_ijk", local_ijks);

            for (int s = 0; s < scalar_channels.size(); s++) {
                pc->addScalarQuantity(scalar_channels[s].second, scalar_data[s]);
            }

            for (int v = 0; v < vec_channels.size(); v++) {
                pc->addVectorQuantity(vec_channels[v].second, vec_data[v]);
            }
        }
    }

}
