#pragma once

#include <Eigen/Core>
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace simple {

using Vec3 = Eigen::Vector3d;

struct Cell {
    Vec3 center = Vec3::Zero();
    double h = 0.0;
    double volume = 0.0;
    int level = 0;
    bool cut = false;
    // Global cell coordinates at this cell's octree level (not finest-level keys).
    std::array<int, 3> key{};
};

struct Face {
    int owner = -1;
    int neighbor = -1;
    int axis = 0;
    double sign = 1.0;
    double area = 0.0;
    double distance = 0.0;
    Vec3 center = Vec3::Zero();
    // Neighbor uses its adjacent periodic image when a face crosses a seam.
    Vec3 delta = Vec3::Zero();
    Vec3 ownerOffset = Vec3::Zero();
    Vec3 neighborOffset = Vec3::Zero();
    // 0 = internal (including periodic), 1 = no slip, 2/3 = pressure inlet/outlet.
    int boundary = 0;
    // Nonzero only for a curved embedded wall; Cartesian faces retain axis/sign.
    Vec3 embeddedNormal = Vec3::Zero();
};

struct Mesh {
    std::vector<Cell> cells;
    std::vector<Face> faces;
    Vec3 extent = Vec3(1.0, 0.125, 0.125);
    int coarseFineFaces = 0;
    bool periodicZ = true;
    std::string backend;
    bool embedded = false;
    std::string geometrySource;
    // Owns the native HADeviceGrid<Tile> and its leaf host holder without making
    // the CPU SIMPLE solver depend on CUDA headers.
    std::shared_ptr<void> nativeStorage;
};

// Channel: y no-slip, z periodic or no-slip, x periodic or pressure inlet/outlet.
// ny is the coarse cell count across y; it must be a power of two >= 8.
// Adaptive mode refines native tiles in 0.25 <= x < 0.75 by one octree level.
// Embedded import may defer faces; its caller must visitFacesAndValidate before
// using the topology. Other callers retain the full validated face array.
Mesh makeOctreeChannel(int ny, bool adaptive, bool periodicX, bool periodicZ = true,
                      const std::string& geometryPath = "",const std::string& geometryMetadata = "",
                      bool buildFaces = true);

// The geometry file contains only the prescribed cut geometry (no flow state).
// Actual adaptive leaf cells still come from the original HADeviceGrid.
Mesh makeEmbeddedOctree(const std::string& geometryPath, bool adaptive);
void retainNativeCells(Mesh& mesh, const std::vector<int>& oldIndices);

// After the final active-cell mapping is fixed, retain complete native host
// Tile bytes in a fresh checked scratch directory. Device topology is unchanged.
void offloadNativeHostTiles(Mesh& mesh, const std::string& freshDirectory);

// Creates one owner-neighbor connection per shared subface. Coarse/fine
// interfaces keep four separate subfaces; no uniform-grid expansion is used.
void buildFacesAndValidate(Mesh& mesh, bool periodicX, bool periodicZ = true);

// Visit the same Cartesian subfaces in the same order without retaining an
// array. All background-mesh checks still run; return the coarse/fine count.
// The callback must leave mesh and its cells unchanged.
int visitFacesAndValidate(const Mesh& mesh, bool periodicX, bool periodicZ,
                          const std::function<void(const Face&)>& visit);

// Throws if face coverage, orientation, 2:1 balance, volume, or cell closure fail.
void validateMesh(const Mesh& mesh);
void dumpMeshCsv(const Mesh& mesh, const std::string& prefix);

// Writes cell-centered velocity to native tile channels 0..2 and pressure to 3,
// copies the modified leaf tiles to the original GPU grid, then writes its native
// binary blob at path. This layout is specific to SIMPLE, not the MAC layout.
// writeBinary=false still synchronizes the physical channels but omits the blob.
void exportNativeFields(Mesh& mesh, const std::vector<Vec3>& velocity,
                        const std::vector<double>& pressure, const std::string& path,
                        bool preserveOperatorMetadata=false,bool writeBinary=true);

} // namespace simple
