#pragma once
#include "SimpleMesh.h"
#include "PoissonTile.h"

namespace simple {
// The caller keeps mesh.nativeStorage alive. GPU operators use the same grid
// that produced the active fluid leaves, not a regenerated Cartesian grid.
HADeviceGrid<Tile>& nativeDeviceGrid(const Mesh& mesh);
}
