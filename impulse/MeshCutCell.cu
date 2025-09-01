#include "MeshCutCell.h"
#include <tbb/parallel_for.h>

// xform is the affine transform from mesh-local to world coordinates.
// launch on launch_types
void CalculateSDFOnNodes(HADeviceGrid<Tile>& grid, int sdf_channel, const MeshSDFAccel& mesh_sdf, const uint8_t launch_types, const Eigen::Transform<T, 3, Eigen::Affine>& xform) {
	//step 1: gather tile corners to a std::vector<Vec>
	std::vector<Vec> corners(grid.hAllTiles.size() * Tile::NODESIZE);
	auto h_acc = grid.hostAccessor();
	//use tbb to parallely fill corners
	tbb::parallel_for(tbb::blocked_range<size_t>(0, grid.hAllTiles.size()), [&](const tbb::blocked_range<size_t>& r) {
		for (size_t tile_idx = r.begin(); tile_idx != r.end(); ++tile_idx) {
			HATileInfo<Tile> info = grid.hAllTiles[tile_idx];
			if (info.mType & launch_types) {
				for (int node_idx = 0; node_idx < Tile::NODESIZE; ++node_idx) {
					auto l_ijk = h_acc.localNodeOffsetToCoord(node_idx);
					auto pos = h_acc.cellCorner(info, l_ijk);
					corners[tile_idx * Tile::NODESIZE + node_idx] = pos;
				}
			}
		}
		});

	//step 2: launch mesh_sdf.query on corners
	std::vector<T> h_sdfs = mesh_sdf.querySDF(corners, xform);
	thrust::device_vector<T> d_sdfs = h_sdfs;

	//step 3: copy sdf back to grid
	auto d_sdf_ptr = d_sdfs.data().get();
	grid.launchTileFunc(
		[=] __device__(HATileAccessor<Tile> acc, const int tile_idx, HATileInfo<Tile>&info) {
		auto& tile = info.tile();
		for (int i = 0; i < Tile::NODESIZE; ++i) {
			int seq_idx = tile_idx * Tile::NODESIZE + i;
			auto l_ijk = acc.localNodeOffsetToCoord(i);
			tile.node(sdf_channel, l_ijk) = d_sdf_ptr[seq_idx];
		}
	},
		-1, launch_types, LAUNCH_SUBTREE
	);
}