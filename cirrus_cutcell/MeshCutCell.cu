#include "MeshCutCell.h"
#include "AMGSolver.h"
#include <tbb/parallel_for.h>
#include "CPUTimer.h"

void CalculateTSDFOnGivenTiles(HADeviceGrid<Tile>& grid,	const thrust::host_vector<HATileInfo<Tile>>& tile_infos, int node_sdf_channel, const MeshSDFAccel& mesh_sdf, const Eigen::Transform<T, 3, Eigen::Affine>& xform, T truncation)
{
	cudaDeviceSynchronize();
	CPUTimer timer;
	timer.start();

	if (tile_infos.empty()) return;
	ASSERT(truncation > 0, "truncation should be positive, got {}", truncation);

	auto h_acc = grid.hostAccessor();

	const int num_tiles = static_cast<int>(tile_infos.size());

	// -------------------------------------------------------------------------
	// Step 1: query tile center sdf, then classify tiles
	// -------------------------------------------------------------------------
	std::vector<Vec> centers(num_tiles);

	tbb::parallel_for(
		tbb::blocked_range<int>(0, num_tiles),
		[&](const tbb::blocked_range<int>& r) {
			for (int local_tile_idx = r.begin(); local_tile_idx != r.end(); ++local_tile_idx) {
				const auto& info = tile_infos[local_tile_idx];

				Coord min_node_coord(0, 0, 0);
				Coord max_node_coord(Tile::DIM, Tile::DIM, Tile::DIM);

				Vec pmin = h_acc.cellCorner(info, min_node_coord);
				Vec pmax = h_acc.cellCorner(info, max_node_coord);
				centers[local_tile_idx] = (pmin + pmax) * (T)0.5;
			}
		});

	std::vector<T> center_sdfs = mesh_sdf.querySDF(centers, xform);
	ASSERT(static_cast<int>(center_sdfs.size()) == num_tiles,
		"center_sdfs size {} mismatch expected {}", center_sdfs.size(), num_tiles);

	thrust::host_vector<HATileInfo<Tile>> far_pos_tiles;
	thrust::host_vector<HATileInfo<Tile>> far_neg_tiles;
	thrust::host_vector<HATileInfo<Tile>> near_tiles;

	far_pos_tiles.reserve(num_tiles);
	far_neg_tiles.reserve(num_tiles);
	near_tiles.reserve(num_tiles);

	for (int local_tile_idx = 0; local_tile_idx < num_tiles; ++local_tile_idx) {
		const auto& info = tile_infos[local_tile_idx];

		const T dx = h_acc.voxelSize(info.mLevel);
		const T tile_extent = dx * (T)Tile::DIM;
		const T r = (T)std::sqrt((T)3) * (T)0.5 * tile_extent;

		const T phi_c = center_sdfs[local_tile_idx];

		if (phi_c > truncation + r) {
			far_pos_tiles.push_back(info);
		}
		else if (phi_c < -truncation - r) {
			far_neg_tiles.push_back(info);
		}
		else {
			near_tiles.push_back(info);
		}
	}

	// -------------------------------------------------------------------------
	// Step 2: fill far tiles directly with +/- truncation
	// -------------------------------------------------------------------------
	auto FillTilesWithConstantTSDF =
		[&](const thrust::host_vector<HATileInfo<Tile>>& fill_tiles, T value) {
		if (fill_tiles.empty()) return;

		thrust::device_vector<HATileInfo<Tile>> d_infos = fill_tiles;
		auto d_infos_ptr = thrust::raw_pointer_cast(d_infos.data());
		const int total_nodes = static_cast<int>(fill_tiles.size()) * Tile::NODESIZE;

		LaunchIndexFunc(
			[=] __device__(int seq_idx) {
			const int local_tile_idx = seq_idx / Tile::NODESIZE;
			const int node_idx = seq_idx % Tile::NODESIZE;

			HATileInfo<Tile> info = d_infos_ptr[local_tile_idx];
			auto& tile = info.tile();
			tile.node(node_sdf_channel, node_idx) = value;
		},
			total_nodes
		);
		};

	FillTilesWithConstantTSDF(far_pos_tiles, truncation);
	FillTilesWithConstantTSDF(far_neg_tiles, -truncation);

	// -------------------------------------------------------------------------
	// Step 3: compute exact sdf only on near tiles, then clamp to [-trunc, trunc]
	// -------------------------------------------------------------------------
	if (!near_tiles.empty()) {
		std::vector<Vec> corners(near_tiles.size() * Tile::NODESIZE);

		tbb::parallel_for(
			tbb::blocked_range<size_t>(0, near_tiles.size()),
			[&](const tbb::blocked_range<size_t>& r) {
				for (size_t local_tile_idx = r.begin(); local_tile_idx != r.end(); ++local_tile_idx) {
					const auto& info = near_tiles[local_tile_idx];
					for (int node_idx = 0; node_idx < Tile::NODESIZE; ++node_idx) {
						Coord r_ijk = h_acc.localNodeOffsetToCoord(node_idx);
						corners[local_tile_idx * Tile::NODESIZE + node_idx] = h_acc.cellCorner(info, r_ijk);
					}
				}
			});

		std::vector<T> h_sdfs = mesh_sdf.querySDF(corners, xform);
		ASSERT(h_sdfs.size() == near_tiles.size() * Tile::NODESIZE,
			"h_sdfs size {} mismatch expected {}", h_sdfs.size(), near_tiles.size() * Tile::NODESIZE);

		thrust::device_vector<HATileInfo<Tile>> d_infos = near_tiles;
		thrust::device_vector<T> d_sdfs = h_sdfs;

		auto d_infos_ptr = thrust::raw_pointer_cast(d_infos.data());
		auto d_sdfs_ptr = thrust::raw_pointer_cast(d_sdfs.data());
		const int total_nodes = static_cast<int>(near_tiles.size()) * Tile::NODESIZE;

		LaunchIndexFunc(
			[=] __device__(int seq_idx) {
			const int local_tile_idx = seq_idx / Tile::NODESIZE;
			const int node_idx = seq_idx % Tile::NODESIZE;

			HATileInfo<Tile> info = d_infos_ptr[local_tile_idx];
			auto& tile = info.tile();

			T sdf = d_sdfs_ptr[seq_idx];
			if (sdf > truncation) sdf = truncation;
			if (sdf < -truncation) sdf = -truncation;

			tile.node(node_sdf_channel, node_idx) = sdf;
		},
			total_nodes
		);
	}

	cudaDeviceSynchronize();
	double elapsed = timer.stop();

	const int total_nodes_all = num_tiles * Tile::NODESIZE;
	const int total_nodes_near = static_cast<int>(near_tiles.size()) * Tile::NODESIZE;

	Info("CalculateSDFOnGivenTiles: total {} tiles, near {} tiles, far+ {} tiles, far- {} tiles",
		num_tiles, near_tiles.size(), far_pos_tiles.size(), far_neg_tiles.size());
	Info("CalculateSDFOnGivenTiles: wrote TSDF on {}M nodes in {} ms, exact queries on {}M nodes, throughput: {}M nodes/s",
		total_nodes_all / 1e6,
		elapsed,
		total_nodes_near / 1e6,
		total_nodes_all / elapsed / 1000.0);
}

// xform is the affine transform from mesh-local to world coordinates.
// launch on launch_types
void CalculateTSDFOnNodes(HADeviceGrid<Tile>& grid, int node_sdf_channel, const MeshSDFAccel& mesh_sdf, const uint8_t launch_types, const Eigen::Transform<T, 3, Eigen::Affine>& xform, const T truncation) {
	CalculateTSDFOnGivenTiles(grid, grid.hAllTiles, node_sdf_channel, mesh_sdf, xform, truncation);

	////step 1: gather tile corners to a std::vector<Vec>
	//std::vector<Vec> corners(grid.hAllTiles.size() * Tile::NODESIZE);
	//auto h_acc = grid.hostAccessor();
	////use tbb to parallely fill corners
	//tbb::parallel_for(tbb::blocked_range<size_t>(0, grid.hAllTiles.size()), [&](const tbb::blocked_range<size_t>& r) {
	//	for (size_t tile_idx = r.begin(); tile_idx != r.end(); ++tile_idx) {
	//		HATileInfo<Tile> info = grid.hAllTiles[tile_idx];
	//		if (info.mType & launch_types) {
	//			for (int node_idx = 0; node_idx < Tile::NODESIZE; ++node_idx) {
	//				auto l_ijk = h_acc.localNodeOffsetToCoord(node_idx);
	//				auto pos = h_acc.cellCorner(info, l_ijk);
	//				corners[tile_idx * Tile::NODESIZE + node_idx] = pos;

	//				//if (info.mTileCoord == Coord(2, 4, 7) && l_ijk == Coord(8, 7, 0)) {
	//				//	auto actual_sdf = (Vec(0.5, 0.5, 0.8) - pos).length() - 0.1;
	//				//	std::vector<Vec> test_pos = { pos };
	//				//	auto query_sdf = mesh_sdf.querySDF(test_pos, xform)[0];
	//				//	printf("corner: %f, %f, %f tile idx %d, seq idx: %d actual sdf: %f query sdf: %f\n", pos[0], pos[1], pos[2], tile_idx, tile_idx * Tile::NODESIZE + node_idx, actual_sdf, query_sdf);
	//				//}

	//			}
	//		}
	//	}
	//	});

	////step 2: launch mesh_sdf.query on corners
	//std::vector<T> h_sdfs = mesh_sdf.querySDF(corners, xform);
	//thrust::device_vector<T> d_sdfs = h_sdfs;

	////step 3: copy sdf back to grid
	//auto d_sdf_ptr = d_sdfs.data().get();
	//grid.launchNodeFuncWithTileIdxOnAllTiles(
	//	[=] __device__(HATileAccessor<Tile> acc, const int tile_idx, HATileInfo<Tile>&info, const Coord &r_ijk) {
	//	auto& tile = info.tile();
	//	int node_idx = acc.localNodeCoordToOffset(r_ijk);
	//	int seq_idx = tile_idx * Tile::NODESIZE + node_idx;
	//	auto sdf_value = d_sdf_ptr[seq_idx];
	//	tile.node(node_sdf_channel, r_ijk) = sdf_value;

	//	//if (info.mTileCoord == Coord(2, 4, 7) && r_ijk == Coord(8, 7, 0)) {
	//	//	printf("write sdf: %f, tile idx: %d, seq idx: %d\n", d_sdf_ptr[seq_idx], tile_idx, seq_idx);
	//	//}
	//},
	//	launch_types
	//);
}

__hostdev__ int CellCornerSDFInsideCount(const Tile& tile, const int node_sdf_channel, const Coord& l_ijk, T isovalue) {
	int inside_cnt = 0;
	for (int di : {0, 1})
	{
		for (int dj : {0, 1})
		{
			for (int dk : {0, 1})
			{
				Coord r_ijk;
				r_ijk[0] = l_ijk[0] + di;
				r_ijk[1] = l_ijk[1] + dj;
				r_ijk[2] = l_ijk[2] + dk;
				if (tile.node(node_sdf_channel, r_ijk) - isovalue < 0)
				{
					inside_cnt++;
				}
			}
		}
	}
	return inside_cnt;
}


__hostdev__ float FracInside(float a, float b)
{
	if (a < 0.0 && b < 0.0)
		return 0.0;
	else if (a < 0.0 && b >= 0.0)
		return b / (b - a);
	else if (a >= 0.0 && b < 0.0)
		return a / (a - b);
	else
		return 1.0;
}

__device__ cuda_vec4_t<T> FaceCornerSDFs(const int node_sdf_channel, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const Coord& l_ijk, const int axis) {
	auto& tile = info.tile();
	cuda_vec4_t<T> face_corner_sdf;
	//negative face of the cell
	if (axis == 0)
	{
		face_corner_sdf.x = tile.node(node_sdf_channel, l_ijk + Coord(0, 0, 0));
		face_corner_sdf.y = tile.node(node_sdf_channel, l_ijk + Coord(0, 0, 1));
		face_corner_sdf.z = tile.node(node_sdf_channel, l_ijk + Coord(0, 1, 1));
		face_corner_sdf.w = tile.node(node_sdf_channel, l_ijk + Coord(0, 1, 0));
	}
	else if (axis == 1)
	{
		face_corner_sdf.x = tile.node(node_sdf_channel, l_ijk + Coord(0, 0, 0));
		face_corner_sdf.y = tile.node(node_sdf_channel, l_ijk + Coord(0, 0, 1));
		face_corner_sdf.z = tile.node(node_sdf_channel, l_ijk + Coord(1, 0, 1));
		face_corner_sdf.w = tile.node(node_sdf_channel, l_ijk + Coord(1, 0, 0));
	}
	else if (axis == 2)
	{
		face_corner_sdf.x = tile.node(node_sdf_channel, l_ijk + Coord(0, 0, 0));
		face_corner_sdf.y = tile.node(node_sdf_channel, l_ijk + Coord(0, 1, 0));
		face_corner_sdf.z = tile.node(node_sdf_channel, l_ijk + Coord(1, 1, 0));
		face_corner_sdf.w = tile.node(node_sdf_channel, l_ijk + Coord(1, 0, 0));
	}
	else {
		CUDA_ASSERT(false, "axis=%d", axis);
	}
	return face_corner_sdf;
}

__hostdev__ float FaceFluidRatio(float phi0, float phi1, float phi2, float phi3)
{
	// calculate vol
	float ret;
	if (phi0 < 0 && phi1 < 0 && phi2 < 0 && phi3 < 0)
	{
		ret = 0;
	}
	else if (phi0 < 0 && phi1 < 0 && phi2 < 0 && phi3 >= 0)
	{
		float edge1 = FracInside(phi3, phi2);
		float edge2 = FracInside(phi3, phi0);
		ret = 0.5 * edge1 * edge2;
	}
	else if (phi0 < 0 && phi1 < 0 && phi2 >= 0 && phi3 < 0)
	{
		float edge1 = FracInside(phi2, phi1);
		float edge2 = FracInside(phi2, phi3);
		ret = 0.5 * edge1 * edge2;
	}
	else if (phi0 < 0 && phi1 < 0 && phi2 >= 0 && phi3 >= 0)
	{
		float edge1 = FracInside(phi2, phi1);
		float edge2 = FracInside(phi3, phi0);
		ret = 0.5 * (edge1 + edge2);
	}
	else if (phi0 < 0 && phi1 >= 0 && phi2 < 0 && phi3 < 0)
	{
		float edge1 = FracInside(phi1, phi0);
		float edge2 = FracInside(phi1, phi2);
		ret = 0.5 * edge1 * edge2;
	}
	else if (phi0 < 0 && phi1 >= 0 && phi2 < 0 && phi3 >= 0)
	{
		float edge1 = FracInside(phi1, phi0);
		float edge2 = FracInside(phi1, phi2);
		float edge3 = FracInside(phi3, phi2);
		float edge4 = FracInside(phi3, phi0);
		ret = 0.5 * edge1 * edge2 + 0.5 * edge3 * edge4;
	}
	else if (phi0 < 0 && phi1 >= 0 && phi2 >= 0 && phi3 < 0)
	{
		float edge1 = FracInside(phi1, phi0);
		float edge2 = FracInside(phi2, phi3);
		ret = 0.5 * (edge1 + edge2);
	}
	else if (phi0 < 0 && phi1 >= 0 && phi2 >= 0 && phi3 >= 0)
	{
		float edge1 = 1.0 - FracInside(phi0, phi1);
		float edge2 = 1.0 - FracInside(phi0, phi3);
		ret = 1.0 - 0.5 * edge1 * edge2;
	}
	else if (phi0 >= 0 && phi1 < 0 && phi2 < 0 && phi3 < 0)
	{
		float edge1 = FracInside(phi0, phi1);
		float edge2 = FracInside(phi0, phi3);
		ret = 0.5 * edge1 * edge2;
	}
	else if (phi0 >= 0 && phi1 < 0 && phi2 < 0 && phi3 >= 0)
	{
		float edge1 = FracInside(phi0, phi1);
		float edge2 = FracInside(phi3, phi2);
		ret = 0.5 * (edge1 + edge2);
	}
	else if (phi0 >= 0 && phi1 < 0 && phi2 >= 0 && phi3 < 0)
	{
		float edge1 = FracInside(phi0, phi1);
		float edge2 = FracInside(phi0, phi3);
		float edge3 = FracInside(phi2, phi1);
		float edge4 = FracInside(phi2, phi3);
		ret = 0.5 * edge1 * edge2 + 0.5 * edge3 * edge4;
	}
	else if (phi0 >= 0 && phi1 < 0 && phi2 >= 0 && phi3 >= 0)
	{
		float edge1 = 1.0 - FracInside(phi1, phi0);
		float edge2 = 1.0 - FracInside(phi1, phi2);
		ret = 1.0 - 0.5 * edge1 * edge2;
	}
	else if (phi0 >= 0 && phi1 >= 0 && phi2 < 0 && phi3 < 0)
	{
		float edge1 = FracInside(phi0, phi3);
		float edge2 = FracInside(phi1, phi2);
		ret = 0.5 * (edge1 + edge2);
	}
	else if (phi0 >= 0 && phi1 >= 0 && phi2 < 0 && phi3 >= 0)
	{
		float edge1 = 1.0 - FracInside(phi2, phi1);
		float edge2 = 1.0 - FracInside(phi2, phi3);
		ret = 1.0 - 0.5 * edge1 * edge2;
	}
	else if (phi0 >= 0 && phi1 >= 0 && phi2 >= 0 && phi3 < 0)
	{
		float edge1 = 1.0 - FracInside(phi3, phi0);
		float edge2 = 1.0 - FracInside(phi3, phi2);
		ret = 1.0 - 0.5 * edge1 * edge2;
	}
	else
	{
		ret = 1;
	}
	if (ret < 0.1 && ret != 0)
		ret = 0.1;
	return ret;
}

__hostdev__ T FaceFluidRatio(const cuda_vec4_t<T>& corner_phis) {
	return FaceFluidRatio(corner_phis.x, corner_phis.y, corner_phis.z, corner_phis.w);
}



void CreateAMGLaplacianSystemWithSolidCutOnNodeSDF(HADeviceGrid<Tile>& grid, const int node_sdf_channel, const int coeff_channel, const T R_matrix_coeff) {
	//1. make sure pure NEUMANN cells defined by phi are set to NEUMANN
	//2. calculate face/diag coeffs for all LEAF/GHOST/NONLEAF cells

	//set cell types to NEUMANN for pure NEUMANN cells
	grid.launchVoxelFuncOnAllTiles(
		[=] __device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk)
	{
		auto& tile = info.tile();
		auto h = acc.voxelSize(info);
		int interior_cnt = CellCornerSDFInsideCount(tile, node_sdf_channel, l_ijk, h * SDF_REL_EPS);
		if (interior_cnt == 8) tile.type(l_ijk) = NEUMANN;
	},
		LEAF);
	//here we have correct cell types for all LEAF cells
	CheckCudaError("CreateLaplacianSystemWithSolidCut: setting NEUMANN cell types failed");

	//set GHOST cell types
	grid.launchVoxelFuncOnAllTiles(
		[=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk)
	{
		auto& tile = info.tile();
		auto g_ijk = acc.composeGlobalCoord(info.mTileCoord, l_ijk);
		auto pg_ijk = acc.parentCoord(g_ijk);
		HATileInfo<Tile> pinfo;
		Coord pl_ijk;
		acc.findVoxel(info.mLevel - 1, pg_ijk, pinfo, pl_ijk);
		if (!pinfo.empty())
			tile.type(l_ijk) = pinfo.tile().type(pl_ijk);
		else
			tile.type(l_ijk) = DIRICHLET;
	},
		GHOST);
	CheckCudaError("CreateLaplacianSystemWithSolidCut: setting GHOST cell types failed");

	//calculate face terms for LEAF and GHOST cells
	grid.launchVoxelFuncOnAllTiles(
		[=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk)
	{
		auto h = acc.voxelSize(info);
		Tile& tile = info.tile();
		uint8_t ttype0 = info.mType;
		uint8_t ctype0 = tile.type(l_ijk);

		if (ttype0 == NONLEAF) {
			// for NONLEAF, we will calculate the coefficients later in the solver
			for (int axis = 0; axis < 3; ++axis) {
				tile(coeff_channel + axis, l_ijk) = 0;
			}
		}
		else {
			//LEAF or GHOST

			auto cell_center = acc.cellCenter(info, l_ijk);
			// iterate neighbors
			acc.iterateSameLevelNeighborVoxels(info, l_ijk,
				[&] __device__(const HATileInfo<Tile> &ninfo, const Coord & nl_ijk, const int axis, const int sgn)
			{
				if (sgn != -1)
					return;
				T coeff = 0;

				uint8_t ttype1;
				uint8_t ctype1;
				if (ninfo.empty())
				{
					ttype1 = ttype0;
					ctype1 = DIRICHLET;
				}
				else
				{
					auto& ntile = ninfo.tile();
					ttype1 = ninfo.mType;
					ctype1 = ntile.type(nl_ijk);
				}
				bool both_leafs = ((ttype0 & LEAF) && (ttype1 & LEAF));
				bool one_leaf_one_ghost = ((ttype0 & LEAF && ttype1 & GHOST) || (ttype0 & GHOST && ttype1 & LEAF));
				bool has_neumann = (ctype0 & NEUMANN || ctype1 & NEUMANN);
				bool has_interior = (ctype0 & INTERIOR || ctype1 & INTERIOR);
				if ((both_leafs || one_leaf_one_ghost) && !has_neumann && has_interior)
				{
					cuda_vec4_t<T> face_corner_sdf = FaceCornerSDFs(node_sdf_channel, acc, info, l_ijk, axis);

					//T face_corner_sdf[4];
					//if (axis == 0)
					//{
					//	face_corner_sdf[0] = tile.node(node_sdf_channel, l_ijk + Coord(0, 0, 0));
					//	face_corner_sdf[1] = tile.node(node_sdf_channel, l_ijk + Coord(0, 0, 1));
					//	face_corner_sdf[2] = tile.node(node_sdf_channel, l_ijk + Coord(0, 1, 1));
					//	face_corner_sdf[3] = tile.node(node_sdf_channel, l_ijk + Coord(0, 1, 0));
					//}
					//else if (axis == 1)
					//{
					//	face_corner_sdf[0] = tile.node(node_sdf_channel, l_ijk + Coord(0, 0, 0));
					//	face_corner_sdf[1] = tile.node(node_sdf_channel, l_ijk + Coord(0, 0, 1));
					//	face_corner_sdf[2] = tile.node(node_sdf_channel, l_ijk + Coord(1, 0, 1));
					//	face_corner_sdf[3] = tile.node(node_sdf_channel, l_ijk + Coord(1, 0, 0));
					//}
					//else if (axis == 2)
					//{
					//	face_corner_sdf[0] = tile.node(node_sdf_channel, l_ijk + Coord(0, 0, 0));
					//	face_corner_sdf[1] = tile.node(node_sdf_channel, l_ijk + Coord(0, 1, 0));
					//	face_corner_sdf[2] = tile.node(node_sdf_channel, l_ijk + Coord(1, 1, 0));
					//	face_corner_sdf[3] = tile.node(node_sdf_channel, l_ijk + Coord(1, 0, 0));
					//}
					
					//coeff = h * FaceFluidRatio(face_corner_sdf.x, face_corner_sdf.y, face_corner_sdf.z, face_corner_sdf.w);
					coeff = h * FaceFluidRatio(face_corner_sdf);

					//{
					//	auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
					//	if (g_ijk == Coord(147, 130, 127)) {
					//		printf("tile %d, local %d %d %d, global %d %d %d, axis %d, sgn %d, coeff %f sdfs %f %f %f %f\n", info.mTileCoord.x(), l_ijk.x(), l_ijk.y(), l_ijk.z(), g_ijk.x(), g_ijk.y(), g_ijk.z(), axis, sgn, coeff, face_corner_sdf.x, face_corner_sdf.y, face_corner_sdf.z, face_corner_sdf.w);
					//	}
					//}
				}



				tile(coeff_channel + axis, l_ijk) = -coeff;
			});
		}
	},
		LEAF | GHOST | NONLEAF);
	CheckCudaError("CreateLaplacianSystemWithSolidCut: calculating face coefficients failed");

	PrepareLaplacianSystemFromLeafAndGhostCellTypesAndFaceCoeffs(grid, coeff_channel, R_matrix_coeff);
}