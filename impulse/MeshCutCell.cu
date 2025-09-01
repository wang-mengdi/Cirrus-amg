#include "MeshCutCell.h"
#include "AMGSolver.h"
#include <tbb/parallel_for.h>

// xform is the affine transform from mesh-local to world coordinates.
// launch on launch_types
void CalculateSDFOnNodes(HADeviceGrid<Tile>& grid, int node_sdf_channel, const MeshSDFAccel& mesh_sdf, const uint8_t launch_types, const Eigen::Transform<T, 3, Eigen::Affine>& xform) {
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
			auto r_ijk = acc.localNodeOffsetToCoord(i);
			tile.node(node_sdf_channel, r_ijk) = d_sdf_ptr[seq_idx];
		}
	},
		-1, launch_types, LAUNCH_SUBTREE
	);
}

__hostdev__ int CornerInteriorCount(const Tile& tile, const int node_sdf_channel, const Coord& l_ijk, T isovalue = 0) {
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

void CreateAMGLaplacianSystemWithSolidCutOnNodeSDF(HADeviceGrid<Tile>& grid, const int node_sdf_channel, const int coeff_channel, const T R_matrix_coeff) {
	//1. make sure pure NEUMANN cells defined by phi are set to NEUMANN
	//2. calculate face/diag coeffs for all LEAF/GHOST/NONLEAF cells

	//set cell types to NEUMANN for pure NEUMANN cells
	grid.launchVoxelFuncOnAllTiles(
		[=] __device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk)
	{
		auto& tile = info.tile();
		int interior_cnt = CornerInteriorCount(tile, node_sdf_channel, l_ijk);
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
					T face_corner_sdf[4];
					if (axis == 0)
					{
						face_corner_sdf[0] = tile.node(node_sdf_channel, l_ijk + Coord(0, 0, 0));
						face_corner_sdf[1] = tile.node(node_sdf_channel, l_ijk + Coord(0, 0, 1));
						face_corner_sdf[2] = tile.node(node_sdf_channel, l_ijk + Coord(0, 1, 1));
						face_corner_sdf[3] = tile.node(node_sdf_channel, l_ijk + Coord(0, 1, 0));
					}
					else if (axis == 1)
					{
						face_corner_sdf[0] = tile.node(node_sdf_channel, l_ijk + Coord(0, 0, 0));
						face_corner_sdf[1] = tile.node(node_sdf_channel, l_ijk + Coord(0, 0, 1));
						face_corner_sdf[2] = tile.node(node_sdf_channel, l_ijk + Coord(1, 0, 1));
						face_corner_sdf[3] = tile.node(node_sdf_channel, l_ijk + Coord(1, 0, 0));
					}
					else if (axis == 2)
					{
						face_corner_sdf[0] = tile.node(node_sdf_channel, l_ijk + Coord(0, 0, 0));
						face_corner_sdf[1] = tile.node(node_sdf_channel, l_ijk + Coord(0, 1, 0));
						face_corner_sdf[2] = tile.node(node_sdf_channel, l_ijk + Coord(1, 1, 0));
						face_corner_sdf[3] = tile.node(node_sdf_channel, l_ijk + Coord(1, 0, 0));
					}
					coeff = h * FaceFluidRatio(face_corner_sdf[0], face_corner_sdf[1], face_corner_sdf[2], face_corner_sdf[3]);
				}
				tile(coeff_channel + axis, l_ijk) = -coeff;
			});
		}
	},
		LEAF | GHOST | NONLEAF);
	CheckCudaError("CreateLaplacianSystemWithSolidCut: calculating face coefficients failed");

	PrepareLaplacianSystemFromLeafAndGhostCellTypesAndFaceCoeffs(grid, coeff_channel, R_matrix_coeff);
}