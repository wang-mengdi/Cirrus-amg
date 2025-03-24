#include "AMGSolver.h"
#include "PoissonGrid.h"
#include "CPUTimer.h"
#include "GMGSolver.h"
//#include "PoissonSolverOptimized.h"

#include "PoissonIOFunc.h"
#include <polyscope/polyscope.h>

__forceinline__ __device__ T NegativeLaplacianCoeff(T h, uint8_t ttype0, uint8_t ttype1, uint8_t ctype0, const uint8_t ctype1) {
    //valid if both are leafs, or one leaf one ghost
    bool both_leafs = ((ttype0 & LEAF) && (ttype1 & LEAF));
    bool one_leaf_one_ghost = ((ttype0 & LEAF && ttype1 & GHOST) || (ttype0 & GHOST && ttype1 & LEAF));

	//valid if none of them is NEUMANN and at least is INTERIOR
    //ignore if one of them is NEUMANN
    bool has_neumann = (ctype0 & NEUMANN || ctype1 & NEUMANN);
	bool has_interior = (ctype0 & INTERIOR || ctype1 & INTERIOR);

	bool valid = (both_leafs || one_leaf_one_ghost) && !has_neumann && has_interior;
    return valid ? h : 0;
}

//will be called on fine tiles
__global__ void CoarsenOffDiagCoefficientsOneStepKernel(HATileAccessor<Tile> acc, HATileInfo<Tile>* fine_tiles, int fine_subtree_level, uint8_t fine_tile_types, int fine_u_channel, int coarse_u_channel, Tile::T R_mat_coeff, uint8_t cell_types) {
    __shared__ T data[3][8 * 8 * 8];
    int bi = blockIdx.x, ti = threadIdx.x;
    auto& finfo = fine_tiles[bi];

    if (!(finfo.subtreeType(fine_subtree_level) & fine_tile_types)) {
        return;
    }
    auto& ftile = finfo.tile();

    for (int axis : {0, 1, 2}) {
        for (int i = 0; i < 4; i++) {
            int vi = i * 128 + ti;
            uint8_t ctype = ftile.type(vi);
            data[axis][vi] = (ctype & cell_types) ? ftile(fine_u_channel + axis, vi) : 0;
        }
    }
    __syncthreads();

    if (ti < 64) {
        Coord fl_ijk(ti / 16 * 2, (ti / 4) % 4 * 2, ti % 4 * 2);
        Coord fg_ijk = acc.localToGlobalCoord(finfo, fl_ijk);
        Coord cg_ijk = acc.parentCoord(fg_ijk);
        HATileInfo<Tile> cinfo; Coord cl_ijk;//coarse tile and local coord
        acc.findVoxel(finfo.mLevel - 1, cg_ijk, cinfo, cl_ijk);
        if (!cinfo.empty()) {
            auto& ctile = cinfo.tile();
            for (int axis : {0, 1, 2}) {
                T sum = 0;
                for (int jj : {0, 1}) {
                    for (int kk : {0, 1}) {
                        Coord fl1_ijk = fl_ijk + acc.rotateCoord(axis, Coord(0, jj, kk));
                        int vi1 = acc.localCoordToOffset(fl1_ijk);
                        sum += data[axis][vi1];

                        //if (cg_ijk == Coord(8, 8, 10) && cinfo.mLevel == 1) {
                        //    printf("axis %d l_ijk %d %d %d vi1 %d data %f sum %f\n", axis, fl1_ijk[0], fl1_ijk[1], fl1_ijk[2], vi1, data[axis][vi1], sum);
                        //}

                    }
                }
                
                //T coeff = finfo.mType == GHOST ? 1 : R_mat_coeff;
                T coeff = R_mat_coeff;
                ctile(coarse_u_channel + axis, cl_ijk) += sum * coeff;
            }
        }

    }
}

void CoarsenTypesAndAMGCoeffs(HADeviceGrid<Tile>& grid, const int coeff_channel, const T R_matrix_coeff) {
    // Before calling this function, cell types for LEAFs must be filled
    // This function will:
    // 1. fill GHOST cell types
    // 2. calculate off-diag and diag coeffs for LEAF and GHOST cells
    // 3. calculate cell types off-diag and diag coeffs for NONLEAF cells
    // all non-zero entries in R equal to R_matrix_coeff

    //step 0: clear coeff channel for all tiles
    grid.launchVoxelFuncOnAllTiles(
        [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();
        for (int axis : {0, 1, 2}) {
            tile(coeff_channel + axis, l_ijk) = 0;
        }
        tile(coeff_channel + 3, l_ijk) = 0;
    },
        LEAF | GHOST | NONLEAF
    );

    //step 1: fill GHOST cell types
    grid.launchVoxelFuncOnAllTiles(
        [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();
        auto g_ijk = acc.composeGlobalCoord(info.mTileCoord, l_ijk);
        auto pg_ijk = acc.parentCoord(g_ijk);
        HATileInfo<Tile> pinfo; Coord pl_ijk;
        acc.findVoxel(info.mLevel - 1, pg_ijk, pinfo, pl_ijk);
        if (!pinfo.empty()) tile.type(l_ijk) = pinfo.tile().type(pl_ijk);
        else tile.type(l_ijk) = DIRICHLET;
    }, GHOST);

    //step 2: calculate face terms on LEAF and GHOST cells
    grid.launchVoxelFuncOnAllTiles(
        [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto h = acc.voxelSize(info);
        Tile& tile = info.tile();
        uint8_t ttype0 = info.mType;
        uint8_t ctype0 = tile.type(l_ijk);

        //iterate neighbors
        acc.iterateSameLevelNeighborVoxels(info, l_ijk,
            [&]__device__(const HATileInfo<Tile>&ninfo, const Coord & nl_ijk, const int axis, const int sgn) {
            if (sgn != -1) return;

            uint8_t ttype1;
            uint8_t ctype1;
            T coeff = 0;
            if (ninfo.empty()) {
                ttype1 = ttype0;
                ctype1 = DIRICHLET;
            }
            else {
                auto& ntile = ninfo.tile();
                ttype1 = ninfo.mType;
                ctype1 = ntile.type(nl_ijk);
            }
            coeff = NegativeLaplacianCoeff(h, ttype0, ttype1, ctype0, ctype1);
            //if (ttype0 == GHOST) coeff *= 0.5;
            tile(coeff_channel + axis, l_ijk) = -coeff;
        });
    }, LEAF | GHOST);

    //step 3: accumulate face terms
    for (int i = grid.mMaxLevel; i >= 0; i--) {
        int num_tiles = grid.hNumTiles[i];
        if (num_tiles > 0) {
            CoarsenOffDiagCoefficientsOneStepKernel << <num_tiles, 128 >> > (
                grid.deviceAccessor(),
                thrust::raw_pointer_cast(grid.dTileArrays[i].data()),
                i,
                LEAF | GHOST,
                coeff_channel,
                coeff_channel,
                R_matrix_coeff,
                INTERIOR | DIRICHLET | NEUMANN
                );
        }
    }

    //step 4: calculate diagonal terms for LEAF cells and coarsen diagonal terms for NONLEAF cells
    grid.launchVoxelFunc(
        [=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
        auto& tile = info.tile();
        auto g_ijk = acc.composeGlobalCoord(info.mTileCoord, l_ijk);

        if (info.mType & (LEAF | GHOST)) {
            auto h = acc.voxelSize(info);
            T diag_coeff = 0;
            if (tile.type(l_ijk) & INTERIOR) {
                acc.iterateSameLevelNeighborVoxels(info, l_ijk,
                    [&]__device__(const HATileInfo<Tile>&ninfo, const Coord & nl_ijk, const int axis, const int sgn) {
                    T c0 = tile(coeff_channel + axis, l_ijk);
                    //if ninfo if empty, we regard it as DIRICHLET and has coeff -h
                    T c1 = ninfo.empty() ? -h : ninfo.tile()(coeff_channel + axis, nl_ijk);
                    T face_term = (sgn == -1) ? c0 : c1;

                    //if (ninfo.mType == GHOST) face_term /= 8;
                    //if (ninfo.mType == GHOST) face_term /= 4;

                    diag_coeff -= face_term;
                });
            }
            tile(coeff_channel + 3, l_ijk) = diag_coeff;
        }
        else {
            //coarsen diagonal terms at NONLEAF cells
            int interior_cnt = 0;
            bool has_ghost_child = false;
            uint8_t ctypes[2][2][2];
            T coff_diag[3][2][2][2];//8 children
            T diag_sum = 0;
            for (int ci = 0; ci < 2; ci++) {
                for (int cj = 0; cj < 2; cj++) {
                    for (int ck = 0; ck < 2; ck++) {
                        Coord cg_ijk(g_ijk[0] * 2 + ci, g_ijk[1] * 2 + cj, g_ijk[2] * 2 + ck);
                        HATileInfo<Tile> cinfo; Coord cl_ijk;
                        acc.findVoxel(info.mLevel + 1, cg_ijk, cinfo, cl_ijk);
                        if (!cinfo.empty()) {
                            if (cinfo.mType == GHOST) has_ghost_child = true;
                            auto& ctile = cinfo.tile();
                            diag_sum += ctile(coeff_channel + 3, cl_ijk);
                            interior_cnt += (ctile.type(cl_ijk) == INTERIOR);
                            ctypes[ci][cj][ck] = ctile.type(cl_ijk);
                            for (int axis : {0, 1, 2}) {
                                coff_diag[axis][ci][cj][ck] = ctile(coeff_channel + axis, cl_ijk);
                            }
                        }
                        else {
                            ctypes[ci][cj][ck] = DIRICHLET;
                            for (int axis : {0, 1, 2}) {
                                coff_diag[axis][ci][cj][ck] = 0;
                            }
                        }

                    }
                }
            }

            for (int ci = 0; ci < 2; ci++) {
                for (int cj = 0; cj < 2; cj++) {
                    for (int ck = 0; ck < 2; ck++) {
                        if (ctypes[0][cj][ck] == INTERIOR && ctypes[1][cj][ck] == INTERIOR) {
                            //it will be added twice
                            diag_sum += coff_diag[0][1][cj][ck];
                        }
                        if (ctypes[ci][0][ck] == INTERIOR && ctypes[ci][1][ck] == INTERIOR) {
                            //it will be added twice
                            diag_sum += coff_diag[1][ci][1][ck];
                        }
                        if (ctypes[ci][cj][0] == INTERIOR && ctypes[ci][cj][1] == INTERIOR) {
                            //it will be added twice
                            diag_sum += coff_diag[2][ci][cj][1];
                        }

                        ////seems not working
                        //Coord coff(ci, cj, ck);
                        //for (int axis : {0, 1, 2}) {
                        //    for (int sgn : {0, 1}) {
                        //        if (sgn == coff[axis]) {
                        //            //test if the child neighbors a ghost cell
                        //            Coord ncg_ijk(g_ijk[0] * 2 + ci, g_ijk[1] * 2 + cj, g_ijk[2] * 2 + ck);
                        //            ncg_ijk[axis] += (sgn == 1) ? 1 : -1;
                        //            HATileInfo<Tile> ncinfo; Coord ncl_ijk;
                        //            acc.findVoxel(info.mLevel + 1, ncg_ijk, ncinfo, ncl_ijk);
                        //            if (!ncinfo.empty() && ncinfo.mType == GHOST) {
                        //                T off0 = coff_diag[axis][ci][cj][ck];
                        //                T off1 = ncinfo.tile()(coeff_channel + axis, ncl_ijk);
                        //                T face_term = (sgn == 1) ? off1 : off0;

                        //                for (int qi : {0, 1}) {
                        //                    for (int qj : {0, 1}) {
                        //                        for (int qk : {0, 1}) {
                        //                            if (ci == qi && cj == qj && ck == qk) continue;
                        //                            if (ctypes[qi][qj][qk] == INTERIOR) {
                        //                                //diag_sum -= face_term / 8;
                        //                                diag_sum -= face_term / 4;
                        //                            }
                        //                        }
                        //                    }
                        //                }

                        //                //diag_sum -= 0.5 * (sgn == 1) ? off1 : off0;
                        //                //diag_sum += 0.5 * (sgn == 1) ? off1 : off0;
                        //                //printf("off0: %f off1: %f\n", off0, off1);
                        //            }
                        //        }
                        //    }
                        //}
                    }
                }
            }

            tile.type(l_ijk) = interior_cnt > 0 ? INTERIOR : DIRICHLET;
            tile(coeff_channel + 3, l_ijk) = diag_sum * R_matrix_coeff;
        }
    },
        -1, LEAF | GHOST | NONLEAF, LAUNCH_SUBTREE, FINE_FIRST);
}

class AMGLaplacianTileData {
public:
    static constexpr int halo = 1;
    static constexpr int SN = 8 + halo * 2;
	uint8_t absttype[SN][SN][SN];    // tile type
    T x[SN][SN][SN];               // x channel
	T diag[8][8][8];			   // Diagonal coefficient
	T off_diag0[9][8][8];		  // Off-diagonal coefficient 0
	T off_diag1[8][9][8];		  // Off-diagonal coefficient 1
	T off_diag2[8][8][9];		  // Off-diagonal coefficient 2
    //uint8_t ttype[SN][SN][SN];    // tile type

    //they will take actual tile local coords
	__device__ uint8_t& absttypeT(Coord l_ijk) { return absttype[l_ijk[0] + halo][l_ijk[1] + halo][l_ijk[2] + halo]; }
    __device__ T& xValueT(Coord l_ijk) { return x[l_ijk[0] + halo][l_ijk[1] + halo][l_ijk[2] + halo]; }
	__device__ T& diagValueT(Coord l_ijk) { return diag[l_ijk[0]][l_ijk[1]][l_ijk[2]]; }
	__device__ T& offDiag0ValueT(Coord l_ijk) { return off_diag0[l_ijk[0]][l_ijk[1]][l_ijk[2]]; }
	__device__ T& offDiag1ValueT(Coord l_ijk) { return off_diag1[l_ijk[0]][l_ijk[1]][l_ijk[2]]; }
	__device__ T& offDiag2ValueT(Coord l_ijk) { return off_diag2[l_ijk[0]][l_ijk[1]][l_ijk[2]]; }
    //take the face coeff of the negative direction of actual tile local voxel l_ijk
	__device__ T& offDiagValueT(int axis, Coord l_ijk) {
		if (axis == 0) return off_diag0[l_ijk[0]][l_ijk[1]][l_ijk[2]];
		if (axis == 1) return off_diag1[l_ijk[0]][l_ijk[1]][l_ijk[2]];
        return off_diag2[l_ijk[0]][l_ijk[1]][l_ijk[2]];
	}

    __device__ T negativeLap(Coord l_ijk) {
        //it's DIRICHLET of NEUMANN
        if (diagValueT(l_ijk) == 0) return 0;

        T x0 = xValueT(l_ijk);
		T sum = x0 * diagValueT(l_ijk);

        ////if (l_ijk == Coord(4, 4, 4)) 
        //{
        //    printf("AMG x0 %f diag %f sum %f\n", x0, diagValueT(l_ijk), sum);
        //}

		for (int axis : { 0, 1, 2 }) {
			for (int sgn : { -1, 1 }) {
				Coord nl_ijk = l_ijk;
				nl_ijk[axis] += sgn;
                //the "upwind" cell, with higher axis coord
                Coord ul_ijk = l_ijk;
                ul_ijk[axis] += (sgn == -1) ? 0 : 1;

			    

                sum += offDiagValueT(axis, ul_ijk) * xValueT(nl_ijk);

                ////if (l_ijk == Coord(4, 4, 4)) 
                //{
                //    T h = 1. / 16;
                //    T induced_u = (xValueT(nl_ijk) - x0) / h;
                //    printf("AMG axis %d sgn %d nl_ijk %d %d %d ul_ijk %d %d %d off %f nx %f induced u %f sum %f\n", axis, sgn, nl_ijk[0], nl_ijk[1], nl_ijk[2], ul_ijk[0], ul_ijk[1], ul_ijk[2], offDiagValueT(axis, ul_ijk), xValueT(nl_ijk), induced_u, sum);
                //}
			}
		}

		return sum;
    }

    __device__ T negativeLapOuterNonLeaf(Coord l_ijk) {
        //this is LEAF, only calculate terms induced by NONLEAF cells
        //it's DIRICHLET of NEUMANN
        if (diagValueT(l_ijk) == 0) return 0;
        T x0 = absttypeT(l_ijk) == NONLEAF ? xValueT(l_ijk) : 0;
        T sum = x0 * diagValueT(l_ijk);
        for (int axis : { 0, 1, 2 }) {
            for (int sgn : { -1, 1 }) {
                Coord nl_ijk = l_ijk;
                nl_ijk[axis] += sgn;
                //the "upwind" cell, with higher axis coord
                Coord ul_ijk = l_ijk;
                ul_ijk[axis] += (sgn == -1) ? 0 : 1;

				T x1 = absttypeT(nl_ijk) == NONLEAF ? xValueT(nl_ijk) : 0;

                sum += offDiagValueT(axis, ul_ijk) * x1;
            }
        }

        return sum;
    }
};

//vi in [0,64)
__device__ static Coord FaceNeighborLocalCoord(int axis, int sgn, const int vi) {
    int axk = (sgn == -1) ? -1 : 8;
    int i = vi / 8, j = vi % 8;
    return axis == 0 ? Coord(axk, i, j) : axis == 1 ? Coord(i, axk, j) : Coord(i, j, axk);
}

//__device__ static Coord FastLocalOffsetToCoord(int vi) {
//    return { vi / 64, (vi / 8) % 8, vi % 8 };
//}

__device__ void LoadAMGLaplacianTileData(const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, AMGLaplacianTileData& shared_data, const int x_channel, const int coeff_channel, int ti) {
    constexpr int SN = 10;
    const int shared_offset = 1;

    auto& tile = info.tile();
	auto h = acc.voxelSize(info);

    for (int i = 0; i < 4; i++) {
        // Voxel index
        int vi = i * 128 + ti;
        Coord l_ijk = acc.localOffsetToCoord(vi);

        // Load x, off-diagonal and diagonal coefficients
        shared_data.absttypeT(l_ijk) = info.mType;
        shared_data.xValueT(l_ijk) = tile.type(l_ijk) == INTERIOR ? tile(x_channel, vi) : 0;
        shared_data.offDiag0ValueT(l_ijk) = tile(coeff_channel, vi);
        shared_data.offDiag1ValueT(l_ijk) = tile(coeff_channel + 1, vi);
        shared_data.offDiag2ValueT(l_ijk) = tile(coeff_channel + 2, vi);

        shared_data.diagValueT(l_ijk) = tile(coeff_channel + 3, vi);  // Load diagonal coefficient
    }

    __syncthreads();


    Coord b_ijk = info.mTileCoord;

    //we use [0, 64) for negative boundaries and [64, 128) for positive boundaries
    int fi, sgn;
    if (ti < 64) {
        fi = ti;
        sgn = -1;
    }
    else {
        fi = ti - 64;
        sgn = 1;
    }

    for (int axis : {0, 1, 2}) {
        Coord fl_ijk = FaceNeighborLocalCoord(axis, sgn, fi);
        //for example, for positive boundary, fl_ijk may be (8,j,k), and the cell inside the center tile is (7,j,k)
        Coord cl_ijk = fl_ijk; cl_ijk[axis] -= sgn;
        T avg_x = 0;
        Coord cl1_ijk = cl_ijk;
        for (int di : {0, 1}) {
            cl1_ijk[0] = (cl_ijk[0] ^ di);
            for (int dj : {0, 1}) {
				cl1_ijk[1] = (cl_ijk[1] ^ dj);
                for (int dk : {0, 1}) {
					cl1_ijk[2] = (cl_ijk[2] ^ dk);
					avg_x += shared_data.xValueT(cl1_ijk);
                }
            }
        }
		avg_x /= 8;

        Coord nb_ijk = b_ijk; nb_ijk[axis] += sgn;
        //for sgn==-1 (load negative boundaries), the local coord in the neighboring tile is 7
        //for sgn==1 (load positive boundaries), the local coord in the neighboring tile is 0
        Coord nl_ijk = fl_ijk; nl_ijk[axis] = (sgn == -1) ? 7 : 0;

        //HATileInfo<Tile> ninfo = acc.tileInfo(info.mLevel, nb_ijk);
        HATileInfo<Tile> ninfo = (sgn == -1) ? tile.mNeighbors[axis] : tile.mNeighbors[axis + 3];

        bool empty = ninfo.empty();
        if (empty) {
            shared_data.absttypeT(fl_ijk) = info.mType;
            shared_data.xValueT(fl_ijk) = 0;
            if (sgn == 1) shared_data.offDiagValueT(axis, fl_ijk) = h;
        }
        else if (ninfo.mType & GHOST) {
            shared_data.absttypeT(fl_ijk) = ninfo.mType;
            //dp/dx calculated with coarse grid equals to fine grid
            auto& ntile = ninfo.tile();
            {
                T V1 = ntile(x_channel, nl_ijk);
                T V0 = avg_x;

                //Coord clc_ijk = cl_ijk; clc_ijk[0] ^= 1; clc_ijk[1] ^= 1; clc_ijk[2] ^= 1;
                //T V0 = (shared_data.xValueT(cl_ijk) + shared_data.xValueT(clc_ijk)) / 2;

                T vg = shared_data.xValueT(cl_ijk) + 0.5 * (V1 - V0);
                //T vg = shared_data.xValueT(cl_ijk) +  (V1 - V0);
                shared_data.xValueT(fl_ijk) = vg;

                if (sgn == 1) shared_data.offDiagValueT(axis, fl_ijk) = ninfo.tile()(coeff_channel + axis, nl_ijk);
            }

            //{
            //    T vH = ninfo.tile()(x_channel, nl_ijk);//larger cell center, which is a corner of the ghost cell

            //    //next we e0xtrapolate the opposite corner of the ghost cell
            //    //Afivo: a framework for quadtree/octree AMR with shared-memory parallelization and geometric multigrid methods

            //    //for example, for positive boundary, fl_ijk may be (8,j,k), and the cell inside the center tile is (7,j,k)
            //    Coord cl_ijk = fl_ijk; cl_ijk[axis] -= sgn;
            //    T vh = shared_data.xValueT(cl_ijk);
            //    Coord cl0_ijk = cl_ijk; cl0_ijk[0] ^= 1; T vh0 = shared_data.xValueT(cl0_ijk);
            //    Coord cl1_ijk = cl_ijk; cl1_ijk[1] ^= 1; T vh1 = shared_data.xValueT(cl1_ijk);
            //    Coord cl2_ijk = cl_ijk; cl2_ijk[2] ^= 1; T vh2 = shared_data.xValueT(cl2_ijk);
            //    //vh - 0.5 * (vh0 - vh) - 0.5 * (vh1 - vh) - 0.5 * (vh2 - vh);
            //    T v1 = 2.5 * vh - 0.5 * (vh0 + vh1 + vh2);

            //    shared_data.xValueT(fl_ijk) = (vH + v1) / 2;
            //    if (sgn == 1) shared_data.offDiagValueT(axis, fl_ijk) = ninfo.tile()(coeff_channel + axis, nl_ijk);
            //}
        }
        else {
			shared_data.absttypeT(fl_ijk) = ninfo.mType;
			auto& ntile = ninfo.tile();
            shared_data.xValueT(fl_ijk) = (ntile.type(nl_ijk) == INTERIOR ? ntile(x_channel, nl_ijk) : 0);
            if (sgn == 1) shared_data.offDiagValueT(axis, fl_ijk) = ninfo.tile()(coeff_channel + axis, nl_ijk);
        }
    }
}

__global__ void NegativeLaplacianSameLevelAMG128Kernel(const HATileAccessor<Tile> acc, HATileInfo<Tile>* tiles, int subtree_level, uint8_t launch_tile_types, int x_channel, int coeff_channel, int Ax_channel) {             
    __shared__ AMGLaplacianTileData shared_data;

    //block idx (tile idx), distinguish from thread idx
    int bi = blockIdx.x;
    //thread idx
    int ti = threadIdx.x;

    //somehow it's faster with the reference symbol
    const auto& info = tiles[bi];

    if (!(info.subtreeType(subtree_level) & launch_tile_types)) return;

    LoadAMGLaplacianTileData(acc, info, shared_data, x_channel, coeff_channel, ti);
    __syncthreads();


    for (int i = 0; i < 4; i++) {
        //voxel idx
        int vi = i * 128 + ti;
		Coord l_ijk = acc.localOffsetToCoord(vi);

		//auto g_ijk = acc.composeGlobalCoord(info.mTileCoord, l_ijk);
  //      if (info.mLevel==1&&g_ijk == Coord(7,8,10)) {
  //          printf("l_ijk: %d %d %d\n", l_ijk[0], l_ijk[1], l_ijk[2]);
  //      }
  //      else continue;

        info.tile()(Ax_channel, vi) = info.tile().type(vi) == INTERIOR ? shared_data.negativeLap(l_ijk) : 0;
    }
}

void NegativeLaplacianSameLevelAMG128(HADeviceGrid<Tile>& grid, thrust::device_vector<HATileInfo<Tile>>& tiles, int launch_tile_num, int subtree_level, uint8_t launch_tile_types, int x_channel, int coeff_channel, int Ax_channel) {
    if (launch_tile_num > 0) {
        NegativeLaplacianSameLevelAMG128Kernel << <launch_tile_num, 128 >> > (grid.deviceAccessor(), thrust::raw_pointer_cast(tiles.data()), subtree_level, launch_tile_types, x_channel, coeff_channel, Ax_channel);
    }
}

//on all leafs of the tree
void AMGFullNegativeLaplacianOnLeafs(HADeviceGrid<Tile>& grid, const int x_channel, const int coeff_channel, const int Ax_channel) {
    PropagateToChildren(grid, x_channel, x_channel, -1, GHOST, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);
    AccumulateToParentsOneStep(grid, x_channel, x_channel, LEAF, 1. / 8, false, INTERIOR | DIRICHLET | NEUMANN);
    NegativeLaplacianSameLevelAMG128(grid, grid.dAllTiles, grid.dAllTiles.size(), -1, LEAF, x_channel, coeff_channel, Ax_channel);
}

__global__ void AMGAddGradientToFace128Kernel(const HATileAccessor<Tile> acc, HATileInfo<Tile>* tiles, int subtree_level, uint8_t launch_tile_types, int x_channel, int coeff_channel, int u_channel) {
    __shared__ AMGLaplacianTileData shared_data;
    //bi: block idx, ti: thread idx
    int bi = blockIdx.x, ti = threadIdx.x;
    //somehow it's faster with the reference symbol
    const auto& info = tiles[bi];
    if (!(info.subtreeType(subtree_level) & launch_tile_types)) return;
    LoadAMGLaplacianTileData(acc, info, shared_data, x_channel, coeff_channel, ti);
    __syncthreads();

    auto& tile = info.tile();
	auto h = acc.voxelSize(info);
    for (int i = 0; i < 4; i++) {
        //voxel idx
        int vi = i * 128 + ti;
        Coord l_ijk = acc.localOffsetToCoord(vi);

        

        for (int axis : {0, 1, 2}) {
			//printf("axis %d l_ijk %d %d %d u %f\n",axis, l_ijk[0], l_ijk[1], l_ijk[2], tile(u_channel + axis, l_ijk));

            Coord dl_ijk = l_ijk; dl_ijk[axis]--;
			T pu = shared_data.xValueT(l_ijk);
			T pd = shared_data.xValueT(dl_ijk);
            //tile(u_channel + axis, l_ijk) -= (pu - pd) * shared_data.offDiagValueT(axis, l_ijk) / (h * h);
            tile(u_channel + axis, l_ijk) += (pu - pd) / h;

    //        {
				//auto g_ijk = acc.composeGlobalCoord(info.mTileCoord, l_ijk);
    //            if (
    //                //(info.mLevel==2&&g_ijk == Coord(16,30,22))
    //                //||
    //                (info.mLevel==1&&g_ijk==Coord(7,8,10))
    //                ) {
				//	printf("calculating grad g_ijk %d %d %d axis %d pu %f pd %f off %f term %f vel %f\n", g_ijk[0], g_ijk[1], g_ijk[2], axis, pu, pd, shared_data.offDiagValueT(axis, l_ijk), (pu - pd) / h, tile(u_channel + axis, l_ijk));
    //            }
    //        }
        }
    }
}

void AMGAddGradientToFace(HADeviceGrid<Tile>& grid, int subtree_level, uint8_t launch_tile_types, int x_channel, int coeff_channel, int u_channel) {
    //first calculate GHOST and NONLEAF values
    PropagateToChildren(grid, x_channel, x_channel, -1, GHOST, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);
    AccumulateToParentsOneStep(grid, x_channel, x_channel, LEAF, 1. / 8, false, INTERIOR | DIRICHLET | NEUMANN);
    //then launch AMGAddGradientToFace128Kernel on all leafs to add grad(x) to u
    AMGAddGradientToFace128Kernel << <grid.dAllTiles.size(), 128 >> > (grid.deviceAccessor(), thrust::raw_pointer_cast(grid.dAllTiles.data()), subtree_level, launch_tile_types, x_channel, coeff_channel, u_channel);
}

void AMGVolumeWeightedDivergenceOnLeafs(HADeviceGrid<Tile>& grid, int u_channel, int x_channel) {
    for (int axis : {0, 1, 2}) {
        PropagateToChildren(grid, u_channel + axis, u_channel + axis, -1, GHOST, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);
        //AccumulateToParentsOneStep(grid, u_channel + axis, u_channel + axis, LEAF, 1. / 8, false, INTERIOR | DIRICHLET | NEUMANN);
    }
    AccumulateFacesToParentsOneStep(grid, u_channel, u_channel, LEAF, 1. / 4, false, INTERIOR | DIRICHLET | NEUMANN);

    grid.launchVoxelFuncOnAllTiles(
        [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto h = acc.voxelSize(info);
        Tile& tile = info.tile();
        uint8_t ctype0 = tile.type(l_ijk);
        T sum = 0;

        //iterate neighbors
        acc.iterateSameLevelNeighborVoxels(info, l_ijk,
            [&]__device__(const HATileInfo<Tile>&ninfo, const Coord & nl_ijk, const int axis, const int sgn) {

            T u0 = tile(u_channel + axis, l_ijk);
            T u1;

            if (ninfo.empty()) {
                u1 = 0;
            }
            else {
                auto& ntile = ninfo.tile();
                u1 = ntile(u_channel + axis, nl_ijk);
            }

            //basically coeff is -h
            //sum += (sgn == -1) ? u0 * coeff * h : -u1 * coeff * h;

            sum += (sgn == -1) ? -u0 * h * h : u1 * h * h;
        });

        //tile(x_channel, l_ijk) = sum;
        tile(x_channel, l_ijk) = (ctype0 & INTERIOR) ? sum : 0;

    }, LEAF);
}

void AMGVolumeWeightedDivergenceOnLeafs(HADeviceGrid<Tile>& grid, int u_channel, int coeff_channel, int x_channel) {
    for (int axis : {0, 1, 2}) {
        PropagateToChildren(grid, u_channel + axis, u_channel + axis, -1, GHOST, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);
        //AccumulateToParentsOneStep(grid, u_channel + axis, u_channel + axis, LEAF, 1. / 8, false, INTERIOR | DIRICHLET | NEUMANN);
    }
    AccumulateFacesToParentsOneStep(grid, u_channel, u_channel, LEAF, 1. / 4, false, INTERIOR | DIRICHLET | NEUMANN);

    grid.launchVoxelFuncOnAllTiles(
        [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto h = acc.voxelSize(info);
        Tile& tile = info.tile();
        uint8_t ctype0 = tile.type(l_ijk);
        T sum = 0;

        //iterate neighbors
        acc.iterateSameLevelNeighborVoxels(info, l_ijk,
            [&]__device__(const HATileInfo<Tile>&ninfo, const Coord & nl_ijk, const int axis, const int sgn) {

            T coeff0 = tile(coeff_channel + axis, l_ijk);
            T u0 = tile(u_channel + axis, l_ijk);
            T coeff1;
            T u1;

            if (ninfo.empty()) {
                u1 = 0;
                coeff1 = 0;
            }
            else {
                auto& ntile = ninfo.tile();
                u1 = ntile(u_channel + axis, nl_ijk);
                coeff1 = ntile(coeff_channel + axis, nl_ijk);
            }
            T coeff = (sgn == -1) ? coeff0 : coeff1;

            //basically coeff is -h
            sum += (sgn == -1) ? u0 * coeff * h : -u1 * coeff * h;

            //sum += (sgn == -1) ? -u0 * h * h : u1 * h * h;

    //        {
    //            //43,13,45
				//auto g_ijk = acc.composeGlobalCoord(info.mTileCoord, l_ijk);
    //            if (info.mLevel == 1 && g_ijk == Coord(7,8,10)) {
    //                printf("g_ijk %d %d %d axis %d sgn %d u0 %f u1 %f term %f sum %f\n", g_ijk[0], g_ijk[1], g_ijk[2], axis, sgn, u0, u1, (sgn == -1) ? u0 * coeff * h : -u1 * coeff * h, sum);
    //            }
    //        }
        });

        tile(x_channel, l_ijk) = sum;
        //tile(x_channel, l_ijk) = (ctype0 & INTERIOR) ? sum : 0;

    }, LEAF);
}

//for single level smoothing, does not consider ttype
__global__ void NegativeLaplacianAndGaussSeidelSameLevelAMG128Kernel(const HATileAccessor<Tile> acc, HATileInfo<Tile>* tiles, int subtree_level, uint8_t launch_tile_types, int x_channel, int coeff_channel, int rhs_channel, int color) {
    __shared__ AMGLaplacianTileData shared_data;


    //block idx (tile idx), distinguish from thread idx
    int bi = blockIdx.x;
    //thread idx
    int ti = threadIdx.x;

    //somehow it's faster with the reference symbol
    const auto& info = tiles[bi];


    if (!(info.subtreeType(subtree_level) & launch_tile_types)) return;

    LoadAMGLaplacianTileData(acc, info, shared_data, x_channel, coeff_channel, ti);

    __syncthreads();


    for (int i = 0; i < 4; i++) {
        //voxel idx
        int vi = i * 128 + ti;
        Coord l_ijk = acc.localOffsetToCoord(vi);
		int li = l_ijk[0], lj = l_ijk[1], lk = l_ijk[2];
        if ((li + lj + lk) % 2 == color) {
            uint8_t ctype = info.tile().type(vi);
            T b = info.tile()(rhs_channel, vi);
            T Ax = shared_data.negativeLap(l_ijk);
			T D = shared_data.diagValueT(l_ijk);
            if (D == 0 && (ctype & INTERIOR)) {
                auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
                printf("D is zero!!!!!!!!!!!!!!! at g_ijk %d %d %d level %d\n", g_ijk[0], g_ijk[1], g_ijk[2], info.mLevel);
            }
            T delta_x = (ctype & INTERIOR) ? (b - Ax) / D : 0;
            info.tile()(x_channel, vi) = shared_data.xValueT(l_ijk) + delta_x * 2.0 / 3;
        }
    }
}

void NegativeLaplacianAndGaussSeidelSameLevelAMG128(HADeviceGrid<Tile>& grid, thrust::device_vector<HATileInfo<Tile>>& tiles, int launch_tile_num, int subtree_level, uint8_t launch_tile_types, int x_channel, int coeff_channel, int rhs_channel, int color) {
    if (launch_tile_num > 0) {
        NegativeLaplacianAndGaussSeidelSameLevelAMG128Kernel << <launch_tile_num, 128 >> > (grid.deviceAccessor(), thrust::raw_pointer_cast(tiles.data()), subtree_level, launch_tile_types, x_channel, coeff_channel, rhs_channel, color);
    }
}



void GaussSeidelAMG(int iters, int order, HADeviceGrid<Tile>& grid, const int level, const int x_channel, const int coeff_channel, const int rhs_channel) {
    //order==0: 0,1
    //order==1: 1,0
    for (int i = 0; i < iters; i++) {
        NegativeLaplacianAndGaussSeidelSameLevelAMG128(grid, grid.dTileArrays[level], grid.hNumTiles[level], level, LEAF, x_channel, coeff_channel, rhs_channel, order);
        NegativeLaplacianAndGaussSeidelSameLevelAMG128(grid, grid.dTileArrays[level], grid.hNumTiles[level], level, LEAF, x_channel, coeff_channel, rhs_channel, 1 - order);
    }
}

__global__ void ResidualAndRestrictAMG128Kernel(HATileAccessor<Tile> acc, HATileInfo<Tile>* fine_tiles, int level, uint8_t launch_tile_types, int fine_x_channel, int fine_coeff_channel, int fine_rhs_channel, int coarse_x_channel, int coarse_residual_channel, T residual_op_coeff) {
    __shared__ AMGLaplacianTileData shared_data;
    __shared__ T residual[8 * 8 * 8];
    //block idx (tile idx), distinguish from thread idx
    int bi = blockIdx.x;
    //thread idx
    int ti = threadIdx.x;

    auto& finfo = fine_tiles[bi];


    if (!(finfo.subtreeType(level) & launch_tile_types)) return;
    auto h = acc.voxelSize(finfo);

    LoadAMGLaplacianTileData(acc, finfo, shared_data, fine_x_channel, fine_coeff_channel, ti);
    __syncthreads();

    //calculate residual
    for (int i = 0; i < 4; i++) {
        //voxel idx
        int vi = i * 128 + ti;
        Coord l_ijk = acc.localOffsetToCoord(vi);
        residual[vi] = finfo.tile()(fine_rhs_channel, vi) - shared_data.negativeLap(l_ijk);
        //finfo.tile()(coarse_residual_channel, vi) = residual[vi];
    }

    __syncthreads();

    if (ti < 64) {
        Coord fl_ijk(ti / 16 * 2, (ti / 4) % 4 * 2, ti % 4 * 2);
        Coord fg_ijk = acc.localToGlobalCoord(finfo, fl_ijk);
        Coord cg_ijk = acc.parentCoord(fg_ijk);
        HATileInfo<Tile> cinfo; Coord cl_ijk;//coarse tile and local coord
        acc.findVoxel(level - 1, cg_ijk, cinfo, cl_ijk);
        if (!cinfo.empty()) {
            T residual_sum = 0;
            T x_sum = 0;
            for (int ii : {0, 1}) {
                for (int jj : {0, 1}) {
                    for (int kk : {0, 1}) {
                        Coord fl1_ijk = fl_ijk + Coord(ii, jj, kk);
                        int vi1 = acc.localCoordToOffset(fl1_ijk);
                        residual_sum += residual[vi1];
                        x_sum += shared_data.xValueT(fl1_ijk);
                    }
                }
            }
            //ghost accumulate
            if (finfo.subtreeType(level) & GHOST) {
                //auto& ctile = cinfo.tile();
                //ctile(coarse_residual_channel, cl_ijk) += sum;
     //           {
					//printf("cg_ijk %d %d %d sum %f\n", cg_ijk[0], cg_ijk[1], cg_ijk[2], sum);
     //           }
            }
            else {
                auto& ctile = cinfo.tile();
                ctile(coarse_residual_channel, cl_ijk) = residual_sum * residual_op_coeff;
                //ctile(coarse_x_channel, cl_ijk) = 0;
                ctile(coarse_x_channel, cl_ijk) = x_sum / 8;
            }
        }

    }
}

void ResidualAndRestrictAMG(HADeviceGrid<Tile>& grid, int fine_x_channel, int fine_coeff_channel, int fine_rhs_channel, int coarse_x_channel, int coarse_residual_channel, int level, uint8_t launch_tile_types, T residual_op_coeff) {
    ResidualAndRestrictAMG128Kernel << <grid.hNumTiles[level], 128 >> > (grid.deviceAccessor(), thrust::raw_pointer_cast(grid.dTileArrays[level].data()), level, launch_tile_types, fine_x_channel, fine_coeff_channel, fine_rhs_channel, coarse_x_channel, coarse_residual_channel, residual_op_coeff);
}

__global__ void DeductFluxOnAbsLeafs128Kernel(HATileAccessor<Tile> acc, HATileInfo<Tile>* tiles, int x_channel, int coeff_channel, int f_channel, int rhs_channel) {
    __shared__ AMGLaplacianTileData shared_data;
    //block idx (tile idx), distinguish from thread idx
    int bi = blockIdx.x;
    //thread idx
    int ti = threadIdx.x;

    auto& info = tiles[bi];
    auto& tile = info.tile();


    if (!(info.mType & LEAF)) return;

    LoadAMGLaplacianTileData(acc, info, shared_data, x_channel, coeff_channel, ti);
    __syncthreads();

    //calculate residual
    for (int i = 0; i < 4; i++) {
        //voxel idx
        int vi = i * 128 + ti;
        Coord l_ijk = acc.localOffsetToCoord(vi);

		//tile(rhs_channel, l_ijk) -= shared_data.negativeLap(l_ijk);
        if (tile.type(l_ijk) & INTERIOR) tile(rhs_channel, l_ijk) = tile(f_channel, l_ijk) - shared_data.negativeLapOuterNonLeaf(l_ijk);
        else tile(rhs_channel, l_ijk) = 0;
    }
}

void DeductFluxOnAbsLeafs(HADeviceGrid<Tile>& grid, int level, int x_channel, int coeff_channel, int f_channel, int rhs_channel) {
    DeductFluxOnAbsLeafs128Kernel << <grid.hNumTiles[level], 128 >> > (
        grid.deviceAccessor(), thrust::raw_pointer_cast(grid.dTileArrays[level].data()),
        x_channel, coeff_channel, f_channel, rhs_channel
        );
}

__global__ void SetLevelZeroAbsTiles128Kernel(HATileAccessor<Tile> acc, HATileInfo<Tile>* tiles, int level, uint8_t abs_tile_types, int x_channel) {
    //block idx (tile idx), distinguish from thread idx
    int bi = blockIdx.x;
    //thread idx
    int ti = threadIdx.x;

    auto& info = tiles[bi];
    auto& tile = info.tile();
    if (!(info.mType & abs_tile_types)) return;

	for (int i = 0; i < 4; i++) {
		int vi = i * 128 + ti;
        Coord l_ijk = acc.localOffsetToCoord(vi);
		tile(x_channel, l_ijk) = 0;
	}
}

void SetLevelZeroAbsTiles(HADeviceGrid<Tile>& grid, int level, uint8_t abs_tile_types, int x_channel) {
	SetLevelZeroAbsTiles128Kernel << <grid.hNumTiles[level], 128 >> > (grid.deviceAccessor(), thrust::raw_pointer_cast(grid.dTileArrays[level].data()), level, abs_tile_types, x_channel);
}

__global__ void ProlongateAndUpdateAMG128Kernel(HATileAccessor<Tile> acc, HATileInfo<Tile>* fine_tiles, int fine_level, uint8_t fine_tile_types, int coarse_x_channel, int fine_x_channel, T prolong_coeff) {
    //block idx (tile idx), distinguish from thread idx
    int bi = blockIdx.x;
    //thread idx
    int ti = threadIdx.x;

    auto& finfo = fine_tiles[bi];
    auto fttype = finfo.subtreeType(fine_level);
    if (!(fttype & fine_tile_types)) return;

    for (int i = 0; i < 4; i++) {
        int fvi = i * 128 + ti;
        Coord fl_ijk = acc.localOffsetToCoord(fvi);
        Coord fg_ijk = acc.localToGlobalCoord(finfo, fl_ijk);
        auto fctype = finfo.tile().type(fvi);

        Coord cg_ijk = acc.parentCoord(fg_ijk);
        HATileInfo<Tile> cinfo; Coord cl_ijk;//coarse tile and local coord
        acc.findVoxel(fine_level - 1, cg_ijk, cinfo, cl_ijk);

        T cx = (cinfo.empty()) ? 0 : cinfo.tile()(coarse_x_channel, cl_ijk);
        //T cx = (cinfo.empty()) ? 0 : cinfo.tile().interiorValue(coarse_x_channel, cl_ijk);
        T fx = finfo.tile()(fine_x_channel, fvi);

        if (fttype & GHOST) {
            finfo.tile()(fine_x_channel, fvi) = (fctype & INTERIOR) ? cx : 0;
        }
        else {
            finfo.tile()(fine_x_channel, fvi) = (fctype & INTERIOR) ? fx + prolong_coeff * cx : 0;
        }
    }
}

void ProlongateAndUpdateAMG128(HADeviceGrid<Tile>& grid, int coarse_x_channel, int fine_x_channel, int fine_level, T prolong_coeff) {
    ProlongateAndUpdateAMG128Kernel << <grid.hNumTiles[fine_level], 128 >> > (grid.deviceAccessor(), thrust::raw_pointer_cast(grid.dTileArrays[fine_level].data()), fine_level, LEAF | GHOST, coarse_x_channel, fine_x_channel, prolong_coeff);
}

//must initialize to zero 
void AMGSolver::VCycle(HADeviceGrid<Tile>& grid, const int x_channel, const int f_channel, const int rhs_channel, const int coeff_channel, int level_iters, int coarsest_iters) {
    //f channel remains unchanged during MG (it will be used in CG iterations)
    //rhs channel is b in SPGrid paper
    grid.launchVoxelFuncOnAllTiles(
        [=]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();
        int vi = acc.localCoordToOffset(l_ijk);
        //initial guess is 0 for LEAF | NONLEAF | GHOST cells
        tile(x_channel, vi) = 0;

        if (info.mType & GHOST) {
            //rhs_channel is 0 for GHOST cells
            tile(rhs_channel, vi) = 0;
        }
        else {
            //rhs_channel is b for LEAF | NONLEAF cells
            tile(rhs_channel, vi) = tile(f_channel, vi);
        }
    },
        LEAF | NONLEAF | GHOST, 4
    );

    //downstroke
    for (int i = grid.mMaxLevel; i > 0; i--) {
        //Info("downstroke level {}", i);
        //{
        //    Info("downstroke level {}", i);
        //    auto holder = grid.getHostTileHolder(LEAF | NONLEAF);
        //    polyscope::init();
        //    IOFunc::AddLeveledPoissonGridCellCentersToPolyscopePointCloud(holder,
        //        {
        //        {-1,"type"} , {x_channel,"x"}, {rhs_channel,"rhs"}, { coeff_channel + 3,"diag" }
        //        },
        //        {}, i, FLT_MAX);
        //    polyscope::show();
        //    polyscope::removeAllStructures();
        //}

        //u^l=0
        //fjinest level already done at the beginning
        GaussSeidelAMG(level_iters, 0, grid, i, x_channel, coeff_channel, rhs_channel);




        //calculate residual and restrict to next level
        //r^l = b^l-Au^l
        ResidualAndRestrictAMG(grid, x_channel, coeff_channel, rhs_channel, x_channel, rhs_channel, i, LEAF, R_restrict_coeff);


    }


     
	GaussSeidelAMG(coarsest_iters / 2, 0, grid, 0, x_channel, coeff_channel, rhs_channel);
	GaussSeidelAMG(coarsest_iters / 2, 1, grid, 0, x_channel, coeff_channel, rhs_channel);



    for (int i = 1; i <= grid.mMaxLevel; i++) {
        ProlongateAndUpdateAMG128(grid, x_channel, x_channel, i, prolong_coeff);


        //smoothing: fine.x = smooth(fine.b)
		GaussSeidelAMG(level_iters, 1, grid, i, x_channel, coeff_channel, rhs_channel);
    }

    // timer.stop("upstroke");
}

void AMGSolver::muCycleStep(int current_level, int repeat_times, HADeviceGrid<Tile>& grid, const int x_channel, const int f_channel, const int rhs_channel, const int coeff_channel, int level_iters, int coarsest_iters) {
    if (current_level == 0) {
        GaussSeidelAMG(coarsest_iters / 2, 0, grid, 0, x_channel, coeff_channel, rhs_channel);
        GaussSeidelAMG(coarsest_iters / 2, 1, grid, 0, x_channel, coeff_channel, rhs_channel);
        return;
    }
    
    GaussSeidelAMG(level_iters, 0, grid, current_level, x_channel, coeff_channel, rhs_channel);
    //will set initial guess of coarser level to 0
    ResidualAndRestrictAMG(grid, x_channel, coeff_channel, rhs_channel, x_channel, rhs_channel, current_level, LEAF, R_restrict_coeff);
    DeductFluxOnAbsLeafs(grid, current_level - 1, x_channel, coeff_channel, f_channel, rhs_channel);
    SetLevelZeroAbsTiles(grid, current_level - 1, NONLEAF, x_channel);

    for (int i = 0; i < repeat_times; i++) {
        muCycleStep(current_level - 1, repeat_times, grid, x_channel, f_channel, rhs_channel, coeff_channel, level_iters, coarsest_iters);
    }

	ProlongateAndUpdateAMG128(grid, x_channel, x_channel, current_level, prolong_coeff);
	GaussSeidelAMG(level_iters, 1, grid, current_level, x_channel, coeff_channel, rhs_channel);

    //{
    //    Info("done with mucycle level {}", current_level);
    //    polyscope::init();
    //    auto holder = grid.getHostTileHolder(LEAF | NONLEAF | GHOST);
    //    IOFunc::AddLeveledPoissonGridCellCentersToPolyscopePointCloud(
    //        holder,
    //        { {-1,"type"}, {x_channel,"x"}, { rhs_channel,"rhs" }, {coeff_channel,"offd0"}, {coeff_channel + 1,"offd1"} ,{coeff_channel + 2,"offd2"},{coeff_channel + 3,"diag"} },
    //        {},
    //        current_level, FLT_MAX
    //    );
    //    polyscope::show();
    //    polyscope::removeAllStructures();
    //}
}

void AMGSolver::muCycle(int repeat_times, HADeviceGrid<Tile>& grid, const int x_channel, const int f_channel, const int rhs_channel, const int coeff_channel, int level_iters, int coarsest_iters)
{
    //f channel remains unchanged during MG (it will be used in CG iterations)
    //rhs channel is b in SPGrid paper
    grid.launchVoxelFuncOnAllTiles(
        [=]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();
        int vi = acc.localCoordToOffset(l_ijk);
        //initial guess is 0 for LEAF | NONLEAF | GHOST cells
        tile(x_channel, vi) = 0;

        if (info.mType & GHOST) {
            //rhs_channel is 0 for GHOST cells
            tile(rhs_channel, vi) = 0;
        }
        else {
            //rhs_channel is b for LEAF | NONLEAF cells
            tile(rhs_channel, vi) = tile(f_channel, vi);
        }
    },
        LEAF | NONLEAF | GHOST, 4
    );
    muCycleStep(grid.mMaxLevel, repeat_times, grid, x_channel, f_channel, rhs_channel, coeff_channel, level_iters, coarsest_iters);
}

__global__ void FASSaveSolutionAndUpdateSource128Kernel(HATileAccessor<Tile> acc, HATileInfo<Tile>* tiles, uint8_t launch_abs_tile_types, int x_channel, int coeff_channel, int rhs_channel, int x0_channel) {
    __shared__ AMGLaplacianTileData shared_data;
    //block idx (tile idx), distinguish from thread idx
    int bi = blockIdx.x;
    //thread idx
    int ti = threadIdx.x;

    auto& info = tiles[bi];
    auto& tile = info.tile();

    if (!(info.mType & launch_abs_tile_types)) return;


    LoadAMGLaplacianTileData(acc, info, shared_data, x_channel, coeff_channel, ti);
    __syncthreads();

    //calculate residual
    for (int i = 0; i < 4; i++) {
        //voxel idx
        int vi = i * 128 + ti;
        Coord l_ijk = acc.localOffsetToCoord(vi);

        //tile(rhs_channel, l_ijk) -= shared_data.negativeLap(l_ijk);
        if (tile.type(l_ijk) & INTERIOR) {
            tile(rhs_channel, l_ijk) = tile(rhs_channel, l_ijk) + shared_data.negativeLapOuterNonLeaf(l_ijk);
            tile(x0_channel, l_ijk) = tile(x_channel, l_ijk);
        }
        else tile(rhs_channel, l_ijk) = 0;
    }
}

void FASSaveSolutionAndUpdateSource128(HADeviceGrid<Tile>& grid, int level, int x_channel, int coeff_channel, int rhs_channel, int x0_channel) {
    FASSaveSolutionAndUpdateSource128Kernel << <grid.hNumTiles[level], 128 >> > (
        grid.deviceAccessor(), thrust::raw_pointer_cast(grid.dTileArrays[level].data()),
        NONLEAF,
        x_channel, coeff_channel, rhs_channel, x0_channel
        );
}

__global__ void FASProlongateAndUpdateAMG128Kernel(HATileAccessor<Tile> acc, HATileInfo<Tile>* fine_tiles, int fine_level, uint8_t fine_tile_types, int coarse_x_channel, int coarse_x0_channel, int fine_x_channel, T prolong_coeff) {
    //block idx (tile idx), distinguish from thread idx
    int bi = blockIdx.x;
    //thread idx
    int ti = threadIdx.x;

    auto& finfo = fine_tiles[bi];
    auto fttype = finfo.subtreeType(fine_level);
    if (!(fttype & fine_tile_types)) return;

    for (int i = 0; i < 4; i++) {
        int fvi = i * 128 + ti;
        Coord fl_ijk = acc.localOffsetToCoord(fvi);
        Coord fg_ijk = acc.localToGlobalCoord(finfo, fl_ijk);
        auto fctype = finfo.tile().type(fvi);

        Coord cg_ijk = acc.parentCoord(fg_ijk);
        HATileInfo<Tile> cinfo; Coord cl_ijk;//coarse tile and local coord
        acc.findVoxel(fine_level - 1, cg_ijk, cinfo, cl_ijk);

        T cx = (cinfo.empty()) ? 0 : cinfo.tile()(coarse_x_channel, cl_ijk);
		T cx0 = (cinfo.empty()) ? 0 : cinfo.tile()(coarse_x0_channel, cl_ijk);
        //T cx = (cinfo.empty()) ? 0 : cinfo.tile().interiorValue(coarse_x_channel, cl_ijk);
        T fx = finfo.tile()(fine_x_channel, fvi);

        if (fttype & GHOST) {
            finfo.tile()(fine_x_channel, fvi) = (fctype & INTERIOR) ? cx : 0;
        }
        else {
            finfo.tile()(fine_x_channel, fvi) = (fctype & INTERIOR) ? fx + prolong_coeff * (cx - cx0) : 0;
        }
    }
}

void FASProlongateAndUpdateAMG128(HADeviceGrid<Tile>& grid, int coarse_x_channel, int coarse_x0_channel, int fine_x_channel, int fine_level, T prolong_coeff) {
    FASProlongateAndUpdateAMG128Kernel << <grid.hNumTiles[fine_level], 128 >> > (
        grid.deviceAccessor(), thrust::raw_pointer_cast(grid.dTileArrays[fine_level].data()),
        fine_level, LEAF | GHOST,
        coarse_x_channel, coarse_x0_channel, fine_x_channel, prolong_coeff);
}

void AMGSolver::FASMuCycleStep(int current_level, int repeat_times, HADeviceGrid<Tile>& grid, const int x_channel, const int rhs_channel, const int x0_channel, const int coeff_channel, int level_iters, int coarsest_iters)
{
    if (current_level == 0) {
        GaussSeidelAMG(coarsest_iters / 2, 0, grid, 0, x_channel, coeff_channel, rhs_channel);
        GaussSeidelAMG(coarsest_iters / 2, 1, grid, 0, x_channel, coeff_channel, rhs_channel);
        return;
    }

    GaussSeidelAMG(level_iters, 0, grid, current_level, x_channel, coeff_channel, rhs_channel);

    //will set initial guess of coarser level to 0
    ResidualAndRestrictAMG(grid, x_channel, coeff_channel, rhs_channel, x_channel, rhs_channel, current_level, LEAF, R_restrict_coeff);
    FASSaveSolutionAndUpdateSource128(grid, current_level - 1, x_channel, coeff_channel, rhs_channel, x0_channel);

    for (int i = 0; i < repeat_times; i++) {
        FASMuCycleStep(current_level - 1, repeat_times, grid, x_channel, rhs_channel, x0_channel, coeff_channel, level_iters, coarsest_iters);
    }

    FASProlongateAndUpdateAMG128(grid, x_channel, x0_channel, x_channel, current_level, prolong_coeff);
    GaussSeidelAMG(level_iters, 1, grid, current_level, x_channel, coeff_channel, rhs_channel);
}

void AMGSolver::FASMuCycle(int repeat_times, HADeviceGrid<Tile>& grid, const int x_channel, const int rhs_channel, const int x0_channel, const int coeff_channel, int level_iters, int coarsest_iters)
{
    //set initial guess to 0
    grid.launchVoxelFuncOnAllTiles(
        [=]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();
        int vi = acc.localCoordToOffset(l_ijk);
        //initial guess is 0 for LEAF | NONLEAF | GHOST cells
        tile(x_channel, vi) = 0;
    },
        LEAF | NONLEAF | GHOST, 4
    );
    FASMuCycleStep(grid.mMaxLevel, repeat_times, grid, x_channel, rhs_channel, x0_channel, coeff_channel, level_iters, coarsest_iters);
}

std::tuple<int, double> AMGSolver::FASMuCycleSolve(int repeat_times, HADeviceGrid<Tile>& grid, int max_iters, double relative_tolerance, int level_iters, int coarsest_iters)
{
    auto x_channel = Tile::x_channel;//0
    auto rhs_channel = Tile::b_channel;//1
    auto x0_channel = Tile::p_channel;//2

    //set initial guess to 0
    grid.launchVoxelFuncOnAllTiles(
        [=]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();
        int vi = acc.localCoordToOffset(l_ijk);
        //initial guess is 0 for LEAF | NONLEAF | GHOST cells
        tile(x_channel, vi) = 0;
    },
        LEAF | NONLEAF | GHOST, 4
    );

	double rhs_norm = sqrt(Dot(grid, rhs_channel, rhs_channel, LEAF));
	double threshold_norm = relative_tolerance * rhs_norm;
	threshold_norm = std::max(threshold_norm, std::numeric_limits<double>::min());

	Info("FASMuCycleSolve iter 0 norm {}", rhs_norm);

    int i = 0;
    double residual_norm;
	for (i = 0; i < max_iters; i++) {
        FASMuCycleStep(grid.mMaxLevel, repeat_times, grid, x_channel, rhs_channel, x0_channel, coeff_channel, level_iters, coarsest_iters);
        //calculate residual to x0 channel
        //x0=-lap(x)
		AMGFullNegativeLaplacianOnLeafs(grid, x_channel, coeff_channel, x0_channel);
		//r=x0-b
        grid.launchVoxelFuncOnAllTiles(
            [=]__device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
            auto& tile = info.tile();
            tile(x0_channel, l_ijk) = tile(rhs_channel, l_ijk) - tile(x0_channel, l_ijk);
        }, LEAF, 4);
		
		residual_norm = sqrt(Dot(grid, x0_channel, x0_channel, LEAF));
		Info("FASMuCycleSolve iter {} norm {}", i + 1, residual_norm);
		if (residual_norm < threshold_norm) {
			break;
		}
	}

	return std::make_tuple(i, residual_norm / rhs_norm);
}

void AMGSolver::prepareTypesAndCoeffs(HADeviceGrid<Tile>& grid)
{
	CoarsenTypesAndAMGCoeffs(grid, coeff_channel, R_matrix_coeff);
}

std::tuple<int, double> AMGSolver::solve(HADeviceGrid<Tile>& grid, bool verbose, int max_iters, double relative_tolerance, int level_iters, int coarsest_iters, int sync_stride, bool is_pure_neumann)
{
    int mu_cycle_repeat_times = 1;

    double rhs_norm2, threshold_norm2, last_residual_norm2;

    //Use 0 as initial guess
    //x0=0
    Fill(grid, Tile::x_channel, (T)0, -1, LEAF, LAUNCH_SUBTREE, INTERIOR | NEUMANN | DIRICHLET);//fill all leafs with 0
    //initial residual is b

    if (sync_stride > 0) {
        //rhs_norm2=r*r
        last_residual_norm2 = rhs_norm2 = Dot(grid, Tile::r_channel, Tile::r_channel, LEAF);
        //if (verbose) Info("ConjugateGradient with initial norm of rhs: {}", sqrt(rhs_norm2));
        if (verbose) Info("ConjugateGradient iter 0 norm {}(1.0)", sqrt(rhs_norm2));

        //if b is zero, just solve to zero
        if (rhs_norm2 == (double)0) {
            //d_x is zero
            //iters=0, relative_error=0
            //x0=0
            return std::make_tuple(0, (double)0);
        }
        //(epsilon*|b|)^2
        threshold_norm2 = relative_tolerance * relative_tolerance * rhs_norm2;
        threshold_norm2 = std::max(threshold_norm2, std::numeric_limits<double>::min());
    }
    if (is_pure_neumann) {
        ReCenterLeafCells(grid, Tile::r_channel, cnt_reducer, mean_d, count_d);
        //ReCenterLeafVoxels(grid, Tile::r_channel, mean_d, count_d);
    }
    ////z0=Minv*r0
    //r_channel->z_channel
	//muCycle(mu_cycle_repeat_times, grid, Tile::z_channel, Tile::r_channel, Tile::Ap_channel, coeff_channel, level_iters, coarsest_iters);
    FASMuCycle(mu_cycle_repeat_times, grid, Tile::z_channel, Tile::r_channel, Tile::Ap_channel, coeff_channel, level_iters, coarsest_iters);

    //p0=z0
    Copy(grid, Tile::z_channel, Tile::p_channel, -1, LEAF, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);


    DotAsync(gamma_d, grid, Tile::z_channel, Tile::r_channel, LEAF);


    int i = 0;
    for (i = 0; i < max_iters; i++) {
        //Ap_k=A*p_k
        //here A=-lap
        AMGFullNegativeLaplacianOnLeafs(grid, Tile::p_channel, coeff_channel, Tile::Ap_channel);
        //AMGFullNegativeLaplacianOnLeafs(grid, Tile::p_channel, coeff_channel, Tile::Ap_channel);

        //alpha_k=gamma_k/(p_k^T*A*p_k)
        DotAsync(fp_d, grid, Tile::p_channel, Tile::Ap_channel, LEAF);//fp_k=p_k^T*A*p_k
        //alpha = gamma / fp
        TernaryOnArray(alpha_d, gamma_d, fp_d, []__device__(double& alpha, double gamma, double fp) { alpha = gamma / fp; });

        //x_{k+1} = x_k + alpha_k * p_k
        //r_{k+1} = r_k - alpha_k * Ap_k
        auto alpha_ptr = alpha_d;
        grid.launchVoxelFuncOnAllTiles(
            [alpha_ptr]__device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
            auto& tile = info.tile();
            int vi = acc.localCoordToOffset(l_ijk);
            if (tile.type(vi) & INTERIOR) {
                //double alpha = *alpha_ptr;
                auto alpha = float(*alpha_ptr);
                tile(Tile::x_channel, vi) += alpha * tile(Tile::p_channel, vi);
                //tile(Tile::r_channel, vi) -= alpha * tile(Tile::Ap_channel, vi);
            }

        }, LEAF, 4);
        grid.launchVoxelFuncOnAllTiles(
            [alpha_ptr]__device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
            auto& tile = info.tile();
            int vi = acc.localCoordToOffset(l_ijk);
            if (tile.type(vi) & INTERIOR) {
                auto alpha = float(*alpha_ptr);
                //tile(Tile::x_channel, vi) += alpha * tile(Tile::p_channel, vi);
                tile(Tile::r_channel, vi) -= alpha * tile(Tile::Ap_channel, vi);
            }

        }, LEAF, 4);

        if (sync_stride > 0 && (i + 1) % sync_stride == 0) {
            double residual_norm2 = Dot(grid, Tile::r_channel, Tile::r_channel, LEAF);
            if (verbose) {
                Info("ConjugateGradient iter {} norm {:e}({:e}), convergence factor={:e}", i+1, sqrt(residual_norm2), sqrt(residual_norm2 / rhs_norm2), 1.0 / sqrt(residual_norm2 / last_residual_norm2));
                last_residual_norm2 = residual_norm2;
            }
            if (residual_norm2 < threshold_norm2) {
                return std::make_tuple(i + 1, sqrt(residual_norm2 / rhs_norm2));
            }


        }
        if (is_pure_neumann) {
            ReCenterLeafCells(grid, Tile::r_channel, cnt_reducer, mean_d, count_d);
            //ReCenterLeafVoxels(grid, Tile::r_channel, mean_d, count_d);
        }

        //z_{k+1} = Minv * r_{k+1}
        //r->z
        //muCycle(mu_cycle_repeat_times, grid, Tile::z_channel, Tile::r_channel, Tile::Ap_channel, coeff_channel, level_iters, coarsest_iters);
        FASMuCycle(mu_cycle_repeat_times, grid, Tile::z_channel, Tile::r_channel, Tile::Ap_channel, coeff_channel, level_iters, coarsest_iters);


        //gamma_old = gamma;
        TernaryOnArray(gamma_d, gamma_d, gamma_old_d, []__device__(double& gamma, double& _, double& gamma_old) { gamma_old = gamma; });

        //gamma_{k+1} = dot(r_{k+1}, z_{k+1})
        DotAsync(gamma_d, grid, Tile::r_channel, Tile::z_channel, LEAF);

        //beta_{k+1} = gamma_{k+1} / gamma_k
        //beta = gamma / gamma_old;
        TernaryOnArray(beta_d, gamma_d, gamma_old_d, []__device__(double& beta, double gamma, double gamma_old) { beta = gamma / gamma_old; });

        auto beta_ptr = beta_d;
        //p_{k+1} = z_{k+1} + beta_{k+1} * p_{k}
        //BinaryTransform(grid, Tile::z_channel, Tile::p_channel, Tile::p_channel, [=]__device__(Tile::T z, Tile::T p) { return z + beta * p; }, -1, LEAF, LAUNCH_SUBTREE);
        grid.launchVoxelFuncOnAllTiles(
            [beta_ptr]__device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
            auto& tile = info.tile();
            int vi = acc.localCoordToOffset(l_ijk);
            if (tile.type(vi) & INTERIOR) {
                //double beta = *beta_ptr;
                auto beta = float(*beta_ptr);
                tile(Tile::p_channel, vi) = tile(Tile::z_channel, vi) + beta * tile(Tile::p_channel, vi);
            }
        },
            LEAF, 1
        );
    }

    if (sync_stride > 0) {
        double residual_norm2 = Dot(grid, Tile::r_channel, Tile::r_channel, LEAF);
        if (verbose) {
            Info("ConjugateGradient iter {} norm {:e}({:e}), convergence rate={:e}", i, sqrt(residual_norm2), sqrt(residual_norm2 / rhs_norm2), sqrt(residual_norm2 / last_residual_norm2));
        }
        return std::make_tuple(i, sqrt(residual_norm2 / rhs_norm2));
    }
    else {
        return std::make_tuple(i, -1);
    }
}
