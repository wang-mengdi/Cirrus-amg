#include "CMGSolver.h"
#include "PoissonSolver.h"

__forceinline__ __device__ T NegativeLaplacianCoeff(T h, uint8_t ctype0, const uint8_t ctype1) {
    int has_neumann = int(ctype0 & NEUMANN || ctype1 & NEUMANN);
    //   //T coeff = one_over_delta_h / h * (1 - has_neumann);
    return has_neumann ? 0 : h;
}

class CMGLaplacianTileData {
public:
    static constexpr int halo = 1;
    static constexpr int SN = 8 + halo * 2;
    T x[SN][SN][SN];               // x channel
    //uint8_t ttype[SN][SN][SN];     // tile type
	uint8_t ctype[SN][SN][SN];     // cell types
        
    //they will take actual tile local coords
    __device__ T& xValueT(Coord l_ijk) { return x[l_ijk[0] + halo][l_ijk[1] + halo][l_ijk[2] + halo]; }
    //__device__ uint8_t& ttypeValue(Coord l_ijk) { return ttype[l_ijk[0] + halo][l_ijk[1] + halo][l_ijk[2] + halo]; }
	__device__ uint8_t& ctypeValue(Coord l_ijk) { return ctype[l_ijk[0] + halo][l_ijk[1] + halo][l_ijk[2] + halo]; }

    //for single-level usage, does not consider ttype
    __device__ T negativeLap(const T h, Coord l_ijk, bool diag) {
        //auto ttype0 = ttypeValue(l_ijk);
        auto ctype0 = ctypeValue(l_ijk);
        T x0 = xValueT(l_ijk);
        T sum = 0;

        for (int axis : {0, 1, 2}) {
            for (int sgn : {-1, 1}) {
                Coord nl_ijk = l_ijk;
                nl_ijk[axis] += sgn;
                
				//auto ttype1 = ttypeValue(nl_ijk);
				auto ctype1 = ctypeValue(nl_ijk);
				T x1 = xValueT(nl_ijk);
                T coeff = NegativeLaplacianCoeff(h, ctype0, ctype1);
				sum += diag ? coeff : coeff * (x0 - x1);

                //printf("CMG axis %d sgn %d ctype0 %d ctype1 %d coeff %f x0 %f x1 %f sum %f\n", axis, sgn, ctype0, ctype1, coeff, x0, x1, sum);
            }
        }

        return ctype0 & INTERIOR ? sum : 0;
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

__device__ void LoadCMGLaplacianTileData(const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, CMGLaplacianTileData& shared_data, const int subtree_level, const int x_channel, int ti) {
    auto& tile = info.tile();

    // Load 4 voxels into the shared memory region from the main tile
    for (int i = 0; i < 4; i++) {
        // Voxel index
        int vi = i * 128 + ti;
        Coord l_ijk = acc.localOffsetToCoord(vi);
        
        shared_data.xValueT(l_ijk) = info.tile()(x_channel, vi);
        //shared_data.ttypeValue(l_ijk) = info.subtreeType(subtree_level);
		shared_data.ctypeValue(l_ijk) = tile.type(vi);
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

        Coord nb_ijk = b_ijk; nb_ijk[axis] += sgn;
        //for sgn==-1 (load negative boundaries), the local coord in the neighboring tile is 7
		//for sgn==1 (load positive boundaries), the local coord in the neighboring tile is 0
		Coord nl_ijk = fl_ijk; nl_ijk[axis] = (sgn == -1) ? 7 : 0;

        //HATileInfo<Tile> ninfo = acc.tileInfo(info.mLevel, nb_ijk);
		HATileInfo<Tile> ninfo = (sgn == -1) ? tile.mNeighbors[axis] : tile.mNeighbors[axis + 3];
            
        bool empty = ninfo.empty();
        if (empty) {
			shared_data.xValueT(fl_ijk) = 0;
			//shared_data.ttypeValue(fl_ijk) = info.subtreeType(subtree_level);
			shared_data.ctypeValue(fl_ijk) = CellType::DIRICHLET;
		}
        else if (ninfo.subtreeType(subtree_level) == GHOST) {
            T vH = ninfo.tile()(x_channel, nl_ijk);//larger cell center, which is a corner of the ghost cell

            //next we extrapolate the opposite corner of the ghost cell
            //Afivo: a framework for quadtree/octree AMR with shared-memory parallelization and geometric multigrid methods

            //for example, for positive boundary, fl_ijk may be (8,j,k), and the cell inside the center tile is (7,j,k)
            Coord cl_ijk = fl_ijk; cl_ijk[axis] -= sgn;
            T vh = shared_data.xValueT(cl_ijk);
            Coord cl0_ijk = cl_ijk; cl0_ijk[0] ^= 1; T vh0 = shared_data.xValueT(cl0_ijk);
            Coord cl1_ijk = cl_ijk; cl1_ijk[1] ^= 1; T vh1 = shared_data.xValueT(cl1_ijk);
            Coord cl2_ijk = cl_ijk; cl2_ijk[2] ^= 1; T vh2 = shared_data.xValueT(cl2_ijk);
            //vh - 0.5 * (vh0 - vh) - 0.5 * (vh1 - vh) - 0.5 * (vh2 - vh);
            T v1 = 2.5 * vh - 0.5 * (vh0 + vh1 + vh2);

            shared_data.xValueT(fl_ijk) = (vH + v1) / 2;
            //shared_data.ttypeValue(fl_ijk) = ninfo.subtreeType(subtree_level);
            shared_data.ctypeValue(fl_ijk) = ninfo.tile().type(nl_ijk);
        }
        else {
            shared_data.xValueT(fl_ijk) = ninfo.tile()(x_channel, nl_ijk);
            //shared_data.ttypeValue(fl_ijk) = ninfo.subtreeType(subtree_level);
            shared_data.ctypeValue(fl_ijk) = ninfo.tile().type(nl_ijk);
        }

        //shared_data.xValueT(fl_ijk) = empty ? 0 : ninfo.tile()(x_channel, nl_ijk);
        //shared_data.ttypeValue(fl_ijk) = empty ? info.subtreeType(subtree_level) : ninfo.subtreeType(subtree_level);
        //shared_data.ctypeValue(fl_ijk) = empty ? CellType::DIRICHLET : ninfo.tile().type(nl_ijk);
    }
}

__global__ void ConservativeNegativeLaplacianSameLevel128Kernel(HATileAccessor<Tile> acc, HATileInfo<Tile>* tiles, int subtree_level, uint8_t launch_tile_types, int x_channel, int Ax_channel, bool calc_diag = false) {
    __shared__ CMGLaplacianTileData shared_data;
    //block idx (tile idx), distinguish from thread idx
    int bi = blockIdx.x;
    //thread idx
    int ti = threadIdx.x;

    auto& info = tiles[bi];


    if (!(info.subtreeType(subtree_level) & launch_tile_types)) return;
	auto h = acc.voxelSize(info);

    LoadCMGLaplacianTileData(acc, info, shared_data, subtree_level, x_channel, ti);
    __syncthreads();


    for (int i = 0; i < 4; i++) {
        //voxel idx
        int vi = i * 128 + ti;
        Coord l_ijk = acc.localOffsetToCoord(vi);
        //
        //auto g_ijk = acc.composeGlobalCoord(info.mTileCoord, l_ijk);
        //if (g_ijk == Coord(47, 126, 83)) {
        //    printf("CMG l_ijk: %d %d %d\n", l_ijk[0], l_ijk[1], l_ijk[2]);
        //}
        //else continue;

		info.tile()(Ax_channel, vi) = shared_data.negativeLap(h, l_ijk, calc_diag);
    }
}

//we need to calculate laplacian either on all leafs, or on a specific level
void ConservativeNegativeLaplacianSameLevel128(HADeviceGrid<Tile>& grid, thrust::device_vector<HATileInfo<Tile>>& tiles, int launch_tile_num, int subtree_level, uint8_t launch_tile_types, int x_channel, int Ax_channel, bool calc_diag) {
    ConservativeNegativeLaplacianSameLevel128Kernel << <launch_tile_num, 128 >> > (grid.deviceAccessor(), thrust::raw_pointer_cast(tiles.data()), subtree_level, launch_tile_types, x_channel, Ax_channel);
}

//on all leafs of the tree
void ConservativeFullNegativeLaplacian(HADeviceGrid<Tile>& grid, const int x_channel, const int Ax_channel, bool calc_diag) {
    //PropagateValues(grid, x_channel, x_channel, -1, GHOST, LAUNCH_SUBTREE);
    //PropagateValuesToGhostTiles(grid, x_channel, x_channel);
    PropagateToChildren(grid, x_channel, x_channel, -1, GHOST, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);
    AccumulateToParentsOneStep(grid, x_channel, x_channel, LEAF, 1. / 8, false, INTERIOR | DIRICHLET | NEUMANN);

    ConservativeNegativeLaplacianSameLevel128(grid, grid.dAllTiles, grid.dAllTiles.size(), -1, LEAF, x_channel, Ax_channel, calc_diag);

    //NegativeLaplacianSameLevel(grid, x_channel, Ax_channel, -1, LEAF | GHOST, LAUNCH_SUBTREE, false);
    //add fine terms stored in ghost cells to parents
    //AccumulateValues(grid, Ax_channel, Ax_channel, -1, LEAF, LAUNCH_SUBTREE, true);
    //AccumulateValuesToLeafTiles(grid, Ax_channel, Ax_channel, true);
    //AccumulateToParentsOneStep(grid, Ax_channel, Ax_channel, GHOST, 1., true, INTERIOR | DIRICHLET | NEUMANN);
}


__global__ void ConservativeNegativeLaplacianAndGaussSeidelSameLevel128Kernel(HATileAccessor<Tile> acc, HATileInfo<Tile>* tiles, int subtree_level, uint8_t launch_tile_types, int x_channel, int rhs_channel, int color) {
    __shared__ CMGLaplacianTileData shared_data;
    //block idx (tile idx), distinguish from thread idx
    int bi = blockIdx.x;
    //thread idx
    int ti = threadIdx.x;

    auto& info = tiles[bi];


    if (!(info.subtreeType(subtree_level) & launch_tile_types)) return;
    auto h = acc.voxelSize(info);

    LoadCMGLaplacianTileData(acc, info, shared_data, subtree_level, x_channel, ti);
    __syncthreads();


    for (int i = 0; i < 4; i++) {
        //voxel idx
        int vi = i * 128 + ti;
        Coord l_ijk = acc.localOffsetToCoord(vi);
        int li = l_ijk[0], lj = l_ijk[1], lk = l_ijk[2];
        if ((li + lj + lk) % 2 == color) {
			auto ctype = shared_data.ctypeValue(l_ijk);
			T b = info.tile()(rhs_channel, vi);
			T Ax = shared_data.negativeLap(h, l_ijk, false);
			T D = shared_data.negativeLap(h, l_ijk, true);

            if (D == 0 && (ctype & INTERIOR)) {
				auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
                printf("D is zero!!!!!!!!!!!!!!! at g_ijk %d %d %d level %d\n", g_ijk[0], g_ijk[1], g_ijk[2], info.mLevel);
            }

			T delta_x = (ctype & INTERIOR) ? (b - Ax) / D : 0;
			info.tile()(x_channel, vi) = shared_data.xValueT(l_ijk) + delta_x;
        }
    }
}

//we need to calculate laplacian either on all leafs, or on a specific level
void ConservativeNegativeLaplacianAndGaussSeidelSameLevel128(HADeviceGrid<Tile>& grid, thrust::device_vector<HATileInfo<Tile>>& tiles, int launch_tile_num, int subtree_level, uint8_t launch_tile_types, int x_channel, int rhs_channel, int color) {
    ConservativeNegativeLaplacianAndGaussSeidelSameLevel128Kernel << <launch_tile_num, 128 >> > (grid.deviceAccessor(), thrust::raw_pointer_cast(tiles.data()), subtree_level, launch_tile_types, x_channel, rhs_channel, color);
}


void GaussSeidelCMG(int iters, int order, HADeviceGrid<Tile>& grid, int level, int x_channel, int rhs_channel) {
    for (int i = 0; i < iters; i++) {
		ConservativeNegativeLaplacianAndGaussSeidelSameLevel128(grid, grid.dTileArrays[level], grid.hNumTiles[level], level, LEAF, x_channel, rhs_channel, order);
        ConservativeNegativeLaplacianAndGaussSeidelSameLevel128(grid, grid.dTileArrays[level], grid.hNumTiles[level], level, LEAF, x_channel, rhs_channel, 1 - order);
	}
}

__global__ void ConservativeResidualAndRestrict128Kernel(HATileAccessor<Tile> acc, HATileInfo<Tile>* fine_tiles, int level, uint8_t launch_tile_types, int fine_x_channel, int fine_rhs_channel, int coarse_residual_channel, T one_over_alpha) {
    __shared__ CMGLaplacianTileData shared_data;
    __shared__ T residual[8 * 8 * 8];
    //block idx (tile idx), distinguish from thread idx
    int bi = blockIdx.x;
    //thread idx
    int ti = threadIdx.x;

    auto& finfo = fine_tiles[bi];


    if (!(finfo.subtreeType(level) & launch_tile_types)) return;
    auto h = acc.voxelSize(finfo);

    LoadCMGLaplacianTileData(acc, finfo, shared_data, level, fine_x_channel, ti);
    __syncthreads();

    //calculate residual
    for (int i = 0; i < 4; i++) {
        //voxel idx
        int vi = i * 128 + ti;
        Coord l_ijk = acc.localOffsetToCoord(vi);
        residual[vi] = finfo.tile()(fine_rhs_channel, vi) - shared_data.negativeLap(h, l_ijk, false);
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
            T sum = 0;
            for (int ii : {0, 1}) {
                for (int jj : {0, 1}) {
                    for (int kk : {0, 1}) {
                        Coord fl1_ijk = fl_ijk + Coord(ii, jj, kk);
                        int vi1 = acc.localCoordToOffset(fl1_ijk);
                        sum += residual[vi1];
                    }
                }
            }
            //ghost accumulate
            if (finfo.subtreeType(level) & GHOST) {
                cinfo.tile()(coarse_residual_channel, cl_ijk) += sum;
			}
			else {
				cinfo.tile()(coarse_residual_channel, cl_ijk) = sum * one_over_alpha;
			}
        }

    }
}

void ConservativeResidualAndRestrict(HADeviceGrid<Tile>& grid, int fine_x_channel, int fine_rhs_channel, int coarse_residual_channel, int level, uint8_t launch_tile_types, T one_over_alpha) {
    ConservativeResidualAndRestrict128Kernel << <grid.hNumTiles[level], 128 >> > (grid.deviceAccessor(), thrust::raw_pointer_cast(grid.dTileArrays[level].data()), level, launch_tile_types, fine_x_channel, fine_rhs_channel, coarse_residual_channel, one_over_alpha);
}

__global__ void ConservativeProlongateAndUpdate128Kernel(HATileAccessor<Tile> acc, HATileInfo<Tile>* fine_tiles, int fine_level, uint8_t fine_tile_types, int coarse_x_channel, int fine_x_channel, T prolong_coeff) {
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

void ConservativeProlongateAndUpdate128(HADeviceGrid<Tile>& grid, int coarse_x_channel, int fine_x_channel, int fine_level, T prolong_coeff) {
    ConservativeProlongateAndUpdate128Kernel << <grid.hNumTiles[fine_level], 128 >> > (grid.deviceAccessor(), thrust::raw_pointer_cast(grid.dTileArrays[fine_level].data()), fine_level, LEAF, coarse_x_channel, fine_x_channel, prolong_coeff);
}

//must initialize to zero 
void CMGSolver::VCycle(HADeviceGrid<Tile>& grid, int x_channel, int f_channel, const int rhs_channel, int level_iters, int coarsest_iters) {
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

    //timer.stop("init"); timer.start();

    //downstroke
    for (int i = grid.mMaxLevel; i > 0; i--) {
        //u^l=0
        //this is already done at the beginning
        //Fill(grid, x_channel, (Tile::T)0, i, LEAF | NONLEAF, LAUNCH_LEVEL);

        //smooth(u^l, b^l)
		GaussSeidelCMG(level_iters, 0, grid, i, x_channel, rhs_channel);

        //calculate residual and restrict to next level
        //r^l = b^l-Au^l
        ConservativeResidualAndRestrict(grid, x_channel, rhs_channel, rhs_channel, i, LEAF, one_over_alpha);
    }

    //smooth bottom
	GaussSeidelCMG(coarsest_iters / 2, 0, grid, 0, x_channel, rhs_channel);
	GaussSeidelCMG(coarsest_iters / 2, 1, grid, 0, x_channel, rhs_channel);

    for (int i = 1; i <= grid.mMaxLevel; i++) {
		ConservativeProlongateAndUpdate128(grid, x_channel, x_channel, i, prolong_coeff);

        //smoothing: fine.x = smooth(fine.b)
		GaussSeidelCMG(level_iters, 1, grid, i, x_channel, rhs_channel);
    }
}

std::tuple<int, double> CMGSolver::solve(HADeviceGrid<Tile>& grid, bool verbose, int max_iters, double relative_tolerance, int level_iters, int coarsest_iters, int sync_stride, bool is_pure_neumann)
{
    double rhs_norm2, threshold_norm2, last_residual_norm2;

    //Use 0 as initial guess
    //x0=0
    Fill(grid, Tile::x_channel, (T)0, -1, LEAF, LAUNCH_SUBTREE, INTERIOR | NEUMANN | DIRICHLET);//fill all leafs with 0
    //initial residual is b

    if (sync_stride > 0) {
        //rhs_norm2=r*r
        last_residual_norm2 = rhs_norm2 = Dot(grid, Tile::r_channel, Tile::r_channel, LEAF);
        if (verbose) Info("ConjugateGradient with initial norm of rhs: {}", sqrt(rhs_norm2));

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
    if (is_pure_neumann) ReCenterLeafCells(grid, Tile::r_channel, cnt_reducer, mean_d, count_d);

    ////z0=Minv*r0
    //r_channel->z_channel
    //VCycle(grid, Tile::z_channel, Tile::r_channel, Tile::Ap_channel, Tile::tmp_channel, Tile::D_channel, level_iters, coarsest_iters);
    VCycle(grid, Tile::z_channel, Tile::r_channel, Tile::Ap_channel, level_iters, coarsest_iters);

    //p0=z0
    Copy(grid, Tile::z_channel, Tile::p_channel, -1, LEAF, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);


    //gamma = Dot(grid, Tile::z_channel, Tile::r_channel, -1, LEAF, LAUNCH_SUBTREE);
    DotAsync(gamma_d, grid, Tile::z_channel, Tile::r_channel, LEAF);
    //Info("initial dotzr: {}", Dot(grid, Tile::z_channel, Tile::r_channel, LEAF));


    int i = 0;
    for (i = 0; i < max_iters; i++) {
        //Ap_k=A*p_k
        //here A=-lap
        ConservativeFullNegativeLaplacian(grid, Tile::p_channel, Tile::Ap_channel);

        //alpha_k=gamma_k/(p_k^T*A*p_k)
        //fp = Dot(grid, Tile::p_channel, Tile::Ap_channel, -1, LEAF, LAUNCH_SUBTREE);//fp_k=p_k^T*A*p_k
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
                Info("ConjugateGradient iter {} norm {:e}({:e}), convergence rate={:e}", i, sqrt(residual_norm2), sqrt(residual_norm2 / rhs_norm2), sqrt(residual_norm2 / last_residual_norm2));
                last_residual_norm2 = residual_norm2;
            }
            if (residual_norm2 < threshold_norm2) {
                return std::make_tuple(i + 1, sqrt(residual_norm2 / rhs_norm2));
            }
        }
        if (is_pure_neumann) ReCenterLeafCells(grid, Tile::r_channel, cnt_reducer, mean_d, count_d);

        //z_{k+1} = Minv * r_{k+1}
        //r->z
        //VCycle(grid, Tile::z_channel, Tile::r_channel, Tile::Ap_channel, Tile::tmp_channel, Tile::D_channel, level_iters, coarsest_iters);
        VCycle(grid, Tile::z_channel, Tile::r_channel, Tile::Ap_channel, level_iters, coarsest_iters);


        //gamma_old = gamma;
        TernaryOnArray(gamma_d, gamma_d, gamma_old_d, []__device__(double& gamma, double& _, double& gamma_old) { gamma_old = gamma; });

        //gamma_{k+1} = dot(r_{k+1}, z_{k+1})
        //gamma = Dot(grid, Tile::z_channel, Tile::r_channel, -1, LEAF, LAUNCH_SUBTREE);
        DotAsync(gamma_d, grid, Tile::r_channel, Tile::z_channel, LEAF);

        //beta_{k+1} = gamma_{k+1} / gamma_k
        //beta = gamma / gamma_old;
        TernaryOnArray(beta_d, gamma_d, gamma_old_d, []__device__(double& beta, double gamma, double gamma_old) { beta = gamma / gamma_old; });

        auto beta_ptr = beta_d;
        //p_{k+1} = z_{k+1} + beta_{k+1} * p_{k}
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
