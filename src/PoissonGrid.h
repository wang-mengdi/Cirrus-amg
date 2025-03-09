#pragma once

//#include "HAGrid.h"
#include "PoissonTile.h"

template <class T, typename Func3>
__global__ void TernaryOnArrayKernel(T* a, T* b, T* c, Func3 f, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        f(a[idx], b[idx], c[idx]);
    }
}

template<class T, class Func3>
void TernaryOnArray(T* d_a, T* d_b, T* d_c, Func3 f, int n = 1, int block_size = 512) {
    if (n == 1) block_size = 1;

    int numBlocks = (n + block_size - 1) / block_size;


    TernaryOnArrayKernel << <numBlocks, block_size >> > (d_a, d_b, d_c, f, n);
}

//All these operators will only perform on interior cells

//apply c[i]=f(a[i],b[i]) on all interior voxels
template<class UnaryOP>
void UnaryTransform(HADeviceGrid<Tile>& grid, const int in_channel, const int out_channel, UnaryOP f, const int level, const uint8_t launch_types, const LaunchMode mode, const uint8_t cell_types = INTERIOR) {
    grid.launchVoxelFunc(
        [=]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();
        if (tile.type(l_ijk) & cell_types) {
            tile(out_channel, l_ijk) = f(tile(in_channel, l_ijk));
            //if (l_ijk == Coord(4, 5, 4)) {
            //    printf("tile[%d,%d,%d]=%f type %d\n", l_ijk[0], l_ijk[1], l_ijk[2], tile(out_channel, l_ijk), tile.type(l_ijk));
            //}
        }
    },
        level, launch_types, mode
    );
}

//apply c[i]=f(a[i],b[i]) on all interior voxels
template<class BinaryOP>
void BinaryTransform(HADeviceGrid<Tile>& grid, const int in1_channel, const int in2_channel, const int out_channel, BinaryOP f, const int level, const uint8_t launch_types, const LaunchMode mode, const uint8_t cell_types = INTERIOR) {
    grid.launchVoxelFunc(
        [=]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();
        if (tile.type(l_ijk) & cell_types) {
            tile(out_channel, l_ijk) = f(tile(in1_channel, l_ijk), tile(in2_channel, l_ijk));
        }
    },
        level, launch_types, mode
    );
}

void CalculateNeighborTiles(HADeviceGrid<Tile>& grid);

void Copy(HADeviceGrid<Tile>& grid, const int in_channel, const int out_channel, const int level, const uint8_t launch_types, const LaunchMode mode, const uint8_t cell_types = INTERIOR);
void Fill(HADeviceGrid<Tile>& grid, const int channel, const Tile::T val, const int level, const uint8_t launch_types, const LaunchMode mode, const uint8_t cell_types = INTERIOR);

void DotAsync(double* d_result, HADeviceGrid<Tile>& grid, const uint8_t in1_channel, const uint8_t in2_channel, const uint8_t launch_tile_types);
double Dot(HADeviceGrid<Tile>& grid, const uint8_t in1_channel, const uint8_t in2_channel, const uint8_t launch_tile_types);
__global__ void ChannelPowerSumKernel128(const int order, HATileAccessor<PoissonTile<T>> acc, HATileInfo<PoissonTile<T>>* infos, int subtree_level, uint8_t launch_tile_types, const int in_channel, double* value_sum, double* weights_sum, bool volume_weighted, bool use_abs, uint8_t launch_cell_types);
double NormSync(HADeviceGrid<Tile>& grid, const int order, const int in_channel, bool volume_weighted, uint8_t launch_cell_types = INTERIOR);

//copy values from parents for tiles specified by propagate_tile_types
//for example, GHOST will propagate ghost values from parents
//this is actually prolongation with sum kernel
//we call it propagate following the convention in SPGrid paper
void PropagateToChildren(HADeviceGrid<Tile>& grid, const int coarse_channel, const int fine_channel, const int target_subtree_level, const uint8_t target_tile_types, const LaunchMode mode, const uint8_t cell_types);
//accumulate values from children (fine levels) to parents (coarse levels)
//will launch parent tiles respective to target_subtree_level and target_tile_types
//calculate on cell_types only, and if child cell is not in cell_type, just ignore its value (set to 0)
//coeff is: parent+=child*coeff
//if additive, will add to parent value, otherwise will overwrite parent value
void AccumulateToParents(HADeviceGrid<Tile>& grid, const int fine_channel, const int coarse_channel, const int target_subtree_level, const uint8_t target_tile_types, const LaunchMode mode, const uint8_t cell_types, const Tile::T coeff, bool additive);

void AccumulateToParentsOneStep(HADeviceGrid<Tile>& grid, const int fine_channel, const int coarse_channel, const uint8_t fine_tile_types, const Tile::T coeff, bool additive, uint8_t cell_types);

__global__ void AccumulateFacesToParentsOneStepKernel(HATileAccessor<Tile> acc, HATileInfo<Tile>* fine_tiles, int fine_subtree_level, uint8_t fine_tile_types, int fine_u_channel, int coarse_u_channel, Tile::T coeff, bool additive, uint8_t cell_types);
void AccumulateFacesToParentsOneStep(HADeviceGrid<Tile>& grid, const int fine_u_channel, const int coarse_u_channel, const uint8_t fine_tile_types, const Tile::T coeff, bool additive, uint8_t cell_types);

//tricky thing: there are two possibilities: (1) ghost tiles do not contain valid data, (2) ghost-leaf faces contain valid data
//for (1), you need to (a) propagate faces before calculating velocity nodes, (b) use findNodeNeighborLeaf in CalcLeafNodeValuesFromFaceCenters
//(c) use findPlusFaceIntpVoxel in InterpolateFaceValue, which basically tries to find the neighbor leaf cell
//for (2), you need to (a) calculate correct ghost values and not propagate, (b) set weight 
void CalcLeafNodeValuesFromFaceCenters(HADeviceGrid<Tile>& grid, const int u_channel, const int node_u_channel);
void CalcLeafNodeValuesFromCellCenters(HADeviceGrid<Tile>& grid, const int cell_channel, const int node_channel);

__device__ Tile::T InterpolateCellValue(const HATileAccessor<Tile>& acc, const Vec& pos, const int cell_channel, const int node_channel);
__device__ Vec InterpolateFaceValue(const HATileAccessor<Tile>& acc, const Vec& pos, const int u_channel, const int node_u_channel);

template<class FuncII>
__hostdev__ void IterateFaceNeighborCellTypes(const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const Coord& l_ijk, const int axis, FuncII f) {
    auto& tile = info.tile();
    uint8_t type0 = tile.type(l_ijk);

    auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
    auto ng_ijk = g_ijk; ng_ijk[axis]--;
    HATileInfo<Tile> ninfo; Coord nl_ijk;
    acc.findVoxel(info.mLevel, ng_ijk, ninfo, nl_ijk);

    if (!ninfo.empty()) {
        if (!ninfo.isLeaf()) {
            for (int offj : {0, 1}) {
                for (int offk : {0, 1}) {
                    Coord child_offset = acc.rotateCoord(axis, Coord(1, offj, offk));
                    Coord nc_ijk = acc.childCoord(ng_ijk, child_offset);
					HATileInfo<Tile> nc_info; Coord ncl_ijk;
					acc.findVoxel(info.mLevel + 1, nc_ijk, nc_info, ncl_ijk);
                    if (!nc_info.empty()) {
                        auto& nctile = nc_info.tile();
						uint8_t type1 = nctile.type(ncl_ijk);
						f(type0, type1);
                    }
                }
            }
        }
        else {
            if (ninfo.isGhost()) {
                //it's coarser
                Coord np_ijk = acc.parentCoord(ng_ijk);
                acc.findVoxel(info.mLevel - 1, np_ijk, ninfo, nl_ijk);

            }
            auto& ntile = ninfo.tile();
			uint8_t type1 = ntile.type(nl_ijk);
			f(type0, type1);
        }
    }
}