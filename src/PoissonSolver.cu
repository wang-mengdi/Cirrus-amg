#include "PoissonSolver.h"
#include "CPUTimer.h"
//#include "GMGSolver.h"

void CalcCellTypesFromLeafs(HADeviceGrid<Tile>& grid) {
    //We already have cell types for leafs

    //step 1: update non-leafs
    grid.launchVoxelFunc(
        [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();
        auto g_ijk = acc.composeGlobalCoord(info.mTileCoord, l_ijk);

        int dirichlet_cnt = 0;
        int interior_cnt = 0;
        int neumann_cnt = 0;

        acc.iterateChildVoxels(info, l_ijk,
            [&]__device__(const HATileInfo<Tile>&c_info, const Coord & c_ijk) {
            if (!c_info.empty() && c_info.isActive()) {
                auto& tile = c_info.tile();
                auto typ = tile.type(c_ijk);
                if (typ & DIRICHLET) dirichlet_cnt++;
                else if (typ & INTERIOR) interior_cnt++;
            }
        });

        //extrapolation actually requires that a NONLEAF cell is marked as INTERIOR
        //if it has some INTERIOR children
        //therefore the face will be masked as valid
        //interior>dirichlet>neumann
        //if (interior_cnt > 0) tile.type(l_ijk) = INTERIOR;
        //else if (dirichlet_cnt > 0) tile.type(l_ijk) = DIRICHLET;
        //else tile.type(l_ijk) = NEUMANN;

        ////dirichlet>interior>neumann
        if (dirichlet_cnt > 0) tile.type(l_ijk) = DIRICHLET;
        else if (interior_cnt > 0) tile.type(l_ijk) = INTERIOR;
        else tile.type(l_ijk) = NEUMANN;
    },
        -1, NONLEAF, LAUNCH_SUBTREE, FINE_FIRST);

    //step 2: update ghosts
    grid.launchVoxelFuncOnAllTiles(
        [=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
        auto& tile = info.tile();
        auto g_ijk = acc.composeGlobalCoord(info.mTileCoord, l_ijk);
        auto p_g_ijk = acc.parentCoord(g_ijk);
        HATileInfo<Tile> p_info; Coord p_l_ijk;
        acc.findVoxel(info.mLevel - 1, p_g_ijk, p_info, p_l_ijk);
        if (!p_info.empty()) {
            tile.type(l_ijk) = p_info.tile().type(p_l_ijk);
        }
        else {
            tile.type(l_ijk) = DIRICHLET;
        }
    }, GHOST);
        //-1, GHOST, LAUNCH_SUBTREE, COARSE_FIRST);
}

void ReCenterLeafCells(HADeviceGrid<Tile>& grid, const int channel, DeviceReducer<double>& cnt_reducer, double* d_mean, double* d_count) {
    int num_tiles = grid.dAllTiles.size();
    cnt_reducer.resize(num_tiles);

    if (num_tiles > 0) {
        ChannelPowerSumKernel128 << <num_tiles, 128 >> > (
            1,//1st order for mean
            grid.deviceAccessor(),
            thrust::raw_pointer_cast(grid.dAllTiles.data()),
            -1, LEAF, //subtree level and launched tile types
            channel,
            grid.dAllTilesReducer.data(), cnt_reducer.data(),
            false, //point wise
            false, //use_abs
            INTERIOR // cell types
            );
        grid.dAllTilesReducer.sumAsyncTo(d_mean);
        cnt_reducer.sumAsyncTo(d_count);
        TernaryOnArray(d_mean, d_count, d_count, []__device__(double& mean, double count, double _) { mean = mean / count; });

        grid.launchVoxelFuncOnAllTiles(
            [=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
            auto& tile = info.tile();
            double mean = *d_mean;
            if (tile.isInterior(l_ijk)) {
                tile(channel, l_ijk) -= mean;
            }
        }, LEAF, 4
        );

		CheckCudaError("ReCenterLeafCells");
    }
}