#include "PoissonGrid.h"



template<class T>
__global__ void FillArrayKernel(T* arr, const T val, const int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        arr[idx] = val;
    }
}

template<class T>
void FillArray(T* d_array, const int n, const T val, const int block_size = 512) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error before FillArray: " << cudaGetErrorString(err) << std::endl;
    }

    int numBlocks = (n + block_size - 1) / block_size;

    FillArrayKernel << <numBlocks, block_size >> > (d_array, val, n);
}


void Copy(HADeviceGrid<Tile>& grid, const int in_channel, const int out_channel, const int level, const uint8_t launch_types, const LaunchMode mode, const uint8_t cell_types) {
    UnaryTransform(grid, in_channel, out_channel, []__device__(Tile::T x) { return x; }, level, launch_types, mode, cell_types);
}

//will fill all voxels specified by level and exec_policy
//no matter if the voxel is active or not
void Fill(HADeviceGrid<Tile>& grid, const int channel, const Tile::T val, const int level, const uint8_t launch_types, const LaunchMode mode, const uint8_t cell_types) {
    UnaryTransform(grid, channel, channel, [=]__device__(Tile::T x) { return val; }, level, launch_types, mode, cell_types);
}

template<typename T>
T CUBDeviceArraySum(const T* device_ptr, const int n) {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    T* d_result;
    cudaMalloc(&d_result, sizeof(T));
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, device_ptr, d_result, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, device_ptr, d_result, n);
    T result;
    cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_temp_storage);
    cudaFree(d_result);
    return result;
}

template<typename T1, typename T2>
void CUBDeviceArraySumAsync(const T1* device_ptr, const int n, T2* d_result) {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, device_ptr, d_result, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, device_ptr, d_result, n);
    cudaFree(d_temp_storage);
}

template<typename T>
T CUBDeviceArrayMax(const T* device_ptr, const int n) {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    T* d_result;
    cudaMalloc(&d_result, sizeof(T));
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, device_ptr, d_result, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, device_ptr, d_result, n);
    T result;
    cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_temp_storage);
    cudaFree(d_result);
    return result;
}

void CalculateNeighborTiles(HADeviceGrid<Tile>& grid) {
    grid.launchTileFunc(
        [=] __device__(HATileAccessor<Tile> acc, const int _, HATileInfo<Tile>&info) {
        auto& tile = info.tile();
        for (int axis : {0, 1, 2}) {
            Coord b_ijk = info.mTileCoord;
            Coord nb_ijk = b_ijk; nb_ijk[axis]--;
            Coord np_ijk = b_ijk; np_ijk[axis]++;

            tile.mNeighbors[axis + 0] = acc.tileInfo(info.mLevel, nb_ijk);
            tile.mNeighbors[axis + 3] = acc.tileInfo(info.mLevel, np_ijk);
        }
    }, -1, LEAF | GHOST | NONLEAF, LAUNCH_SUBTREE);
}

template<typename T>
struct Vec4Type;

template<>
struct Vec4Type<float> {
    using Type = float4;
};

template<>
struct Vec4Type<double> {
    using Type = double4;
};


//follow the same launch convention as launchVoxelFunc
__global__ void Dot128Kernel(HATileAccessor<Tile> acc, HATileInfo<Tile>* infos, const uint8_t in1_channel, const uint8_t in2_channel, double* sum, int subtree_level, uint8_t launch_types) {
    int bi = blockIdx.x;
    int ti = threadIdx.x;
    const auto& info = infos[bi];
    

    if (!(info.subtreeType(subtree_level) & launch_types)) {
        if (ti == 0) sum[bi] = 0;
        return;
    }

    auto& tile = info.tile();
    auto data1AsVec4 = reinterpret_cast<typename Vec4Type<T>::Type*>(tile.mData[in1_channel]);
    auto data2AsVec4 = reinterpret_cast<typename Vec4Type<T>::Type*>(tile.mData[in2_channel]);
    auto data1 = data1AsVec4[ti];
    auto data2 = data2AsVec4[ti];
	double thread_dot = data1.x * data2.x + data1.y * data2.y + data1.z * data2.z + data1.w * data2.w;
    

    typedef cub::BlockReduce<double, 128> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    double block_sum = BlockReduce(temp_storage).Sum(thread_dot);

    if (ti==0) {
        sum[bi] = block_sum;
    }
}

//follow the launch convention of launchVoxelFunc
void DotAsync(double* d_result, HADeviceGrid<Tile>& grid, const uint8_t in1_channel, const uint8_t in2_channel, const uint8_t launch_tile_types) {
    int num_tiles = grid.dAllTiles.size();
    FillArray(grid.dAllTilesReducer.data(), grid.dAllTiles.size(), 0.0);
    if (num_tiles > 0) {
        Dot128Kernel << <num_tiles, 128 >> > (
            grid.deviceAccessor(),
            thrust::raw_pointer_cast(grid.dAllTiles.data()),
            in1_channel, in2_channel,
            grid.dAllTilesReducer.data(),
            -1, launch_tile_types
            );
		grid.dAllTilesReducer.sumAsyncTo(d_result);
    }

    CheckCudaError("DotAsync end");
}

//follow the launch convention of launchVoxelFunc
double Dot(HADeviceGrid<Tile>& grid, const uint8_t in1_channel, const uint8_t in2_channel, const uint8_t launch_tile_types) {
    double* d_result;
    cudaMalloc(&d_result, sizeof(double));
    
	DotAsync(d_result, grid, in1_channel, in2_channel, launch_tile_types);

    double result;
    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    return result;
}

//it will calculate the ordered pointwise/volume weighted sum on launched tiles and INTERIOR cells
//-1: linf
//1: l1
//2: l2
//if weights_sum is not null, then compute the sum of weights
//if volume_weighted is set, then weight by h^3, otherwise pointwise (weight by 1)
//if use_abs is set, then use abs value
__global__ void ChannelPowerSumKernel128(const int order, HATileAccessor<Tile> acc, HATileInfo<Tile>* infos, int subtree_level, uint8_t launch_tile_types, const int in_channel, double* value_sum, double* weights_sum, bool volume_weighted, bool use_abs, uint8_t launch_cell_types) {
    int bi = blockIdx.x;
    int ti = threadIdx.x;
    const auto& info = infos[bi];

    if (!(info.subtreeType(subtree_level) & launch_tile_types)) {
        if (ti == 0) {
            value_sum[bi] = 0;
            if (weights_sum) weights_sum[bi] = 0;
        }
        return;
    }

    auto& tile = info.tile();
    auto tile_data_as_vec4 = reinterpret_cast<typename Vec4Type<T>::Type*>(tile.mData[in_channel]);
    auto data = tile_data_as_vec4[ti];
	auto tile_types_as_uchar4 = reinterpret_cast<uchar4*>(tile.mCellType);
	auto type = tile_types_as_uchar4[ti];

	data.x = (type.x & launch_cell_types) ? data.x : 0;
	data.y = (type.y & launch_cell_types) ? data.y : 0;
	data.z = (type.z & launch_cell_types) ? data.z : 0;
	data.w = (type.w & launch_cell_types) ? data.w : 0;
    int thread_cnt =
        ((type.x & launch_cell_types) != 0) +
        ((type.y & launch_cell_types) != 0) +
        ((type.z & launch_cell_types) != 0) +
        ((type.w & launch_cell_types) != 0);


    if (use_abs) {
		data.x = abs(data.x);
		data.y = abs(data.y);
		data.z = abs(data.z);
		data.w = abs(data.w);
    }

    double thread_value_sum;
    if (order == -1) {
        thread_value_sum = max(max(data.x, data.y), max(data.z, data.w));
    }
    else if (order == 1) {
		thread_value_sum = data.x + data.y + data.z + data.w;
	}
	else if (order == 2) {
		thread_value_sum = data.x * data.x + data.y * data.y + data.z * data.z + data.w * data.w;
	}
    else {
		thread_value_sum = pow(data.x, order) + pow(data.y, order) + pow(data.z, order) + pow(data.w, order);
    }

    double block_value_sum, block_weight_sum;

    typedef cub::BlockReduce<double, 128> BlockReduce;
    __shared__ typename BlockReduce::TempStorage value_sum_storage;
    __shared__ typename BlockReduce::TempStorage weight_sum_storage;

    if (order == -1) {
        //use max instead of sum
		block_value_sum = BlockReduce(value_sum_storage).Reduce(thread_value_sum, cub::Max());
		if (weights_sum) {
            block_weight_sum = 1;//maximum of count
		}
    }
    else {
        block_value_sum = BlockReduce(value_sum_storage).Sum(thread_value_sum);
        if (weights_sum) {
            block_weight_sum = BlockReduce(weight_sum_storage).Sum((double)thread_cnt);
        }
    }

    if (ti == 0) {
        T h = acc.voxelSize(info);
        T single_weight = volume_weighted ? h * h * h : 1;
        value_sum[bi] = block_value_sum * single_weight;
		if (weights_sum) weights_sum[bi] = block_weight_sum * single_weight;

		//printf("tile level %d coord %d %d %d value_sum %f weights_sum %f wingle weight %f bloick_weight_sum %f thread cnt %d type %x\n", info.mLevel, info.mTileCoord[0], info.mTileCoord[1], info.mTileCoord[2], value_sum[bi], weights_sum ? weights_sum[bi] : 0, single_weight, block_weight_sum, thread_cnt, type);

    }
}

//on LEAF tiles and INTERIOR cells
double NormSync(HADeviceGrid<Tile>& grid, const int order, const int in_channel, bool volume_weighted, uint8_t launch_cell_types) {
    if (order == -1) {
        Assert(!volume_weighted, "Linf norm does not support volume weighted, it only works with point-wise norm");
    }

	size_t num_tiles = grid.dAllTiles.size();
    if (num_tiles > 0) {
        if (order == -1) {
            ChannelPowerSumKernel128 << <num_tiles, 128 >> > (
                order, grid.deviceAccessor(),
                thrust::raw_pointer_cast(grid.dAllTiles.data()),
                -1, LEAF, //subtree level and launched tile types
                in_channel,
                grid.dAllTilesReducer.data(), nullptr,
                false, //volume weighted
                true, //use_abs
                launch_cell_types // cell types
                );
            return grid.dAllTilesReducer.maxSync();
        }
        else {
            DeviceReducer<double> weights_sum_reducer(grid.dAllTiles.size());
            ChannelPowerSumKernel128 << <num_tiles, 128 >> > (
                order, grid.deviceAccessor(),
                thrust::raw_pointer_cast(grid.dAllTiles.data()),
                -1, LEAF, //subtree level and launched tile types
                in_channel,
                grid.dAllTilesReducer.data(), weights_sum_reducer.data(),
                volume_weighted, //volume weighted
                true, //use_abs
                launch_cell_types // cell types
                );
            double value_sum = CUBDeviceArraySum(grid.dAllTilesReducer.data(), num_tiles);
            double weights_sum = CUBDeviceArraySum(weights_sum_reducer.data(), num_tiles);

			//Info("NormSync order: {} volume_weighted {} value_sum {} weights_sum {}", order,volume_weighted, value_sum, weights_sum);

            if (order == 1) return value_sum / weights_sum;
            else if (order == 2) return sqrt(value_sum / weights_sum);
            else return pow(value_sum / weights_sum, 1.0 / order);
        }
    }
    else return 0;
}

void PropagateToChildren(HADeviceGrid<Tile>& grid, const int coarse_channel, const int fine_channel, const int target_subtree_level, const uint8_t target_tile_types, const LaunchMode mode, const uint8_t cell_types) {
    //can be used to propagate one single level or iterating through the whole tree
    //if it's the whole tree, must propagate from coarse to fine
    grid.launchVoxelFunc(
        [=]__device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & fine_l_ijk) {
        auto& tile = info.tile();
        if (!(tile.type(fine_l_ijk) & cell_types)) {
            //tile(fine_channel, fine_l_ijk) = Tile::BACKGROUND_VALUE;
            return;
        }
        auto fine_g_ijk = acc.localToGlobalCoord(info, fine_l_ijk);
        auto coarse_g_ijk = acc.parentCoord(fine_g_ijk);
        HATileInfo<Tile> coarse_info; Coord coarse_l_ijk;
        acc.findVoxel(info.mLevel - 1, coarse_g_ijk, coarse_info, coarse_l_ijk);

        auto val = Tile::BACKGROUND_VALUE;
        if (!coarse_info.empty()) {
            auto& coarse_tile = coarse_info.tile();
            val = coarse_tile(coarse_channel, coarse_l_ijk);
            if (!(coarse_tile.type(coarse_l_ijk) & cell_types)) {
                val = Tile::BACKGROUND_VALUE;
            }
        }
        tile(fine_channel, fine_l_ijk) = val;
    },
        target_subtree_level, target_tile_types, mode, COARSE_FIRST);
}

//will be called on fine tiles
__global__ void AccumulateToParentsOneStepKernel(HATileAccessor<Tile> acc, HATileInfo<Tile>* fine_tiles, int fine_subtree_level, uint8_t fine_tile_types, int fine_channel, int coarse_channel, Tile::T coeff, bool additive, uint8_t cell_types) {
    __shared__ T data[8 * 8 * 8];
	int bi = blockIdx.x;
	int ti = threadIdx.x;
	auto& finfo = fine_tiles[bi];

	if (!(finfo.subtreeType(fine_subtree_level) & fine_tile_types)) {
		return;
	}
	auto& ftile = finfo.tile();
    
    for (int i = 0; i < 4; i++) {
        int vi = i * 128 + ti;
		uint8_t ctype = ftile.type(vi);
        data[vi] = (ctype & cell_types) ? ftile(fine_channel, vi) : 0;
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
            T sum = 0;
            for (int ii : {0, 1}) {
                for (int jj : {0, 1}) {
                    for (int kk : {0, 1}) {
                        Coord fl1_ijk = fl_ijk + Coord(ii, jj, kk);
                        int vi1 = acc.localCoordToOffset(fl1_ijk);
                        sum += data[vi1];
                    }
                }
            }
            //ghost accumulate
            if (additive) {
                ctile(coarse_channel, cl_ijk) += sum * coeff;
            }
            else {
                ctile(coarse_channel, cl_ijk) = sum * coeff;
            }
        }

    }
} 

void AccumulateToParentsOneStep(HADeviceGrid<Tile>& grid, const int fine_channel, const int coarse_channel, const uint8_t fine_tile_types, const Tile::T coeff, bool additive, uint8_t cell_types) {
    Assert(fine_tile_types == GHOST || fine_tile_types == LEAF, "AccumulateToParentsOneStep can only be carried out on GHOST or LEAF tiles");

	int num_fine_tiles = grid.dAllTiles.size();
    AccumulateToParentsOneStepKernel << <num_fine_tiles, 128 >> > (
        grid.deviceAccessor(),
        thrust::raw_pointer_cast(grid.dAllTiles.data()),
        -1, fine_tile_types, fine_channel, coarse_channel, coeff, additive, cell_types
        );
}

//will be called on fine tiles
__global__ void AccumulateFacesToParentsOneStepKernel(HATileAccessor<Tile> acc, HATileInfo<Tile>* fine_tiles, int fine_subtree_level, uint8_t fine_tile_types, int fine_u_channel, int coarse_u_channel, Tile::T coeff, bool additive, uint8_t cell_types) {
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
                if (additive) {
                    ctile(coarse_u_channel + axis, cl_ijk) += sum * coeff;
                }
                else {
                    ctile(coarse_u_channel + axis, cl_ijk) = sum * coeff;
                }
            }
        }

    }
}

void AccumulateFacesToParentsOneStep(HADeviceGrid<Tile>& grid, const int fine_u_channel, const int coarse_u_channel, const uint8_t fine_tile_types, const Tile::T coeff, bool additive, uint8_t cell_types) {
	Assert(fine_tile_types == GHOST || fine_tile_types == LEAF, "AccumulateFacesToParentsOneStep can only be carried out on GHOST or LEAF tiles");

	int num_fine_tiles = grid.dAllTiles.size();
    AccumulateFacesToParentsOneStepKernel << <num_fine_tiles, 128 >> > (
        grid.deviceAccessor(),
        thrust::raw_pointer_cast(grid.dAllTiles.data()),
        -1, fine_tile_types, fine_u_channel, coarse_u_channel, coeff, additive, cell_types
        );
}

void AccumulateToParents(HADeviceGrid<Tile>& grid, const int fine_channel, const int coarse_channel, const int target_subtree_level, const uint8_t target_tile_types, const LaunchMode mode, const uint8_t cell_types, const Tile::T coeff, bool additive) {
    grid.launchVoxelFunc(
        [=]__device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & coarse_l_ijk) {
        auto& tile = info.tile();
        auto coarse_g_ijk = acc.localToGlobalCoord(info, coarse_l_ijk);

        //if (info.mLevel == 2 && coarse_g_ijk == Coord(6, 31, 15)) {
        //    printf("accumulating to parent level %d g_ijk %d %d %d type %d update types %d\n", info.mLevel, coarse_g_ijk[0], coarse_g_ijk[1], coarse_g_ijk[2], tile.type(coarse_l_ijk), cell_types);
        //}

        if (!(tile.type(coarse_l_ijk) & cell_types)) {
            return;
        }
        Tile::T sum = 0;

        for (int cid = 0; cid < acc.NUMCHILDREN; cid++) {
            auto fine_g_ijk = acc.childCoord(coarse_g_ijk, cid);
            HATileInfo<Tile> fine_info; Coord fine_l_ijk;
            acc.findVoxel(info.mLevel + 1, fine_g_ijk, fine_info, fine_l_ijk);
            if (!fine_info.empty()) {
                auto& fine_tile = fine_info.tile();
                if (fine_tile.type(fine_l_ijk) & cell_types) {
                    sum += fine_tile(fine_channel, fine_l_ijk);
                }
                //sum += fine_tile.interiorValue(fine_channel, fine_l_ijk);
            }
        }



        if (additive) tile(coarse_channel, coarse_l_ijk) += sum * coeff;
		else tile(coarse_channel, coarse_l_ijk) = sum * coeff;
    },
        target_subtree_level, target_tile_types, mode, FINE_FIRST);
}

void CalcLeafNodeValuesFromFaceCenters(HADeviceGrid<Tile>& grid, const int u_channel, const int node_u_channel) {
    //for (int axis : {0, 1, 2}) {
    //    PropagateValues(grid, u_channel + axis, u_channel + axis, -1, GHOST, LAUNCH_SUBTREE);
    //}

    auto calcNode = [=]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile> &info, const Coord & l_ijk, int axis) -> Tile::T {
        HATileInfo<Tile> n_info; Coord n_l_ijk;
        Tile::T sum = 0, weight_sum = 0;


        //Coord off(0, 0, 0);
        //int axis1 = acc.rotateAxis(axis, 1);
        //int axis2 = acc.rotateAxis(axis, 2);
        for (int offj : {-1, 0}) {
            for (int offk : {-1, 0}) {
                Coord off_ijk = acc.rotateCoord(axis, Coord(0, offj, offk));
                //off[axis1] = off1;
                //off[axis2] = off2;
                acc.findNodeNeighborLeaf(info, l_ijk, off_ijk, n_info, n_l_ijk);
				//acc.findNodeNeighborLeafOrGhost(info, l_ijk, off_ijk, n_info, n_l_ijk);

                if (!n_info.empty()) {
                    auto h = acc.voxelSize(n_info);
                    auto weight = 1.0 / h;
                    //in order to be valid, the node must be a corner of this neighbor cell
                    //which means, if this neighbor cell is coarser, the node coords must be divisible by 2,4,8...
					//for the vector intp case, the maximum difference of levels is 2, so the worst case is that it's divisible by 2 but not by 4
                    if (n_info.mLevel < info.mLevel) {
                        int k = (1 << (info.mLevel - n_info.mLevel));
                        if (l_ijk[0] % k != 0 || l_ijk[1] % k != 0 || l_ijk[2] % k != 0) {
                            return NODATA;
                        }
                    }
                    auto& n_tile = n_info.tile();
                    //sum += n_tile.interiorValue(u_channel + axis, n_l_ijk) * weight;
					sum += n_tile(u_channel + axis, n_l_ijk) * weight;
                    weight_sum += weight;

                }
            }
        }
        return weight_sum > 0 ? sum / weight_sum : 0;
    };

    grid.launchNodeFunc(
        [=]__device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& r_ijk) {
        auto& tile = info.tile();
        for (int axis : {0, 1, 2}) {
            tile.node(node_u_channel + axis, r_ijk) = calcNode(acc, info, r_ijk, axis);
        }
    },
        -1, LEAF, LAUNCH_SUBTREE
    );

    //fix all nodes that can be divided by k and not valid
    auto fixJunctions = [&](int level, int k) {
        grid.launchNodeFunc(
            [=]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & r_ijk) {
            auto& tile = info.tile();
            for (int axis : {0, 1, 2}) {
                if (tile.node(node_u_channel + axis, r_ijk) == NODATA && r_ijk[0] % k == 0 && r_ijk[1] % k == 0 && r_ijk[2] % k == 0) {
                    Tile::T sum = 0;
                    for (int offi : {-1, 1}) {
                        int n_i = (r_ijk[0] % (2 * k) != 0) ? r_ijk[0] + offi * k : r_ijk[0];
                        for (int offj : {-1, 1}) {
                            int n_j = (r_ijk[1] % (2 * k) != 0) ? r_ijk[1] + offj * k : r_ijk[1];
                            for (int offk : {-1, 1}) {
                                int n_k = (r_ijk[2] % (2 * k) != 0) ? r_ijk[2] + offk * k : r_ijk[2];
                                Coord n_r_ijk(n_i, n_j, n_k);
                                sum += tile.node(node_u_channel + axis, n_r_ijk);
                            }
                        }
                    }

                    tile.node(node_u_channel + axis, r_ijk) = sum / 8;
                }
            }
        },
            level, LEAF, LAUNCH_SUBTREE
        );
        };

    for (int i = 0; i <= grid.mMaxLevel; i++) {
		fixJunctions(i, 2);
        fixJunctions(i, 1);
    }
}

void CalcLeafNodeValuesFromCellCenters(HADeviceGrid<Tile>& grid, const int cell_channel, const int node_channel) {
    //PropagateValues(grid, cell_channel, cell_channel, -1, GHOST, LAUNCH_SUBTREE);
    PropagateToChildren(grid, cell_channel, cell_channel, -1, GHOST, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);

    auto calcNode = [=]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile> &info, const Coord & l_ijk) -> Tile::T {
        auto g_ijk = acc.localToGlobalCoord(info, l_ijk);

        HATileInfo<Tile> n_info; Coord n_l_ijk;
        Tile::T sum = 0, weight_sum = 0;
        for (int offx : {-1, 0}) {
            for (int offy : {-1, 0}) {
                for (int offz : {-1, 0}) {
                    Coord off(offx, offy, offz);
                    acc.findNodeNeighborLeaf(info, l_ijk, off, n_info, n_l_ijk);

                    if (!n_info.empty()) {
                        auto h = acc.voxelSize(n_info);
                        auto weight = 1.0 / h;

                        if (n_info.mLevel < info.mLevel) {
                            int k = (1 << (info.mLevel - n_info.mLevel));
                            if (l_ijk[0] % k != 0 || l_ijk[1] % k != 0 || l_ijk[2] % k != 0) {
                                return NODATA;
                            }
                        }

                        auto& n_tile = n_info.tile();
                        sum += n_tile(cell_channel, n_l_ijk) * weight;
                        weight_sum += weight;
                    }

                }
            }
        }

        return weight_sum > 0 ? sum / weight_sum : 0;
    };

    grid.launchVoxelFunc(
        [=]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
        auto& tile = info.tile();
        int thread_idx = acc.localCoordToOffset(l_ijk);
        for (int idx : {thread_idx * 2, thread_idx * 2 + 1}) {
            if (0 <= idx && idx < Tile::NODESIZE) {

                Coord node_l_ijk = acc.localNodeOffsetToCoord(idx);
                tile.node(node_channel, node_l_ijk) = calcNode(acc, info, node_l_ijk);
            }
        }
    },
        -1, LEAF, LAUNCH_SUBTREE
    );

    //fix all nodes that can be divided by k and not valid
    auto fixJunctions = [&](int level, int k) {
        grid.launchNodeFunc(
            [=]__device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & r_ijk) {
            auto& tile = info.tile();
            if (tile.node(node_channel, r_ijk) == NODATA && r_ijk[0] % k == 0 && r_ijk[1] % k == 0 && r_ijk[2] % k == 0) {
                Tile::T sum = 0;
                for (int offi : {-1, 1}) {
                    int n_i = (r_ijk[0] % (2 * k) != 0) ? r_ijk[0] + offi * k : r_ijk[0];
                    for (int offj : {-1, 1}) {
                        int n_j = (r_ijk[1] % (2 * k) != 0) ? r_ijk[1] + offj * k : r_ijk[1];
                        for (int offk : {-1, 1}) {
                            int n_k = (r_ijk[2] % (2 * k) != 0) ? r_ijk[2] + offk * k : r_ijk[2];
                            Coord n_r_ijk(n_i, n_j, n_k);
                            sum += tile.node(node_channel, n_r_ijk);
                        }
                    }
                }

                tile.node(node_channel, r_ijk) = sum / 8;
            }
        },
            level, LEAF, LAUNCH_SUBTREE
        );
        };

    for (int i = 0; i <= grid.mMaxLevel; i++) {
        fixJunctions(i, 4);
        fixJunctions(i, 2);
        fixJunctions(i, 1);
    }
}


__device__ Tile::T InterpolateCellValue(const HATileAccessor<Tile>& acc, const Vec& pos, const int cell_channel, const int node_channel) {
    HATileInfo<Tile> info; Coord l_ijk; Vec frac;
    if (acc.findLeafVoxelAndFrac(pos, info, l_ijk, frac)) {
        auto b_ijk = info.mTileCoord;
        auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
        // printf("find leaf at level %d g_ijk %d %d %d l_ijk %d %d %d frac %f %f %f\n", info.mLevel, g_ijk[0], g_ijk[1], g_ijk[2], l_ijk[0], l_ijk[1], l_ijk[2], frac[0], frac[1], frac[2]);
        auto& tile = info.tile();
        return tile.cellInterp(cell_channel, node_channel, l_ijk, frac);
    }
    else return Tile::BACKGROUND_VALUE;
}

__device__ Vec InterpolateFaceValue(const HATileAccessor<Tile>& acc, const Vec& pos, const int u_channel, const int node_u_channel) {
    Vec vec;

    HATileInfo<Tile> info; Coord l_ijk; Vec frac;
    acc.findLeafVoxelAndFrac(pos, info, l_ijk, frac);
    for (int axis = 0; axis < 3; axis++) {
        //printf("=============================interpolate face value at axis=%d\n", axis);
        //printf("pos: %f %f %f\n", pos[0], pos[1], pos[2]);
        //printf("pos0*12800: %lf\n", pos[0] * 12800);

        Tile::T v0, v1, w;
        w = frac[axis];
        if (!info.empty()) {
            auto& tile = info.tile();
            v0 = tile.faceInterp(u_channel, node_u_channel, axis, l_ijk, frac);
        }
        else v0 = 0;

		auto cell_ctr = acc.cellCenter(info, l_ijk);
		auto n_pos = pos; n_pos[axis] += (1 - frac[axis]) * acc.voxelSize(info);

        HATileInfo<Tile> n_info; Coord n_l_ijk; Vec n_frac;
        acc.findPlusFaceIntpVoxel(pos, axis, info, l_ijk, n_info, n_l_ijk, n_frac);
        if (!n_info.empty()) {
            auto& n_tile = n_info.tile();

            v1 = n_tile.faceInterp(u_channel, node_u_channel, axis, n_l_ijk, n_frac);
        }
        else v1 = 0;

        //printf("calc: %lf\n", (1 - w) * 10300 + w * 10350);
        vec[axis] = (1 - w) * v0 + w * v1;
    }
    return vec;
}

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
            if (tile.type(l_ijk) & INTERIOR) {
                tile(channel, l_ijk) -= mean;
            }
        }, LEAF, 4
        );

        CheckCudaError("ReCenterLeafCells");
    }
}

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
        [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
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