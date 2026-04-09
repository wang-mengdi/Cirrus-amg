#pragma once

#include "PoissonGrid.h"
#include <cub/cub.cuh>
#include <cub/block/block_reduce.cuh>

struct CubMin { template<typename T> __device__ T operator()(const T& a, const T& b) const { return a < b ? a : b; } };
struct CubMax { template<typename T> __device__ T operator()(const T& a, const T& b) const { return a > b ? a : b; } };

template<class FuncTT>
__global__ void MarkRegionOfInterestWithChannelMinAndMax128Kernel(HATileAccessor<Tile> acc, HATileInfo<Tile>* infos, int subtree_level, uint8_t launch_types, const int data_channel, FuncTT func_interested, bool calc_locked) {
	int bi = blockIdx.x;
	int ti = threadIdx.x;

	const auto& info = infos[bi];

	if (!(info.subtreeType(subtree_level) & launch_types)) {
		if (ti == 0) {
			auto& tile = info.tile();
			tile.mIsInterestArea = false;
			if (calc_locked) tile.mIsLockedRefine = false;
		}
		return;
	}

	auto& tile = info.tile();
	auto dataAsFloat4 = reinterpret_cast<float4*>(tile.mData[data_channel]);
	float4 value = dataAsFloat4[ti];

	T thread_min = min(min(value.x, value.y), min(value.z, value.w));
	T thread_max = max(max(value.x, value.y), max(value.z, value.w));

	typedef cub::BlockReduce<T, 128> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage_min;
	__shared__ typename BlockReduce::TempStorage temp_storage_max;

	T block_min = BlockReduce(temp_storage_min).Reduce(thread_min, CubMin());
	T block_max = BlockReduce(temp_storage_max).Reduce(thread_max, CubMax());

	if (ti == 0) {
		if (func_interested(block_min, block_max)) {
			tile.mIsInterestArea = true;
			if (calc_locked) {
				tile.mIsLockedRefine = true;
			}
		}
		else {
			tile.mIsInterestArea = false;
			if (calc_locked) {
				tile.mIsLockedRefine = false;
			}
		}
	}
}
