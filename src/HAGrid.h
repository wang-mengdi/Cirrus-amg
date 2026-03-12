#pragma once

#include "HAAccessor.h"

#include <vector>
//to print std::vector
#include <fmt/ranges.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <cstdint>
#include <cstring>
#include <type_traits>

void CheckCudaError(const std::string& message);

template <typename T>
class DeviceReducer {
public:
	thrust::device_vector<T> d_data;         // 使用 device_vector 来存储数据
	thrust::device_vector<char> d_temp_sum;  // 使用 device_vector 来存储临时存储区
	thrust::device_vector<char> d_temp_max;  // 使用 device_vector 来存储最大值的临时存储区
	size_t n;                                // 数据长度
	size_t temp_bytes_sum;                   // 临时存储空间字节数
	size_t temp_bytes_max;                   // 最大值临时存储空间字节数

	DeviceReducer() : n(0), temp_bytes_sum(0), temp_bytes_max(0) {}
	DeviceReducer(size_t len) {
		resize(len);
	}

	T* data(void) {
		return thrust::raw_pointer_cast(d_data.data());
	}

	void fill(const T& value) {
		if (n == 0) return;
		thrust::fill(d_data.begin(), d_data.end(), value);
		CheckCudaError("DeviceReducer: thrust::fill");
	}

	void resize(size_t len) {
		n = len;

		if (n > 0) {
			d_data.resize(n);

			// Allocate temp storage for sum
			cub::DeviceReduce::Sum(nullptr, temp_bytes_sum, data(), data(), n);
			d_temp_sum.resize(temp_bytes_sum);

			// Allocate temp storage for max
			cub::DeviceReduce::Max(nullptr, temp_bytes_max, data(), data(), n);
			d_temp_max.resize(temp_bytes_max);
		}
	}

	void sumAsyncTo(T* d_result) {
		if (n > 0) {
			cub::DeviceReduce::Sum(thrust::raw_pointer_cast(d_temp_sum.data()), temp_bytes_sum, data(), d_result, n);
			CheckCudaError("DeviceReducer: cub::DeviceReduce::Sum");
		}
	}

	T sumSync(void) {
		T* d_result;
		cudaMalloc((void**)&d_result, sizeof(T));
		if (n > 0) {
			cub::DeviceReduce::Sum(thrust::raw_pointer_cast(d_temp_sum.data()), temp_bytes_sum, data(), d_result, n);
			CheckCudaError("DeviceReducer: cub::DeviceReduce::Sum");
		}
		T result;
		cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
		cudaFree(d_result);
		return result;
	}

	void maxAsyncTo(T* d_result) {
		if (n > 0) {
			cub::DeviceReduce::Max(thrust::raw_pointer_cast(d_temp_max.data()), temp_bytes_max, data(), d_result, n);
			CheckCudaError("DeviceReducer: cub::DeviceReduce::Max");
		}
	}

	T maxSync(void) {
		T* d_result;
		cudaMalloc((void**)&d_result, sizeof(T));
		if (n > 0) {
			cub::DeviceReduce::Max(thrust::raw_pointer_cast(d_temp_max.data()), temp_bytes_max, data(), d_result, n);
			CheckCudaError("DeviceReducer: cub::DeviceReduce::Max");
		}
		T result;
		cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
		cudaFree(d_result);
		return result;
	}

	void printData() const {
		if (n > 0) {
			std::vector<T> h_data(n);

			cudaMemcpy(&h_data[0], thrust::raw_pointer_cast(d_data.data()), n * sizeof(T), cudaMemcpyDeviceToHost);
			CheckCudaError("DeviceReducer: cudaMemcpy for d_data");

			fmt::print("Device data: {}\n", h_data);
		}
	}
};


template<class Tile>
class HAHostTileHolder {
public:
	using Coord = typename Tile::CoordType;
	using T = typename Tile::T;

	T mH0;
	int mNumLevels;
	int mMaxLevel;

	std::vector<Tile> mHostTiles;
	std::vector<std::vector<HATileInfo<Tile>>> mHostLevels;

	HAHostTileHolder() {}
	HAHostTileHolder(const T h0, const int num_layers, const int max_level) :
		mH0(h0), mNumLevels(num_layers), mMaxLevel(max_level)
	{
		mHostLevels.resize(num_layers);
	}

	HACoordAccessor<Tile> coordAccessor(void) const {
		return HACoordAccessor<Tile>(mH0);
	}

	//NOTE: this function iterates over all tiles in the level
	//It is slow and should be used for debugging only
	T cellValue(const int level, const Coord g_ijk, const int channel) {
		auto acc = coordAccessor();
		Coord b_ijk, l_ijk;
		acc.decomposeGlobalCoord(g_ijk, b_ijk, l_ijk);
		for (auto& info : mHostLevels[level]) {
			if (info.mTileCoord == b_ijk) {
				auto& tile = info.tile();

				if (channel == -1) return tile.type(l_ijk);
				else {
					ASSERT(0 <= channel && channel < Tile::num_channels, "Invalid channel {}", channel);
					return tile(channel, l_ijk);
				}
			}
		}
		return 0;
	}

	int numberOfLeafTiles(void)const {
		int cnt = 0;
		for (int i = 0; i <= mMaxLevel; i++) {
			for (auto& info : mHostLevels[i]) {
				if (info.isLeaf()) cnt++;
			}
		}
		return cnt;
	}

	void iterateLeafTiles(std::function<void(HATileInfo<Tile>&)> f) {
		for (int i = 0; i <= mMaxLevel; i++) {
			for (auto& info : mHostLevels[i]) {
				if (info.isLeaf()) f(info);
			}
		}
	}

	//f(info, l_ijk)
	template<class FuncBC>
	void iterateLeafCells(FuncBC f) {
		for (int level = 0; level <= mMaxLevel; level++) {
			for (auto& info : mHostLevels[level]) {
				if (info.isLeaf()) {

					for (int i = 0; i < Tile::DIM; i++) {
						for (int j = 0; j < Tile::DIM; j++) {
							for (int k = 0; k < Tile::DIM; k++) {
								f(info, Coord(i, j, k));
							}
						}
					}
				}
			}
		}
	}


	template<class FuncBC>
	void iterateLevelCells(const int level, FuncBC f) {
		for (auto& info : mHostLevels[level]) {
			for (int i = 0; i < Tile::DIM; i++) {
				for (int j = 0; j < Tile::DIM; j++) {
					for (int k = 0; k < Tile::DIM; k++) {
						f(info, Coord(i, j, k));
					}
				}
			}
		}
	}
};

//A=accessor, B=block(tile), C=coord
//FuncABC means f(acc, tile_info, ijk)

template<class Tile, class FuncABC>
__global__ void LaunchVoxelsOnAllTilesHelperKernel(HATileAccessor<Tile> acc, FuncABC f, HATileInfo<Tile>* tiles, uint8_t launch_types, int num_groups) {
	using Coord = typename Tile::Coord;
	HATileInfo<Tile> info = tiles[blockIdx.x];
	int ti = threadIdx.x;
	if (info.mType & launch_types) {
		for (int i = 0; i < num_groups; i++) {
			int vi = i * blockDim.x + ti;
			Coord l_ijk = acc.localOffsetToCoord(vi);
			f(acc, info, l_ijk);
		}

		//Coord l_ijk = Coord(threadIdx.x + i_offset, threadIdx.y, threadIdx.z);
		//f(acc, info, l_ijk);
	}
}

//A=accessor, B=block(tile), C=coord
//FuncABC means f(acc, tile_info, ijk)
template<class Tile, class FuncABC>
__global__ void LaunchVoxelsHelperKernel(HATileAccessor<Tile> acc, FuncABC f, HATileInfo<Tile>* tiles, const int subtree_level, uint8_t launch_types, int i_offset) {
	using Coord = typename Tile::Coord;
	HATileInfo<Tile> tile_info = tiles[blockIdx.x];
	Coord l_ijk = Coord(threadIdx.x + i_offset, threadIdx.y, threadIdx.z);
	//if (l_ijk == Coord(4, 5, 4)) {
	//	printf("LaunchVoxelsHelperKernel l_ijk %d %d %d\n", l_ijk[0], l_ijk[1], l_ijk[2]);
	//}
	if (tile_info.subtreeType(subtree_level) & launch_types) f(acc, tile_info, l_ijk);
}

//blocksize = Tile::SIZE
template<class Tile, class FuncABC>
__global__ void LaunchNodesHelperKernel(HATileAccessor<Tile> acc, FuncABC f, HATileInfo<Tile>* tiles, const int subtree_level, uint8_t launch_types) {
	using Coord = typename Tile::Coord;
	//auto idx = threadIdx.x;
	for (auto idx : { threadIdx.x * 2, threadIdx.x * 2 + 1 }) {
		if (idx < (Tile::DIM + 1) * (Tile::DIM + 1) * (Tile::DIM + 1)) {
			Coord r_ijk = acc.localNodeOffsetToCoord(idx);
			HATileInfo<Tile> tile_info = tiles[blockIdx.x];
			if (tile_info.subtreeType(subtree_level) & launch_types) f(acc, tile_info, r_ijk);
		}
	}
}

//blocksize = Tile::SIZE
template<class Tile, class FuncAIBC>
__global__ void LaunchNodesWithTileIdxHelperKernel(HATileAccessor<Tile> acc, FuncAIBC f, HATileInfo<Tile>* tiles, const int subtree_level, uint8_t launch_types) {
	using Coord = typename Tile::Coord;
	//auto idx = threadIdx.x;
	for (auto idx : { threadIdx.x * 2, threadIdx.x * 2 + 1 }) {
		if (idx < (Tile::DIM + 1) * (Tile::DIM + 1) * (Tile::DIM + 1)) {
			Coord r_ijk = acc.localNodeOffsetToCoord(idx);
			HATileInfo<Tile> tile_info = tiles[blockIdx.x];
			if (tile_info.subtreeType(subtree_level) & launch_types) f(acc, blockIdx.x, tile_info, r_ijk);
		}
	}
}

template<class Tile, class FuncAIB>
__global__ void LaunchTilesHelperKernel(HATileAccessor<Tile> acc, FuncAIB f, HATileInfo<Tile>* tiles, int num_tiles, const int subtree_level, uint8_t launch_types) {
	auto idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_tiles) {
		HATileInfo<Tile> tile_info = tiles[idx];
		if (tile_info.subtreeType(subtree_level) & launch_types) f(acc, idx, tile_info);
	}
}

template<class Tile, class FuncAB>
__global__ void MarkRefineFlagOnLeafsWithLevelTargetHelperKernel(HATileAccessor<Tile> acc, FuncAB level_target, HATileInfo<Tile>* tiles, int num_tiles, int* refine_flg_dev) {
	using Coord = typename Tile::Coord;
	auto idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_tiles) {
		HATileInfo<Tile> info = tiles[idx];
		int i = info.mLevel;
		if (!(info.mType & LEAF)) {
			refine_flg_dev[idx] = 0;
			return;
		}

		auto& tile = info.tile();
		//a leaf tile need to be refined if it doesn't reach level target, or there is a neighbor leaf on +2 level
		//that means, it has a ghost child, whose REFINE_FLAG is set
		//the flags of ghost tiles will be set in the next step

		bool to_refine = false;
		//case 1: if the target level is higher than the current level, refine
		if (level_target(acc, info) > info.mLevel) {
			to_refine = true;
		}

		//case 2: check ghost children
		//there are no ghost children of the max level
		if (i < acc.mMaxLevel) {
			acc.iterateChildCoords(info.mTileCoord,
				[&]__device__(const Coord & child_ijk) {
				auto& child_info = acc.tileInfo(i + 1, child_ijk);
				if (child_info.isGhost()) {
					auto& child_tile = child_info.tile();
					if (child_tile.mStatus & REFINE_FLAG) {
						to_refine = true;
					}
				}
			});
		}

		//we will not refine if the maximum capacity of levels is reached
		if (i + 1 >= acc.mNumLevels) to_refine = false;
		tile.setMask(REFINE_FLAG, to_refine);
		refine_flg_dev[idx] = to_refine;
	}
}

//calculate DELETE flags for LEAF tiles, and unset COARSEN flags for convenience
template<class Tile, class FuncAB>
__global__ void MarkCoarsenAndDeleteFlagOnLeafsWithLevelTargetHelperKernel(HATileAccessor<Tile> acc, FuncAB level_target, HATileInfo<Tile>* tiles, int num_tiles) {
	using Coord = typename Tile::Coord;
	auto idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_tiles) {
		HATileInfo<Tile> info = tiles[idx];
		int i = info.mLevel;
		if (!(info.mType & LEAF)) {
			return;
		}

		auto& tile = info.tile();

		bool to_delete = level_target(acc, info) < info.mLevel;

		//if there is a same-level NONLEAF neighbor that cannot be coarsen, cannot delete
		acc.iterateNeighborCoords(info.mTileCoord, [&](const Coord& nb_ijk) {//nb for neighbor block
			auto ninfo = acc.tileInfoCopy(info.mLevel, nb_ijk);
			if (ninfo.mType & NONLEAF) {
				auto& ntile = ninfo.tile();
				if (!(ntile.mStatus & COARSEN_FLAG)) {
					to_delete = false;
				}
			}
			});

		tile.setMask(COARSEN_FLAG, false);
		tile.setMask(DELETE_FLAG, to_delete);
	}
}

template<class Tile>
class HADeviceGrid {
	static constexpr DataHolder AllocSide = DEVICE;
public:
	using Coord = typename Tile::Coord;
	using T = typename Tile::T;

	cudaStream_t mStream;
	T mH0;
	uint32_t mNumLevels = 0;
	int mMaxLevel;

	bool mCompressedFlag = true;

	//all tiles are allocated on the device
	//but tile info may be sotred on host or device
	//technically, the metadata can reside on HOST/DEVICE
	//and data (tiles) can reside on HOST/DEVICE
	//the combination of DEVICE-HOST is meaningless
	//here we have HOST-DEVICE and DEVICE-DEVICE combinations
	//note that we DO NOT have HOST-HOST combination
	//therefore, all modifications to the grid are done with allocating and freeing DEVICE memory


	thrust::host_vector<uint32_t> hNumTiles;
	thrust::host_vector<uint32_t> hLog2Hashes;
	std::vector<thrust::host_vector<HATileInfo<Tile>>> hHashTables;
	std::vector<thrust::host_vector<HATileInfo<Tile>>> hTileArrays;
	//pointers to host data, helping building accessors
	//these pointers are calculated when initializing the grid
	std::vector<HATileInfo<Tile>* > hHashTablePtrs;

	bool mDeviceSyncFlag;//if device layers are synchronized with host layers


	//device data structure
	thrust::device_vector<uint32_t> dNumTiles;
	thrust::device_vector<uint32_t> dLog2Hashes;
	std::vector<thrust::device_vector<HATileInfo<Tile>>> dHashTables;
	std::vector<thrust::device_vector<HATileInfo<Tile>>> dTileArrays;
	//pointers to device data, helping building accessors
	//these pointers are calculated when synchronizing the device
	thrust::device_vector<HATileInfo<Tile>*> dHashTablePtrs;


	//auxillary
	//thrust::device_vector<HATileInfo<Tile>> dLeafTiles;
	//thrust::device_vector<HATileInfo<Tile>> dGhostTiles;
	thrust::host_vector<HATileInfo<Tile>> hAllTiles;
	thrust::device_vector<HATileInfo<Tile>> dAllTiles;
	DeviceReducer<double> dAllTilesReducer;// for dot etc

	HADeviceGrid() :
		mH0(0),
		mNumLevels(0),
		mMaxLevel(-1),
		mCompressedFlag(true)
	{
		cudaStreamCreate(&mStream);
	}
	HADeviceGrid(const T h0, const thrust::host_vector<uint32_t>& levels_log2_hash) :
		mH0(h0),
		mMaxLevel(-1),
		mCompressedFlag(true),//it's empty
		mDeviceSyncFlag(true)//empty
	{
		cudaStreamCreate(&mStream);

		mNumLevels = levels_log2_hash.size();

		hNumTiles.resize(mNumLevels);
		hLog2Hashes.resize(mNumLevels);
		hHashTables.resize(mNumLevels);
		hTileArrays.resize(mNumLevels);
		hHashTablePtrs.resize(mNumLevels);
		//hTileArrayPtrs.resize(mNumLevels);

		//hLog2Hashes = std::vector<uint32_t>(levels_log2_hash);
		hLog2Hashes = levels_log2_hash;

		for (int i = 0; i < mNumLevels; i++) {
			hNumTiles[i] = 0;
			hHashTables[i].resize(1u << hLog2Hashes[i]);
			hTileArrays[i].resize(1u << hLog2Hashes[i]);

			hHashTablePtrs[i] = thrust::raw_pointer_cast(hHashTables[i].data());
			//hTileArrayPtrs[i] = thrust::raw_pointer_cast(hTileArrays[i].data());
		}
	}
	HADeviceGrid(const T h0, const std::initializer_list<uint32_t>& levels_log2_hash) :
		HADeviceGrid(h0, thrust::host_vector<uint32_t>(levels_log2_hash))
	{
	}

	HADeviceGrid(const HADeviceGrid<Tile>&) = delete;
	HADeviceGrid& operator=(const HADeviceGrid<Tile>&) = delete;

	~HADeviceGrid() {
		ASSERT(mCompressedFlag, "Grid must be compressed before destruction");
		for (int i = 0; i < mNumLevels; i++) {
			for (int j = 0; j < hNumTiles[i]; j++) {
				auto info = hTileArrays[i][j];
				if (!info.empty()) {
					if (AllocSide == DEVICE) cudaFree(info.mTilePtr);
					else std::free(info.mTilePtr);
				}
			}
		}
		cudaStreamDestroy(mStream);
	}

	void setTilesFromHolder(const HAHostTileHolder<Tile>& holder) {
		ASSERT(mCompressedFlag, "setTilesFromHolder requires compressed grid");
		ASSERT(holder.mNumLevels == mNumLevels, "setTilesFromHolder: num layers mismatch");
		mH0 = holder.mH0;
		mNumLevels = holder.mNumLevels;
		mMaxLevel = holder.mMaxLevel;
		mCompressedFlag = false;
		for (int i = 0; i < mNumLevels; i++) {
			hNumTiles[i] = holder.mHostLevels[i].size();
			for (int j = 0; j < hNumTiles[i]; j++) {
				auto& info = holder.mHostLevels[i][j];
				auto& tile = info.tile();
				auto type = info.mType;
				auto coord = info.mTileCoord;
				setTileHost(i, coord, tile, type);
			}
		}
	}

	std::shared_ptr<HADeviceGrid<Tile>> deepCopy(void){
		ASSERT(mCompressedFlag, "deepCopy requires compressed grid");
		auto ptr = std::make_shared<HADeviceGrid<Tile>>(mH0, hLog2Hashes);
		auto& grid1 = *ptr;
		for (int i = 0; i <= mMaxLevel; i++) {
			for (int j = 0; j < hNumTiles[i]; j++) {
				auto info = hTileArrays[i][j];
				grid1.setTileHost(i, info.mTileCoord, info.getTile(DEVICE), info.mType);
				//grid1.setTileHost(i, info.mTileCoord, Tile(), info.mType);
				//Info("copyStructure: level={} coord={} type={}", i, info.mTileCoord, info.mType);
			}
		}
		grid1.compressHost();
		grid1.syncHostAndDevice();
		grid1.spawnGhostTiles();
		return ptr;
	}

	int numTotalTiles(void) const {
		ASSERT(mCompressedFlag, "numTotalTiles requires compressed grid");
		int num = 0;
		for (int i = 0; i < mNumLevels; i++) {
			num += hNumTiles[i];
		}
		return num;
	}
	int numTotalLeafTiles(void) const {
		ASSERT(mCompressedFlag, "numTotalLeafs requires compressed grid");
		int num = 0;
		for (int i = 0; i < mNumLevels; i++) {
			for (int j = 0; j < hNumTiles[i]; j++) {
				if (hTileArrays[i][j].isLeaf()) num++;
			}
		}
		return num;
	}

	//b_ijk for block coord, which is the beginning voxel coord / 8
	//this operation only changes the host data structure
	//need to sync
	HATileInfo<Tile> setTileHost(const int level, const Coord& b_ijk, const Tile& data, uint8_t type) {
		auto h_acc = hostAccessor();
		auto& info = h_acc.tileInfo(level, b_ijk);

		mCompressedFlag = false;
		mDeviceSyncFlag = false;

		if (info.empty()) {
			//create a new tile
			Tile* ptr = nullptr;
			if (AllocSide == DEVICE) cudaMalloc((void**)&ptr, sizeof(Tile));
			else ptr = (Tile*)std::malloc(sizeof(Tile));
			info = HATileInfo<Tile>(type, level, ptr, b_ijk);
		}

		//modify the existing tile
		if (AllocSide == DEVICE) cudaMemcpy(info.mTilePtr, &data, sizeof(Tile), cudaMemcpyHostToDevice);
		else std::memcpy(info.mTilePtr, &data, sizeof(Tile));
		Tile* ptr = info.mTilePtr;
		info = HATileInfo<Tile>(type, level, ptr, b_ijk);

		return info;
	}

	void removeTileHost(const int level, const Coord& b_ijk) {
		auto& h_acc = hostAccessor();
		auto i = h_acc.tileIdx(level, b_ijk);
		if (hHashTables[level][i].empty()) return;
		auto j = i;
		uint32_t HASH_MASK = (1u << hLog2Hashes[level]) - 1u;
		while (true) {
			j = (j + 1) & HASH_MASK;
			if (hHashTables[level][j].empty()) break;
			auto k = h_acc.hash(hHashTables[level][j].mTileCoord, hLog2Hashes[level]);
			if ((j > i && (k <= i || k > j)) || (j < i && (k <= i && k > j))) {
				//mHostHashTable[i] = mHostHashTable[j];
				std::swap(hHashTables[level][i], hHashTables[level][j]);//we need to deconstruct j later
				i = j;
			}
		}
		if (AllocSide == DEVICE) cudaFree(hHashTables[level][i].mTilePtr);
		else std::free(hHashTables[level][i].mTilePtr);
		hHashTables[level][i] = HATileInfo<Tile>();//set to empty
		//mHostHashTable[i].mTilePtr = nullptr;
		mCompressedFlag = false;
		mDeviceSyncFlag = false;
	}

	std::shared_ptr<HAHostTileHolder<Tile>> getHostTileHolder(const uint8_t tile_types, int max_level = -1) {
		if (max_level == -1) max_level = mMaxLevel;
		auto holder = std::make_shared<HAHostTileHolder<Tile>>(mH0, mNumLevels, mMaxLevel);
		holder->mHostTiles.clear();
		int num_leafs = 0;
		for (int i = 0; i <= max_level; i++) {
			holder->mHostLevels[i].clear();
			int level_leafs = 0;
			for (int j = 0; j < hNumTiles[i]; j++) {
				HATileInfo<Tile> info = hTileArrays[i][j];
				if (info.mType & tile_types) {
					level_leafs++;
					num_leafs++;
				}
			}
			holder->mHostLevels[i].reserve(level_leafs);
		}
		holder->mHostTiles.reserve(num_leafs);
		for (int i = 0; i <= max_level; i++) {
			for (int j = 0; j < hNumTiles[i]; j++) {
				HATileInfo<Tile> info = hTileArrays[i][j];
				if (info.mType & tile_types) {
					holder->mHostTiles.push_back(info.getTile(DEVICE));
					info.mTilePtr = &(holder->mHostTiles.back());
					holder->mHostLevels[i].push_back(info);

					//Warn("getHostTileHolder take tile {} with type {} at level {} required types {}", info.mTileCoord, info.mType, i, tile_types);
				}
			}
		}
		return holder;
	}

	std::shared_ptr<HAHostTileHolder<Tile>> getHostTileHolderForLeafs(void) {
		return getHostTileHolder(LEAF);
	}

	int maxProbeLength(void) {
		auto h_acc = hostAccessor();
		int max_len = 0;
		for (int i = 0; i < mNumLevels; i++) {
			for (int j = 0; j < hNumTiles[i]; j++) {
				auto info = hTileArrays[i][j];
				auto hash_idx = h_acc.hash(info.mTileCoord, hLog2Hashes[i]);
				auto tile_idx = h_acc.tileIdx(i, info.mTileCoord);
				int len = tile_idx - hash_idx;
				max_len = std::max(max_len, len);
			}
		}
		return max_len;
	}

	int hashTableDeviceBytes(void) {
		int total_bytes = 0;
		for (int i = 0; i < mNumLevels; i++) {
			total_bytes += dHashTables[i].size() * sizeof(HATileInfo<Tile>);
		}
		return total_bytes;
	}

	void compressHost(bool verbose = true) {
		std::vector<int> load_rate;
		mMaxLevel = -1;
		for (int i = 0; i < mNumLevels; i++) {
			hNumTiles[i] = thrust::copy_if(
				hHashTables[i].begin(), hHashTables[i].end(),
				hTileArrays[i].begin(),
				[&](const HATileInfo<Tile>& info) ->bool {return info.mTilePtr != nullptr; }//pred
			) - hTileArrays[i].begin();
			uint32_t HASH_SIZE = (1u << hLog2Hashes[i]);
			load_rate.push_back((hNumTiles[i] + 0.0) / HASH_SIZE * 100);
			if (hNumTiles[i] != 0) mMaxLevel = i;
		}
		mCompressedFlag = true;
		if (verbose) {
			Info("Grid compressed {} layers, with each {} tiles and load rate {}/%", mNumLevels, hNumTiles, load_rate);
			Info("Max probe length: {}", maxProbeLength());
		}
	}
	void syncHostAndDevice(void) {
		ASSERT(mCompressedFlag, "syncHostAndDevice requires compressed grid");

		dNumTiles = hNumTiles;
		dLog2Hashes = hLog2Hashes;
		dHashTables.resize(mNumLevels);
		dTileArrays.resize(mNumLevels);
		for (int i = 0; i < mNumLevels; i++) {
			dHashTables[i] = hHashTables[i];
			dTileArrays[i] = hTileArrays[i];
		}

		thrust::host_vector<HATileInfo<Tile>*> host_layer_hash_table_ptrs;
		for (int i = 0; i < mNumLevels; i++) {
			host_layer_hash_table_ptrs.push_back(thrust::raw_pointer_cast(dHashTables[i].data()));
		}
		dHashTablePtrs = host_layer_hash_table_ptrs;


		hAllTiles.clear();
		for (int layer = 0; layer < mNumLevels; layer++) {
			for (int j = 0; j < hNumTiles[layer]; j++) {
				const HATileInfo<Tile>& info = hTileArrays[layer][j];
				hAllTiles.push_back(info);
			}
		}

		dAllTiles = hAllTiles;

		dAllTilesReducer.resize(dAllTiles.size());

		mDeviceSyncFlag = true;
	}

	std::vector<uint8_t> dumpBinaryBlob(uint8_t tile_types = (LEAF | GHOST | NONLEAF),
		int max_level = -1) const
	{
		ASSERT(mCompressedFlag, "dumpBinaryBlob requires compressed grid (call compressHost first).");
		if (max_level < 0) max_level = mMaxLevel;

		using CoordT = Coord;

		static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
		static_assert(std::is_trivially_copyable_v<CoordT>, "Coord must be trivially copyable");
		static_assert(std::is_trivially_copyable_v<Tile>, "Tile must be trivially copyable for raw dump");

		struct Header {
			uint32_t magic;      // 'HAGR'
			uint32_t version;    // 1
			uint32_t num_levels; // mNumLevels
			int32_t  max_level;  // stored max level
			uint32_t tile_size;  // sizeof(Tile)
			uint32_t coord_size; // sizeof(Coord)
			T        h0;         // mH0
			uint64_t tile_count; // #records
		};

		// Count tiles
		uint64_t tile_count = 0;
		for (int lv = 0; lv <= max_level; ++lv) {
			for (uint32_t j = 0; j < hNumTiles[lv]; ++j) {
				const auto& info = hTileArrays[lv][j];
				if (info.mType & tile_types) tile_count++;
			}
		}

		Header hdr{};
		hdr.magic = 0x52474148u; // 'HAGR' little-endian
		hdr.version = 1;
		hdr.num_levels = mNumLevels;
		hdr.max_level = max_level;
		hdr.tile_size = (uint32_t)sizeof(Tile);
		hdr.coord_size = (uint32_t)sizeof(CoordT);
		hdr.h0 = mH0;
		hdr.tile_count = tile_count;

		const size_t bytes_header = sizeof(Header);
		const size_t bytes_hashes = sizeof(uint32_t) * hdr.num_levels;
		const size_t bytes_record =
			sizeof(uint32_t) +  // level
			sizeof(uint8_t) +  // type
			sizeof(CoordT) +  // tileCoord
			sizeof(Tile);       // tile bytes

		const size_t total = bytes_header + bytes_hashes + (size_t)tile_count * bytes_record;

		std::vector<uint8_t> blob(total);
		uint8_t* p = blob.data();

		auto write_raw = [&](const void* src, size_t n) {
			std::memcpy(p, src, n);
			p += n;
			};

		write_raw(&hdr, sizeof(Header));

		// write hLog2Hashes
		for (uint32_t lv = 0; lv < hdr.num_levels; ++lv) {
			uint32_t v = hLog2Hashes[lv];
			write_raw(&v, sizeof(uint32_t));
		}

		// write tile records (from compressed tile arrays)
		for (int lv = 0; lv <= max_level; ++lv) {
			for (uint32_t j = 0; j < hNumTiles[lv]; ++j) {
				const auto& info = hTileArrays[lv][j];
				if (!(info.mType & tile_types)) continue;

				uint32_t level_u32 = (uint32_t)info.mLevel;
				uint8_t  type_u8 = info.mType;
				CoordT   coord = info.mTileCoord;

				// Copy device tile bytes to host
				Tile tile_host = info.getTile(DEVICE);

				write_raw(&level_u32, sizeof(uint32_t));
				write_raw(&type_u8, sizeof(uint8_t));
				write_raw(&coord, sizeof(CoordT));
				write_raw(&tile_host, sizeof(Tile));
			}
		}

		ASSERT((size_t)(p - blob.data()) == total, "dumpBinaryBlob size mismatch");
		return blob;
	}

	static std::shared_ptr<HADeviceGrid<Tile>> loadBinaryBlob(const uint8_t* data, size_t size)
	{
		using CoordT = Coord;

		static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
		static_assert(std::is_trivially_copyable_v<CoordT>, "Coord must be trivially copyable");
		static_assert(std::is_trivially_copyable_v<Tile>, "Tile must be trivially copyable for raw load");

		struct Header {
			uint32_t magic;
			uint32_t version;
			uint32_t num_levels;
			int32_t  max_level;
			uint32_t tile_size;
			uint32_t coord_size;
			T        h0;
			uint64_t tile_count;
		};

		auto require = [&](bool cond, const char* msg) {
			ASSERT(cond, "loadBinaryBlob: {}", msg);
			};

		require(size >= sizeof(Header), "buffer too small");
		const uint8_t* p = data;

		auto read_raw = [&](void* dst, size_t n) {
			require((size_t)(p - data) + n <= size, "buffer truncated");
			std::memcpy(dst, p, n);
			p += n;
			};

		Header hdr{};
		read_raw(&hdr, sizeof(Header));

		require(hdr.magic == 0x52474148u, "bad magic");
		require(hdr.version == 1, "unsupported version");
		require(hdr.tile_size == sizeof(Tile), "Tile size mismatch");
		require(hdr.coord_size == sizeof(CoordT), "Coord size mismatch");
		require(hdr.num_levels > 0, "num_levels invalid");

		thrust::host_vector<uint32_t> log2_hashes;
		log2_hashes.resize(hdr.num_levels);
		for (uint32_t lv = 0; lv < hdr.num_levels; ++lv) {
			uint32_t v = 0;
			read_raw(&v, sizeof(uint32_t));
			log2_hashes[lv] = v;
		}

		auto grid_ptr = std::make_shared<HADeviceGrid<Tile>>(hdr.h0, log2_hashes);

		for (uint64_t t = 0; t < hdr.tile_count; ++t) {
			uint32_t level_u32 = 0;
			uint8_t  type_u8 = 0;
			CoordT   coord{};
			Tile     tile_host{};

			read_raw(&level_u32, sizeof(uint32_t));
			read_raw(&type_u8, sizeof(uint8_t));
			read_raw(&coord, sizeof(CoordT));
			read_raw(&tile_host, sizeof(Tile));

			grid_ptr->setTileHost((int)level_u32, coord, tile_host, type_u8);
		}

		// rebuild compressed arrays and sync device structures
		grid_ptr->compressHost(false);
		grid_ptr->syncHostAndDevice();

		return grid_ptr;
	}

	static std::shared_ptr<HADeviceGrid<Tile>> loadBinaryBlob(const std::vector<uint8_t>& blob) {
		return loadBinaryBlob(blob.data(), blob.size());
	}


	void spawnGhostTiles(bool verbose = false, std::vector<HATileInfo<Tile>>* additive_new_tile_infos = nullptr) {
		using Acc = HACoordAccessor<Tile>;
		//note that we will NOT spawn ghost tiles on level 0
		//because it's the root level, and they will not have parents
		ASSERT(mCompressedFlag, "Grid must be compressed and synced before spawn ghost tiles");
		auto h_acc = hostAccessor();
		for (int level = mMaxLevel; level > 0; level--) {
			for (int i = 0; i < hNumTiles[level]; i++) {
				auto& info = hTileArrays[level][i];
				if (info.isActive()) {
					Coord b_ijk = info.mTileCoord;
					Acc::iterateNeighborCoords(b_ijk, [&](const Coord& n_ijk) {
						auto& n_info = h_acc.tileInfo(level, n_ijk);
						//auto& n_info = grid.mHostLayers[level].tileInfo(n_ijk);
						if (n_info.empty()) {
							Coord p_ijk = Acc::parentCoord(n_ijk);
							auto& p_info = h_acc.tileInfo(level - 1, p_ijk);
							//auto& p_info = grid.mHostLayers[level - 1].tileInfo(p_ijk);
							if (p_info.isLeaf()) {
								auto child_info = setTileHost(level, n_ijk, Tile(), GHOST);
								if (additive_new_tile_infos) {
									additive_new_tile_infos->push_back(child_info);
								}
								//grid.mHostLayers[level].setTile(n_ijk, Tile(), level, GHOST);
							}
						}
						}
					);
				}
			}
		}
		compressHost(verbose);
		syncHostAndDevice();
	}

	//should be called after each topology change
	void rebuild(bool verbose = false) {
		compressHost(verbose);
		//syncHostAndDevice();
		spawnGhostTiles(verbose);

	}

	template<class FuncABC>
	void launchVoxelFunc(FuncABC f, int level, const uint8_t launch_types, LaunchMode mode = LAUNCH_LEVEL, const LaunchOrder order = COARSE_FIRST, const int num_groups = 1) {
		//launch a voxel function f(Accessor, TileInfo, l_ijk) on specified tiles

		//launch_types is a bit mask containing all tile types we want to launch
		//if mode is LAUNCH_LEVEL, we will authentically launch specified types
		//otherwise, all non-leafs on that level will be considered as leafs when launching

		//if mode is set to LAUNCH_LEVEL, then only the specified level will be launched
		//otherwise, we will launch on levels [0,level], which is a sub-tree with specified finest level
		//and all tiles at this finest level are regarded as leafs
		//if level is set to -1, LAUNCH_SUBTREE is forced, and all levels will be launched

		//if order is set to COARSE_FIRST, then we will launch from coarse to fine
		//otherwise, we will launch from fine to coarse

		//if num_groups=1, all 512 voxels are launched in the same block
		//if num_groups=2, 256 voxels are launched in the same block
		//...4,8

		ASSERT(num_groups == 1 || num_groups == 2 || num_groups == 4 || num_groups == 8, "launchVoxelFunc invalid num groups: {}", num_groups);

		if (level == -1) {
			if (mode == LAUNCH_LEVEL) {
				Info("launchVoxelFunc level=-1, overwrite mode=LAUNCH_SUBTREE");
			}
			mode = LAUNCH_SUBTREE;
			level = mNumLevels - 1;
		}
		int start, end, step;
		if (mode == LAUNCH_LEVEL) start = level, end = level + 1, step = 1;
		else {
			if (order == COARSE_FIRST) start = 0, end = level + 1, step = 1;
			else start = level, end = -1, step = -1;
		}

		int subtree_level = (mode == LAUNCH_LEVEL) ? -1 : level;
		for (int i = start; i != end; i += step) {
			for (int gi = 0; gi < num_groups; gi++) {
				int offset = gi * Tile::DIM / num_groups;
				if (hNumTiles[i] == 0) continue;
				LaunchVoxelsHelperKernel << <hNumTiles[i], dim3(Tile::DIM / num_groups, Tile::DIM, Tile::DIM) >> > (
					deviceAccessor(),
					f,
					thrust::raw_pointer_cast(dTileArrays[i].data()),
					subtree_level,
					launch_types,
					offset
					);

				CheckCudaError(fmt::format("launchVoxelFunc level={} gi={} num tiles {} num_groups={}", i, gi, hNumTiles[i], num_groups));
			}
		}
	}

	template<class FuncABC>
	void launchVoxelFuncOnTiles(FuncABC f, thrust::device_vector<HATileInfo<Tile>> &tiles, const int num_launched_tiles, const uint8_t launch_types, const int num_groups = 1) {
		//for (int gi = 0; gi < num_groups; gi++) {
			//int offset = gi * Tile::DIM / num_groups;
			if (num_launched_tiles == 0) return;
			LaunchVoxelsOnAllTilesHelperKernel << <num_launched_tiles, Tile::SIZE / num_groups >> > (
				deviceAccessor(),
				f,
				thrust::raw_pointer_cast(tiles.data()),
				launch_types,
				num_groups
				);
		//}
	}

	template<class FuncABC>
	void launchVoxelFuncOnAllTiles(FuncABC f, const uint8_t launch_types, const int num_groups = 1) {
		launchVoxelFuncOnTiles(f, dAllTiles, dAllTiles.size(), launch_types, num_groups);
	}

	//template<class FuncABC>
	//void launchNodeFuncOnTiles(FuncABC f, thrust::device_vector<HATileInfo<Tile>>& tiles, const int num_launched_tiles, const uint8_t launch_types) {
	//	if (num_launched_tiles == 0) return;
	//	LaunchNodesHelperKernel << <num_launched_tiles, Tile::SIZE >> > (
	//		deviceAccessor(),
	//		f,
	//		thrust::raw_pointer_cast(tiles.data()),
	//		-1,
	//		launch_types
	//		);
	//}

	template<class FuncAIBC>
	void launchNodeFuncWithTileIdxOnAllTiles(FuncAIBC f, const uint8_t launch_types) {
		LaunchNodesWithTileIdxHelperKernel << <dAllTiles.size(), Tile::SIZE >> > (deviceAccessor(), f, thrust::raw_pointer_cast(dAllTiles.data()), -1, launch_types);
		//launchNodeFuncOnTiles(f, dAllTiles, dAllTiles.size(), launch_types);
	}

	template<class FuncABC>
	void launchNodeFunc(FuncABC f, int level, const uint8_t launch_types, LaunchMode mode = LAUNCH_LEVEL, const LaunchOrder order = COARSE_FIRST) {
		//same as launchVoxelFunc but we apply functions on nodes instead of voxels(cells)

		if (level == -1) {
			if (mode == LAUNCH_LEVEL) {
				Info("launchVoxelFunc level=-1, overwrite mode=LAUNCH_SUBTREE");
			}
			mode = LAUNCH_SUBTREE;
			level = mNumLevels - 1;
		}
		int start, end, step;
		if (mode == LAUNCH_LEVEL) start = level, end = level + 1, step = 1;
		else {
			if (order == COARSE_FIRST) start = 0, end = level + 1, step = 1;
			else start = level, end = -1, step = -1;
		}

		int subtree_level = (mode == LAUNCH_LEVEL) ? -1 : level;
		for (int i = start; i != end; i += step) {
			if (hNumTiles[i] == 0) continue;
			LaunchNodesHelperKernel << <hNumTiles[i], Tile::SIZE >> > (
				deviceAccessor(),
				f,
				thrust::raw_pointer_cast(dTileArrays[i].data()),
				subtree_level,
				launch_types
				);
		}
	}

	template<class FuncAIB>
	void launchTileFunc(FuncAIB f, int level, const uint8_t launch_types, LaunchMode mode = LAUNCH_LEVEL, const LaunchOrder order = COARSE_FIRST, int blockSize = 512) {
		//launch a voxel function f(Accessor, TileInfo, l_ijk) on specified tiles

		//launch_types is a bit mask containing all tile types we want to launch

		//if mode is set to LAUNCH_LEVEL, then only the specified level will be launched
		//otherwise, we will launch on levels [0,level], which is a sub-tree with specified finest level
		//and all tiles at this finest level are regarded as leafs
		//if level is set to -1, LAUNCH_SUBTREE is forced, and all levels will be launched

		//if order is set to COARSE_FIRST, then we will launch from coarse to fine
		//otherwise, we will launch from fine to coarse

		if (level == -1) {
			mode = LAUNCH_SUBTREE;
			level = mNumLevels - 1;
		}
		int start, end, step;
		if (mode == LAUNCH_LEVEL) start = level, end = level + 1, step = 1;
		else {
			if (order == COARSE_FIRST) start = 0, end = level + 1, step = 1;
			else start = level, end = -1, step = -1;
		}

		int subtree_level = (mode == LAUNCH_LEVEL) ? -1 : level;
		for (int i = start; i != end; i += step) {
			if (hNumTiles[i] == 0) continue;
			LaunchTilesHelperKernel << <(hNumTiles[i] + blockSize - 1) / blockSize, blockSize >> > (
				deviceAccessor(),
				f,
				thrust::raw_pointer_cast(dTileArrays[i].data()),
				hNumTiles[i],
				subtree_level,
				launch_types
				);
		}
	}

	HATileAccessor<Tile> hostAccessor(void) {
		return HATileAccessor<Tile>(
			mH0,
			mNumLevels,
			mMaxLevel,
			thrust::raw_pointer_cast(hNumTiles.data()),
			thrust::raw_pointer_cast(hLog2Hashes.data()),
			thrust::raw_pointer_cast(hHashTablePtrs.data())//,
			//thrust::raw_pointer_cast(hTileArrayPtrs.data())
		);
	}

	//this can't be a const member function because that requires the pointers to be const
	HATileAccessor<Tile> deviceAccessor(void) {
		return HATileAccessor<Tile>(
			mH0,
			mNumLevels,
			mMaxLevel,
			thrust::raw_pointer_cast(dNumTiles.data()),
			thrust::raw_pointer_cast(dLog2Hashes.data()),
			thrust::raw_pointer_cast(dHashTablePtrs.data())//,
			//thrust::raw_pointer_cast(dTileArrayPtrs.data())
		);
	}

	template<class ABFunc>
	std::vector<int> refineStep(const ABFunc& level_target, bool verbose, std::vector<HATileInfo<Tile>>* additive_new_tile_infos = nullptr) {
		//must be called after ghost tiles are properly spawned
		//then, refine leafs for one step
		//this may need to be called multiple times to reach the target level
		ASSERT(mDeviceSyncFlag, "Grid must be synced before refine step");

		constexpr int blockSize = 512;
		std::vector<int> level_refine_cnts(mNumLevels, 0);

		for (int i = mMaxLevel; i >= 0; i--) {
			thrust::host_vector<int> refine_host(hNumTiles[i]);
			thrust::device_vector<int> refine_flg_dev = refine_host;
			ASSERT(refine_flg_dev.size() == hNumTiles[i], "refine_flg_dev size {} mismatch num tiles {}", refine_flg_dev.size(), hNumTiles[i]);

			if (hNumTiles[i] == 0) continue;

			auto refine_flg_dev_ptr = thrust::raw_pointer_cast(refine_flg_dev.data());
			
			//mark refine flags on leafs
			MarkRefineFlagOnLeafsWithLevelTargetHelperKernel << <(hNumTiles[i] + blockSize - 1) / blockSize, blockSize >> > (
				deviceAccessor(),
				level_target,
				thrust::raw_pointer_cast(dTileArrays[i].data()),
				hNumTiles[i],
				refine_flg_dev_ptr
				);
			refine_host = refine_flg_dev;

			//mark refine flags on ghosts, they will be passed to parents
			launchTileFunc(
				[=]__device__(HATileAccessor<Tile>&acc, const uint32_t tile_idx, HATileInfo<Tile>&info) {
				auto& tile = info.tile();
				bool refine_flag = false;
				acc.iterateNeighborCoords(info.mTileCoord,
					[&](const Coord& n_ijk) {
						auto& n_info = acc.tileInfo(i, n_ijk);
						if (n_info.isLeaf()) {
							auto& n_tile = n_info.tile();
							if (n_tile.mStatus & REFINE_FLAG) {
								refine_flag = true;
							}
						}
					});
				tile.setMask(REFINE_FLAG, refine_flag);
			},
				i, GHOST, LAUNCH_LEVEL
			);

			using Acc = HACoordAccessor<Tile>;
			//we will refine all required tiles to the next level
			//thrust::host_vector<int> refine_host = refine_flg_dev;
			auto h_acc = hostAccessor();
			for (int j = 0; j < refine_host.size(); j++) {
				if (refine_host[j]) {
					level_refine_cnts[i]++;

					//this seems strange, but the compress() function will write mHostTileArray based on hash table
					//so we must modify the hash table element
					auto& tmp_info = hTileArrays[i][j];
					auto& info = h_acc.tileInfo(i, tmp_info.mTileCoord);

					info.mType = NONLEAF;
					auto tile = info.getTile(DEVICE);

					for (int ci = 0; ci < Acc::NUMCHILDREN; ci++) {
						Coord offset = Acc::childIndexToOffset(ci);
						Coord c_ijk = Acc::childCoord(info.mTileCoord, offset);

						auto child_info = setTileHost(i + 1, c_ijk, tile.childTile(offset), LEAF);
						if (additive_new_tile_infos) {
							additive_new_tile_infos->push_back(child_info);
						}
					}
				}
			}

		}

		compressHost(verbose);
		syncHostAndDevice();

		return level_refine_cnts;
	}

	template<class ABFunc>
	void iterativeRefine(ABFunc level_target, bool verbose = true) {
		while (true) {
			//auto refine_cnts = RefineLeafsOneStep(grid, level_target, verbose);
			auto refine_cnts = refineStep(level_target, verbose);
			spawnGhostTiles(verbose);
			if (verbose) Info("Refine {} tiles on each layer", refine_cnts);
			auto cnt = std::accumulate(refine_cnts.begin(), refine_cnts.end(), 0);
			if (cnt == 0) break;
		}
	}

	//return number of deleted tiles each layer
	template<class ABFunc>
	std::vector<int> coarsenStep(ABFunc level_target, bool verbose) {
		//level_target function is only available on leaf tiles
		//COARSEN_FLAG means you can delete its children and make the tile a LEAF
		//DELETE_FLAG means you can delete the tile itself

		constexpr int blockSize = 512;

		//upstroke from fine to coarse, calculate COARSEN flags based on DELETE flags
		for (int level = mNumLevels - 1; level >= 0; level--) {
			//mark DELETE flag for leaf tiles and COARSEN flag for non-leaf tiles
			//we're not launching GHOST here
			//if there is a same-level neighbor NONLEAF tile that can't be coarsen, we can't delete a LEAF
			//therefore, COARSEN flags are calculated prior to DELETE

			//1: calculate COARSEN flags for NONLEAF tiles and unset DELETE flags for convenience
			//if all 8 children are deletable LEAF tiles, we can mark COARSEN flag
			launchTileFunc(
				[=] __device__(HATileAccessor<Tile>&acc, uint32_t tile_idx, HATileInfo<Tile>&info) {
				auto& tile = info.tile();
				bool to_coarsen = true;
				acc.iterateChildCoords(info.mTileCoord, [&](const Coord& c_ijk) {
					auto cinfo = acc.tileInfoCopy(info.mLevel + 1, c_ijk);//c for child
					if (cinfo.isLeaf()) {
						const auto& ctile = cinfo.tile();
						if (!(ctile.mStatus & DELETE_FLAG)) {
							to_coarsen = false;
						}
					}
					else {
						//we're only launching NONLEAF tiles so the child here must be NONLEAF
						//and its DELETE flag is false
						to_coarsen = false;
					}
					});
				tile.setMask(COARSEN_FLAG, to_coarsen);
				tile.setMask(DELETE_FLAG, false);
			},
				level, NONLEAF, LAUNCH_LEVEL
			);

			//2: calculate DELETE flags for LEAF tiles, and unset COARSEN flags for convenience
			if (hNumTiles[level] == 0) continue;
			MarkCoarsenAndDeleteFlagOnLeafsWithLevelTargetHelperKernel << <hNumTiles[level] / blockSize + 1, blockSize >> > (
				deviceAccessor(),
				level_target,
				thrust::raw_pointer_cast(dTileArrays[level].data()),
				hNumTiles[level]
				);
		}

		//downstroke from coarse to fine, propagate DELETE flags from COARSEN flags
		for (int level = 0; level < mNumLevels; level++) {
			//3: propagate DELETE flags for LEAF and GHOST based on COASREN flags
			//in the upstroke pass, COARSEN flag is calculated on all NONLEAF tiles and unset on all LEAF tiles
			//if a tile is marked as COARSEN, we can delete its children and mark it as LEAF
			//if DELETE flag is set for a tile, we also set the COARSEN to further delete its children
			launchTileFunc(
				[=] __device__(HATileAccessor<Tile>&acc, uint32_t tile_idx, HATileInfo<Tile>&info) {
				auto& tile = info.tile();
				bool to_delete = false;

				auto pb_ijk = acc.parentCoord(info.mTileCoord);
				auto pinfo = acc.tileInfoCopy(info.mLevel - 1, pb_ijk);
				if (!pinfo.empty()) {
					auto& ptile = pinfo.tile();
					if (ptile.mStatus & COARSEN_FLAG) {
						to_delete = true;
					}
				}

				tile.setMask(DELETE_FLAG, to_delete);
				if (to_delete) {
					tile.setMask(COARSEN_FLAG, true);
				}
			},
				level, LEAF | GHOST, LAUNCH_LEVEL
			);

			//4: additionally update DELETE flags for GHOST tiles
			//if a GHOST tile does not have a neighbor LEAF tile that will continue to exist, we mark it as DELETE
			//(if the GHOST tile's parent is going to be deleted, the delete flag is already propagated to it)
			launchTileFunc(
				[=] __device__(HATileAccessor<Tile>&acc, uint32_t tile_idx, HATileInfo<Tile>&info) {
				auto& tile = info.tile();
				bool to_delete = true;
				acc.iterateNeighborCoords(info.mTileCoord, [&](const Coord& nb_ijk) {
					auto ninfo = acc.tileInfoCopy(info.mLevel, nb_ijk);
					if (ninfo.mType & LEAF) {
						auto& ntile = ninfo.tile();
						if (!(ntile.mStatus & DELETE_FLAG)) {
							to_delete = false;
						}
					}
					});
				if (to_delete) tile.setMask(DELETE_FLAG, true);
			},
				level, GHOST, LAUNCH_LEVEL
			);
		}

		//delete tiles with DELETE flag and mark existing COARSEN tiles as leaf
		std::vector<int> deleted_tiles(mNumLevels, 0);
		for (int level = 0; level < mNumLevels; level++) {
			thrust::host_vector<int> stat_h(hNumTiles[level]);
			thrust::device_vector<int> stat_d = stat_h;
			auto stat_d_ptr = thrust::raw_pointer_cast(stat_d.data());
			launchTileFunc(
				[=] __device__(HATileAccessor<Tile>&acc, uint32_t tile_idx, HATileInfo<Tile>&info) {
				auto& tile = info.tile();
				stat_d_ptr[tile_idx] = tile.mStatus;
			},
				level, LEAF | NONLEAF | GHOST, LAUNCH_LEVEL
			);
			stat_h = stat_d;

			int level_deleted = 0;
			auto h_acc = hostAccessor();
			for (int i = 0; i < stat_h.size(); i++) {
				auto b_ijk = hTileArrays[level][i].mTileCoord;
				if (stat_h[i] & DELETE_FLAG) {
					removeTileHost(level, b_ijk);
					level_deleted++;
				}
				else if (stat_h[i] & COARSEN_FLAG) {
					//note that compressHost will overwrite tile array with hash table
					//so we need to modify the info in the hash table
					auto& info = h_acc.tileInfo(level, b_ijk);
					info.mType = LEAF;
				}
			}
			deleted_tiles[level] = level_deleted;
		}

		compressHost(verbose);
		syncHostAndDevice();
		return deleted_tiles;
	}



};


