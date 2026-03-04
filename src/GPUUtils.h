#pragma once
#include <cuda_runtime.h>
#include <NanoVDB.h>


#if defined(__CUDACC__) || defined(__HIP__)
// Only define __hostdev__ when using NVIDIA CUDA or HIP compiler
#define __hostdev__ __host__ __device__
#else
#define __hostdev__
#endif

//execuate in range [0, N)
template <typename Func>
__global__ void ForEachKernel(Func f, const int N, const int numGroups) {
	int base = blockIdx.x * (blockDim.x * numGroups);

	for (int i = 0; i < numGroups; i++) {
		int idx = base + i * blockDim.x + threadIdx.x;
		if (idx < N) {
			f(idx);
		}
	}

	//int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//if (idx < N) {
	//	f(idx);
	//}
}

template<typename Func>
void LaunchIndexFunc(Func f, const int N, const int blockSize = 512, const int numGroups = 4) {
	if (N == 0) return;
	Assert(blockSize % numGroups == 0);
	int numBlocks = (N + blockSize - 1) / blockSize;
	ForEachKernel << <numBlocks, blockSize / numGroups >> > (f, N, numGroups);
}

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
