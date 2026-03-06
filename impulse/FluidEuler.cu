#include "FluidEuler.h"
#include "Random.h"

#include <cub/cub.cuh>
#include <cub/block/block_reduce.cuh>
#include <thrust/execution_policy.h>
#include <tbb/parallel_for.h>


#include <cmath>
#include <limits>
#include <memory>
#include <cstdint>
#include <unordered_set>

void SanityCheckCoeffs(HADeviceGrid<Tile>& grid, uint8_t launch_types) {
	Info("Performing sanity check on coeffs...");
	grid.launchVoxelFuncOnAllTiles(
		[=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
		auto& tile = info.tile();
		auto val = tile(ProjChnls::c0 + 3, l_ijk);
		auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
		CUDA_ASSERT(isfinite(val), "SanityCheckCoeffs failed at non-finite value %f at level %d channel %d cell %d %d %d type %d tile type %d", val, info.mLevel, ProjChnls::c0 + 3, g_ijk[0], g_ijk[1], g_ijk[2], tile.type(l_ijk), info.mType);
		if (info.mType != GHOST && tile.type(l_ijk) & INTERIOR) {
			CUDA_ASSERT(val > 0, "SanityCheckCoeffs failed at non-positive value %f at level %d channel %d cell %d %d %d type %d tile type %d", val, info.mLevel, ProjChnls::c0 + 3, g_ijk[0], g_ijk[1], g_ijk[2], tile.type(l_ijk), info.mType);
		}
	}, launch_types
	);
}

struct CoordHash {
	size_t operator()(const nanovdb::Coord& c) const noexcept {
		size_t h = std::hash<int>()(c[0]);
		h ^= std::hash<int>()(c[1]) + 0x9e3779b9 + (h << 6) + (h >> 2);
		h ^= std::hash<int>()(c[2]) + 0x9e3779b9 + (h << 6) + (h >> 2);
		return h;
	}
};

void SanityCheckTiles(HADeviceGrid<Tile>& grid) {
	//check if there are any repeat tiles in the grid
	for (int level = 0; level < grid.mNumLevels; level++) {
		std::unordered_set<Coord, CoordHash> unique_coords;
		for (int i = 0; i < grid.hNumTiles[level]; i++) {
			const auto& info = grid.hTileArrays[level][i];
			ASSERT(info.mTilePtr != nullptr, "Tile {} pointer is null at level {} tile {}", info.mTileCoord, level, i);
			//the tile coordinate should be unique in the level
			bool insert_success = unique_coords.insert(info.mTileCoord).second;
			if (!insert_success) exit(1);
			ASSERT(insert_success, "Duplicate tile coordinate {} at level {} tile {}", info.mTileCoord, level, i);
			
			
			//if (level == 5) Info("level {} tile {} coord {} type {} pointer {}", level, i, info.mTileCoord, info.mType, (void*)info.mTilePtr);
		}
	}
}

double CellPointRMSNormOnHostTiles(
	const std::shared_ptr<HAHostTileHolder<Tile>>& holder_ptr,
	int channel,                 // scalar channel index
	int level,                   // -1 for all levels, otherwise specific level
	uint8_t tile_types,          // e.g., LEAF | GHOST | NONLEAF
	int norm_type                // 1 (L1 mean), 2 (RMS), -1 (Linf)
) {
	const auto& holder = *holder_ptr;
	using Coord = typename Tile::CoordType;

	double acc = 0.0; // for L1/L2 accumulation
	double mx = 0.0;  // for Linf
	size_t count = 0;

	auto process_level = [&](int lv) {
		for (const auto& info : holder.mHostLevels[lv]) {
			if (!(info.mType & tile_types)) continue;

			auto& tile = info.tile();

			for (int i = 0; i < Tile::DIM; ++i) {
				for (int j = 0; j < Tile::DIM; ++j) {
					for (int k = 0; k < Tile::DIM; ++k) {
						Coord l_ijk(i, j, k);

						double x = tile(channel, l_ijk);
						if (!std::isfinite(x)) continue;

						double ax = std::abs(x);

						if (norm_type == -1) {
							if (ax > mx) mx = ax;
						}
						else if (norm_type == 1) {
							acc += ax;
							count++;
						}
						else if (norm_type == 2) {
							acc += x * x;
							count++;
						}
					}
				}
			}
		}
		};

	int beg = 0, end = holder.mMaxLevel;
	if (level != -1) beg = end = level;

	for (int lv = beg; lv <= end; ++lv) process_level(lv);

	if (norm_type == -1) return mx;

	if (count == 0) return 0.0;

	if (norm_type == 1) return acc / count;
	if (norm_type == 2) return std::sqrt(acc / count);

	return std::numeric_limits<double>::quiet_NaN();
}

double NodePointRMSNormOnHostTiles(
	const std::shared_ptr<HAHostTileHolder<Tile>>& holder_ptr,
	int channel,                 // scalar node channel
	int level,                   // -1 for all levels
	uint8_t tile_types,          // LEAF | GHOST | NONLEAF
	int norm_type                // 1 (L1 mean), 2 (RMS), -1 (Linf)
) {
	const auto& holder = *holder_ptr;
	using Coord = typename Tile::CoordType;

	double acc = 0.0;
	double mx = 0.0;
	size_t count = 0;

	auto process_level = [&](int lv) {
		for (const auto& info : holder.mHostLevels[lv]) {

			if (!(info.mType & tile_types)) continue;

			auto& tile = info.tile();

			for (int i = 0; i <= Tile::DIM; ++i) {
				for (int j = 0; j <= Tile::DIM; ++j) {
					for (int k = 0; k <= Tile::DIM; ++k) {

						Coord r_ijk(i, j, k);

						double x = tile.node(channel, r_ijk);
						if (!std::isfinite(x)) continue;

						double ax = std::abs(x);

						if (norm_type == -1) {
							if (ax > mx) mx = ax;
						}
						else if (norm_type == 1) {
							acc += ax;
							count++;
						}
						else if (norm_type == 2) {
							acc += x * x;
							count++;
						}
					}
				}
			}
		}
		};

	int beg = 0, end = holder.mMaxLevel;
	if (level != -1) beg = end = level;

	for (int lv = beg; lv <= end; ++lv)
		process_level(lv);

	if (norm_type == -1) return mx;

	if (count == 0) return 0.0;

	if (norm_type == 1) return acc / count;
	if (norm_type == 2) return std::sqrt(acc / count);

	return std::numeric_limits<double>::quiet_NaN();
}

__device__ Vec NFMErodedAdvectionPoint(const int axis, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const Coord& l_ijk) {
	auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
	auto ng_ijk = g_ijk; ng_ijk[axis]--;
	HATileInfo<Tile> ninfo; Coord nl_ijk;
	acc.findVoxel(info.mLevel, ng_ijk, ninfo, nl_ijk);

	int effect_level = info.mLevel;
	Coord effect_g_ijk = g_ijk;

	if (!ninfo.empty() && (ninfo.mType & GHOST)) {
		effect_level = info.mLevel - 1;
		effect_g_ijk = acc.parentCoord(g_ijk);
	}
	return acc.faceCenterGlobal(axis, effect_level, effect_g_ijk);
}





void ClearAllNeumannNeighborFaces(HADeviceGrid<Tile>& grid, const int u_channel)
{
	grid.launchVoxelFunc(
		[=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
		for (int axis : {0, 1, 2}) {
			bool to_set = false;
			IterateFaceNeighborCellTypes(acc, info, l_ijk, axis, [&](const uint8_t type0, const uint8_t type1) {
				if ((type0 & NEUMANN) || (type1 & NEUMANN) || ((type0 & DIRICHLET) && (type1 & DIRICHLET))) {
					to_set = true;
				}
				});
			if (to_set) {
				info.tile()(u_channel + axis, l_ijk) = 0;
			}
		}
	}, -1, LEAF, LAUNCH_SUBTREE
	);
}

void MarkOldParticlesAsInvalid(thrust::device_vector<Particle>& particles, const T current_time, const T particle_life) {
	auto particles_ptr = thrust::raw_pointer_cast(particles.data());
	LaunchIndexFunc([=] __device__(int idx) {
		auto& p = particles_ptr[idx];

		if (current_time - p.start_time > particle_life) {
			p.pos = Vec(NODATA, NODATA, NODATA);
		}

	}, particles.size(), 128);
}

// Sample N marker particles outside the mesh.
// For each candidate:
// 1. Randomly pick one mesh vertex.
// 2. Sample a point uniformly inside a world-space ball of radius r centered at that vertex.
// 3. Reject the point if it is inside the mesh (sdf < 0).
// 4. Accept the point if it is outside or exactly on the surface (sdf >= 0).
thrust::device_vector<MarkerParticle> SampleMarkerParticlesOutsideMesh(const MeshSDFAccel& mesh_sdf, const Eigen::Transform<T, 3, Eigen::Affine>& mesh_to_world, int N,	T r, T birth_time, RandomGenerator& rng, int batch_size)
{
	ASSERT(N >= 0);
	ASSERT(r > T(0));
	ASSERT(batch_size > 0);
	ASSERT(mesh_sdf.V_.rows() > 0);

	if (N == 0) {
		return thrust::device_vector<MarkerParticle>();
	}

	std::vector<MarkerParticle> h_particles;
	h_particles.reserve(static_cast<size_t>(N));

	std::vector<Vec> candidates;
	candidates.reserve(static_cast<size_t>(batch_size));

	while (static_cast<int>(h_particles.size()) < N) {
		const int remaining = N - static_cast<int>(h_particles.size());
		const int curr_batch_size = std::min(batch_size, remaining);

		candidates.clear();
		candidates.resize(static_cast<size_t>(curr_batch_size));

		// Generate candidate points in world space
		for (int k = 0; k < curr_batch_size; ++k) {
			// Randomly pick one vertex in mesh-local space
			const int vid = rng.rand(0, static_cast<int>(mesh_sdf.V_.rows()) - 1);

			Eigen::Matrix<T, 3, 1> p_mesh(
				mesh_sdf.V_(vid, 0),
				mesh_sdf.V_(vid, 1),
				mesh_sdf.V_(vid, 2));

			// Transform the vertex to world space
			const Eigen::Matrix<T, 3, 1> center_world = mesh_to_world * p_mesh;

			// Sample a random direction by rejection from the unit ball
			Eigen::Matrix<T, 3, 1> dir;
			while (true) {
				dir << T(rng.uniform(-1.0, 1.0)),
					T(rng.uniform(-1.0, 1.0)),
					T(rng.uniform(-1.0, 1.0));

				const T norm2 = dir.squaredNorm();
				if (norm2 > T(1e-12) && norm2 <= T(1)) {
					dir /= std::sqrt(norm2);
					break;
				}
			}

			// Sample radius with volume-uniform distribution
			const T u = T(rng.uniform(0.0, 1.0));
			const T rho = r * std::cbrt(u);

			// Final candidate point in world space
			const Eigen::Matrix<T, 3, 1> sample_world = center_world + rho * dir;
			candidates[static_cast<size_t>(k)] = Vec(sample_world(0), sample_world(1), sample_world(2));
		}

		// Query SDF for the whole batch
		const std::vector<T> sdf = mesh_sdf.querySDF(candidates, mesh_to_world);
		ASSERT(static_cast<int>(sdf.size()) == curr_batch_size);

		// Keep points outside the mesh or exactly on the surface
		for (int k = 0; k < curr_batch_size; ++k) {
			if (sdf[static_cast<size_t>(k)] < T(0)) {
				continue;
			}

			MarkerParticle p;
			p.pos = candidates[static_cast<size_t>(k)];
			p.birth_time = birth_time;
			h_particles.push_back(p);
		}
	}

	ASSERT(static_cast<int>(h_particles.size()) >= N);
	h_particles.resize(static_cast<size_t>(N));

	Info("Sampled {} marker particles outside the mesh", N);

	return thrust::device_vector<MarkerParticle>(h_particles.begin(), h_particles.end());
}