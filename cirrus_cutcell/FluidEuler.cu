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

size_t SmartResizeParticlesForInsert(thrust::device_vector<Particle>& particles, const size_t insert_num) {
	size_t old_capacity = particles.capacity();
	size_t new_capacity = particles.size() + insert_num;

	size_t diff;
	if (new_capacity > old_capacity) {
		diff = new_capacity - old_capacity;
		size_t alloc_capacity = old_capacity + diff * 2;
		particles.reserve(alloc_capacity);

		//Info("alloc capacity: {} particles capacity: {}", alloc_capacity, particles.capacity());
	}
	else {
		diff = 0;
	}

	return diff;
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

__global__ void CalculateReseedingNumbersOnLeafTiles128Kernel(const T current_time, HATileAccessor<Tile> acc, HATileInfo<Tile>* tile_infos, FluidParams params, const int tmp_channel, const int node_sdf_channel, const int num_particles_per_cell, int* reseed_number_per_cell) {
	int bidx = blockIdx.x;
	auto& info = tile_infos[bidx];
	auto& tile = info.tile();

	if (!(info.mType & LEAF)) return;

	auto sample_threshold_sdf = params.mRelativeParticleSampleBandwidth * acc.voxelSize(params.mFineLevel);

	//int reseed_threshold = num_particles_per_cell / 2;
	int reseed_threshold = 1;

	for (int i = threadIdx.x; i < Tile::SIZE; i += blockDim.x) {
		auto l_ijk = acc.localOffsetToCoord(i);
		int reseed_num = 0;

		if (tile.type(l_ijk) == INTERIOR) {
			//calculate the average value of its 8 nodes
			T node_sdf_sum = 0;
			for (int offi : {0, 1}) {
				for (int offj : {0, 1}) {
					for (int offk : {0, 1}) {
						auto r_ijk = l_ijk + Coord(offi, offj, offk);
						auto node_sdf = tile.node(node_sdf_channel, r_ijk);
						node_sdf_sum += node_sdf;
					}
				}
			}
			T node_sdf_avg = node_sdf_sum / 8.0;
			if (node_sdf_avg <= sample_threshold_sdf) {
				int num_particles = tile(tmp_channel, l_ijk);
				if (num_particles < reseed_threshold) {
					reseed_num = num_particles_per_cell - num_particles;
					//int total_idx = bidx * Tile::SIZE + i;
				}
			}
		}
		reseed_number_per_cell[bidx * Tile::SIZE + i] = reseed_num;
	}
}

void ReseedParticles(HADeviceGrid<Tile>& grid, const FluidParams& params, const int tmp_channel, const int node_sdf_channel, const double current_time, const int num_particles_per_cell, thrust::device_vector<Particle>& particles) {
	//CPUTimer timer; timer.start();
	static uint64_t global_particle_counter = 0;

	static RandomGenerator rng;
	CountParticleNumberInLeafCells(grid, particles, tmp_channel);

	//cudaDeviceSynchronize(); timer.stop("count particles"); timer.start();

	//int reseed_threshold = num_particles_per_cell / 2;
	//int reseed_threshold = 

	static thrust::device_vector<int> cell_particle_num_d;
	if (cell_particle_num_d.size() < grid.dAllTiles.size() * Tile::SIZE) {
		cell_particle_num_d.resize(grid.dAllTiles.size() * Tile::SIZE);
	}
	//Assert(cell_particle_num_d.size() == grid.dAllTiles.size() * Tile::SIZE,"");

	thrust::fill(cell_particle_num_d.begin(), cell_particle_num_d.end(), 0);
	auto cell_particle_num_d_ptr = thrust::raw_pointer_cast(cell_particle_num_d.data());

	auto info_ptr = thrust::raw_pointer_cast(grid.dAllTiles.data());
	auto acc_d = grid.deviceAccessor();
	CalculateReseedingNumbersOnLeafTiles128Kernel << <grid.dAllTiles.size(), 128 >> > (current_time, acc_d, info_ptr, params, tmp_channel, node_sdf_channel, num_particles_per_cell, cell_particle_num_d_ptr);

	//cudaDeviceSynchronize(); timer.stop("calc reseeding number of each voxel"); timer.start();

	static thrust::host_vector<int> cell_particle_num_h;
	cell_particle_num_h = cell_particle_num_d;
	//Info("cell_particle_num_h: {}", cell_particle_num_h);

	auto acc = grid.hostAccessor();
	thrust::host_vector<Particle> reseed_particles_h;

	for (int idx = 0; idx < cell_particle_num_h.size(); idx++) {
		if (cell_particle_num_h[idx] > 0) {
			int tile_idx = idx / Tile::SIZE;
			int l_idx = idx % Tile::SIZE;
			auto& info = grid.hAllTiles[tile_idx];
			auto l_ijk = acc.localOffsetToCoord(l_idx);

			//sample random particles
			auto bbox = acc.cellBBox(info, l_ijk);
			auto minPoint = bbox.min();
			auto maxPoint = bbox.max();

			for (int i = 0; i < cell_particle_num_h[idx]; i++) {
				auto x = rng.uniform(minPoint[0], maxPoint[0]);
				auto y = rng.uniform(minPoint[1], maxPoint[1]);
				auto z = rng.uniform(minPoint[2], maxPoint[2]);

				Particle p;
				p.global_idx = global_particle_counter++;
				p.pos = Vec(x, y, z);
				p.impulse = Vec(0., 0., 0.);
				p.matT() = Eigen::Matrix3<T>::Identity();
				p.start_time = current_time;
				reseed_particles_h.push_back(p);
			}
		}
	}

	//cudaDeviceSynchronize(); timer.stop(fmt::format("generating host reseeding list {}", reseed_particles_h.size())); timer.start();

	int diff = SmartResizeParticlesForInsert(particles, reseed_particles_h.size());

	//cudaDeviceSynchronize(); timer.stop(fmt::format("resize particles {} more", diff)); timer.start();

	particles.insert(particles.end(), reseed_particles_h.begin(), reseed_particles_h.end());

	//cudaDeviceSynchronize(); timer.stop("insert particles to device"); timer.start();

	// auto holder_ptr = grid.getHostTileHolderForLeafs();


	// thrust::host_vector<Particle> reseed_particles_h;

	// auto acc = grid.hostAccessor();
	// holder_ptr->iterateLeafCells([&](const HATileInfo<Tile>& info, const Coord& l_ijk) {
	// 	auto& tile = info.tile();
	// 	Vec pos = acc.cellCenter(info, l_ijk);
	// 	if (tile.type(l_ijk) == INTERIOR && params.isInParticleGenerationRegion(acc, info, l_ijk)) {

	// 		int num_particles = tile(tmp_channel, l_ijk);
	// 		if (num_particles < reseed_threshold) {
	// 			//sample random particles
	// 			auto bbox = acc.voxelBBox(info, l_ijk);
	// 			auto minPoint = bbox.min();
	// 			auto maxPoint = bbox.max();

	// 			for (int i = 0; i < num_particles_per_cell - num_particles; i++) {
	// 				auto x = rng.uniform(minPoint[0], maxPoint[0]);
	// 				auto y = rng.uniform(minPoint[1], maxPoint[1]);
	// 				auto z = rng.uniform(minPoint[2], maxPoint[2]);

	// 				Particle p;
	// 				p.pos = Vec(x, y, z);
	// 				p.impulse = Vec(0., 0., 0.);
	// 				p.matT = Eigen::Matrix3<T>::Identity();
	// 				p.start_time = current_time;
	// 				reseed_particles_h.push_back(p);
	// 			}
	// 		}
	// 	}
	// 	});

	//particles.insert(particles.end(), reseed_particles_h.begin(), reseed_particles_h.end());
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

__device__ Vec SemiLagrangianBackwardPosition(const HATileAccessor<Tile>& acc, const int fine_level, const int coarse_level, const Vec& pos, const T dt, const int u_channel) {
	Vec v0;
	auto success1 = KernelIntpVelocityMAC2(acc, fine_level, coarse_level, pos, u_channel, v0);
	auto pos1 = pos - 0.5 * dt * v0;
	Vec v1;
	auto success2 = KernelIntpVelocityMAC2(acc, fine_level, coarse_level, pos1, u_channel, v1);
	auto pos2 = pos - dt * v1;
	return pos2;
}

thrust::device_vector<MarkerParticle> SampleMarkerParticlesOutsideMeshBand(const MeshSDFAccel& mesh_sdf, const Eigen::Transform<T, 3, Eigen::Affine>& mesh_to_world, T dx, T relative_bandwidth, T samples_per_tile, T birth_time, RandomGenerator& rng)
{
	ASSERT(dx > T(0));
	ASSERT(relative_bandwidth >= T(0));
	ASSERT(samples_per_tile >= T(0));
	ASSERT(mesh_sdf.V_.rows() > 0);
	ASSERT(mesh_sdf.F_.rows() > 0);
	ASSERT(mesh_sdf.face_area_.rows() == mesh_sdf.F_.rows());
	ASSERT(mesh_sdf.FN_.rows() == mesh_sdf.F_.rows());
	ASSERT(mesh_sdf.total_area_ >= T(0));
	AssertRigidTransform(mesh_to_world);

	if (mesh_sdf.total_area_ <= T(0) || samples_per_tile <= T(0)) {
		ASSERT(false, "Mesh has non-positive total area or samples_per_tile is non-positive, cannot sample marker particles.");
		return thrust::device_vector<MarkerParticle>();
	}

	const T sample_bandwidth = relative_bandwidth * dx;
	const T tile_width = T(8) * dx;
	const T tile_area = tile_width * tile_width;
	const T samples_per_area = samples_per_tile / tile_area;

	// Expected total number of samples.
	const T expected_total = mesh_sdf.total_area_ * samples_per_area;
	int target_N = static_cast<int>(std::ceil(expected_total));

	if (target_N <= 0) {
		ASSERT(false, "Expected total number of marker particles is non-positive, cannot sample marker particles. Check if the mesh has positive area and samples_per_tile is positive.");
		return thrust::device_vector<MarkerParticle>();
	}

	std::vector<Vec> candidates;
	candidates.reserve(static_cast<size_t>(target_N));

	// Precompute normal transform for world-space offset.
	// This handles general affine transforms correctly.
	const Eigen::Matrix<T, 3, 3> normal_xform = mesh_to_world.linear().inverse().transpose();

	// Generate candidates face by face.
	for (int fid = 0; fid < mesh_sdf.F_.rows(); ++fid) {
		const T area = mesh_sdf.face_area_(fid);
		if (area <= T(0)) continue;

		const T expected_count = samples_per_area * area;
		int count = static_cast<int>(std::floor(expected_count));
		const T frac = expected_count - T(count);

		if (T(rng.uniform(0.0, 1.0)) < frac) {
			++count;
		}

		if (count <= 0) continue;

		const int i0 = mesh_sdf.F_(fid, 0);
		const int i1 = mesh_sdf.F_(fid, 1);
		const int i2 = mesh_sdf.F_(fid, 2);

		const Eigen::Matrix<T, 3, 1> a = mesh_sdf.V_.row(i0).transpose();
		const Eigen::Matrix<T, 3, 1> b = mesh_sdf.V_.row(i1).transpose();
		const Eigen::Matrix<T, 3, 1> c = mesh_sdf.V_.row(i2).transpose();

		Eigen::Matrix<T, 3, 1> n_local = mesh_sdf.FN_.row(fid).transpose();
		Eigen::Matrix<T, 3, 1> n_world = normal_xform * n_local;
		n_world = n_world.normalized();

		for (int s = 0; s < count; ++s) {
			// Uniform sample on triangle.
			const T u = T(rng.uniform(0.0, 1.0));
			const T v = T(rng.uniform(0.0, 1.0));
			const T su = std::sqrt(u);

			const T w0 = T(1) - su;
			const T w1 = su * (T(1) - v);
			const T w2 = su * v;

			const Eigen::Matrix<T, 3, 1> p_local = w0 * a + w1 * b + w2 * c;
			const Eigen::Matrix<T, 3, 1> p_world_surface = mesh_to_world * p_local;

			T offset = T(0);
			if (sample_bandwidth > T(0)) {
				offset = sample_bandwidth * T(rng.uniform(0.0, 1.0));
			}

			const Eigen::Matrix<T, 3, 1> p_world = p_world_surface + offset * n_world;
			candidates.emplace_back(p_world(0), p_world(1), p_world(2));
		}
	}

	if (candidates.empty()) {
		return thrust::device_vector<MarkerParticle>();
	}

	const std::vector<T> sdf = mesh_sdf.querySDF(candidates, mesh_to_world);
	ASSERT(sdf.size() == candidates.size());

	std::vector<MarkerParticle> h_particles;
	h_particles.reserve(candidates.size());

	for (size_t i = 0; i < candidates.size(); ++i) {
		if (sdf[i] < T(0)) continue;

		MarkerParticle p;
		p.pos = candidates[i];
		p.birth_time = birth_time;
		h_particles.push_back(p);
	}

	Info("Sampled {} marker particles outside the mesh within bandwidth {}, total {} faces and area {} avg {} particles per face", h_particles.size(), sample_bandwidth, mesh_sdf.F_.rows(), mesh_sdf.total_area_, (h_particles.size() + 0.) / mesh_sdf.F_.rows());

	return thrust::device_vector<MarkerParticle>(h_particles.begin(), h_particles.end());
}

void FluidEuler::mixFluidAndSolidVelocityOnFaces(HADeviceGrid<Tile>& grid, const double current_time, const double dt, const int coeff_channel, const int u_fluid_channel, const int u_mix_channel) {
	auto params = mParams;
	grid.launchVoxelFuncOnAllTiles(
		[=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
		auto& tile = info.tile();
		auto h = acc.voxelSize(info);
		for (int axis : {0, 1, 2}) {
			T fluid_ratio = -tile(coeff_channel + axis, l_ijk) / h;
			T solid_ratio = 1 - fluid_ratio;
			T mix_vel = 0;
			if (fluid_ratio > 0) {
				mix_vel += fluid_ratio * tile(u_fluid_channel + axis, l_ijk);
			}
			if (solid_ratio > 0) {
				mix_vel += solid_ratio * params.solidFaceCenterVelocity(current_time, dt, acc, info, l_ijk, axis);
			}

			tile(u_mix_channel + axis, l_ijk) = mix_vel;
		}
	}, LEAF
	);
}

void FluidEuler::extrapolateFluidVelocityForAdvection(HADeviceGrid<Tile>& grid, const int iteration_times, const int u_fluid_channel, const int coeff_channel)
{
	//at the beginning we have correct velocities at LEAF tiles
	 
	//fill GHOST tiles with NaN to expose bugs
	FillChannelsInGridWithValue(grid, std::numeric_limits<T>::quiet_NaN(), GHOST, { u_fluid_channel, u_fluid_channel + 1, u_fluid_channel + 2 });


	//mark faces that has no fluid contribution as invalid
	grid.launchVoxelFuncOnAllTiles(
		[=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
		auto& tile = info.tile();
		auto h = acc.voxelSize(info);
		for (int axis : {0, 1, 2}) {
			T fluid_ratio = -tile(coeff_channel + axis, l_ijk) / h;
			if (fluid_ratio <= 0) {
				tile(u_fluid_channel + axis, l_ijk) = NODATA;
			}
		}
	}, LEAF | NONLEAF | GHOST
	);

	//iterate from finest level to corasest level
	for (int level = grid.mMaxLevel; level >= 0; level--) {
		//iteratively extrapolate
		for (int i = 0; i < iteration_times; i++) {
			//for each invalid face node on LEAF/NONLEAF tiles, check its 6 same-level neighbors
			grid.launchVoxelFunc(
				[=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
				auto& tile = info.tile();
				for (int axis : {0, 1, 2}) {
					if (tile(u_fluid_channel + axis, l_ijk) == NODATA) {
						T nb_valid_sum = 0; int nb_valid_cnt = 0;
						acc.iterateSameLevelNeighborVoxels(info, l_ijk,
							[&]__device__(const HATileInfo<Tile>&ninfo, const Coord & nl_ijk, const int axis, const int sgn) {
							if (!ninfo.empty()) {
								if (ninfo.mType & (LEAF | NONLEAF)) {
									auto ntile = ninfo.tile();
									if (ntile(u_fluid_channel + axis, nl_ijk) != NODATA) {
										nb_valid_sum += ntile(u_fluid_channel + axis, nl_ijk);
										nb_valid_cnt++;
									}
								}
							}
						});
						if (nb_valid_cnt > 1) {
							tile(u_fluid_channel + axis, l_ijk) = nb_valid_sum / nb_valid_cnt;
						}
					}
				}
			}, level, LEAF | NONLEAF, LAUNCH_LEVEL
			);
		}

		int num_fine_tiles = grid.hNumTiles[level];
		if (num_fine_tiles == 0) continue;
		AverageFaceVelocitiesToParentsOneStepKernel << <num_fine_tiles, 128 >> > (
			grid.deviceAccessor(),
			thrust::raw_pointer_cast(grid.dTileArrays[level].data()),
			-1, LEAF | NONLEAF, u_fluid_channel, INTERIOR | DIRICHLET | NEUMANN
			);
	}

	//set all NODATA velocities to 0 to avoid issues in advection stencils
		//mark faces that has no fluid contribution as invalid
	grid.launchVoxelFuncOnAllTiles(
		[=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
		auto& tile = info.tile();
		auto h = acc.voxelSize(info);
		for (int axis : {0, 1, 2}) {
			if (tile(u_fluid_channel + axis, l_ijk) == NODATA) {
				tile(u_fluid_channel + axis, l_ijk) = 0;
			}
		}
	}, LEAF | NONLEAF | GHOST
	);

}

void FluidEuler::iterativeNodeSDFAndRefineNarrowBand(HADeviceGrid<Tile>& grid, const T current_time, const T solid_relative_bandwidth, const T fluid_relative_bandwidth)
{
	auto params = mParams;
	auto xform = params.meshToWorldTransform(current_time);
	auto h_acc = grid.hostAccessor();
	auto truncation = h_acc.voxelSize(params.mFineLevel) * (max(solid_relative_bandwidth, fluid_relative_bandwidth) + 1);//should be slightly larger than dx*bandwidth, because the refinment predicate uses > and <
	ASSERT(max(solid_relative_bandwidth, fluid_relative_bandwidth) > sqrt(3));
	CalculateTSDFOnNodes(grid, BufChnls::sdf, *mMeshSDFAccel, LEAF | NONLEAF | GHOST, xform, truncation);
	while (true) {
		auto levelTarget = [=]__device__(const HATileAccessor<Tile> &acc, const HATileInfo<Tile> &info) ->int {
			auto h = acc.voxelSize(params.mFineLevel);

			//calculate the min and max value of node sdf in the tile
			T min_sdf = std::numeric_limits<T>::infinity();
			T max_sdf = -std::numeric_limits<T>::infinity();
			auto& tile = info.tile();
			for (int i = 0; i <= 8; i++) {
				for (int j = 0; j <= 8; j++) {
					for (int k = 0; k <= 8; k++) {
						auto phi = tile.node(BufChnls::sdf, Coord(i, j, k));
						max_sdf = max(max_sdf, phi);
						min_sdf = min(min_sdf, phi);
					}
				}
			}
			//if sdf range is completely larger than h * fluid_relative_bandwidth (positive, outside of solid),
			//or completely smaller than -h * solid_relative_bandwidth (negative, inside of fluid)
			//then keep the target level
			//otherwise we refine it to finest level
			if ((min_sdf > h * fluid_relative_bandwidth) || (max_sdf < -h * solid_relative_bandwidth)) {
				return info.mLevel;
			}
			else {
				return params.mFineLevel;
			}
		};

		std::vector<HATileInfo<Tile>> refined_tiles;
		auto refine_cnts = grid.refineStep(levelTarget, false, &refined_tiles);
		{
			auto err = cudaDeviceSynchronize();
			ASSERT(err == cudaSuccess, "cuda error after refinestep: {}", cudaGetErrorString(err));
		}
		grid.spawnGhostTiles(false, &refined_tiles);
		{
			auto err = cudaDeviceSynchronize();
			ASSERT(err == cudaSuccess, "cuda error after spawnghost: {}", cudaGetErrorString(err));
		}
		CalculateTSDFOnGivenTiles(grid, refined_tiles, BufChnls::sdf, *mMeshSDFAccel, xform, truncation);

		auto cnt = std::accumulate(refine_cnts.begin(), refine_cnts.end(), 0);
		if (cnt == 0) break;
	}
}

void FluidEuler::project(HADeviceGrid<Tile>& grid, const T current_time, const T dt) {
	//will use u_mix as a temporary channel

	CPUTimer total_projection_timer; total_projection_timer.start();

	//AMG
	{
		CalculateNeighborTiles(grid);
		//calculate mix velocities
		mixFluidAndSolidVelocityOnFaces(grid, current_time, dt, ProjChnls::c0, BufChnls::u, ProjChnls::u_mix);
		AMGVolumeWeightedDivergenceWithoutCoeffOnLeafs(grid, ProjChnls::u_mix, ProjChnls::b);

		//Info("before proj div pt linf: {}", NormSync(grid, -1, ProjChnls::b, false));

		AMGSolver solver(ProjChnls::c0, 0.5, 1, 1);
		//solver.prepareTypesAndCoeffs(grid);

		cudaDeviceSynchronize(); CPUTimer projection_solve_timer; projection_solve_timer.start();
		auto [iters, err] = solver.solve(grid, false, 100, 1e-7, 2, 10, 1, mParams.mIsPureNeumann);
		cudaDeviceSynchronize(); projection_solve_time = projection_solve_timer.stop();
		double total_cells = grid.numTotalTiles() * Tile::SIZE;
		double cells_per_second = (total_cells + 0.0) / (projection_solve_time / 1000.0);
		Info("Total {:.5}M cells, {}ms, AMGPCG speed {:.5} M cells /s at {} iters", total_cells / (1024.0 * 1024), projection_solve_time, cells_per_second / (1024.0 * 1024), iters);

		//Info("pressure pt l2: {}", NormSync(grid, 2, ProjChnls::x, false));

		//AMGAddFaceWeightedGradientToFace(grid, -1, LEAF, ProjChnls::x, ProjChnls::c0, BufChnls::u);
		//add gradp to fluid velocity without coeffs
		//solid velocity is not stored
		AMGAddGradientToFace(grid, -1, LEAF, ProjChnls::x, ProjChnls::c0, BufChnls::u);

		//mix and calculate divergence again for validation
		mixFluidAndSolidVelocityOnFaces(grid, current_time, dt, ProjChnls::c0, BufChnls::u, BufChnls::u);


		AMGVolumeWeightedDivergenceWithoutCoeffOnLeafs(grid, BufChnls::u, ProjChnls::b);
		Info("after proj div pt linf: {}", NormSync(grid, -1, ProjChnls::b, false));

		{
			Warn("after proj umax = {}, vmax = {}, wmax = {}", NormSync(grid, -1, BufChnls::u, false), NormSync(grid, -1, BufChnls::u + 1, false), NormSync(grid, -1, BufChnls::u + 2, false));
		}
	}

	cudaDeviceSynchronize();  total_projection_time = total_projection_timer.stop("total projection");

	//{
	//	//show velocity on polyscope before proj
	//	polyscope::init();
	//	polyscope::removeAllStructures();
	//	auto holder = grid.getHostTileHolderForLeafs();
	//	//AddLeveledPoissonGridCellCentersToPolyscopePointCloud
	//	//AddPoissonGridCellCentersToPolyscopePointCloud
	//	IOFunc::AddLeveledPoissonGridCellCentersToPolyscopePointCloud(holder, { { -1,"type" }, { ProjChnls::c0 + 3, "c3" }, {ProjChnls::x, "pressure"}, { ProjChnls::b, "divergence" } }, { {BufChnls::u, "velocity"}, {ProjChnls::u_mix, "u_mix"} });
	//	//IOFunc::AddLeveledPoissonGridCellCentersToPolyscopePointCloud(holder, { { -1,"type" }, { BufChnls::vor, "vorticity" } }, { { BufChnls::u, "velocity" } });
	//	IOFunc::AddMarkerParticlesToPolyscope(marker_particles_d, "marker_particles");
	//	IOFunc::AddTilesToPolyscopeVolumetricMesh(grid, LEAF, "leaf_tiles");
	//	auto xform = mParams.meshToWorldTransform(current_time);
	//	Eigen::Matrix<T, -1, 3> V_world =
	//		(xform * mMeshSDFAccel->V_.transpose()).transpose();
	//	auto* psMesh = polyscope::registerSurfaceMesh("mesh", V_world, mMeshSDFAccel->F_);
	//	polyscope::show();
	//}


}

void FluidEuler::adaptAndAdvect(DriverMetaData& metadata, std::vector<std::shared_ptr<HADeviceGrid<Tile>>> grid_ptrs) {
	const double dt = metadata.dt;
	const double current_time = metadata.current_time;


	cudaDeviceSynchronize(); CPUTimer adapt_and_advect_timer; adapt_and_advect_timer.start();

	int advection_u_channel = BufChnls::u;

	//saved intermediate velocities
	int n = grid_ptrs.size() - 1;
	//we only need to prepare the last grid at this time
	auto& last_grid = *grid_ptrs[n - 1];

	thrust::host_vector<HATileAccessor<Tile>> accs_h;
	for (int i = 0; i < n; i++) {
		accs_h.push_back(grid_ptrs[i]->deviceAccessor());
	}
	thrust::device_vector<HATileAccessor<Tile>> accs_d = accs_h;
	auto accs_d_ptr = thrust::raw_pointer_cast(accs_d.data());
	thrust::device_vector<double> time_steps_d = time_steps;
	auto time_steps_d_ptr = thrust::raw_pointer_cast(time_steps_d.data());

	MarkParticlesOutsideFluidRegionAsInvalid(pfm_particles_d, last_grid);
	MarkOldParticlesAsInvalid(pfm_particles_d, current_time, mParams.mParticleLife);
	EraseInvalidParticles(pfm_particles_d);

	ReseedParticles(last_grid, mParams, BufChnls::tmp, BufChnls::sdf, current_time, mParams.mSampleNumPerCell, pfm_particles_d);
	//cudaDeviceSynchronize(); timer.stop("Reseeding particles"); timer.start();


	//{
	//	polyscope::init();
	//	IOFunc::AddTilesToPolyscopeVolumetricMesh(last_grid, LEAF, "leaf tiles");
	//	IOFunc::AddParticleSystemToPolyscope(particles, "particles");
	//	polyscope::show();
	//}


	//reset impulse for all particles
	if (time_step_counter % mParams.mFlowMapStride == 0) {
		//auto holder_ptr = last_grid.getHostTileHolderForLeafs();
		//GenerateParticlesUniformlyOnFinestLevel(holder_ptr, 2, particles);

		//with midpoint velocity on, we have to create a copy
		//nfm_query_grid_ptr = grid_ptrs[n - 1]->deepCopy();

		//without midpoint velocity, we can directly use the last grid
		//nfm_query_grid_ptr = grid_ptrs[n - 1];
		ResetParticleImpulse(last_grid, mParams.mFineLevel, mParams.mCoarseLevel, advection_u_channel, pfm_particles_d);
	}
	else {
		int fine_level = mParams.mFineLevel;
		int coarse_level = mParams.mCoarseLevel;

		int back_traced_steps = time_step_counter % mParams.mFlowMapStride;
		int nfm_start_idx = n - back_traced_steps - 1;
		auto particles_d_ptr = thrust::raw_pointer_cast(pfm_particles_d.data());
		LaunchIndexFunc([=] __device__(int idx) {
			auto& particle = particles_d_ptr[idx];

			if (particle.start_time == current_time) {

				Vec psi = particle.pos;
				Eigen::Matrix3<T> matT;
				NFMBackMarchPsiAndT(accs_d_ptr, fine_level, coarse_level, time_steps_d_ptr, advection_u_channel, nfm_start_idx, n - 1, psi, matT);
				Vec m0;
				bool success = KernelIntpVelocityMAC2(accs_d_ptr[nfm_start_idx], fine_level, coarse_level, psi, advection_u_channel, m0);


				//NFMBackQueryImpulseAndT(accs_d_ptr, fine_level, coarse_level, time_steps_d_ptr, u_channel, last_u_node_channel, nfm_start_idx, n - 1, psi, m0, matT);

				particle.impulse = m0;
				particle.matT() = matT;
			}
		}, pfm_particles_d.size(), 128);
	}



	static thrust::device_vector<int> tile_prefix_sum_d;
	static thrust::device_vector<ParticleRecord> records_d;
	HistogramSortParticlesAtGivenLevel(last_grid, mParams.mFineLevel, BufChnls::tmp, pfm_particles_d, tile_prefix_sum_d, records_d);
	OptimizedAdvectParticlesAndSingleStepGradMRK4ForwardAtGivenLevel(last_grid, mParams.mFineLevel, advection_u_channel, dt, tile_prefix_sum_d, records_d);
	EraseInvalidParticles(pfm_particles_d);

	Warn("after particle advection {} particles", pfm_particles_d.size());

	//Info("end here");
	//exit(-1);

	//{
	//	polyscope::init();
	//	auto holder = last_grid.getHostTileHolderForLeafs();
	//	IOFunc::AddParticleSystemToPolyscope(particles, "particles");
	//	//IOFunc::AddPoissonGridCellCentersToPolyscopePointCloud(holder, { { -1,"type" } }, { {Tile::u_channel,"vel"} });
	//	//IOFunc::AddPoissonGridFaceCentersToPolyscopePointCloud(holder, { {Tile::u_channel, "vel"} });
	//	polyscope::show();
	//}

	cudaDeviceSynchronize(); CPUTimer grid_adaptation_timer; grid_adaptation_timer.start();

	auto& grid = *grid_ptrs[n];

	RefineWithParticles(grid, pfm_particles_d, mParams.mCoarseLevel, mParams.mFineLevel, BufChnls::counter, false);
	CoarsenWithParticles(grid, pfm_particles_d, mParams.mCoarseLevel, mParams.mFineLevel, BufChnls::counter, false);
	CheckCudaError("adapt with particles");
	{
		//after the grid structure is set, calculate type and initialize AMG coeffs
		iterativeNodeSDFAndRefineNarrowBand(grid, current_time, mParams.mRelativeRefineBandwidth, mParams.mRelativeRefineBandwidth);
		buildTypesAndAMGCoeffsFromNodeSDFs(grid, current_time);
	}

	//{
	//	//show velocity on polyscope before proj
	//	polyscope::init();
	//	polyscope::removeAllStructures();
	//	auto holder = last_grid.getHostTileHolderForLeafs();
	//	//AddLeveledPoissonGridCellCentersToPolyscopePointCloud
	//	//AddPoissonGridCellCentersToPolyscopePointCloud
	//	//IOFunc::AddLeveledPoissonGridCellCentersToPolyscopePointCloud(holder, { { -1,"type" }, { ProjChnls::c0 + 3, "c3" }, {ProjChnls::x, "pressure"}, { ProjChnls::b, "divergence" } }, { {BufChnls::u, "velocity"} });
	//	//IOFunc::AddLeveledPoissonGridCellCentersToPolyscopePointCloud(holder, { { -1,"type" }, { BufChnls::vor, "vorticity" } }, { { BufChnls::u, "velocity" } });

	//	//IOFunc::AddParticlesToPolyscope(pfm_particles_d, "pfm_particles");
	//	IOFunc::AddLeveledTilesToPolyscopeVolumetricMesh(last_grid, LEAF, "leaf_tiles");
	//	auto xform = mParams.meshToWorldTransform(current_time);
	//	Eigen::Matrix<T, -1, 3> V_world =
	//		(xform * mMeshSDFAccel->V_.transpose()).transpose();
	//	auto* psMesh = polyscope::registerSurfaceMesh("mesh", V_world, mMeshSDFAccel->F_);
	//	polyscope::show();
	//}

	cudaDeviceSynchronize(); grid_adaptation_time = grid_adaptation_timer.stop("Grid adaptation");

	//ParticleImpulseToGridMACIntp(grid, particles, u_channel, next_uw_channel);
	HistogramSortParticlesAtGivenLevel(grid, mParams.mFineLevel, BufChnls::counter, pfm_particles_d, tile_prefix_sum_d, records_d);
	OptimizedP2GTransferAtGivenLevel(grid, mParams.mFineLevel, advection_u_channel, AdvChnls::u_weight, tile_prefix_sum_d, records_d);
	EraseInvalidParticles(pfm_particles_d);

	CheckCudaError("pfm p2g");
	//Info("max impulse after pfm: {}", VelocityLinf(grid, u_channel, -1, LEAF, LAUNCH_SUBTREE));

	//advect dye and NFM
	{
		auto last_acc = last_grid.deviceAccessor();
		//auto nfm_query_acc = nfm_query_grid_ptr->deviceAccessor();
		auto params = mParams;

		int fine_level = mParams.mFineLevel;
		int coarse_level = mParams.mCoarseLevel;

		int back_traced_steps = time_step_counter % mParams.mFlowMapStride;
		int nfm_start_idx = n - back_traced_steps - 1;

		grid.launchVoxelFuncOnAllTiles(
			[=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
			auto& tile = info.tile();
			//if (!tile.isInterior(l_ijk)) return;

			{
				//grid velocity advection
				for (int axis : {0, 1, 2}) {
					if (tile(AdvChnls::u_weight + axis, l_ijk) < 1 - 1e-3)
					{
						//Vec psi = acc.faceCenter(axis, info, l_ijk);
						Vec psi = NFMErodedAdvectionPoint(axis, acc, info, l_ijk);
						Vec m0; Eigen::Matrix3<T> matT;

						//NFMBackQueryImpulseAndT(accs_d_ptr, info.mLevel, coarse_level, time_steps_d_ptr, u_channel, last_u_node_channel, nfm_start_idx, n, psi, m0, matT);
						//NFMBackQueryImpulseAndT(accs_d_ptr, fine_level, coarse_level, time_steps_d_ptr, u_channel, last_u_node_channel, nfm_start_idx, n, psi, m0, matT);

						NFMBackMarchPsiAndT(accs_d_ptr, fine_level, coarse_level, time_steps_d_ptr, advection_u_channel, nfm_start_idx, n, psi, matT);

						KernelIntpVelocityMAC2(accs_d_ptr[nfm_start_idx], fine_level, coarse_level, psi, advection_u_channel, m0);

						Vec m1 = MatrixTimesVec(matT.transpose(), m0);

						tile(BufChnls::u + axis, l_ijk) = m1[axis];
					}
				}
			}
		}, LEAF, 4
		);
	}


	CalcCellTypesFromLeafs(grid);

	cudaDeviceSynchronize(); adapt_and_advect_time = adapt_and_advect_timer.stop("Adaptation and advection");
	CheckCudaError("nfm advection");

	//Info("max impulse after nfm: {}", VelocityLinf(grid, u_channel, -1, LEAF, LAUNCH_SUBTREE));
}

void FluidEuler::applyViscosity(HADeviceGrid<Tile>& grid, const T dt, const T nu)
{
	for (int axis : {0, 1, 2}) {
		//copy cell type and 
	}
}

void FluidEuler::Advance(DriverMetaData& metadata) {

	cudaDeviceSynchronize(); CPUTimer total_advance_timer; total_advance_timer.start();

	//Info("before advance"); PrintMemoryInfo();

	if (grid_ptrs.size() < mParams.mFlowMapStride + 1) {
		//create a new grid
		auto nxt_ptr = grid_ptrs.back()->deepCopy();
		grid_ptrs.push_back(nxt_ptr);
		time_steps.push_back(metadata.dt);
	}
	else {
		auto nxt_ptr = grid_ptrs[0];
		grid_ptrs.erase(grid_ptrs.begin());
		grid_ptrs.push_back(nxt_ptr);
		time_steps.erase(time_steps.begin());
		time_steps.push_back(metadata.dt);
	}

	auto& grid = *grid_ptrs.back();
	auto& last_grid = *grid_ptrs[grid_ptrs.size() - 2];
	//double dt = metadata.dt;

	fmt::print("\n");
	Pass("Advance frame {} current time {} dt {} step counter {}", metadata.current_frame, metadata.current_time, metadata.dt, time_step_counter);
	ASSERT(metadata.dt > 0, "dt should be positive");


	adaptAndAdvect(metadata, grid_ptrs);


	applyExternalForce(grid, metadata.dt);



	//if (time_step_counter == 65) {
	//	amg_solver_open_visualization = 1;
	//}
	//else {
	//	amg_solver_open_visualization = 0;
	//}


	//projection
	project(grid, metadata.current_time, metadata.dt);





	CalculateVelocityAndVorticityMagnitudeOnLeafCellCenters(grid, mParams.mFineLevel, mParams.mCoarseLevel, BufChnls::u, BufChnls::u_cell, BufChnls::vor);

	//{
	//	//show velocity on polyscope before proj
	//	polyscope::init();
	//	polyscope::removeAllStructures();
	//	auto holder = grid.getHostTileHolderForLeafs();
	//	IOFunc::AddPoissonGridCellCentersToPolyscopePointCloud(holder, { { -1,"type" }, { BufChnls::vor, "vorticity" }, {ProjChnls::x, "pressure"}, { ProjChnls::b, "divergence" } }, { {BufChnls::u, "velocity"} });
	//	//IOFunc::AddLeveledPoissonGridCellCentersToPolyscopePointCloud(holder, { { -1,"type" }, { BufChnls::vor, "vorticity" } }, { { BufChnls::u, "velocity" } });
	//	polyscope::show();
	//}

	//extrapolateFluidVelocityForAdvection(grid, mParams.mExtrapolationIters, BufChnls::u, ProjChnls::c0);

	CheckCudaError("Advance");
	time_step_counter++;


	PrintMemoryInfo();

	cudaDeviceSynchronize(); total_advance_time = total_advance_timer.stop("Total advance");
}