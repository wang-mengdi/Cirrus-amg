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
			T fluid_ratio = tile(coeff_channel + axis, l_ijk);
			T solid_ratio = 1 - fluid_ratio;
			T mix_vel;
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

void FluidEuler::project(HADeviceGrid<Tile>& grid, const T current_time, const T dt) {
	//auto c0_channel = ProjChnls::c0;

	{
		//contaminate channels for debugging
		grid.launchVoxelFuncOnAllTiles(
			[=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
			info.tile()(ProjChnls::b, l_ijk) = NODATA;
		}, LEAF | GHOST | NONLEAF, 4
		);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

	}

	//AMG
	{
		CalculateNeighborTiles(grid);
		//calculate mix velocities
		mixFluidAndSolidVelocityOnFaces(grid, current_time, dt, ProjChnls::c0, BufChnls::u, ProjChnls::u_mix);
		AMGVolumeWeightedDivergenceWithoutCoeffOnLeafs(grid, ProjChnls::u_mix, ProjChnls::b);

		//Info("before proj div pt linf: {}", NormSync(grid, -1, ProjChnls::b, false));

		AMGSolver solver(ProjChnls::c0, 0.5, 1, 1);
		//solver.prepareTypesAndCoeffs(grid);

		CPUTimer timer;
		timer.start();
		auto [iters, err] = solver.solve(grid, false, 100, 1e-7, 2, 10, 1, mParams.mIsPureNeumann);
		cudaDeviceSynchronize();
		double elapsed = timer.stop("AMGPCG");
		double total_cells = grid.numTotalTiles() * Tile::SIZE;
		double cells_per_second = (total_cells + 0.0) / (elapsed / 1000.0);
		Info("Total {:.5}M cells, AMGPCG speed {:.5} M cells /s at {} iters", total_cells / (1024.0 * 1024), cells_per_second / (1024.0 * 1024), iters);
		projection_time = elapsed;

		//Info("pressure pt l2: {}", NormSync(grid, 2, ProjChnls::x, false));

		//AMGAddFaceWeightedGradientToFace(grid, -1, LEAF, ProjChnls::x, ProjChnls::c0, BufChnls::u);
		//add gradp to fluid velocity without coeffs
		//solid velocity is not stored
		AMGAddGradientToFace(grid, -1, LEAF, ProjChnls::x, ProjChnls::c0, BufChnls::u);

		//mix and calculate divergence again for validation
		mixFluidAndSolidVelocityOnFaces(grid, current_time, dt, ProjChnls::c0, BufChnls::u, ProjChnls::u_mix);
		AMGVolumeWeightedDivergenceWithoutCoeffOnLeafs(grid, BufChnls::u, ProjChnls::b);
		Info("after proj div pt linf: {}", NormSync(grid, -1, ProjChnls::b, false));

		{
			Warn("after proj umax = {}, vmax = {}, wmax = {}", NormSync(grid, -1, BufChnls::u, false), NormSync(grid, -1, BufChnls::u + 1, false), NormSync(grid, -1, BufChnls::u + 2, false));
		}
	}

	//{
	//	//show velocity on polyscope before proj
	//	polyscope::init();
	//	polyscope::removeAllStructures();
	//	auto holder = grid.getHostTileHolderForLeafs();
	//	//AddLeveledPoissonGridCellCentersToPolyscopePointCloud
	//	//AddPoissonGridCellCentersToPolyscopePointCloud
	//	IOFunc::AddLeveledPoissonGridCellCentersToPolyscopePointCloud(holder, { { -1,"type" }, { ProjChnls::c0 + 3, "c3" }, {ProjChnls::x, "pressure"}, { ProjChnls::b, "divergence" } }, { {BufChnls::u, "velocity"} });
	//	//IOFunc::AddLeveledPoissonGridCellCentersToPolyscopePointCloud(holder, { { -1,"type" }, { BufChnls::vor, "vorticity" } }, { { BufChnls::u, "velocity" } });
	//	polyscope::show();
	//}
}
