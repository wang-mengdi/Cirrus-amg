#include "SolverTests.h"
#include "GMGSolver.h"
#include "CMGSolver.h"
#include "AMGSolver.h"
#include "PoissonIOFunc.h"
#include "Common.h"
#include "GPUTimer.h"
#include "Random.h"
#include <polyscope/polyscope.h>
#include <polyscope/point_cloud.h>

// extern int laplacian_total_tile_counts;

namespace SolverTests
{
	__hostdev__ int SolverTestsLevelTarget(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const TestGrids grid_name)
	{
		// uniform grids
		if (grid_name == TestGrids::uniform8)
		{
			return 0;
		}
		else if (grid_name == TestGrids::uniform32)
		{
			return 2;
		}
		else if (grid_name == TestGrids::uniform128)
		{
			return 4;
		}
		else if (grid_name == TestGrids::uniform256)
		{
			return 5;
		}
		else if (grid_name == TestGrids::uniform512)
		{
			return 6;
		}
		else if (grid_name == TestGrids::staircase12)
		{
			auto bbox = acc.tileBBox(info);
			int desired_level = 0;
			if (bbox.max()[0] >= 0.75)
				return 2; // slow converging, if 0.25 not converging
			else
				return 1;
		}
		else if (grid_name == TestGrids::staircase21)
		{
			auto bbox = acc.tileBBox(info);
			int desired_level = 0;
			if (bbox.min()[0] <= 0.25)
				return 2; // slow converging, if 0.25 not converging
			else
				return 1;
		}
		else if (grid_name == TestGrids::staircase34)
		{
			auto bbox = acc.tileBBox(info);
			int desired_level = 0;
			if (bbox.max()[0] >= 0.75)
				return 3; // slow converging, if 0.25 not converging
			else
				return 4;
		}
		else if (grid_name == TestGrids::staircase43)
		{
			// 64^3 at small x, 128^3 at large x
			auto bbox = acc.tileBBox(info);
			int desired_level = 0;
			if (bbox.min()[0] <= 0.25)
				return 4; // slow converging, if 0.25 not converging
			else
				return 3;
		}
		else if (grid_name == TestGrids::twosources67)
		{
			int desired_level = 0;
			auto bbox = acc.tileBBox(info);
			const Vec pointSrc1(0.51, 0.49, 0.54);
			const Vec pointSrc2(0.93, 0.08, 0.91);
			if (bbox.isInside(pointSrc2))
				desired_level = 6;
			if (bbox.isInside(pointSrc1))
				desired_level = 7;
			// if (bbox.isInside(pointSrc2)) desired_level = 3;
			// if (bbox.isInside(pointSrc1)) desired_level = 4;
			return desired_level;
		}
		else if (grid_name == TestGrids::twosources_deform)
		{
			// two sources for testing 3d deformation
			// refine at (0.35,0.35,0.35)
			// it's for testing the 3D deformation
			int desired_level = 0;
			auto bbox = acc.tileBBox(info);
			const Vec pointSrc1(0.35, 0.35, 0.35);
			const Vec pointSrc2(0.8, 0.2, 0.6);
			if (bbox.isInside(pointSrc2))
				desired_level = 5;
			if (bbox.isInside(pointSrc1))
				desired_level = 6;
			// if (bbox.isInside(pointSrc2)) desired_level = 3;
			// if (bbox.isInside(pointSrc1)) desired_level = 4;
			return desired_level;
		}
		else if (grid_name == TestGrids::centersource)
		{
			// try to test nfm advection with 3d deformation

			int desired_level = 0;
			auto bbox = acc.tileBBox(info);
			double eps = 1e-6;
			const Vec pointSrc1(0.5 - eps, 0.5 - eps, 0.5 - eps);
			if (bbox.isInside(pointSrc1))
				desired_level = 6;

			return desired_level;
		}
	}

	void TestAMGLaplacianAndFluxConsistency(TestGrids grid_name)
	{
		fmt::print("==========================================================\n");
		Info("Test lap=div(grad) on grid {}", ToString(grid_name));

		uint32_t scale = 8;
		float h = 1.0 / scale;
		// 0:8, 1:16, 2:32, 3:64, 4:128, 5:256, 6:512, 7:1024
		HADeviceGrid<Tile> grid(h, { 16, 16, 16, 16, 16, 16, 18, 16, 16, 16 });
		grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
		grid.rebuild();
		grid.iterativeRefine([=] __device__(const HATileAccessor<Tile> &acc, HATileInfo<Tile> &info)
		{
			return SolverTestsLevelTarget(acc, info, grid_name);
		}, false);
		int num_cells = grid.numTotalLeafTiles() * Tile::SIZE;
		int total_hash_bytes = grid.hashTableDeviceBytes();
		Info("Total {}M cells, hash table {}GB", num_cells / (1024.0 * 1024), total_hash_bytes / (1024.0 * 1024 * 1024));

		int x_channel = 0;
		int coeff_channel = 1;
		int u_channel = 5;
		int nlap_channel = 8;
		int lap_channel = 9;
		int diff_channel = 10;

		// load cell types, load solution to grdt_channel
		auto cell_type = [=] __device__(const HATileAccessor<Tile> &acc, const HATileInfo<Tile> &info, const nanovdb::Coord & l_ijk) -> uint8_t
		{
			bool is_dirichlet = false;
			bool is_neumann = false;
			acc.iterateSameLevelNeighborVoxels(info, l_ijk,
				[&] __device__(const HATileInfo<Tile> &n_info, const Coord & n_l_ijk, const int axis, const int sgn)
			{
				if (n_info.empty())
				{
					if (axis == 0 && sgn == -1)
						is_neumann = true;
					else
						is_dirichlet = true;
				}
			});
			if (is_neumann)
				return CellType::NEUMANN;
			else if (is_dirichlet)
				return CellType::DIRICHLET;
			else
				return CellType::INTERIOR;
		};
		// solution f(x)
		auto f = [=] __device__(const Vec & pos) -> T
		{
			return sin(-2.5 * pos[0] + pos[1] + 7 * pos[2]);
		};
		grid.launchVoxelFuncOnAllTiles(
			[=] __device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk)
		{
			auto& tile = info.tile();
			tile.type(l_ijk) = cell_type(acc, info, l_ijk);
			if (tile.type(l_ijk) & INTERIOR)
				tile(x_channel, l_ijk) = f(acc.cellCenter(info, l_ijk));
			else
				tile(x_channel, l_ijk) = 0;
			for (int axis : {0, 1, 2})
			{
				tile(u_channel + axis, l_ijk) = 0;
			}
		},
			LEAF | GHOST);
		CalcCellTypesFromLeafs(grid);

		// use AMG to calculate Laplacian from x_channel to lap_channel
		{
			CalculateNeighborTiles(grid);
			AMGSolver solver(coeff_channel, 0.5, 1, 1);
			solver.prepareTypesAndCoeffs(grid);
			AMGFullNegativeLaplacianOnLeafs(grid, x_channel, coeff_channel, nlap_channel);
		}

		// calculate div(grad)
		{
			// grad(p)
			AMGAddGradientToFace(grid, -1, LEAF | GHOST, x_channel, coeff_channel, u_channel);
			// div(u)
			AMGVolumeWeightedDivergenceOnLeafs(grid, u_channel, coeff_channel, lap_channel);
		}

		// calculate difference from x and grdt in r_channel
		grid.launchVoxelFuncOnAllTiles(
			[=] __device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk)
		{
			auto& tile = info.tile();
			if (tile.type(l_ijk) & INTERIOR)
			{
				tile(diff_channel, l_ijk) = tile(lap_channel, l_ijk) + tile(nlap_channel, l_ijk);
			}
			else
			{
				tile(diff_channel, l_ijk) = 0;
			}
		},
			LEAF);

		auto holder = grid.getHostTileHolderForLeafs();
		polyscope::init();
		IOFunc::AddPoissonGridCellCentersToPolyscopePointCloud(holder,
			{ {-1, "type"}, {x_channel, "x"}, {lap_channel, "div(grad)"}, {nlap_channel, "nlap"}, {diff_channel, "diff"} },
			{});
		polyscope::show();

		auto linf_norm = NormSync(grid, -1, diff_channel, false);
		if (linf_norm < 1e-5)
		{
			Pass("Test passed with Linf norm of grdt-x: {}\n\n", linf_norm);
		}
		else
		{
			Error("Test failed with Linf norm of grdt-x: {}\n\n", linf_norm);
		}
	}

	__host__ void TestNeumannDirichletRecovery(TestGrids grid_name, const std::string algorithm)
	{
		// algorithm: "cmg"/"amg"
		fmt::print("==========================================================\n");
		Info("Test Poisson Solver on Neumann/Dirichlet BC on grid {} with given function for {}", ToString(grid_name), algorithm);
		Assert(algorithm == "cmg" || algorithm == "amg", "algorithm should be cmg or amg");

		uint32_t scale = 8;
		float h = 1.0 / scale;
		// 0:8, 1:16, 2:32, 3:64, 4:128, 5:256, 6:512, 7:1024
		HADeviceGrid<Tile> grid(h, { 16, 16, 16, 16, 16, 16, 20, 16, 16, 16 });
		grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
		grid.rebuild();

		grid.iterativeRefine([=] __device__(const HATileAccessor<Tile> &acc, HATileInfo<Tile> &info)
		{
			return SolverTestsLevelTarget(acc, info, grid_name);
		}, false);
		int num_cells = grid.numTotalLeafTiles() * Tile::SIZE;
		int total_hash_bytes = grid.hashTableDeviceBytes();
		Info("Total {}M cells, hash table {}GB", num_cells / (1024.0 * 1024), total_hash_bytes / (1024.0 * 1024 * 1024));

		Assert(Tile::num_channels >= 10, "Tile::num_channels should be >= 10 for this test");
		// channels 0,1,2,3,4: channels implicitly required by GMGPCG
		// channels 5: grdt channel
		// channels 6,7,8,9: coeffs channel for AMG
		int grdt_channel = 5;
		int coeff_channel = 6;

		// load cell types, load solution to grdt_channel
		auto cell_type = [=] __device__(const HATileAccessor<Tile> &acc, const HATileInfo<Tile> &info, const nanovdb::Coord & l_ijk) -> uint8_t
		{
			bool is_dirichlet = false;
			bool is_neumann = false;
			acc.iterateSameLevelNeighborVoxels(info, l_ijk,
				[&] __device__(const HATileInfo<Tile> &n_info, const Coord & n_l_ijk, const int axis, const int sgn)
			{
				if (n_info.empty())
				{
					if (axis == 0 && sgn == -1)
					{
						is_neumann = true;
					}
					else
					{
						is_dirichlet = true;
					}
				}
			});
			if (is_neumann)
				return CellType::NEUMANN;
			else if (is_dirichlet)
				return CellType::DIRICHLET;
			else
				return CellType::INTERIOR;
		};
		// solution f(x)
		auto f = [=] __device__(const Vec & pos)
		{
			return exp(pos[0]) * sin(pos[1]);
		};
		grid.launchVoxelFunc(
			[=] __device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk)
		{
			auto& tile = info.tile();
			tile.type(l_ijk) = cell_type(acc, info, l_ijk);
			if (tile.type(l_ijk) & INTERIOR)
				tile(grdt_channel, l_ijk) = f(acc.cellCenter(info, l_ijk));
			else
				tile(grdt_channel, l_ijk) = 0;
		},
			-1, LEAF, LAUNCH_SUBTREE);
		CalcCellTypesFromLeafs(grid);
		CalculateNeighborTiles(grid);

		if (algorithm == "cmg")
		{
			CalculateNeighborTiles(grid);
			ConservativeFullNegativeLaplacian(grid, grdt_channel, Tile::b_channel);

			_sleep(200);
			CMGSolver solver(1.0, 1.0);
			// Copy(grid, Tile::b_channel, Tile::phi_channel, -1, LEAF, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);
			CPUTimer<std::chrono::microseconds> timer;
			timer.start();
			auto [iters, err] = solver.solve(grid, true, 1000, 1e-6, 1, 10, 1, false);
			CheckCudaError("CMGPCG solve");
			float elapsed = timer.stop("CMGPCG Async");
			int total_cells = grid.numTotalTiles() * Tile::SIZE;
			float cells_per_second = (total_cells + 0.0) / (elapsed / 1e6);
			Info("Total {:.5}M cells, CMGPCG Async speed {:.5} M cells /s", total_cells / (1024.0 * 1024), cells_per_second / (1024.0 * 1024));
			Info("CMGPCG solved in {} iterations with error {}, average iteration throughput {:.5}M cell/s", iters, err, cells_per_second * iters / (1024.0 * 1024));
		}
		else if (algorithm == "amg")
		{
			CalculateNeighborTiles(grid);

			AMGSolver solver(coeff_channel, 0.5, 1, 1);
			solver.prepareTypesAndCoeffs(grid);
			AMGFullNegativeLaplacianOnLeafs(grid, grdt_channel, coeff_channel, Tile::b_channel);

			// auto holder = grid.getHostTileHolder(LEAF | NONLEAF);
			// polyscope::init();
			// IOFunc::AddLeveledPoissonGridCellCentersToPolyscopePointCloud(holder,
			//     {
			//     {solver.coeff_channel,"offd0"}, {solver.coeff_channel + 1,"offd1"} ,{solver.coeff_channel + 2,"offd2"},{solver.coeff_channel + 3,"diag"}, {-1,"type"}
			//     },
			//     {}, -1, FLT_MAX);
			// polyscope::show();

			_sleep(200);

			CPUTimer<std::chrono::microseconds> timer;
			timer.start();
			auto [iters, err] = solver.solve(grid, true, 1000, 1e-6, 2, 10, 1, false);
			CheckCudaError("AMGPCG solve");
			float elapsed = timer.stop("AMGPCG Async");
			int total_cells = grid.numTotalLeafTiles() * Tile::SIZE;
			float cells_per_second = (total_cells + 0.0) / (elapsed / 1e6);
			Info("Total {:.5}M cells, AMGPCG Async speed {:.5} M cells /s", total_cells / (1024.0 * 1024), cells_per_second / (1024.0 * 1024));
			Info("AMGPCG solved in {} iterations with error {}, average iteration throughput {:.5}M cell/s", iters, err, cells_per_second * iters / (1024.0 * 1024));
		}
		_sleep(200);

		// calculate difference from x and grdt in r_channel
		grid.launchVoxelFuncOnAllTiles(
			[=] __device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk)
		{
			auto& tile = info.tile();
			if (tile.type(l_ijk) & INTERIOR)
			{
				tile(Tile::r_channel, l_ijk) = tile(grdt_channel, l_ijk) - tile(Tile::x_channel, l_ijk);
			}
			else
			{
				tile(Tile::r_channel, l_ijk) = 0;
			}
		},
			LEAF);

		Info("linf: {}", NormSync(grid, -1, Tile::r_channel, false));
		Info("weighted L1: {}", NormSync(grid, 1, Tile::r_channel, true));
		Info("weighted rms: {}", NormSync(grid, 2, Tile::r_channel, true));

		auto linf_norm = NormSync(grid, -1, Tile::r_channel, true);
		if (linf_norm < 1e-4)
		{
			Pass("Test passed with Linf norm of grdt-x: {}\n\n", linf_norm);
		}
		else
		{
			Error("Test failed with Linf norm of grdt-x: {}\n\n", linf_norm);
		}
	}

	__host__ void TestStaticPressureUAAMG(TestGrids grid_name, const std::string algorithm)
	{
		// algorithm: "cmg"/"amg"
		// A Fast Unsmoothed Aggregation Algebraic Multigrid Framework for the Large - Scale Simulation of Incompressible Flow
		fmt::print("==========================================================\n");
		Info("Test Poisson Solver with Static Pressure on grid {} with given function for {}", ToString(grid_name), algorithm);
		Assert(algorithm == "cmg" || algorithm == "amg", "algorithm should be cmg or amg");

		uint32_t scale = 8;
		float h = 1.0 / scale;
		// 0:8, 1:16, 2:32, 3:64, 4:128, 5:256, 6:512, 7:1024
		HADeviceGrid<Tile> grid(h, { 16, 16, 16, 16, 16, 16, 18, 16, 16, 16 });
		grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
		grid.rebuild();

		grid.iterativeRefine([=] __device__(const HATileAccessor<Tile> &acc, HATileInfo<Tile> &info)
		{
			return SolverTestsLevelTarget(acc, info, grid_name);
		}, false);
		int num_cells = grid.numTotalLeafTiles() * Tile::SIZE;
		int total_hash_bytes = grid.hashTableDeviceBytes();
		Info("Total {}M cells, hash table {}GB", num_cells / (1024.0 * 1024), total_hash_bytes / (1024.0 * 1024 * 1024));

		Assert(Tile::num_channels >= 10, "Tile::num_channels should be >= 10 for this test");
		// channels 0,1,2,3,4: channels implicitly required by GMGPCG
		// channels 5: grdt channel
		// channels 6,7,8,9: coeffs channel for AMG
		int grdt_channel = 5;
		int coeff_channel = 6;

		// load cell types, load solution to grdt_channel
		auto cell_type = [=] __device__(const HATileAccessor<Tile> &acc, const HATileInfo<Tile> &info, const nanovdb::Coord & l_ijk) -> uint8_t
		{
			bool is_dirichlet = false;
			bool is_neumann = false;
			acc.iterateSameLevelNeighborVoxels(info, l_ijk,
				[&] __device__(const HATileInfo<Tile> &n_info, const Coord & n_l_ijk, const int axis, const int sgn)
			{
				if (n_info.empty())
				{
					if (axis == 0 && sgn == -1)
					{
						is_neumann = true;
					}
					else
					{
						is_dirichlet = true;
					}
				}
			});
			if (is_neumann)
				return CellType::NEUMANN;
			else if (is_dirichlet)
				return CellType::DIRICHLET;
			else
				return CellType::INTERIOR;
		};
		// solution f(x)
		auto f = [=] __device__(const Vec & pos)
		{
			return exp(pos[0]) * sin(pos[1]);
		};
		grid.launchVoxelFunc(
			[=] __device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk)
		{
			auto& tile = info.tile();
			tile.type(l_ijk) = cell_type(acc, info, l_ijk);
			if (tile.type(l_ijk) & INTERIOR)
				tile(grdt_channel, l_ijk) = f(acc.cellCenter(info, l_ijk));
			else
				tile(grdt_channel, l_ijk) = 0;
		},
			-1, LEAF, LAUNCH_SUBTREE);
		CalcCellTypesFromLeafs(grid);
		CalculateNeighborTiles(grid);

		if (algorithm == "cmg")
		{
			CalculateNeighborTiles(grid);
			ConservativeFullNegativeLaplacian(grid, grdt_channel, Tile::b_channel);

			_sleep(200);
			CMGSolver solver(1.0, 1.0);
			// Copy(grid, Tile::b_channel, Tile::phi_channel, -1, LEAF, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);
			CPUTimer<std::chrono::microseconds> timer;
			timer.start();
			auto [iters, err] = solver.solve(grid, true, 1000, 1e-6, 1, 10, 1, false);
			CheckCudaError("CMGPCG solve");
			float elapsed = timer.stop("CMGPCG Async");
			int total_cells = grid.numTotalTiles() * Tile::SIZE;
			float cells_per_second = (total_cells + 0.0) / (elapsed / 1e6);
			Info("Total {:.5}M cells, CMGPCG Async speed {:.5} M cells /s", total_cells / (1024.0 * 1024), cells_per_second / (1024.0 * 1024));
			Info("CMGPCG solved in {} iterations with error {}, average iteration throughput {:.5}M cell/s", iters, err, cells_per_second * iters / (1024.0 * 1024));
		}
		else if (algorithm == "amg")
		{
			CalculateNeighborTiles(grid);

			AMGSolver solver(coeff_channel, 0.5, 1.0, 1);
			solver.prepareTypesAndCoeffs(grid);
			AMGFullNegativeLaplacianOnLeafs(grid, grdt_channel, coeff_channel, Tile::b_channel);

			_sleep(200);

			CPUTimer<std::chrono::microseconds> timer;
			timer.start();
			auto [iters, err] = solver.solve(grid, true, 1000, 1e-6, 1, 10, 1, false);
			CheckCudaError("AMGPCG solve");
			float elapsed = timer.stop("AMGPCG Async");
			int total_cells = grid.numTotalLeafTiles() * Tile::SIZE;
			float cells_per_second = (total_cells + 0.0) / (elapsed / 1e6);
			Info("Total {:.5}M cells, AMGPCG Async speed {:.5} M cells /s", total_cells / (1024.0 * 1024), cells_per_second / (1024.0 * 1024));
			Info("AMGPCG solved in {} iterations with error {}, average iteration throughput {:.5}M cell/s", iters, err, cells_per_second * iters / (1024.0 * 1024));
		}
		_sleep(200);

		// calculate difference from x and grdt in r_channel
		grid.launchVoxelFuncOnAllTiles(
			[=] __device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk)
		{
			auto& tile = info.tile();
			if (tile.type(l_ijk) & INTERIOR)
			{
				tile(Tile::r_channel, l_ijk) = tile(grdt_channel, l_ijk) - tile(Tile::x_channel, l_ijk);
			}
			else
			{
				tile(Tile::r_channel, l_ijk) = 0;
			}
		},
			LEAF);
		auto linf_norm = NormSync(grid, -1, Tile::r_channel, false);
		if (linf_norm < 1e-5)
		{
			Pass("Test passed with Linf norm of grdt-x: {}\n\n", linf_norm);
		}
		else
		{
			Error("Test failed with Linf norm of grdt-x: {}\n\n", linf_norm);
		}
	}

	__hostdev__ int ConvergenceTestLevelTarget(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const ConvergenceTestGridName grid_name, const int min_level, const int max_level)
	{
		if (grid_name == ConvergenceTestGridName::uniform)
		{
			return max_level;
		}
		else if (grid_name == ConvergenceTestGridName::sphere_shell_05)
		{
			auto bbox = acc.tileBBox(info);
			auto bmin = bbox.min();
			auto bmax = bbox.max();
			const Vec ctr(0.5, 0.5, 0.5);
			constexpr T radius = 0.5 / 2;
			int inside_cnt = 0;
			for (int di : {0, 1})
			{
				for (int dj : {0, 1})
				{
					for (int dk : {0, 1})
					{
						Vec vpos;
						vpos[0] = bmin[0] + di * (bmax[0] - bmin[0]);
						vpos[1] = bmin[1] + dj * (bmax[1] - bmin[1]);
						vpos[2] = bmin[2] + dk * (bmax[2] - bmin[2]);
						if ((vpos - ctr).length() < radius)
						{
							inside_cnt++;
						}
					}
				}
			}
			if (inside_cnt == 0 || inside_cnt == 8)
				return min_level;
			else
				return max_level;
		}
		return max_level;
	}

	class UniformGridCase
	{
	public:
		__hostdev__ static int target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const int min_level, const int max_level)
		{
			return max_level;
		}
		__hostdev__ static uint8_t type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk)
		{
			return CellType::INTERIOR;
		}
	};

	class SphereShell05GridCase
	{
	public:
		__hostdev__ static int target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const int min_level, const int max_level)
		{
			auto bbox = acc.tileBBox(info);
			auto bmin = bbox.min();
			auto bmax = bbox.max();
			const Vec ctr(0.5, 0.5, 0.5);
			constexpr T radius = 0.5 / 2;
			int inside_cnt = 0;
			for (int di : {0, 1})
			{
				for (int dj : {0, 1})
				{
					for (int dk : {0, 1})
					{
						Vec vpos;
						vpos[0] = bmin[0] + di * (bmax[0] - bmin[0]);
						vpos[1] = bmin[1] + dj * (bmax[1] - bmin[1]);
						vpos[2] = bmin[2] + dk * (bmax[2] - bmin[2]);
						if ((vpos - ctr).length() < radius)
						{
							inside_cnt++;
						}
					}
				}
			}
			if (inside_cnt == 0 || inside_cnt == 8)
				return min_level;
			else
				return max_level;
		}
		__hostdev__ static uint8_t type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk)
		{
			return CellType::INTERIOR;
		}
	};
	class SphereSolid05GridCase
	{
	public:
		__hostdev__ static T phi(const Vec& pos)
		{
			const Vec ctr(0.5, 0.5, 0.5);
			constexpr T radius = 0.5 / 2;
			return (pos - ctr).length() - radius;
		}

		__hostdev__ static int target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const int min_level, const int max_level)
		{
			auto bbox = acc.tileBBox(info);
			auto bmin = bbox.min();
			auto bmax = bbox.max();
			int inside_cnt = 0;
			for (int di : {0, 1})
			{
				for (int dj : {0, 1})
				{
					for (int dk : {0, 1})
					{
						Vec vpos;
						vpos[0] = bmin[0] + di * (bmax[0] - bmin[0]);
						vpos[1] = bmin[1] + dj * (bmax[1] - bmin[1]);
						vpos[2] = bmin[2] + dk * (bmax[2] - bmin[2]);
						if (phi(vpos) < 0)
						{
							inside_cnt++;
						}
					}
				}
			}
			if (inside_cnt == 0 || inside_cnt == 8)
				return min_level;
			else
				return max_level;
		}
		__hostdev__ static uint8_t type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk)
		{
			auto pos = acc.cellCenter(info, l_ijk);
			auto dx = acc.voxelSize(info);
			Vec bmin(pos[0] - 0.5 * dx, pos[1] - 0.5 * dx, pos[2] - 0.5 * dx);
			int inside_cnt = 0;
			for (int di : {0, 1})
			{
				for (int dj : {0, 1})
				{
					for (int dk : {0, 1})
					{
						Vec vpos;
						vpos[0] = bmin[0] + di * dx;
						vpos[1] = bmin[1] + dj * dx;
						vpos[2] = bmin[2] + dk * dx;
						if (phi(vpos) < 0)
						{
							inside_cnt++;
						}
					}
				}
			}
			if (inside_cnt == 8)
				return CellType::NEUMANN;
			else
				return CellType::INTERIOR;
		}
	};
	class SphereAir05GridCase
	{
	public:
		__hostdev__ static int target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const int min_level, const int max_level)
		{
			auto bbox = acc.tileBBox(info);
			auto bmin = bbox.min();
			auto bmax = bbox.max();
			const Vec ctr(0.5, 0.5, 0.5);
			constexpr T radius = 0.5 / 2;
			int inside_cnt = 0;
			for (int di : {0, 1})
			{
				for (int dj : {0, 1})
				{
					for (int dk : {0, 1})
					{
						Vec vpos;
						vpos[0] = bmin[0] + di * (bmax[0] - bmin[0]);
						vpos[1] = bmin[1] + dj * (bmax[1] - bmin[1]);
						vpos[2] = bmin[2] + dk * (bmax[2] - bmin[2]);
						if ((vpos - ctr).length() < radius)
						{
							inside_cnt++;
						}
					}
				}
			}
			if (inside_cnt == 0 || inside_cnt == 8)
				return min_level;
			else
				return max_level;
		}
		__hostdev__ static uint8_t type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk)
		{
			auto pos = acc.cellCenter(info, l_ijk);
			const Vec ctr(0.5, 0.5, 0.5);
			constexpr T radius = 0.5 / 2;
			if ((pos - ctr).length() < radius)
				return CellType::DIRICHLET;
			else
				return CellType::INTERIOR;
		}
	};
	class StarShellGerrisGridCase
	{
	public:
		__hostdev__ static int target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const int min_level, const int max_level)
		{
			auto bbox = acc.tileBBox(info);
			auto bmin = bbox.min();
			auto bmax = bbox.max();
			int inside_cnt = 0;
			for (int di : {0, 1})
			{
				for (int dj : {0, 1})
				{
					for (int dk : {0, 1})
					{
						Vec vpos;
						vpos[0] = bmin[0] + di * (bmax[0] - bmin[0]);
						vpos[1] = bmin[1] + dj * (bmax[1] - bmin[1]);
						vpos[2] = bmin[2] + dk * (bmax[2] - bmin[2]);

						T x = vpos[0] - 0.5;
						T y = vpos[1] - 0.5;
						T z = vpos[2] - 0.5;
						T r = sqrt(x * x + y * y + z * z);
						T theta = acos(z / r);
						T phi = atan2(y, x);

						T r0 = 0.237 + 0.079 * cos(6 * theta) * cos(6 * phi);

						// T r0 = 0.237 + 0.079 * cos(6 * theta);

						if (r < r0)
						{
							inside_cnt++;
						}
					}
				}
			}
			if (inside_cnt == 0 || inside_cnt == 8)
				return min_level;
			else
				return max_level;
		}
		__hostdev__ static uint8_t type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk)
		{
			return CellType::INTERIOR;
		}
	};

	class StarSolidGerrisGridCase
	{
	public:
		__hostdev__ static T phi(const Vec& pos)
		{
			T x = pos[0] - 0.5;
			T y = pos[1] - 0.5;
			T z = pos[2] - 0.5;
			T r = sqrt(x * x + y * y + z * z);
			T theta = acos(z / r);
			T phi = atan2(y, x);

			T r0 = 0.237 + 0.079 * cos(6 * theta) * cos(6 * phi);
			return r - r0;
		}
		__hostdev__ static int target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const int min_level, const int max_level)
		{
			auto bbox = acc.tileBBox(info);
			auto bmin = bbox.min();
			auto bmax = bbox.max();
			int inside_cnt = 0;
			for (int di : {0, 1})
			{
				for (int dj : {0, 1})
				{
					for (int dk : {0, 1})
					{
						Vec vpos;
						vpos[0] = bmin[0] + di * (bmax[0] - bmin[0]);
						vpos[1] = bmin[1] + dj * (bmax[1] - bmin[1]);
						vpos[2] = bmin[2] + dk * (bmax[2] - bmin[2]);

						if (phi(vpos) < 0)
						{
							inside_cnt++;
						}
					}
				}
			}
			if (inside_cnt == 0 || inside_cnt == 8)
				return min_level;
			else
				return max_level;
		}

		__hostdev__ static uint8_t type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk)
		{
			auto pos = acc.cellCenter(info, l_ijk);
			auto dx = acc.voxelSize(info);
			Vec bmin(pos[0] - 0.5 * dx, pos[1] - 0.5 * dx, pos[2] - 0.5 * dx);
			const Vec ctr(0.5, 0.5, 0.5);
			constexpr T radius = 0.5 / 2;
			int inside_cnt = 0;
			for (int di : {0, 1})
			{
				for (int dj : {0, 1})
				{
					for (int dk : {0, 1})
					{
						Vec vpos;
						vpos[0] = bmin[0] + di * dx;
						vpos[1] = bmin[1] + dj * dx;
						vpos[2] = bmin[2] + dk * dx;

						if (phi(vpos) < 0)
						{
							inside_cnt++;
						}
					}
				}
			}
			if (inside_cnt == 8)
				return CellType::NEUMANN;
			else
				return CellType::INTERIOR;
		}
	};

	class GerrisSinFunc
	{
	public:
		static constexpr T PI = 3.14159265358979323846;
		static constexpr int k = 3, l = 3;
		__hostdev__ static T f(const Vec pos)
		{
			auto x = pos[0] - 0.5;
			auto y = pos[1] - 0.5;
			return sin(k * PI * x) * sin(l * PI * y);
		}
		__hostdev__ static Vec df(const Vec pos)
		{
			auto x = pos[0] - 0.5;
			auto y = pos[1] - 0.5;
			T dfdx = k * PI * cos(k * PI * x) * sin(l * PI * y);
			T dfdy = l * PI * sin(k * PI * x) * cos(l * PI * y);
			return { dfdx, dfdy, 0 };
		}
		__hostdev__ static T lapf(const Vec pos)
		{
			auto x = pos[0] - 0.5;
			auto y = pos[1] - 0.5;
			return -PI * PI * (k * k + l * l) * sin(k * PI * x) * sin(l * PI * y);
		}
	};
	class AthenaSinFunc
	{
	public:
		// rho0=2 in their paper, but we don't have that
		static constexpr T PI = 3.14159265358979323846, Lx = 1.0, Ly = 1.0, Lz = 1.0, G = 1.0, A = 1.0, phi0 = 0;
		__hostdev__ static T f(const Vec pos)
		{
			auto x = pos[0];
			auto y = pos[1];
			auto z = pos[2];
			// phi = -4*pi*G*A / ( (2*pi/Lx)^2 + (2*pi/Ly)^2 + (2*pi/Lz)^2 )  * sin(2*pi*x/Lx) * sin(2*pi*y/Ly) * sin(2*pi*z/Lz) + phi0
			return -4 * PI * G * A /
				((2 * PI / Lx) * (2 * PI / Lx) + (2 * PI / Ly) * (2 * PI / Ly) + (2 * PI / Lz) * (2 * PI / Lz)) *
				sin(2 * PI * x / Lx) * sin(2 * PI * y / Ly) * sin(2 * PI * z / Lz) +
				phi0;
		}
		__hostdev__ static Vec df(const Vec pos)
		{
			auto x = pos[0];
			auto y = pos[1];
			auto z = pos[2];
			T kx = 2 * PI / Lx;
			T ky = 2 * PI / Ly;
			T kz = 2 * PI / Lz;
			T ksum = kx * kx + ky * ky + kz * kz;
			T dfdx = -4 * PI * G * A / ksum * kx * cos(kx * x) * sin(ky * y) * sin(kz * z);
			T dfdy = -4 * PI * G * A / ksum * ky * sin(kx * x) * cos(ky * y) * sin(kz * z);
			T dfdz = -4 * PI * G * A / ksum * kz * sin(kx * x) * sin(ky * y) * cos(kz * z);
			return { dfdx, dfdy, dfdz };
		}
		__hostdev__ static T lapf(const Vec pos)
		{
			auto x = pos[0];
			auto y = pos[1];
			auto z = pos[2];
			return 4 * PI * G * A * sin(2 * PI * x / Lx) * sin(2 * PI * y / Ly) * sin(2 * PI * z / Lz);
		}
	};

	__device__ bool QueryEffectiveBoundaryDirection(const HATileAccessor<Tile>& acc, const int level, const Coord& g_ijk, int& boundary_axis, int& boundary_off)
	{
		boundary_axis = -1;
		boundary_off = 0;

		for (int axis : {0, 1, 2})
		{
			for (int off : {-1, 1})
			{
				auto ng_ijk = g_ijk;
				ng_ijk[axis] += off;
				HATileInfo<Tile> ninfo;
				Coord nl_ijk;
				acc.findVoxel(level, ng_ijk, ninfo, nl_ijk);
				if (ninfo.empty())
				{
					boundary_axis = axis;
					boundary_off = off;
					return true;
				}
			}
		}
		return false;
	}

	__device__ bool QueryEffectiveBoundaryDirection(const HATileAccessor<Tile>& acc, int chk_level, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk, int& boundary_axis, int& boundary_off)
	{
		chk_level = min(chk_level, info.mLevel);
		int level_diff = info.mLevel - chk_level;

		auto g_ijk = acc.localToGlobalCoord(info, l_ijk);

		g_ijk = Coord(g_ijk[0] >> level_diff, g_ijk[1] >> level_diff, g_ijk[2] >> level_diff);

		// printf("level %d chk level %d diff %d g_ijk %d %d %d\n", info.mLevel, chk_level, level_diff, g_ijk[0], g_ijk[1], g_ijk[2]);
		return QueryEffectiveBoundaryDirection(acc, chk_level, g_ijk, boundary_axis, boundary_off);
	}

	template <class TestFunc>
	void SetAnalyticalTestWithAllDirichletAnalyticalBC(HADeviceGrid<Tile>& grid, const int rhs_channel, const int grdt_channel, const std::string algorithm)
	{
	}

	template <class TestFunc>
	void SetAnalyticalTestWithAllNeumannAnalyticalBC(HADeviceGrid<Tile>& grid, int min_level, int max_level, const int rhs_channel, const int grdt_channel)
	{
		auto cell_type = [=] __device__(const HATileAccessor<Tile> &acc, const HATileInfo<Tile> &info, const nanovdb::Coord & l_ijk) -> uint8_t
		{
			bool is_dirichlet = false;
			bool is_neumann = false;
			acc.iterateSameLevelNeighborVoxels(info, l_ijk,
				[&] __device__(const HATileInfo<Tile> &n_info, const Coord & n_l_ijk, const int axis, const int sgn)
			{
				if (n_info.empty())
				{
					is_neumann = true;
				}
			});
			if (is_neumann)
				return CellType::NEUMANN;
			else if (is_dirichlet)
				return CellType::DIRICHLET;
			else
				return CellType::INTERIOR;
		};
		grid.launchVoxelFuncOnAllTiles(
			[=] __device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk)
		{
			auto& tile = info.tile();
			if (tile.type(l_ijk) == INTERIOR)
			{
				// tile.type(l_ijk) = cell_type(acc, info, l_ijk);
				int boundary_axis, boundary_off;
				if (QueryEffectiveBoundaryDirection(acc, min_level, info, l_ijk, boundary_axis, boundary_off))
				{
					tile.type(l_ijk) = NEUMANN;
				}
				// tile.type(l_ijk) =
			}
		},
			LEAF);
		CalcCellTypesFromLeafs(grid);

		grid.launchVoxelFuncOnAllTiles(
			[=] __device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk)
		{
			auto& tile = info.tile();
			auto pos = acc.cellCenter(info, l_ijk);
			auto h = acc.voxelSize(info);

			if (tile.type(l_ijk) & INTERIOR)
			{
				tile(rhs_channel, l_ijk) = -TestFunc::lapf(pos) * h * h * h;
				tile(grdt_channel, l_ijk) = TestFunc::f(pos);

				// integral of -lap
				T rhs_correction = 0;
				acc.iterateSameLevelNeighborVoxels(info, l_ijk,
					[&] __device__(const HATileInfo<Tile> &ninfo, const Coord & nl_ijk, const int axis, const int sgn)
				{
					Vec npos = acc.cellCenter(ninfo, nl_ijk);
					bool nb_neumann = false, nb_dirichlet = false;
					Vec dirichlet_pos, neumann_pos;

					if (ninfo.empty())
					{
						nb_dirichlet = true;
						dirichlet_pos = npos;
					}
					else if (ninfo.mType == LEAF)
					{
						auto& ntile = ninfo.tile();
						if (ntile.type(nl_ijk) & NEUMANN)
						{
							nb_neumann = true;
							neumann_pos = (pos + npos) * 0.5;
						}
						if (ntile.type(nl_ijk) & DIRICHLET)
						{
							nb_dirichlet = true;
							dirichlet_pos = npos;
						}
					}
					else if (ninfo.mType == NONLEAF)
					{
						auto& ntile = ninfo.tile();
						if (ntile.type(nl_ijk) & NEUMANN)
						{
							nb_neumann = true;
							neumann_pos = (pos + npos) * 0.5;
						}
						if (ntile.type(nl_ijk) & DIRICHLET)
						{
							nb_dirichlet = true;
							dirichlet_pos = npos;
						}

						//{
						//    auto ng_ijk = acc.localToGlobalCoord(ninfo, nl_ijk);
						//    int interior_cnt = 0;
						//    int neumann_cnt = 0;
						//    int dirichlet_cnt = 0;
						//    T children_term = 0;
						//    for (int ci : {0, 1}) {
						//        for (int cj : {0, 1}) {
						//            for (int ck : {0, 1}) {
						//                Coord ncg_ijk(ng_ijk[0] * 2 + ci, ng_ijk[1] * 2 + cj, ng_ijk[2] * 2 + ck);
						//                HATileInfo<Tile> ncinfo; Coord ncl_ijk;
						//                acc.findVoxel(ninfo.mLevel + 1, ncg_ijk, ncinfo, ncl_ijk);
						//                if (!ncinfo.empty()) {
						//                    auto& nctile = ncinfo.tile();
						//                    if (nctile.type(ncl_ijk) & INTERIOR) {
						//                        interior_cnt++;
						//                    }
						//                    else {

						//                        if (nctile.type(ncl_ijk) & NEUMANN) neumann_cnt++;
						//                        if (nctile.type(ncl_ijk) & DIRICHLET) dirichlet_cnt++;

						//                        //basically there is a term (p_i-(pnc0+...pnc7)/8) / h*h^2
						//                        //so there is a rhs term -pnc_j / 8 * h, if it's non-interior
						//                        //but this also has issues.
						//                        //the neumann/dirichlet cells will affect the face coefficient
						//                        //so this correction is again not precise
						//                        children_term -= TestFunc::f(acc.cellCenter(ncinfo, ncl_ijk)) / 8 * h;
						//                    }
						//                }
						//            }
						//        }
						//    }

						//    if (interior_cnt > 0) {
						//        rhs_correction += children_term;
						//    }
						//    else if (dirichlet_cnt > 0) {
						//        nb_dirichlet = true;
						//        dirichlet_pos = npos;
						//    }
						//    else if (neumann_cnt > 0) {
						//        nb_neumann = true;
						//        neumann_pos = (pos + npos) * 0.5;
						//    }
						//}
					}
					else if (ninfo.mType == GHOST)
					{
						auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
						auto pg_ijk = acc.parentCoord(g_ijk);
						Vec ppos = acc.cellCenterGlobal(info.mLevel - 1, pg_ijk);

						auto& ntile = ninfo.tile();
						auto ng_ijk = acc.localToGlobalCoord(ninfo, nl_ijk);
						auto npg_ijk = acc.parentCoord(ng_ijk);
						Vec nppos = acc.cellCenterGlobal(info.mLevel - 1, npg_ijk);

						if (ntile.type(nl_ijk) & NEUMANN)
						{
							nb_neumann = true;
							neumann_pos = (ppos + nppos) * 0.5;
						}
						else if (ntile.type(nl_ijk) & DIRICHLET)
						{
							nb_dirichlet = true;
							dirichlet_pos = nppos;
						}
					}

					if (nb_neumann)
					{
						rhs_correction += (-sgn) * TestFunc::df(neumann_pos)[axis] * h * h;

						//                 {
						// auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
						//                     if (info.mLevel == 3 && g_ijk == Coord(28, 34, 47)) {
						//                         printf("g_ijk = %d %d %d, rhs_correction = %f\n", g_ijk[0], g_ijk[1], g_ijk[2], rhs_correction);
						//                     }
						//                 }
					}
					else if (nb_dirichlet)
					{
						// rhs_correction += (-sgn) * TestFunc::f(dirichlet_pos) * h;
						rhs_correction += TestFunc::f(dirichlet_pos) * h;
					}
				});

				tile(rhs_channel, l_ijk) -= rhs_correction;

				//{
				//    auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
				//    if (info.mLevel == 3 && g_ijk == Coord(28, 34, 47)) {
				//        printf("g_ijk = %d %d %d, rhs = %f laplacian=%f h=%f\n", g_ijk[0], g_ijk[1], g_ijk[2], tile(rhs_channel, l_ijk), TestFunc::lapf(pos), h);
				//    }
				//}
			}
			else
			{
				tile(rhs_channel, l_ijk) = 0;
				tile(grdt_channel, l_ijk) = 0;
			}
		},
			LEAF, 8);
	}

	template <class GridCase>
	std::shared_ptr<HADeviceGrid<Tile>> CreateTestGridCase(const std::string grid_name, const int min_level, const int max_level)
	{
		uint32_t scale = 8;
		float h = 1.0 / scale;
		// 0:8, 1:16, 2:32, 3:64, 4:128, 5:256, 6:512, 7:1024
		auto grid_ptr = std::make_shared<HADeviceGrid<Tile>>(h, thrust::host_vector{ 16, 16, 16, 16, 16, 16, 20, 20, 16, 16 });
		auto& grid = *grid_ptr;
		grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
		grid.rebuild();
		grid.iterativeRefine([=] __device__(const HATileAccessor<Tile> &acc, HATileInfo<Tile> &info)
		{
			return GridCase::target(acc, info, min_level, max_level);
		}, false);
		int num_cells = grid.numTotalLeafTiles() * Tile::SIZE;
		int total_hash_bytes = grid.hashTableDeviceBytes();
		Info("Total {}M cells, hash table {}GB", num_cells / (1024.0 * 1024), total_hash_bytes / (1024.0 * 1024 * 1024));

		grid.launchVoxelFuncOnAllTiles(
			[=] __device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk)
		{
			auto& tile = info.tile();
			tile.type(l_ijk) = GridCase::type(acc, info, l_ijk);
		},
			LEAF);
		return grid_ptr;
	}

	// return <grid, is_pure_neumann>
	std::tuple<std::shared_ptr<HADeviceGrid<Tile>>, bool> CreateTestGrid(const std::string grid_name, const int min_level, const int max_level, const std::string bc_name, const int rhs_channel, const int grdt_channel)
	{
		std::shared_ptr<HADeviceGrid<Tile>> grid_ptr;
		if (grid_name == "uniform")
		{
			grid_ptr = CreateTestGridCase<UniformGridCase>(grid_name, min_level, max_level);
		}
		else if (grid_name == "sphere_shell_05")
		{
			grid_ptr = CreateTestGridCase<SphereShell05GridCase>(grid_name, min_level, max_level);
		}
		else if (grid_name == "sphere_solid_05")
		{
			grid_ptr = CreateTestGridCase<SphereSolid05GridCase>(grid_name, min_level, max_level);
		}
		else if (grid_name == "star_shell")
		{
			grid_ptr = CreateTestGridCase<StarShellGerrisGridCase>(grid_name, min_level, max_level);
		}
		else
		{
			Assert(false, "grid_name {} not supported", grid_name);
		}
		auto& grid = *grid_ptr;

		bool is_pure_neumann = false;

		if (bc_name == "gerris_sin")
		{
			is_pure_neumann = true;
			SetAnalyticalTestWithAllNeumannAnalyticalBC<GerrisSinFunc>(grid, min_level, max_level, rhs_channel, grdt_channel);
		}
		else if (bc_name == "athena_sin")
		{
			is_pure_neumann = true;
			SetAnalyticalTestWithAllNeumannAnalyticalBC<AthenaSinFunc>(grid, min_level, max_level, rhs_channel, grdt_channel);
		}
		else
		{
			Assert(false, "bc_name {} not supported", bc_name);
		}

		CalcCellTypesFromLeafs(grid);
		CalculateNeighborTiles(grid);

		//     {
		//         auto holder = grid.getHostTileHolderForLeafs();
		//         polyscope::init();
		// IOFunc::AddPoissonGridCellCentersToPolyscopePointCloud(holder,
		//	{ {-1,"type"}, {rhs_channel, "rhs"}, {grdt_channel,"grdt"} },
		//	{}
		//);
		// polyscope::show();
		//         IOFunc::OutputPoissonGridAsStructuredVTI(
		//             holder,
		//             { {-1, "type"}, {rhs_channel, "rhs"}, {grdt_channel, "grdt"} },
		//             {  },
		//             fmt::format("output/{}_input.vti", ToString(grid_name))
		//         );
		//     }

		return std::make_tuple(grid_ptr, is_pure_neumann);
	}

	class MultiGridParams {
	public:
		std::string algorithm = "amg";
		int mu_repeat_times = 1;
		int level_iters = 1;
		int bottom_iters = 10;
		double omega = 1.5;
	};

	//solve from b channel to x channel
	//return: iters, rel error, solve time (microseconds)
	std::tuple<int, double, double> SolveLinearSystem(HADeviceGrid<Tile>& grid, const int coeff_channel, bool is_pure_neumann, int max_iters, double rel_tolerance, int sync_stride, const MultiGridParams params, bool verbose) {
		CalculateNeighborTiles(grid);

		if (params.algorithm == "cmg") {
			CMGSolver solver(1.0, 1.0);
			CPUTimer<std::chrono::microseconds> timer;
			timer.start();
			auto [iters, err] = solver.solve(grid, verbose, max_iters, rel_tolerance, params.level_iters, params.bottom_iters, sync_stride, is_pure_neumann);
			cudaDeviceSynchronize();
			CheckCudaError("CMG solve");
			double elapsed = timer.stop("CMG Solve");
			return std::make_tuple(iters, err, elapsed);
		}
		else if (params.algorithm == "amg") {
			AMGSolver solver(coeff_channel, 0.5, 1, 1);
			solver.omega = params.omega;
			solver.mu_cycle_repeat_times = params.mu_repeat_times;
			solver.prepareTypesAndCoeffs(grid);

			CPUTimer<std::chrono::microseconds> timer;
			timer.start();
			auto [iters, err] = solver.solve(grid, verbose, max_iters, rel_tolerance, params.level_iters, params.bottom_iters, sync_stride, is_pure_neumann);
			cudaDeviceSynchronize();
			CheckCudaError("AMG solve");
			double elapsed = timer.stop("AMG Solve");
			return std::make_tuple(iters, err, elapsed);
		}
		else if (params.algorithm == "amg_vcycle") {
			AMGSolver solver(coeff_channel, 0.5, 0.5, 1);
			solver.omega = params.omega;
			solver.mu_cycle_repeat_times = params.mu_repeat_times;
			solver.prepareTypesAndCoeffs(grid);

			CPUTimer<std::chrono::microseconds> timer;
			timer.start();
			auto [iters, err] = solver.FASMuCycleSolve(params.mu_repeat_times, grid, verbose, max_iters, rel_tolerance, params.level_iters, params.bottom_iters);
			CheckCudaError("FAS MuCycle Solve");
			float elapsed = timer.stop("FAS MuCycle");

			Info("iter {} err {}", iters, err);

			return std::make_tuple(iters, err, elapsed);
		}
		else {
			Assert(false, "SolveLinearSystem algorithm {} not supported", params.algorithm);
		}
	}

	void TestIterativeConvergenceWithAnalyticalSolution(HADeviceGrid<Tile>& grid, const std::string algorithm, const int coeff_channel, const int b0_channel, const int grdt_channel, int error_channel, bool is_pure_neumann)
	{
		Info("Test Iter-Error with analytical solution on algorithm {}", algorithm);

		grid.launchVoxelFuncOnAllTiles(
			[=] __device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk)
		{
			auto& tile = info.tile();
			tile(b0_channel, l_ijk) = tile(Tile::b_channel, l_ijk);
		},
			LEAF);

		auto solve_system = [&](const std::string algorithm, int max_iters)->std::tuple<int,double> {
			if (algorithm == "cmg") {

			}
			else if (algorithm == "amg") {
				AMGSolver solver(coeff_channel, 0.5, 1, 1);
				solver.omega = 1;
				solver.mu_cycle_repeat_times = 1;
				solver.prepareTypesAndCoeffs(grid);
				// AMGFullNegativeLaplacianOnLeafs(grid, grdt_channel, coeff_channel, Tile::b_channel);

				_sleep(200);

				CPUTimer<std::chrono::microseconds> timer;
				timer.start();
				auto [iters, err] = solver.solve(grid, false, max_iters, 0.0, 2, 10, 1, is_pure_neumann);
				CheckCudaError("AMGPCG solve");
				return std::make_pair(iters, err);
			}
			else {
				Assert(false, "TestIterativeConvergenceWithAnalyticalSolution algorithm {} not supported", algorithm);
			}
			};

		for (int max_iters = 0;; max_iters++)
		{
			CalculateNeighborTiles(grid);

			grid.launchVoxelFuncOnAllTiles(
				[=] __device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk)
			{
				auto& tile = info.tile();
				tile(Tile::b_channel, l_ijk) = tile(b0_channel, l_ijk);
			},
				LEAF);

			MultiGridParams params;
			params.algorithm = algorithm;
			params.omega = 1.0;
			params.mu_repeat_times = 1;
			params.level_iters = 2;
			params.bottom_iters = 10;
			
			auto [iters, err, elapsed] = SolveLinearSystem(grid, coeff_channel, is_pure_neumann, max_iters, 0, 1, params, false);

			//auto [iters, err] = solve_system(algorithm, max_iters);


			grid.launchVoxelFuncOnAllTiles(
				[=] __device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk)
			{
				auto& tile = info.tile();
				if (tile.type(l_ijk) & INTERIOR)
				{
					tile(error_channel, l_ijk) = tile(grdt_channel, l_ijk) - tile(Tile::x_channel, l_ijk);
				}
				else
				{
					tile(error_channel, l_ijk) = 0;
				}
			},
				LEAF);
			Info("Solve {} iters, linf {} weighted L1 {} weighted L2 {} full weighted L2 {} pointwise L2 {}",
				iters,
				NormSync(grid, -1, error_channel, false, INTERIOR),
				NormSync(grid, 1, error_channel, true, INTERIOR),
				NormSync(grid, 2, error_channel, true, INTERIOR),
				NormSync(grid, 2, error_channel, true, INTERIOR | NEUMANN | DIRICHLET),
				NormSync(grid, 2, error_channel, false, INTERIOR)

			);

			if (err < 1e-6)
				break;
		}
	}

	void TestSolverWithAnalyticalSolution(HADeviceGrid<Tile>& grid, const std::string algorithm, const double omega, const int coeff_channel, const int grdt_channel, int error_channel, bool is_pure_neumann)
	{
		MultiGridParams params;
		params.algorithm = algorithm;
		params.omega = omega;
		params.mu_repeat_times = 1;
		params.level_iters = 2;
		params.bottom_iters = 10;


		auto [iters, err, elapsed] = SolveLinearSystem(grid, coeff_channel, is_pure_neumann, 30, 1e-6, 1, params, true);
		//auto [iters, err, elapsed] = SolveLinearSystem(grid, coeff_channel, is_pure_neumann, 6, 1e-6, -1, params, false);
		int total_cells = grid.numTotalLeafTiles() * Tile::SIZE;
		float cells_per_second = (total_cells + 0.0) / (elapsed / 1e6);
		Info("Total {:.5}M cells, {} speed {:.5} M cells /s", total_cells / (1024.0 * 1024), algorithm, cells_per_second / (1024.0 * 1024));
		Info("{} solved in {} iterations with error {}, average iteration throughput {:.5}M cell/s", algorithm, iters, err, cells_per_second * iters / (1024.0 * 1024));

		// calculate difference from x and grdt in r_channel
		grid.launchVoxelFuncOnAllTiles(
			[=] __device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk)
		{
			auto& tile = info.tile();
			if (tile.type(l_ijk) & INTERIOR)
			{
				tile(error_channel, l_ijk) = tile(grdt_channel, l_ijk) - tile(Tile::x_channel, l_ijk);
			}
			else
			{
				tile(error_channel, l_ijk) = 0;
			}
		},
			LEAF);

		//{
		//    auto holder = grid.getHostTileHolder(LEAF);
		//    polyscope::init();
		//    IOFunc::AddLeveledPoissonGridCellCentersToPolyscopePointCloud(holder,
		//        {
		//        {-1, "type"}, {Tile::x_channel,"x"} ,{grdt_channel,"grdt"},{Tile::r_channel, "r"}
		//        },
		//        {}, -1, FLT_MAX);
		//    polyscope::show();
		//}

		//     {
		// auto holder = grid.getHostTileHolderForLeafs();
		//         IOFunc::OutputPoissonGridAsStructuredVTI(
		//             holder,
		//             { {-1, "type"}, {Tile::b_channel, "rhs"}, {grdt_channel, "grdt"}, {Tile::x_channel, "x"}, {Tile::r_channel, "r"} },
		//             {  },
		//             fmt::format("output/{}_result.vti", algorithm)
		//         );
		//     }

		Info("linf: {}", NormSync(grid, -1, error_channel, false));
		Info("full weighted L1: {}", NormSync(grid, 1, error_channel, true, INTERIOR | NEUMANN | DIRICHLET));
		Info("full weighted L2: {}", NormSync(grid, 2, error_channel, true, INTERIOR | NEUMANN | DIRICHLET));

		auto weighted_rms_error = NormSync(grid, 2, error_channel, true, INTERIOR | NEUMANN | DIRICHLET);
		if (weighted_rms_error < 1e-4)
		{
			Pass("Test passed with weighted_rms_error of grdt-x: {}\n\n", weighted_rms_error);
		}
		else
		{
			Warn("Test failed with weighted_rms_error of grdt-x: {}\n\n", weighted_rms_error);
		}
	}

	void TestDiscretizedLaplacian(const std::string grid_name, const int min_level, const int max_level, const std::string bc_name)
	{
		int x_channel = 0;
		int analytical_nlap_channel = 1;
		int numerical_nlap_channel = 2;
		int residual_channel = 3;
		int coeff_channel = 6;

		auto [grid_ptr, is_pure_neumann] = CreateTestGrid(grid_name, min_level, max_level, bc_name, analytical_nlap_channel, x_channel);
		auto& grid = *grid_ptr;

		{
			CalculateNeighborTiles(grid);

			AMGSolver solver(coeff_channel, 0.5, 1, 1);
			solver.prepareTypesAndCoeffs(grid);
			AMGFullNegativeLaplacianOnLeafs(grid, x_channel, coeff_channel, numerical_nlap_channel);
		}

		// calculate difference from x and grdt in r_channel
		grid.launchVoxelFuncOnAllTiles(
			[=] __device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk)
		{
			auto& tile = info.tile();
			if (tile.type(l_ijk) & INTERIOR)
			{
				auto h = acc.voxelSize(info);
				tile(residual_channel, l_ijk) = (tile(analytical_nlap_channel, l_ijk) - tile(numerical_nlap_channel, l_ijk)) / (h * h * h);
			}
			else
			{
				tile(residual_channel, l_ijk) = 0;
			}
		},
			LEAF);

		auto holder = grid.getHostTileHolder(LEAF);
		polyscope::init();
		IOFunc::AddLeveledPoissonGridCellCentersToPolyscopePointCloud(holder,
			{ {-1, "type"}, {x_channel, "x"}, {analytical_nlap_channel, "analytical"}, {numerical_nlap_channel, "numerical"}, {residual_channel, "residual"} },
			{}, -1, FLT_MAX);
		polyscope::show();

		{
			auto holder = grid.getHostTileHolderForLeafs();
			IOFunc::OutputPoissonGridAsStructuredVTI(
				holder,
				{ {-1, "type"}, {analytical_nlap_channel, "analytical"}, {numerical_nlap_channel, "numerical"}, {residual_channel, "r"}, {x_channel, "x"} },
				{},
				fmt::format("output/laplacian_test_{}_levels_{}_{}.vti", grid_name, min_level, max_level));
		}

		Info("Test Laplacian linf error on grid {} levels {}~{}", grid_name, min_level, max_level);
		// auto linf_norm = SingleChannelLinfSync(grid, residual_channel, LEAF);
		auto linf_norm = NormSync(grid, -1, residual_channel, false);
		if (linf_norm < 1e-4)
		{
			Pass("Test passed with Linf norm of grdt-x: {}\n\n", linf_norm);
		}
		else
		{
			Warn("Test failed with Linf norm of grdt-x: {}\n\n", linf_norm);
		}
	}

	void TestSolverErrorWithAllNeumannBC(const std::string grid_name, const double omega, const int min_level, const int max_level, const std::string bc_name, const std::string algorithm)
	{
		Info("TestSolverErrorWithAllNeumannBC on grid {}, min level {} max level {} with given function of {} for {}", grid_name, min_level, max_level, bc_name, algorithm);

		int grdt_channel = 5;
		int coeff_channel = 6;             // 6,7,8,9
		int rhs_channel = Tile::b_channel; // 1
		int b0_channel = 10;
		// x:0
		int error_channel = 2;

		auto [grid_ptr, is_pure_neumann] = CreateTestGrid(grid_name, min_level, max_level, bc_name, rhs_channel, grdt_channel);
		auto& grid = *grid_ptr;

		TestSolverWithAnalyticalSolution(grid, algorithm, omega, coeff_channel, grdt_channel, error_channel, is_pure_neumann);
		// TestIterativeConvergenceWithAnalyticalSolution(grid, algorithm, coeff_channel, b0_channel, grdt_channel, error_channel, is_pure_neumann);


		fmt::print("\n");

		//{
		//    auto holder = grid.getHostTileHolderForLeafs();


		//    //IOFunc::OutputPoissonGridAsStructuredVTI(
		//    //    holder,
		//    //    { {-2,"level"}, { -1, "type" }, {rhs_channel, "rhs"}, {grdt_channel, "grdt"}, {Tile::x_channel, "x"}, {Tile::r_channel, "r"} },
		//    //    {  },
		//    //    fmt::format("output/analytical_{}_levels{}_{}_{}_{}.vti", grid_name, min_level, max_level, bc_name, algorithm)
		//    //);

		//	polyscope::init();
		//	IOFunc::AddPoissonGridCellCentersToPolyscopePointCloud(
		//		holder,
		//		{ {-1, "type"}, {rhs_channel, "rhs"}, {grdt_channel, "grdt"}, {Tile::x_channel, "x"}, {error_channel, "error"} },
		//		{}
		//	);
		//	polyscope::show();

		//	Info("test: {}", holder->cellValue(6, Coord(331,371,331), grdt_channel));//0.0670311

		//}

		// IOFunc::OutputTilesAsVTU(holder, "output/tiles.vtu");
	}

	void TestSolutionItersErrorWithAllNeumannBC(const std::string grid_name, const double omega, const int min_level, const int max_level, const std::string bc_name, const std::string algorithm)
	{
		Info("TestSolutionItersErrorWithAllNeumannBC on grid {}, min level {} max level {} with given function of {} for {}", grid_name, min_level, max_level, bc_name, algorithm);

		int grdt_channel = 5;
		int coeff_channel = 6;             // 6,7,8,9
		int rhs_channel = Tile::b_channel; // 1
		int b0_channel = 10;
		// x:0
		int error_channel = 2;

		auto [grid_ptr, is_pure_neumann] = CreateTestGrid(grid_name, min_level, max_level, bc_name, rhs_channel, grdt_channel);
		auto& grid = *grid_ptr;

		Info("Test Iter-Error with analytical solution on algorithm {}", algorithm);

		grid.launchVoxelFuncOnAllTiles(
			[=] __device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk)
		{
			auto& tile = info.tile();
			tile(b0_channel, l_ijk) = tile(Tile::b_channel, l_ijk);
		},
			LEAF);

		for (int max_iters = 0; max_iters <= 20; max_iters++)
		{
			CalculateNeighborTiles(grid);

			grid.launchVoxelFuncOnAllTiles(
				[=] __device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk)
			{
				auto& tile = info.tile();
				tile(Tile::b_channel, l_ijk) = tile(b0_channel, l_ijk);
			},
				LEAF);

			MultiGridParams params;
			params.algorithm = algorithm;
			params.omega = 1.0;
			params.mu_repeat_times = 1;
			params.level_iters = 2;
			params.bottom_iters = 10;

			auto [iters, err, elapsed] = SolveLinearSystem(grid, coeff_channel, is_pure_neumann, max_iters, 0, -1, params, false);

			//auto [iters, err] = solve_system(algorithm, max_iters);


			grid.launchVoxelFuncOnAllTiles(
				[=] __device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk)
			{
				auto& tile = info.tile();
				if (tile.type(l_ijk) & INTERIOR)
				{
					tile(error_channel, l_ijk) = tile(grdt_channel, l_ijk) - tile(Tile::x_channel, l_ijk);
				}
				else
				{
					tile(error_channel, l_ijk) = 0;
				}
			},
				LEAF);
			Info("Solve {} iters, linf {} weighted L1 {} weighted L2 {} full weighted L2 {} pointwise L2 {}",
				iters,
				NormSync(grid, -1, error_channel, false, INTERIOR),
				NormSync(grid, 1, error_channel, true, INTERIOR),
				NormSync(grid, 2, error_channel, true, INTERIOR),
				NormSync(grid, 2, error_channel, true, INTERIOR | NEUMANN | DIRICHLET),
				NormSync(grid, 2, error_channel, false, INTERIOR)

			);

			//if (err < 1e-6)
			//	break;
		}


		fmt::print("\n");

	}


	void TestIterativeResidualReduction(HADeviceGrid<Tile>& grid, const std::string algorithm, const int max_iters, const int b0_channel, const int coeff_channel,  const int final_x_channel, bool is_pure_neumann)
	{
		Info("Iteratively smooth out residual for algorithm {}", algorithm);

		CalculateNeighborTiles(grid);
		grid.launchVoxelFuncOnAllTiles(
			[=] __device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk)
		{
			auto& tile = info.tile();
			tile(final_x_channel, l_ijk) = 0;
		},
			LEAF | NONLEAF | GHOST);

		auto calc_residual = [&](int x_channel, int rhs_channel, int r_channel) {
			AMGFullNegativeLaplacianOnLeafs(grid, x_channel, coeff_channel, r_channel);
			grid.launchVoxelFuncOnAllTiles(
				[=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk)
			{
				auto& tile = info.tile();
				tile(r_channel, l_ijk) = tile(rhs_channel, l_ijk) - tile(r_channel, l_ijk);
			},
				LEAF);
			};

		if (algorithm == "amg_vcycle") {
			AMGSolver solver(coeff_channel, 0.5, 0.5, 1);
			solver.omega = 2. / 3;
			solver.prepareTypesAndCoeffs(grid);
			int x_channel = Tile::x_channel;//0
			int b_channel = Tile::b_channel;//1
			int x0_channel = 2;
			int b_copy_channel = 4;
			int iter_r_channel = 3;

			for (int iter = 1; iter <= max_iters; iter++) {
				//calculate residual to Tile::b_channel for this iteration
				calc_residual(final_x_channel, b0_channel, b_channel);
				grid.launchVoxelFuncOnAllTiles(
					[=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
					auto& tile = info.tile();
					tile(x_channel, l_ijk) = 0;

					tile(b_channel, l_ijk) *= 1000;

					tile(b_copy_channel, l_ijk) = tile(b_channel, l_ijk);
				}, LEAF);

				double pt_l2 = NormSync(grid, 2, b_channel, false);
				Info("Residual pointwise L2 norm at step {}: {}", iter - 1, pt_l2);

				solver.FASMuCycleStep(grid.mMaxLevel, 1, grid, x_channel, b_channel, x0_channel, coeff_channel, 1, 10);

				{
					calc_residual(x_channel, b_copy_channel, iter_r_channel);
					Info("after FAS iter {} residual pointwise L2 norm: {}\n", iter, NormSync(grid, 2, iter_r_channel, false));

					auto holder = grid.getHostTileHolder(LEAF);
					polyscope::init();
					IOFunc::AddLeveledPoissonGridCellCentersToPolyscopePointCloud(
						holder,
						{ {-1,"type"}, {b_copy_channel, "iter rhs"}, {x_channel, "iter x"}, {iter_r_channel, "iter r"},
						{coeff_channel + 0, "x-"}, {coeff_channel + 1,"y-"},{coeff_channel + 2,"z-"},{coeff_channel + 3,"diag"}
						},
						{}
					);
					polyscope::show();
				}

				//accumulate x to final_x_channel
				grid.launchVoxelFuncOnAllTiles(
					[=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
					auto& tile = info.tile();
					//tile(final_x_channel, l_ijk) += tile(x_channel, l_ijk);
					tile(final_x_channel, l_ijk) += tile(x_channel, l_ijk) / 1000.0;
				}, LEAF);
			}
			Info("Residual pointwise L2 norm at step {}: {}", max_iters, NormSync(grid, 2, iter_r_channel, false));
		}
		else {
			Assert(false, "TestIterativeResidualReduction algorithm {} not supported", algorithm);
		}
	}

	__hostdev__ float FracInside(float a, float b)
	{
		if (a < 0.0 && b < 0.0)
			return 0.0;
		else if (a < 0.0 && b >= 0.0)
			return b / (b - a);
		else if (a >= 0.0 && b < 0.0)
			return a / (a - b);
		else
			return 1.0;
	}

	__hostdev__ float FaceFluidRatio(float phi0, float phi1, float phi2, float phi3)
	{
		// calculate vol
		float ret;
		if (phi0 < 0 && phi1 < 0 && phi2 < 0 && phi3 < 0)
		{
			ret = 0;
		}
		else if (phi0 < 0 && phi1 < 0 && phi2 < 0 && phi3 >= 0)
		{
			float edge1 = FracInside(phi3, phi2);
			float edge2 = FracInside(phi3, phi0);
			ret = 0.5 * edge1 * edge2;
		}
		else if (phi0 < 0 && phi1 < 0 && phi2 >= 0 && phi3 < 0)
		{
			float edge1 = FracInside(phi2, phi1);
			float edge2 = FracInside(phi2, phi3);
			ret = 0.5 * edge1 * edge2;
		}
		else if (phi0 < 0 && phi1 < 0 && phi2 >= 0 && phi3 >= 0)
		{
			float edge1 = FracInside(phi2, phi1);
			float edge2 = FracInside(phi3, phi0);
			ret = 0.5 * (edge1 + edge2);
		}
		else if (phi0 < 0 && phi1 >= 0 && phi2 < 0 && phi3 < 0)
		{
			float edge1 = FracInside(phi1, phi0);
			float edge2 = FracInside(phi1, phi2);
			ret = 0.5 * edge1 * edge2;
		}
		else if (phi0 < 0 && phi1 >= 0 && phi2 < 0 && phi3 >= 0)
		{
			float edge1 = FracInside(phi1, phi0);
			float edge2 = FracInside(phi1, phi2);
			float edge3 = FracInside(phi3, phi2);
			float edge4 = FracInside(phi3, phi0);
			ret = 0.5 * edge1 * edge2 + 0.5 * edge3 * edge4;
		}
		else if (phi0 < 0 && phi1 >= 0 && phi2 >= 0 && phi3 < 0)
		{
			float edge1 = FracInside(phi1, phi0);
			float edge2 = FracInside(phi2, phi3);
			ret = 0.5 * (edge1 + edge2);
		}
		else if (phi0 < 0 && phi1 >= 0 && phi2 >= 0 && phi3 >= 0)
		{
			float edge1 = 1.0 - FracInside(phi0, phi1);
			float edge2 = 1.0 - FracInside(phi0, phi3);
			ret = 1.0 - 0.5 * edge1 * edge2;
		}
		else if (phi0 >= 0 && phi1 < 0 && phi2 < 0 && phi3 < 0)
		{
			float edge1 = FracInside(phi0, phi1);
			float edge2 = FracInside(phi0, phi3);
			ret = 0.5 * edge1 * edge2;
		}
		else if (phi0 >= 0 && phi1 < 0 && phi2 < 0 && phi3 >= 0)
		{
			float edge1 = FracInside(phi0, phi1);
			float edge2 = FracInside(phi3, phi2);
			ret = 0.5 * (edge1 + edge2);
		}
		else if (phi0 >= 0 && phi1 < 0 && phi2 >= 0 && phi3 < 0)
		{
			float edge1 = FracInside(phi0, phi1);
			float edge2 = FracInside(phi0, phi3);
			float edge3 = FracInside(phi2, phi1);
			float edge4 = FracInside(phi2, phi3);
			ret = 0.5 * edge1 * edge2 + 0.5 * edge3 * edge4;
		}
		else if (phi0 >= 0 && phi1 < 0 && phi2 >= 0 && phi3 >= 0)
		{
			float edge1 = 1.0 - FracInside(phi1, phi0);
			float edge2 = 1.0 - FracInside(phi1, phi2);
			ret = 1.0 - 0.5 * edge1 * edge2;
		}
		else if (phi0 >= 0 && phi1 >= 0 && phi2 < 0 && phi3 < 0)
		{
			float edge1 = FracInside(phi0, phi3);
			float edge2 = FracInside(phi1, phi2);
			ret = 0.5 * (edge1 + edge2);
		}
		else if (phi0 >= 0 && phi1 >= 0 && phi2 < 0 && phi3 >= 0)
		{
			float edge1 =1.0 - FracInside(phi2, phi1);
			float edge2 =1.0 - FracInside(phi2, phi3);
			ret = 1.0 - 0.5 * edge1 * edge2;
		}
		else if (phi0 >= 0 && phi1 >= 0 && phi2 >= 0 && phi3 < 0)
		{
			float edge1 = 1.0 - FracInside(phi3, phi0);
			float edge2 = 1.0 - FracInside(phi3, phi2);
			ret = 1.0 - 0.5 * edge1 * edge2;
		}
		else
		{
			ret = 1;
		}
		if (ret < 0.1 && ret != 0)
			ret = 0.1;
		return ret;
	}

	void TestSolverErrorSolid(const std::string grid_name, const int min_level, const int max_level)
	{
		std::shared_ptr<HADeviceGrid<Tile>> grid_ptr;
		// grid with type
		if (grid_name == "sphere")
		{
			grid_ptr = CreateTestGridCase<SphereSolid05GridCase>(grid_name, min_level, max_level);
		}
		else
		{
			Assert(false, "grid_name {} not supported", grid_name);
		}
		auto& grid = *grid_ptr;
		bool is_pure_neumann = false;

		grid.launchVoxelFuncOnAllTiles(
			[=] __device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk)
		{
			auto& tile = info.tile();
			if (tile.type(l_ijk) == INTERIOR)
			{
				int boundary_axis, boundary_off;
				if (QueryEffectiveBoundaryDirection(acc, min_level, info, l_ijk, boundary_axis, boundary_off))
				{
					if (boundary_axis == 1 && boundary_off == 1)
						tile.type(l_ijk) = DIRICHLET;
					else
						tile.type(l_ijk) = NEUMANN;
				}
			}
		},
			LEAF);
		CalcCellTypesFromLeafs(grid);
		CalculateNeighborTiles(grid);

		// amg with coef
		int coeff_channel = 6;
		float R_matrix_coeff = 0.5;
		AMGSolver solver(coeff_channel, R_matrix_coeff, 1, 1);
		// step 0: clear coeff channel for all tiles
		grid.launchVoxelFuncOnAllTiles(
			[=] __device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk)
		{
			auto& tile = info.tile();
			for (int axis : {0, 1, 2})
			{
				tile(coeff_channel + axis, l_ijk) = 0;
			}
			tile(coeff_channel + 3, l_ijk) = 0;
		},
			LEAF | GHOST | NONLEAF);
		// step 1: fill GHOST cell types
		grid.launchVoxelFuncOnAllTiles(
			[=] __device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk)
		{
			auto& tile = info.tile();
			auto g_ijk = acc.composeGlobalCoord(info.mTileCoord, l_ijk);
			auto pg_ijk = acc.parentCoord(g_ijk);
			HATileInfo<Tile> pinfo;
			Coord pl_ijk;
			acc.findVoxel(info.mLevel - 1, pg_ijk, pinfo, pl_ijk);
			if (!pinfo.empty())
				tile.type(l_ijk) = pinfo.tile().type(pl_ijk);
			else
				tile.type(l_ijk) = DIRICHLET;
		},
			GHOST);
		// step 2: calculate face terms on LEAF and GHOST cells
		grid.launchVoxelFuncOnAllTiles(
			[=] __device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk)
		{
			auto h = acc.voxelSize(info);
			Tile& tile = info.tile();
			uint8_t ttype0 = info.mType;
			uint8_t ctype0 = tile.type(l_ijk);
			auto cell_center = acc.cellCenter(info, l_ijk);
			// iterate neighbors
			acc.iterateSameLevelNeighborVoxels(info, l_ijk,
				[&] __device__(const HATileInfo<Tile> &ninfo, const Coord & nl_ijk, const int axis, const int sgn)
			{
				if (sgn != -1)
					return;
				T coeff = 0;

				uint8_t ttype1;
				uint8_t ctype1;
				if (ninfo.empty())
				{
					ttype1 = ttype0;
					ctype1 = DIRICHLET;
				}
				else
				{
					auto& ntile = ninfo.tile();
					ttype1 = ninfo.mType;
					ctype1 = ntile.type(nl_ijk);
				}
				bool both_leafs = ((ttype0 & LEAF) && (ttype1 & LEAF));
				bool one_leaf_one_ghost = ((ttype0 & LEAF && ttype1 & GHOST) || (ttype0 & GHOST && ttype1 & LEAF));
				bool has_neumann = (ctype0 & NEUMANN || ctype1 & NEUMANN);
				bool has_interior = (ctype0 & INTERIOR || ctype1 & INTERIOR);
				if ((both_leafs || one_leaf_one_ghost) && !has_neumann && has_interior)
				{
					Vec face_corner[4];
					if (axis == 0)
					{
						face_corner[0] = cell_center + Vec(-0.5, -0.5, -0.5) * h;
						face_corner[1] = cell_center + Vec(-0.5, -0.5, 0.5) * h;
						face_corner[2] = cell_center + Vec(-0.5, 0.5, 0.5) * h;
						face_corner[3] = cell_center + Vec(-0.5, 0.5, -0.5) * h;
					}
					else if (axis == 1)
					{
						face_corner[0] = cell_center + Vec(-0.5, -0.5, -0.5) * h;
						face_corner[1] = cell_center + Vec(-0.5, -0.5, 0.5) * h;
						face_corner[2] = cell_center + Vec(0.5, -0.5, 0.5) * h;
						face_corner[3] = cell_center + Vec(0.5, -0.5, -0.5) * h;
					}
					else if (axis == 2)
					{
						face_corner[0] = cell_center + Vec(-0.5, -0.5, -0.5) * h;
						face_corner[1] = cell_center + Vec(-0.5, 0.5, -0.5) * h;
						face_corner[2] = cell_center + Vec(0.5, 0.5, -0.5) * h;
						face_corner[3] = cell_center + Vec(0.5, -0.5, -0.5) * h;
					}
					float phi[4];
					for (int i = 0; i < 4; i++)
						phi[i] = SphereSolid05GridCase::phi(face_corner[i]);
					coeff = h * FaceFluidRatio(phi[0], phi[1], phi[2], phi[3]);
				}
				tile(coeff_channel + axis, l_ijk) = -coeff;
			});
		},
			LEAF | GHOST);

		//step 3: accumulate face terms
		for (int i = grid.mMaxLevel; i >= 0; i--) {
			int num_tiles = grid.hNumTiles[i];
			if (num_tiles > 0) {
				CoarsenOffDiagCoefficientsOneStepKernel << <num_tiles, 128 >> > (
					grid.deviceAccessor(),
					thrust::raw_pointer_cast(grid.dTileArrays[i].data()),
					i,
					LEAF | GHOST,
					coeff_channel,
					coeff_channel,
					R_matrix_coeff,
					INTERIOR | DIRICHLET | NEUMANN
					);
			}
		}

		//step 4: calculate diagonal terms for LEAF cells and coarsen diagonal terms for NONLEAF cells
		grid.launchVoxelFunc(
			[=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
			auto& tile = info.tile();
			auto g_ijk = acc.composeGlobalCoord(info.mTileCoord, l_ijk);

			if (info.mType & (LEAF | GHOST)) {
				auto h = acc.voxelSize(info);
				T diag_coeff = 0;
				if (tile.type(l_ijk) & INTERIOR) {
					acc.iterateSameLevelNeighborVoxels(info, l_ijk,
						[&]__device__(const HATileInfo<Tile>&ninfo, const Coord & nl_ijk, const int axis, const int sgn) {
						T c0 = tile(coeff_channel + axis, l_ijk);
						//if ninfo if empty, we regard it as DIRICHLET and has coeff -h
						T c1 = ninfo.empty() ? -h : ninfo.tile()(coeff_channel + axis, nl_ijk);
						T face_term = (sgn == -1) ? c0 : c1;

						//if (ninfo.mType == GHOST) face_term /= 8;
						//if (ninfo.mType == GHOST) face_term /= 4;

						diag_coeff -= face_term;
					});
				}
				tile(coeff_channel + 3, l_ijk) = diag_coeff;
			}
			else {
				//coarsen diagonal terms at NONLEAF cells
				int interior_cnt = 0;
				bool has_ghost_child = false;
				uint8_t ctypes[2][2][2];
				T coff_diag[3][2][2][2];//8 children
				T diag_sum = 0;
				for (int ci = 0; ci < 2; ci++) {
					for (int cj = 0; cj < 2; cj++) {
						for (int ck = 0; ck < 2; ck++) {
							Coord cg_ijk(g_ijk[0] * 2 + ci, g_ijk[1] * 2 + cj, g_ijk[2] * 2 + ck);
							HATileInfo<Tile> cinfo; Coord cl_ijk;
							acc.findVoxel(info.mLevel + 1, cg_ijk, cinfo, cl_ijk);
							if (!cinfo.empty()) {
								if (cinfo.mType == GHOST) has_ghost_child = true;
								auto& ctile = cinfo.tile();
								diag_sum += ctile(coeff_channel + 3, cl_ijk);
								interior_cnt += (ctile.type(cl_ijk) == INTERIOR);
								ctypes[ci][cj][ck] = ctile.type(cl_ijk);
								for (int axis : {0, 1, 2}) {
									coff_diag[axis][ci][cj][ck] = ctile(coeff_channel + axis, cl_ijk);
								}
							}
							else {
								ctypes[ci][cj][ck] = DIRICHLET;
								for (int axis : {0, 1, 2}) {
									coff_diag[axis][ci][cj][ck] = 0;
								}
							}

						}
					}
				}

				for (int ci = 0; ci < 2; ci++) {
					for (int cj = 0; cj < 2; cj++) {
						for (int ck = 0; ck < 2; ck++) {
							if (ctypes[0][cj][ck] == INTERIOR && ctypes[1][cj][ck] == INTERIOR) {
								//it will be added twice
								diag_sum += coff_diag[0][1][cj][ck];
							}
							if (ctypes[ci][0][ck] == INTERIOR && ctypes[ci][1][ck] == INTERIOR) {
								//it will be added twice
								diag_sum += coff_diag[1][ci][1][ck];
							}
							if (ctypes[ci][cj][0] == INTERIOR && ctypes[ci][cj][1] == INTERIOR) {
								//it will be added twice
								diag_sum += coff_diag[2][ci][cj][1];
							}
						}
					}
				}

				tile.type(l_ijk) = interior_cnt > 0 ? INTERIOR : DIRICHLET;
				tile(coeff_channel + 3, l_ijk) = diag_sum * R_matrix_coeff;
			}
		},
			-1, LEAF | GHOST | NONLEAF, LAUNCH_SUBTREE, FINE_FIRST);

		//rhs
		int rhs_channel = 1;
		int u_channel = 10;
		int v_channel = 11;
		int w_channel = 12;
	
		grid.launchVoxelFuncOnAllTiles(
			[=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk)
		{
			float vel_val = -1.0;
			Tile& tile = info.tile();
			tile(u_channel, l_ijk) = 0.0f;
			tile(w_channel, l_ijk) = 0.0f;
			auto g_ijk = acc.composeGlobalCoord(info.mTileCoord, l_ijk);

			if (tile.type(l_ijk) == NEUMANN || g_ijk[1] == 1)
			{
				tile(v_channel, l_ijk) = 0;
			}
			else
			{
				tile(v_channel, l_ijk) = vel_val;
			}
		}, LEAF);
		AMGVolumeWeightedDivergenceOnLeafs(grid, u_channel, coeff_channel, rhs_channel);

		int b_copy_channel = 14;
		grid.launchVoxelFuncOnAllTiles(
			[=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk)
		{
			auto& tile = info.tile();
			auto pos = acc.cellCenter(info, l_ijk);
			auto h = acc.voxelSize(info);
			if (tile.type(l_ijk) & INTERIOR)
			{
				acc.iterateSameLevelNeighborVoxels(info, l_ijk,
					[&] __device__(const HATileInfo<Tile> &ninfo, const Coord & nl_ijk, const int axis, const int sgn)
				{
					if (axis != 1 || sgn != 1)
						return;
					if (!ninfo.empty())
					{
						auto& ntile = ninfo.tile();
						uint8_t ttype1 = ninfo.mType;
						uint8_t ctype1 = ntile.type(nl_ijk);
						if (ctype1 == DIRICHLET)
						{
							float val;
							if (ttype1 == GHOST)
							{
								auto npos = acc.cellCenter(info, nl_ijk);
								val = npos[1] + 0.5 * h;
							}
							else
							{
								auto npos = acc.cellCenter(info, nl_ijk);
								val = npos[1];
							}
							float coeff = ntile(coeff_channel + 1, nl_ijk);
							tile(rhs_channel, l_ijk) -= coeff * val;
						}
					}

				});
			}
			else {
				tile(Tile::b_channel, l_ijk) = 0;
			}
			tile(b_copy_channel, l_ijk) = tile(Tile::b_channel, l_ijk);
		}, LEAF);

		// grdt
		int grdt_channel = 5;
		grid.launchVoxelFuncOnAllTiles(
			[=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk)
		{
			auto& tile = info.tile();
			if (tile.type(l_ijk) & INTERIOR)
			{
				auto pos = acc.cellCenter(info, l_ijk);
				tile(grdt_channel, l_ijk) = pos[1];
			}
		}, LEAF);

		// solve
		solver.omega = 1.5;
		auto [iters, err] = solver.solve(grid, true, 100, 1e-6, 2, 10, 1, is_pure_neumann);
		cudaDeviceSynchronize();
		
		AMGAddGradientToFace(grid, -1, LEAF | GHOST, grdt_channel, coeff_channel, u_channel);
		
		AMGFullNegativeLaplacianOnLeafs(grid, grdt_channel, coeff_channel, Tile::b_channel);
		// error
		int error_channel = 13;
		grid.launchVoxelFuncOnAllTiles(
			[=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk)
		{
			auto& tile = info.tile();
			if (tile.type(l_ijk) & INTERIOR)
			{
				tile(error_channel, l_ijk) = tile(grdt_channel, l_ijk) - tile(Tile::x_channel, l_ijk);
			}
			else
			{
				tile(error_channel, l_ijk) = 0;
			}
		},
			LEAF);

		auto holder = grid.getHostTileHolder(LEAF | GHOST);
		polyscope::init();
		IOFunc::AddLeveledPoissonGridCellCentersToPolyscopePointCloud(holder,
			{ {-1, "type"}, {coeff_channel, "x-"} , {coeff_channel + 1, "y-"}, {coeff_channel + 2, "z-"}, {coeff_channel + 3, "diag"}, {b_copy_channel, "b"} , {grdt_channel, "grdt"}, {Tile::x_channel, "pressure"}, {error_channel, "error"}, {u_channel, "u"}, {v_channel, "v"}, 
			{w_channel, "w"}},
			{}, -1, FLT_MAX);
		//polyscope::show();

		//Info("linf: {}", NormSync(grid, -1, error_channel, false));
		Info("volume-weighted RMS: {}", NormSync(grid, 2, error_channel, true));

		Info("u: {}", holder->cellValue(3, Coord(48, 30, 30), u_channel));
		Info("p: {} {} {} {} {}", holder->cellValue(3, Coord(48, 30, 30), grdt_channel), holder->cellValue(4, Coord(95, 60, 60), grdt_channel), holder->cellValue(4, Coord(95, 60, 61), grdt_channel), holder->cellValue(4, Coord(95, 61, 60), grdt_channel), holder->cellValue(4, Coord(95, 61, 61), grdt_channel));
		Info("coeff: {} {} {} {}", holder->cellValue(4, Coord(96, 60, 60), coeff_channel), holder->cellValue(4, Coord(96, 60, 61), coeff_channel), holder->cellValue(4, Coord(96, 61, 60), coeff_channel), holder->cellValue(4, Coord(96, 61, 61), coeff_channel));
}

	void TestRecoveryNew(const std::string grid_name, const int min_level, const int max_level)
	{
		std::shared_ptr<HADeviceGrid<Tile>> grid_ptr;
		// grid with type
		if (grid_name == "sphere")
		{
			grid_ptr = CreateTestGridCase<SphereSolid05GridCase>(grid_name, min_level, max_level);
		}
		else
		{
			Assert(false, "grid_name {} not supported", grid_name);
		}
		auto& grid = *grid_ptr;
		bool is_pure_neumann = false;

		grid.launchVoxelFuncOnAllTiles(
			[=] __device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk)
		{
			auto& tile = info.tile();
			if (tile.type(l_ijk) == INTERIOR)
			{
				int boundary_axis, boundary_off;
				if (QueryEffectiveBoundaryDirection(acc, min_level, info, l_ijk, boundary_axis, boundary_off))
				{
					if (boundary_axis == 1 && boundary_off == 1) {

						//tile.type(l_ijk) = DIRICHLET;
					}
					else {
						tile.type(l_ijk) = NEUMANN;
					}
				}
			}
		},
			LEAF);
		CalcCellTypesFromLeafs(grid);
		CalculateNeighborTiles(grid);
		
		//0,1,2,3,4: AMGPCG
		//5: grdt
		//6,7,8,9: coeff
		//10: b0
		//11: x0
		int b0_channel = 1;
		int final_x_channel = 0;
		
		//at the end:
		//1: -lap(grdt)
		//2: -lap(x)
		//3: -lap(grdt) - (-lap(x))
		//4: grdt-x

		int lap_grdt_channel = 1;
		int lap_x_channel = 2;
		int lap_diff_channel = 3;
		int x_diff_channel = 4;

		// amg with coef
		int coeff_channel = 6;
		float R_matrix_coeff = 0.5;
		AMGSolver solver(coeff_channel, R_matrix_coeff, 1, 1);

		solver.prepareTypesAndCoeffs(grid);

		//  grdt
		int grdt_channel = 5;
		grid.launchVoxelFuncOnAllTiles(
			[=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk)
		{
			auto& tile = info.tile();
			if (tile.type(l_ijk) & INTERIOR)
			{
				auto pos = acc.cellCenter(info, l_ijk);
				tile(grdt_channel, l_ijk) = pos[1];
			}
		}, LEAF);

		// rhs
		AMGFullNegativeLaplacianOnLeafs(grid, grdt_channel, coeff_channel, b0_channel);

		// solve
		int Nx = 8 << max_level;
		Info("nx: {}", Nx);
		//solver.omega = 2.0 / (1 + CommonConstants::pi / Nx);
		Info("theory value: {}", 2.0 / (1 + CommonConstants::pi / Nx));
		solver.omega = 1.5;
		auto [iters, err] = solver.solve(grid, true, 100, 1e-6, 2, 10, 1, is_pure_neumann);
		cudaDeviceSynchronize();

		//TestIterativeResidualReduction(grid, "amg_vcycle", 10, b0_channel, coeff_channel, final_x_channel, is_pure_neumann);

		//lap(x)
		AMGFullNegativeLaplacianOnLeafs(grid, grdt_channel, coeff_channel, lap_grdt_channel);
		AMGFullNegativeLaplacianOnLeafs(grid, final_x_channel, coeff_channel, lap_x_channel);

		// error
		grid.launchVoxelFuncOnAllTiles(
			[=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk)
		{
			auto& tile = info.tile();
			if (tile.type(l_ijk) & INTERIOR)
			{
				tile(lap_diff_channel, l_ijk) = tile(lap_grdt_channel, l_ijk) - tile(lap_x_channel, l_ijk);
				tile(x_diff_channel, l_ijk) = tile(grdt_channel, l_ijk) - tile(final_x_channel, l_ijk);
			}
			else
			{
				tile(lap_diff_channel, l_ijk) = 0;
				tile(x_diff_channel, l_ijk) = 0;
			}
		},
			LEAF);

		//auto holder = grid.getHostTileHolder(LEAF);
		//polyscope::init();
		//IOFunc::AddLeveledPoissonGridCellCentersToPolyscopePointCloud(holder,
		//	{ {-1, "type"}, {coeff_channel, "x-"} , {coeff_channel + 1, "y-"}, {coeff_channel + 2, "z-"}, {coeff_channel + 3, "diag"}, {grdt_channel, "grdt"}, {final_x_channel, "pressure"},
		//	{x_diff_channel, "x_diff"}, {lap_diff_channel, "lap_diff"}, {lap_grdt_channel, "lap(grdt)"}, {lap_x_channel, "lap(x)"}},
		//	{}, -1, FLT_MAX);
		//polyscope::show();

		Info("tile size: {}", sizeof(Tile));
		Info("linf: {}", NormSync(grid, -1, x_diff_channel, false));
	}
}