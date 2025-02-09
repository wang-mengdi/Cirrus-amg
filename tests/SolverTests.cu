#include "SolverTests.h"
#include "PoissonSolver.h"
#include "GMGSolver.h"
#include "CMGSolver.h"
#include "AMGSolver.h"
#include "PoissonIOFunc.h"
#include "Common.h"
#include "GPUTimer.h"
#include "Random.h"
#include <polyscope/polyscope.h>
#include <polyscope/point_cloud.h>


//extern int laplacian_total_tile_counts;

namespace SolverTests {
    __hostdev__ int SolverTestsLevelTarget(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const TestGrids grid_name) {
        //uniform grids
        if (grid_name == TestGrids::uniform128) {
            return 4;
        }
        else if (grid_name == TestGrids::uniform256) {
            return 5;
        }
        else if (grid_name == TestGrids::uniform512) {
            return 6;
        }
        else if (grid_name == TestGrids::staircase34) {
			//64^3 at small x, 128^3 at large x
            auto bbox = acc.tileBBox(info);
            int desired_level = 0;
            if (bbox.min()[0] <= 0.25) return 4;//slow converging, if 0.25 not converging
            else return 3;
        }
        else if (grid_name == TestGrids::twosources67) {
            int desired_level = 0;
            auto bbox = acc.tileBBox(info);
            const Vec pointSrc1(0.51, 0.49, 0.54);
            const Vec pointSrc2(0.93, 0.08, 0.91);
            if (bbox.isInside(pointSrc2)) desired_level = 6;
            if (bbox.isInside(pointSrc1)) desired_level = 7;
            //if (bbox.isInside(pointSrc2)) desired_level = 3;
            //if (bbox.isInside(pointSrc1)) desired_level = 4;
            return desired_level;
        }
        else if (grid_name == TestGrids::twosources_deform) {
            //two sources for testing 3d deformation
            //refine at (0.35,0.35,0.35)
            //it's for testing the 3D deformation
            int desired_level = 0;
            auto bbox = acc.tileBBox(info);
            const Vec pointSrc1(0.35, 0.35, 0.35);
            const Vec pointSrc2(0.8, 0.2, 0.6);
            if (bbox.isInside(pointSrc2)) desired_level = 5;
            if (bbox.isInside(pointSrc1)) desired_level = 6;
            //if (bbox.isInside(pointSrc2)) desired_level = 3;
            //if (bbox.isInside(pointSrc1)) desired_level = 4;
            return desired_level;
        }
        else if (grid_name == TestGrids::centersource) {
            //try to test nfm advection with 3d deformation

            int desired_level = 0;
            auto bbox = acc.tileBBox(info);
            double eps = 1e-6;
            const Vec pointSrc1(0.5 - eps, 0.5 - eps, 0.5 - eps);
            if (bbox.isInside(pointSrc1)) desired_level = 6;

            return desired_level;
        }
    }

    void TestAMGLaplacianAndFluxConsistency(TestGrids grid_name) {
        fmt::print("==========================================================\n");
		Info("Test lap=div(grad) on grid {}", ToString(grid_name));

        uint32_t scale = 8;
        float h = 1.0 / scale;
        //0:8, 1:16, 2:32, 3:64, 4:128, 5:256, 6:512, 7:1024
        HADeviceGrid<Tile> grid(h, { 16,16,16,16,16,16,18,16,16,16 });
        grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
        grid.rebuild();
        IterativeRefine(grid, [=]__device__(const HATileAccessor<Tile>&acc, HATileInfo<Tile>&info) { return SolverTestsLevelTarget(acc, info, grid_name); }, false);
        int num_cells = grid.numTotalLeafTiles() * Tile::SIZE;
        int total_hash_bytes = grid.hashTableDeviceBytes();
        Info("Total {}M cells, hash table {}GB", num_cells / (1024.0 * 1024), total_hash_bytes / (1024.0 * 1024 * 1024));

        int x_channel = 0;
        int coeff_channel = 1;
        int u_channel = 5;
        int nlap_channel = 8;
        int lap_channel = 9;
        int diff_channel = 10;

        //load cell types, load solution to grdt_channel
        auto cell_type = [=]__device__(const HATileAccessor<Tile> &acc, const HATileInfo<Tile> &info, const nanovdb::Coord & l_ijk)->uint8_t {
            bool is_dirichlet = false;
            bool is_neumann = false;
            acc.iterateSameLevelNeighborVoxels(info, l_ijk,
                [&]__device__(const HATileInfo<Tile>&n_info, const Coord & n_l_ijk, const int axis, const int sgn) {
                if (n_info.empty()) {
                    if (axis == 0 && sgn == -1) is_neumann = true;
                    else is_dirichlet = true;
                }
            });
            if (is_neumann) return CellType::NEUMANN;
            else if (is_dirichlet) return CellType::DIRICHLET;
            else return CellType::INTERIOR;
        };
        //solution f(x)
        auto f = [=]__device__(const Vec & pos) ->T {
            return sin(-2.5 * pos[0] + pos[1] + 7 * pos[2]);
        };
        grid.launchVoxelFunc(
            [=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
            auto& tile = info.tile();
            tile.type(l_ijk) = cell_type(acc, info, l_ijk);
            if (tile.type(l_ijk) & INTERIOR) tile(x_channel, l_ijk) = f(acc.cellCenter(info, l_ijk));
            else tile(x_channel, l_ijk) = 0;
            for (int axis : {0, 1, 2}) {
				tile(u_channel + axis, l_ijk) = 0;
            }

        }, -1, LEAF|GHOST, LAUNCH_SUBTREE
        );
        CalcCellTypesFromLeafs(grid);

        //use AMG to calculate Laplacian from x_channel to lap_channel
        {
            CalculateNeighborTiles(grid);
            AMGSolver solver(coeff_channel, 0.5, 1, 1);
            solver.prepareTypesAndCoeffs(grid);
            AMGFullNegativeLaplacianOnLeafs(grid, x_channel, coeff_channel, nlap_channel);
        }

        //calculate div(grad)
        {
            //grad(p)
            AMGAddGradientToFace(grid, -1, LEAF | GHOST, x_channel, coeff_channel, u_channel);
            //div(u)
            AMGVolumeWeightedDivergenceOnLeafs(grid, u_channel, coeff_channel, x_channel);
        }

        //calculate difference from x and grdt in r_channel
        grid.launchVoxelFuncOnAllTiles(
            [=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
            auto& tile = info.tile();
            if (tile.type(l_ijk) & INTERIOR) {
                tile(diff_channel, l_ijk) = tile(lap_channel, l_ijk) + tile(nlap_channel, l_ijk);
            }
            else {
                tile(diff_channel, l_ijk) = 0;
            }
        }, LEAF
        );

        //auto holder = grid.getHostTileHolderForLeafs();
        //polyscope::init();
        //IOFunc::AddPoissonGridCellCentersToPolyscopePointCloud(holder,
        //    { {-1,"type"}, { x_channel, "x" }, {lap_channel,"div(grad)"}, {nlap_channel,"nlap"},{diff_channel,"diff"} },
        //    {}
        //);
        //polyscope::show();

        auto linf_norm = SingleChannelLinfSync(grid, diff_channel, LEAF);
        if (linf_norm < 1e-5) {
            Pass("Test passed with Linf norm of grdt-x: {}\n\n", linf_norm);
        }
        else {
            Error("Test failed with Linf norm of grdt-x: {}\n\n", linf_norm);
        }
    }

    __host__ void TestNeumannDirichletRecovery(TestGrids grid_name, const std::string algorithm) {
        //algorithm: "cmg"/"amg"
        fmt::print("==========================================================\n");
        Info("Test Poisson Solver on Neumann/Dirichlet BC on grid {} with given function for {}", ToString(grid_name), algorithm);
        Assert(algorithm == "cmg" || algorithm == "amg", "algorithm should be cmg or amg");
        

        uint32_t scale = 8;
        float h = 1.0 / scale;
        //0:8, 1:16, 2:32, 3:64, 4:128, 5:256, 6:512, 7:1024
        HADeviceGrid<Tile> grid(h, { 16,16,16,16,16,16,18,16,16,16 });
        grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
        grid.rebuild();

        IterativeRefine(grid, [=]__device__(const HATileAccessor<Tile>&acc, HATileInfo<Tile>&info) { return SolverTestsLevelTarget(acc, info, grid_name); }, false);
		int num_cells = grid.numTotalLeafTiles() * Tile::SIZE;
        int total_hash_bytes = grid.hashTableDeviceBytes();
		Info("Total {}M cells, hash table {}GB", num_cells / (1024.0 * 1024), total_hash_bytes / (1024.0 * 1024 * 1024));

        Assert(Tile::num_channels >= 10, "Tile::num_channels should be >= 10 for this test");
        //channels 0,1,2,3,4: channels implicitly required by GMGPCG
        //channels 5: grdt channel
        //channels 6,7,8,9: coeffs channel for AMG
        int grdt_channel = 5;
        int coeff_channel = 6;

        //load cell types, load solution to grdt_channel
        auto cell_type = [=]__device__(const HATileAccessor<Tile> &acc, const HATileInfo<Tile> &info, const nanovdb::Coord & l_ijk)->uint8_t {
            bool is_dirichlet = false;
            bool is_neumann = false;
            acc.iterateSameLevelNeighborVoxels(info, l_ijk,
                [&]__device__(const HATileInfo<Tile>&n_info, const Coord & n_l_ijk, const int axis, const int sgn) {
                if (n_info.empty()) {
                    if (axis == 0 && sgn == -1) {
                        is_neumann = true;
                    }
                    else {
                        is_dirichlet = true;
                    }
                }
            });
            if (is_neumann) return CellType::NEUMANN;
            else if (is_dirichlet) return CellType::DIRICHLET;
            else return CellType::INTERIOR;
        };
        //solution f(x)
        auto f = [=]__device__(const Vec & pos) {
            return exp(pos[0]) * sin(pos[1]);
        };
        grid.launchVoxelFunc(
            [=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
            auto& tile = info.tile();
            tile.type(l_ijk) = cell_type(acc, info, l_ijk);
            if (tile.type(l_ijk) & INTERIOR) tile(grdt_channel, l_ijk) = f(acc.cellCenter(info, l_ijk));
            else tile(grdt_channel, l_ijk) = 0;
        }, -1, LEAF, LAUNCH_SUBTREE
        );
        CalcCellTypesFromLeafs(grid);
        CalculateNeighborTiles(grid);
        


        if (algorithm == "cmg") {
            CalculateNeighborTiles(grid);
            ConservativeFullNegativeLaplacian(grid, grdt_channel, Tile::b_channel);

            _sleep(200);
            CMGSolver solver(1.0, 1.0);
            //Copy(grid, Tile::b_channel, Tile::phi_channel, -1, LEAF, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);
            CPUTimer<std::chrono::microseconds> timer;
            timer.start();
            auto [iters, err] = solver.solve(grid, true, 1000, 1e-6, 1, 10, 1, false);
            CheckCudaError("CMGPCG solve");
            float elapsed = timer.stop("CMGPCG Async");
            int total_cells = grid.numTotalTiles() * Tile::SIZE;
            float cells_per_second = (total_cells + 0.0) / (elapsed / 1e6);
            Info("Total {:.5}M cells, CMGPCG Async speed {:.5} M cells /s", total_cells / (1024.0 * 1024), cells_per_second / (1024.0 * 1024));
            Info("CMGPCG solved in {} iterations with error {}, average iteration throughput {:.5}M cell/s", iters, err, cells_per_second* iters / (1024.0 * 1024));
        }
        else if (algorithm == "amg") {
            CalculateNeighborTiles(grid);
            

            AMGSolver solver(coeff_channel, 0.5, 1, 1);
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


        //calculate difference from x and grdt in r_channel
        grid.launchVoxelFuncOnAllTiles(
            [=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
            auto& tile = info.tile();
            if (tile.type(l_ijk) & INTERIOR) {
                tile(Tile::r_channel, l_ijk) = tile(grdt_channel, l_ijk) - tile(Tile::x_channel, l_ijk);
            }
            else {
                tile(Tile::r_channel, l_ijk) = 0;
            }
        }, LEAF
        );
		auto linf_norm = SingleChannelLinfSync(grid, Tile::r_channel, LEAF);
        if (linf_norm < 1e-5) {
			Pass("Test passed with Linf norm of grdt-x: {}\n\n", linf_norm);
		}
        else {
			Error("Test failed with Linf norm of grdt-x: {}\n\n", linf_norm);
        }
    }


    __host__ void TestStaticPressureUAAMG(TestGrids grid_name, const std::string algorithm) {
        //algorithm: "cmg"/"amg"
        //A Fast Unsmoothed Aggregation Algebraic Multigrid Framework for the Large - Scale Simulation of Incompressible Flow
        fmt::print("==========================================================\n");
        Info("Test Poisson Solver with Static Pressure on grid {} with given function for {}", ToString(grid_name), algorithm);
        Assert(algorithm == "cmg" || algorithm == "amg", "algorithm should be cmg or amg");


        uint32_t scale = 8;
        float h = 1.0 / scale;
        //0:8, 1:16, 2:32, 3:64, 4:128, 5:256, 6:512, 7:1024
        HADeviceGrid<Tile> grid(h, { 16,16,16,16,16,16,18,16,16,16 });
        grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
        grid.rebuild();

        IterativeRefine(grid, [=]__device__(const HATileAccessor<Tile>&acc, HATileInfo<Tile>&info) { return SolverTestsLevelTarget(acc, info, grid_name); }, false);
        int num_cells = grid.numTotalLeafTiles() * Tile::SIZE;
        int total_hash_bytes = grid.hashTableDeviceBytes();
        Info("Total {}M cells, hash table {}GB", num_cells / (1024.0 * 1024), total_hash_bytes / (1024.0 * 1024 * 1024));

        Assert(Tile::num_channels >= 10, "Tile::num_channels should be >= 10 for this test");
        //channels 0,1,2,3,4: channels implicitly required by GMGPCG
        //channels 5: grdt channel
        //channels 6,7,8,9: coeffs channel for AMG
        int grdt_channel = 5;
        int coeff_channel = 6;

        //load cell types, load solution to grdt_channel
        auto cell_type = [=]__device__(const HATileAccessor<Tile> &acc, const HATileInfo<Tile> &info, const nanovdb::Coord & l_ijk)->uint8_t {
            bool is_dirichlet = false;
            bool is_neumann = false;
            acc.iterateSameLevelNeighborVoxels(info, l_ijk,
                [&]__device__(const HATileInfo<Tile>&n_info, const Coord & n_l_ijk, const int axis, const int sgn) {
                if (n_info.empty()) {
                    if (axis == 0 && sgn == -1) {
                        is_neumann = true;
                    }
                    else {
                        is_dirichlet = true;
                    }
                }
            });
            if (is_neumann) return CellType::NEUMANN;
            else if (is_dirichlet) return CellType::DIRICHLET;
            else return CellType::INTERIOR;
        };
        //solution f(x)
        auto f = [=]__device__(const Vec & pos) {
            return exp(pos[0]) * sin(pos[1]);
        };
        grid.launchVoxelFunc(
            [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
            auto& tile = info.tile();
            tile.type(l_ijk) = cell_type(acc, info, l_ijk);
            if (tile.type(l_ijk) & INTERIOR) tile(grdt_channel, l_ijk) = f(acc.cellCenter(info, l_ijk));
            else tile(grdt_channel, l_ijk) = 0;
        }, -1, LEAF, LAUNCH_SUBTREE
        );
        CalcCellTypesFromLeafs(grid);
        CalculateNeighborTiles(grid);



        if (algorithm == "cmg") {
            CalculateNeighborTiles(grid);
            ConservativeFullNegativeLaplacian(grid, grdt_channel, Tile::b_channel);

            _sleep(200);
            CMGSolver solver(1.0, 1.0);
            //Copy(grid, Tile::b_channel, Tile::phi_channel, -1, LEAF, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);
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
        else if (algorithm == "amg") {
            CalculateNeighborTiles(grid);


            AMGSolver solver(coeff_channel, 0.5, 1, 1);
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


        //calculate difference from x and grdt in r_channel
        grid.launchVoxelFuncOnAllTiles(
            [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
            auto& tile = info.tile();
            if (tile.type(l_ijk) & INTERIOR) {
                tile(Tile::r_channel, l_ijk) = tile(grdt_channel, l_ijk) - tile(Tile::x_channel, l_ijk);
            }
            else {
                tile(Tile::r_channel, l_ijk) = 0;
            }
        }, LEAF
        );
        auto linf_norm = SingleChannelLinfSync(grid, Tile::r_channel, LEAF);
        if (linf_norm < 1e-5) {
            Pass("Test passed with Linf norm of grdt-x: {}\n\n", linf_norm);
        }
        else {
            Error("Test failed with Linf norm of grdt-x: {}\n\n", linf_norm);
        }
    }
}