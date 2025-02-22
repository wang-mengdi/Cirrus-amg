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
        if (grid_name == TestGrids::uniform8) {
            return 0;
        }
        else if (grid_name == TestGrids::uniform32) {
            return 2;
        }
        else if (grid_name == TestGrids::uniform128) {
            return 4;
        }
        else if (grid_name == TestGrids::uniform256) {
            return 5;
        }
        else if (grid_name == TestGrids::uniform512) {
            return 6;
        }
        else if (grid_name == TestGrids::staircase12) {
            auto bbox = acc.tileBBox(info);
            int desired_level = 0;
            if (bbox.max()[0] >= 0.75) return 2;//slow converging, if 0.25 not converging
            else return 1;
        }
        else if (grid_name == TestGrids::staircase21) {
            auto bbox = acc.tileBBox(info);
            int desired_level = 0;
            if (bbox.min()[0] <= 0.25) return 2;//slow converging, if 0.25 not converging
            else return 1;
        }
        else if (grid_name == TestGrids::staircase34) {
            auto bbox = acc.tileBBox(info);
            int desired_level = 0;
            if (bbox.max()[0] >= 0.75) return 3;//slow converging, if 0.25 not converging
            else return 4;
        }
        else if (grid_name == TestGrids::staircase43) {
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
        grid.launchVoxelFuncOnAllTiles(
            [=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
            auto& tile = info.tile();
            tile.type(l_ijk) = cell_type(acc, info, l_ijk);
            if (tile.type(l_ijk) & INTERIOR) tile(x_channel, l_ijk) = f(acc.cellCenter(info, l_ijk));
            else tile(x_channel, l_ijk) = 0;
            for (int axis : {0, 1, 2}) {
                tile(u_channel + axis, l_ijk) = 0;
            }

        }, LEAF | GHOST);
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
            AMGVolumeWeightedDivergenceOnLeafs(grid, u_channel, coeff_channel, lap_channel);
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

        auto holder = grid.getHostTileHolderForLeafs();
        polyscope::init();
        IOFunc::AddPoissonGridCellCentersToPolyscopePointCloud(holder,
            { {-1,"type"}, { x_channel, "x" }, {lap_channel,"div(grad)"}, {nlap_channel,"nlap"},{diff_channel,"diff"} },
            {}
        );
        polyscope::show();

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
        HADeviceGrid<Tile> grid(h, { 16,16,16,16,16,16,20,16,16,16 });
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

            //auto holder = grid.getHostTileHolder(LEAF | NONLEAF);
            //polyscope::init();
            //IOFunc::AddLeveledPoissonGridCellCentersToPolyscopePointCloud(holder,
            //    {
            //    {solver.coeff_channel,"offd0"}, {solver.coeff_channel + 1,"offd1"} ,{solver.coeff_channel + 2,"offd2"},{solver.coeff_channel + 3,"diag"}, {-1,"type"}
            //    },
            //    {}, -1, FLT_MAX);
            //polyscope::show();

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
        if (linf_norm < 1e-4) {
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

    __hostdev__ int ConvergenceTestLevelTarget(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const ConvergenceTestGridName grid_name, const int min_level, const int max_level) {
        if (grid_name == ConvergenceTestGridName::uniform) {
            return max_level;
        }
        else if (grid_name == ConvergenceTestGridName::sphere_shell_05) {
            auto bbox = acc.tileBBox(info);
            auto bmin = bbox.min();
            auto bmax = bbox.max();
            const Vec ctr(0.5, 0.5, 0.5);
            constexpr T radius = 0.5 / 2;
            int inside_cnt = 0;
            for (int di : {0, 1}) {
                for (int dj : {0, 1}) {
                    for (int dk : {0, 1}) {
                        Vec vpos;
                        vpos[0] = bmin[0] + di * (bmax[0] - bmin[0]);
                        vpos[1] = bmin[1] + dj * (bmax[1] - bmin[1]);
                        vpos[2] = bmin[2] + dk * (bmax[2] - bmin[2]);
                        if ((vpos - ctr).length() < radius) {
                            inside_cnt++;
                        }
                    }
                }
            }
            if (inside_cnt == 0 || inside_cnt == 8) return min_level;
            else return max_level;
        }
        return max_level;

    }

    class UniformGridCase {
    public:
        __hostdev__ static int target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const int min_level, const int max_level) {
            return max_level;
        }
		__hostdev__ static uint8_t type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) {
			return CellType::INTERIOR;
		}
    };

    class SphereShell05GridCase {
    public:
        __hostdev__ static int target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const int min_level, const int max_level) {
            auto bbox = acc.tileBBox(info);
            auto bmin = bbox.min();
            auto bmax = bbox.max();
            const Vec ctr(0.5, 0.5, 0.5);
            constexpr T radius = 0.5 / 2;
            int inside_cnt = 0;
            for (int di : { 0, 1 }) {
                for (int dj : { 0, 1 }) {
                    for (int dk : { 0, 1 }) {
                        Vec vpos;
                        vpos[0] = bmin[0] + di * (bmax[0] - bmin[0]);
                        vpos[1] = bmin[1] + dj * (bmax[1] - bmin[1]);
                        vpos[2] = bmin[2] + dk * (bmax[2] - bmin[2]);
                        if ((vpos - ctr).length() < radius) {
                            inside_cnt++;
                        }
                    }
                }
            }
            if (inside_cnt == 0 || inside_cnt == 8) return min_level;
            else return max_level;
        }
        __hostdev__ static uint8_t type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) {
            return CellType::INTERIOR;
        }
    };


    class GerrisSinFunc {
    public:
        static constexpr T PI = 3.14159265358979323846;
        static constexpr int k = 3, l = 3;
        __hostdev__ static T f(const Vec pos) {
            auto x = pos[0] - 0.5;
            auto y = pos[1] - 0.5;
            return sin(k * PI * x) * sin(l * PI * y);
        }
        __hostdev__ static Vec df(const Vec pos) {
            auto x = pos[0] - 0.5;
            auto y = pos[1] - 0.5;
            T dfdx = k * PI * cos(k * PI * x) * sin(l * PI * y);
            T dfdy = l * PI * sin(k * PI * x) * cos(l * PI * y);
			return { dfdx, dfdy, 0 };
        }
        __hostdev__ static T lapf(const Vec pos) {
            auto x = pos[0] - 0.5;
            auto y = pos[1] - 0.5;
            return -PI * PI * (k * k + l * l) * sin(k * PI * x) * sin(l * PI * y);
        }
    };
    class AthenaSinFunc {
    public:
        //rho0=2 in their paper, but we don't have that
        static constexpr T PI = 3.14159265358979323846, Lx = 1.0, Ly = 1.0, Lz = 1.0, G = 1.0, A = 1.0, phi0 = 0;
        __hostdev__ static T f(const Vec pos) {
            auto x = pos[0];
            auto y = pos[1];
            auto z = pos[2];
            //phi = -4*pi*G*A / ( (2*pi/Lx)^2 + (2*pi/Ly)^2 + (2*pi/Lz)^2 )  * sin(2*pi*x/Lx) * sin(2*pi*y/Ly) * sin(2*pi*z/Lz) + phi0
            return
                -4 * PI * G * A
                /
                ((2 * PI / Lx) * (2 * PI / Lx) + (2 * PI / Ly) * (2 * PI / Ly) + (2 * PI / Lz) * (2 * PI / Lz))
                *
                sin(2 * PI * x / Lx) * sin(2 * PI * y / Ly) * sin(2 * PI * z / Lz)
                +
                phi0;
        }
        __hostdev__ static Vec df(const Vec pos) {
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
        __hostdev__ static T lapf(const Vec pos) {
            auto x = pos[0];
            auto y = pos[1];
            auto z = pos[2];
            return  4 * PI * G * A * sin(2 * PI * x / Lx) * sin(2 * PI * y / Ly) * sin(2 * PI * z / Lz);
        }
    };

    template<class TestFunc>
    void SetAnalyticalTestWithAllNeumannAnalyticalBC(HADeviceGrid<Tile>& grid, const int rhs_channel, const int grdt_channel) {
        auto cell_type = [=]__device__(const HATileAccessor<Tile> &acc, const HATileInfo<Tile> &info, const nanovdb::Coord & l_ijk)->uint8_t {
            bool is_dirichlet = false;
            bool is_neumann = false;
            acc.iterateSameLevelNeighborVoxels(info, l_ijk,
                [&]__device__(const HATileInfo<Tile>&n_info, const Coord & n_l_ijk, const int axis, const int sgn) {
                if (n_info.empty()) {
                    is_neumann = true;
                }
            });
            if (is_neumann) return CellType::NEUMANN;
            else if (is_dirichlet) return CellType::DIRICHLET;
            else return CellType::INTERIOR;
        };
        grid.launchVoxelFuncOnAllTiles(
            [=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
            auto& tile = info.tile();
            tile.type(l_ijk) = cell_type(acc, info, l_ijk);
        }, LEAF
        );
        CalcCellTypesFromLeafs(grid);

        grid.launchVoxelFuncOnAllTiles(
            [=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
            auto& tile = info.tile();
            auto pos = acc.cellCenter(info, l_ijk);
            auto h = acc.voxelSize(info);


            if (tile.type(l_ijk) & INTERIOR) {
                tile(rhs_channel, l_ijk) = -TestFunc::lapf(pos) * h * h * h;
                tile(grdt_channel, l_ijk) = TestFunc::f(pos);

                //integral of -lap
                T rhs_correction = 0;
                acc.iterateSameLevelNeighborVoxels(info, l_ijk,
                    [&]__device__(const HATileInfo<Tile>&ninfo, const Coord & nl_ijk, const int axis, const int sgn) {
					Vec npos = acc.cellCenter(ninfo, nl_ijk);
                    bool nb_neumann = false, nb_dirichlet = false;
                    Vec dirichlet_pos, neumann_pos;


                    if (ninfo.empty()) {
                        nb_dirichlet = true;
                        dirichlet_pos = npos;
                    }
                    else if (ninfo.mType == LEAF) {
                        auto& ntile = ninfo.tile();
                        if (ntile.type(nl_ijk) & NEUMANN) {
                            nb_neumann = true;
                            neumann_pos = (pos + npos) * 0.5;
                        }
                        if (ntile.type(nl_ijk) & DIRICHLET) {
                            nb_dirichlet = true;
                            dirichlet_pos = npos;
                        }
                    }
                    else if (ninfo.mType == GHOST) {
                        auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
						auto pg_ijk = acc.parentCoord(g_ijk);
                        Vec ppos = acc.cellCenterGlobal(info.mLevel - 1, pg_ijk);

                        auto& ntile = ninfo.tile();
                        auto ng_ijk = acc.localToGlobalCoord(ninfo, nl_ijk);
                        auto npg_ijk = acc.parentCoord(ng_ijk);
                        Vec nppos = acc.cellCenterGlobal(info.mLevel - 1, npg_ijk);

                        if (ntile.type(nl_ijk) & NEUMANN) {
                            nb_neumann = true;
                            neumann_pos = (ppos + nppos) * 0.5;
                        }
                        else if (ntile.type(nl_ijk) & DIRICHLET) {
                            nb_dirichlet = true;
                            dirichlet_pos = nppos;
                        }
                    }

                    if (nb_neumann) {
                        rhs_correction += (-sgn) * TestFunc::df(neumann_pos)[axis] * h * h;
                    }
                    else if (nb_dirichlet) {
                        rhs_correction += (-sgn) * TestFunc::f(dirichlet_pos) * h;
                    }
                });
                tile(rhs_channel, l_ijk) -= rhs_correction;

            }
            else {
                tile(rhs_channel, l_ijk) = 0;
                tile(grdt_channel, l_ijk) = 0;
            }
        }, LEAF
        );
    }

    template<class GridCase>
    std::shared_ptr<HADeviceGrid<Tile>> CreateTestGridCase(const std::string grid_name, const int min_level, const int max_level) {
        uint32_t scale = 8;
        float h = 1.0 / scale;
        //0:8, 1:16, 2:32, 3:64, 4:128, 5:256, 6:512, 7:1024
        auto grid_ptr = std::make_shared<HADeviceGrid<Tile>>(h, thrust::host_vector{ 16,16,16,16,16,16,20,16,16,16 });
        auto& grid = *grid_ptr;
        grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
        grid.rebuild();
		IterativeRefine(grid, [=]__device__(const HATileAccessor<Tile>&acc, HATileInfo<Tile>&info) { return GridCase::target(acc, info, min_level, max_level); }, false);
        int num_cells = grid.numTotalLeafTiles() * Tile::SIZE;
        int total_hash_bytes = grid.hashTableDeviceBytes();
        Info("Total {}M cells, hash table {}GB", num_cells / (1024.0 * 1024), total_hash_bytes / (1024.0 * 1024 * 1024));

        grid.launchVoxelFuncOnAllTiles(
			[=] __device__(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) {
			auto& tile = info.tile();
			tile.type(l_ijk) = GridCase::type(acc, info, l_ijk);
		}, LEAF
		);
        return grid_ptr;
    }

    //return <grid, is_pure_neumann>
    std::tuple<std::shared_ptr<HADeviceGrid<Tile>>, bool> CreateTestGrid(const std::string grid_name, const int min_level, const int max_level, const std::string bc_name, const int rhs_channel, const int grdt_channel) {
		std::shared_ptr<HADeviceGrid<Tile>> grid_ptr;
        if (grid_name == "uniform") {
			grid_ptr = CreateTestGridCase<UniformGridCase>(grid_name, min_level, max_level);
		}
		else if (grid_name == "sphere_shell_05") {
			grid_ptr = CreateTestGridCase<SphereShell05GridCase>(grid_name, min_level, max_level);
		}
        else {
            Assert(false, "grid_name {} not supported", grid_name);
        }
		auto& grid = *grid_ptr;

        bool is_pure_neumann = false;

        if (bc_name == "gerris_sin") {
            is_pure_neumann = true;
            SetAnalyticalTestWithAllNeumannAnalyticalBC<GerrisSinFunc>(grid, rhs_channel, grdt_channel);
        }
        else if (bc_name == "athena_sin") {
            is_pure_neumann = true;
			SetAnalyticalTestWithAllNeumannAnalyticalBC<AthenaSinFunc>(grid, rhs_channel, grdt_channel);
        }
        else {
			Assert(false, "bc_name {} not supported", bc_name);
        }

        CalcCellTypesFromLeafs(grid);
        CalculateNeighborTiles(grid);

   //     {
   //         auto holder = grid.getHostTileHolderForLeafs();
   //         polyscope::init();
			//IOFunc::AddPoissonGridCellCentersToPolyscopePointCloud(holder,
			//	{ {-1,"type"}, {rhs_channel, "rhs"}, {grdt_channel,"grdt"} },
			//	{}
			//);
			//polyscope::show();
   //         IOFunc::OutputPoissonGridAsStructuredVTI(
   //             holder,
   //             { {-1, "type"}, {rhs_channel, "rhs"}, {grdt_channel, "grdt"} },
   //             {  },
   //             fmt::format("output/{}_input.vti", ToString(grid_name))
   //         );
   //     }

		return std::make_tuple(grid_ptr, is_pure_neumann);
    }

    void TestSolverWithAnalyticalSolution(HADeviceGrid<Tile>& grid, const std::string algorithm, const int coeff_channel, const int grdt_channel, bool is_pure_neumann) {
        if (algorithm == "cmg") {
            CalculateNeighborTiles(grid);
            ConservativeFullNegativeLaplacian(grid, grdt_channel, Tile::b_channel);

            _sleep(200);
            CMGSolver solver(1.0, 1.0);
            //Copy(grid, Tile::b_channel, Tile::phi_channel, -1, LEAF, LAUNCH_SUBTREE, INTERIOR | DIRICHLET | NEUMANN);
            CPUTimer<std::chrono::microseconds> timer;
            timer.start();
            auto [iters, err] = solver.solve(grid, false, 1000, 1e-6, 2, 10, 1, is_pure_neumann);
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
            //AMGFullNegativeLaplacianOnLeafs(grid, grdt_channel, coeff_channel, Tile::b_channel);



            _sleep(200);

            CPUTimer<std::chrono::microseconds> timer;
            timer.start();
            auto [iters, err] = solver.solve(grid, false, 1000, 1e-6, 2, 10, 1, is_pure_neumann);
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

        //auto holder = grid.getHostTileHolder(LEAF);
        //polyscope::init();
        //IOFunc::AddLeveledPoissonGridCellCentersToPolyscopePointCloud(holder,
        //    {
        //    {-1, "type"}, {Tile::x_channel,"x"} ,{grdt_channel,"grdt"},{Tile::r_channel, "r"}
        //    },
        //    {}, -1, FLT_MAX);
        //polyscope::show();

   //     {
			//auto holder = grid.getHostTileHolderForLeafs();
   //         IOFunc::OutputPoissonGridAsStructuredVTI(
   //             holder,
   //             { {-1, "type"}, {Tile::b_channel, "rhs"}, {grdt_channel, "grdt"}, {Tile::x_channel, "x"}, {Tile::r_channel, "r"} },
   //             {  },
   //             fmt::format("output/{}_result.vti", algorithm)
   //         );
   //     }

        auto linf_norm = SingleChannelLinfSync(grid, Tile::r_channel, LEAF);
        if (linf_norm < 1e-4) {
            Pass("Test passed with Linf norm of grdt-x: {}\n\n", linf_norm);
        }
        else {
            Warn("Test failed with Linf norm of grdt-x: {}\n\n", linf_norm);
        }
    }

    void TestAnalyticalConvergence(const std::string grid_name, const int min_level, const int max_level, const std::string bc_name, const std::string algorithm) {
		Info("Test Poisson Solver on grid {}, {}~{} levels with given function of {} for {}", grid_name, min_level, max_level, bc_name, algorithm);


        int grdt_channel = 5;
        int coeff_channel = 6;
        int rhs_channel = Tile::b_channel;

        auto [grid_ptr, is_pure_neumann] = CreateTestGrid(grid_name, min_level, max_level, bc_name, rhs_channel, grdt_channel);
		auto& grid = *grid_ptr;


		TestSolverWithAnalyticalSolution(grid, algorithm, coeff_channel, grdt_channel, is_pure_neumann);

        //auto holder = grid.getHostTileHolderForLeafs();
        //IOFunc::OutputPoissonGridAsStructuredVTI(
        //    holder,
        //    { {-1, "type"}, {rhs_channel, "rhs"}, {grdt_channel, "grdt"}, {Tile::x_channel, "x"}, {Tile::r_channel, "r"} },
        //    {  },
        //    fmt::format("output/analytical_{}_{}_{}_{}.vti", ToString(grid_name), max_level, bc_name, algorithm)
        //);
       


        //IOFunc::OutputTilesAsVTU(holder, "output/tiles.vtu");
    }
}