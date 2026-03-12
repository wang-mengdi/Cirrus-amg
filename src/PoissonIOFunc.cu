#include "PoissonIOFunc.h"


//#include <vtkNew.h>
//#include <vtkPoints.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <vtkAMRBox.h>
//#include <vtkCellData.h>
//#include <vtkUnstructuredGrid.h>

#include <vtkOverlappingAMR.h>
#include <vtkNonOverlappingAMR.h>
#include <vtkUniformGridAMR.h>
#include <vtkXMLUniformGridAMRWriter.h>
#include <vtkXMLUniformGridAMRReader.h>
#include <vtkUniformGrid.h>
#include "vtkCompositeDataWriter.h"
#include "vtkXMLHierarchicalBoxDataWriter.h"
#include <vtkXMLImageDataWriter.h>

#include <vtkCellData.h>
#include <vtkNew.h>


//#include <zlib.h>

//#include <AMReX.H>
//#include <AMReX_AmrCore.H>
//#include <AMReX_AmrLevel.H>
//#include <AMReX_FArrayBox.H>
//#include <AMReX_PlotFileUtil.H>
//#include <AMReX_MultiFab.H>
//#include <AMReX_BoxArray.H>
//#include <AMReX_DistributionMapping.H>
//#include <AMReX_Geometry.H>
//#include <AMReX_ParallelDescriptor.H>
//#include <AMReX_Vector.H>
//#include <AMReX_PlotFileUtil.H>

//#include <hdf5.h>

#include <polyscope/polyscope.h>
#include <polyscope/point_cloud.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/volume_mesh.h>


namespace IOFunc {
//#include <vector>
//#include <fstream>
//#include <boost/filesystem.hpp>

    //namespace bf = boost::filesystem;
    //namespace bf = 

    void WriteHAHostTileHolderToFile(const HAHostTileHolder<Tile>& holder, const fs::path& filepath) {
        std::ofstream out(filepath.string(), std::ios::binary);
        if (!out) {
            throw std::runtime_error("Failed to open file for writing: " + filepath.string());
        }

        // 写入简单成员变量
        out.write(reinterpret_cast<const char*>(&holder.mH0), sizeof(holder.mH0));
        out.write(reinterpret_cast<const char*>(&holder.mNumLevels), sizeof(holder.mNumLevels));
        out.write(reinterpret_cast<const char*>(&holder.mMaxLevel), sizeof(holder.mMaxLevel));

        // 写入 mHostTiles
        size_t numTiles = holder.mHostTiles.size();
        out.write(reinterpret_cast<const char*>(&numTiles), sizeof(numTiles));
        for (const auto& tile : holder.mHostTiles) {
            out.write(reinterpret_cast<const char*>(&tile), sizeof(tile));
        }

        // 写入 mHostLevels
        size_t numLevels = holder.mHostLevels.size();
        out.write(reinterpret_cast<const char*>(&numLevels), sizeof(numLevels));
        for (const auto& level : holder.mHostLevels) {
            size_t levelSize = level.size();
            out.write(reinterpret_cast<const char*>(&levelSize), sizeof(levelSize));
            for (const auto& tileInfo : level) {
                // 写入 mTileCoord 和其他成员变量
                out.write(reinterpret_cast<const char*>(&tileInfo.mTileCoord), sizeof(tileInfo.mTileCoord));
                out.write(reinterpret_cast<const char*>(&tileInfo.mLevel), sizeof(tileInfo.mLevel));
                out.write(reinterpret_cast<const char*>(&tileInfo.mType), sizeof(tileInfo.mType));

                // 写入 mTilePtr 的索引
                intptr_t tileIndex = tileInfo.mTilePtr ? (tileInfo.mTilePtr - holder.mHostTiles.data()) : -1;
                out.write(reinterpret_cast<const char*>(&tileIndex), sizeof(tileIndex));
            }
        }

        out.close();
    }

    

    HAHostTileHolder<Tile> ReadHAHostTileHolderFromFile(const fs::path& filepath) {
        std::ifstream in(filepath.string(), std::ios::binary);
        if (!in) {
            throw std::runtime_error("Failed to open file for reading: " + filepath.string());
        }

        HAHostTileHolder<Tile> holder;

        // 读取简单成员变量
        in.read(reinterpret_cast<char*>(&holder.mH0), sizeof(holder.mH0));
        in.read(reinterpret_cast<char*>(&holder.mNumLevels), sizeof(holder.mNumLevels));
        in.read(reinterpret_cast<char*>(&holder.mMaxLevel), sizeof(holder.mMaxLevel));

        // 读取 mHostTiles
        size_t numTiles;
        in.read(reinterpret_cast<char*>(&numTiles), sizeof(numTiles));
        holder.mHostTiles.resize(numTiles);
        for (auto& tile : holder.mHostTiles) {
            in.read(reinterpret_cast<char*>(&tile), sizeof(tile));
        }

        // 读取 mHostLevels
        size_t numLevels;
        in.read(reinterpret_cast<char*>(&numLevels), sizeof(numLevels));
        holder.mHostLevels.resize(numLevels);
        for (auto& level : holder.mHostLevels) {
            size_t levelSize;
            in.read(reinterpret_cast<char*>(&levelSize), sizeof(levelSize));
            level.resize(levelSize);
            for (auto& tileInfo : level) {
                // 读取 mTileCoord 和其他成员变量
                in.read(reinterpret_cast<char*>(&tileInfo.mTileCoord), sizeof(tileInfo.mTileCoord));
                in.read(reinterpret_cast<char*>(&tileInfo.mLevel), sizeof(tileInfo.mLevel));
                in.read(reinterpret_cast<char*>(&tileInfo.mType), sizeof(tileInfo.mType));

                // 读取 mTilePtr 的索引并重建指针
                intptr_t tileIndex;
                in.read(reinterpret_cast<char*>(&tileIndex), sizeof(tileIndex));
                tileInfo.mTilePtr = (tileIndex >= 0) ? &holder.mHostTiles[tileIndex] : nullptr;
            }
        }

        in.close();

        return holder;
    }

    void OutputMarkerParticleSystemAsVTU(std::shared_ptr<thrust::host_vector<MarkerParticle>> particles_ptr, fs::path path) {
        fmt::print("Output Particle System to vtu file: {}\n", path.string());
        auto& particles = *particles_ptr;

        // setup VTK
        vtkNew<vtkXMLUnstructuredGridWriter> writer;
        vtkNew<vtkUnstructuredGrid> unstructured_grid;

        // Use vtkFloatArray for position data (float instead of double)
        vtkNew<vtkFloatArray> positions;
        positions->SetNumberOfComponents(3);  // 3D points
        positions->SetNumberOfTuples(particles.size());
        positions->SetName("Positions");

        //// Use vtkTypeInt64Array for global_idx
        //vtkNew<vtkTypeInt64Array> global_idx_array;
        //global_idx_array->SetName("global_idx");
        //global_idx_array->SetNumberOfComponents(1);  // Scalar data
        //global_idx_array->SetNumberOfTuples(particles.size());

        for (int i = 0; i < particles.size(); i++) {
            auto p = particles[i].pos;  // Assume pos is a float array or convertible to float
            //auto global_idx = particles[i].global_idx;

            positions->SetTuple3(i, p[0], p[1], p[2]);  // Add float position data
            //global_idx_array->SetTuple1(i, global_idx); // Add global_idx data
        }

        // Set points for the grid
        vtkNew<vtkPoints> nodes;
        nodes->SetData(positions);
        unstructured_grid->SetPoints(nodes);

        // Add global_idx array as point data
        //unstructured_grid->GetPointData()->AddArray(global_idx_array);

        // Write the output file
        writer->SetFileName(path.string().c_str());
        writer->SetInputData(unstructured_grid);
        writer->SetDataModeToBinary();  // Optional: Use binary mode for smaller file size
        writer->Write();
    }

    void OutputParticleSystemAsVTU(std::shared_ptr<thrust::host_vector<Particle>> particles_ptr, fs::path path) {
        CPUTimer timer; timer.start();
        //fmt::print("Output Particle System to vtu file: {}\n", path.string());
        auto& particles = *particles_ptr;

        // setup VTK
        vtkNew<vtkXMLUnstructuredGridWriter> writer;
        vtkNew<vtkUnstructuredGrid> unstructured_grid;

        // Use vtkFloatArray for position data (float instead of double)
        vtkNew<vtkFloatArray> positions;
        positions->SetNumberOfComponents(3);  // 3D points
        positions->SetNumberOfTuples(particles.size());
        positions->SetName("Positions");

        //// Use vtkTypeInt64Array for global_idx
        //vtkNew<vtkTypeInt64Array> global_idx_array;
        //global_idx_array->SetName("global_idx");
        //global_idx_array->SetNumberOfComponents(1);  // Scalar data
        //global_idx_array->SetNumberOfTuples(particles.size());

        for (size_t i = 0; i < particles.size(); i++) {
            auto p = particles[i].pos;  // Assume pos is a float array or convertible to float
            //auto global_idx = particles[i].global_idx;

            positions->SetTuple3(i, p[0], p[1], p[2]);  // Add float position data
            //global_idx_array->SetTuple1(i, global_idx); // Add global_idx data
        }

        // Set points for the grid
        vtkNew<vtkPoints> nodes;
        nodes->SetData(positions);
        unstructured_grid->SetPoints(nodes);

        // Add global_idx array as point data
        //unstructured_grid->GetPointData()->AddArray(global_idx_array);

        // Write the output file
        writer->SetFileName(path.string().c_str());
        writer->SetInputData(unstructured_grid);
        writer->SetDataModeToBinary();  // Optional: Use binary mode for smaller file size
        writer->Write();
        
		auto elapsed = timer.stop();
		Pass("Finished writing particle system to VTU file: {} ({} ms)", path.string(), elapsed);
    }

  //  void OutputMarkerParticleSystemAsVTU(const thrust::device_vector<MarkerParticle>& particles_d, const fs::path& path) {
  //      fmt::print("Output Particle System to vtu file: {}\n", path.string());
		//auto particles_ptr = std::make_shared<thrust::host_vector<MarkerParticle>>(particles_d);
  //      OutputMarkerParticleSystemAsVTU(particles_ptr, path);
  //  }

    void AddMarkerParticlesToPolyscope(thrust::device_vector<MarkerParticle> particles_d, std::string name) {
		thrust::host_vector<MarkerParticle> particles = particles_d;

        std::vector<Vec> positions;
        //std::vector<Vec> impulses;

        positions.reserve(particles.size());
		//impulses.reserve(particles.size());

        for (int i = 0; i < particles.size(); i++) {
            auto p = particles[i].pos;
            //auto v = particles[i].impulse;
            positions.emplace_back(p[0], p[1], p[2]);
            //impulses.emplace_back(v[0], v[1], v[2]);
        }

        polyscope::PointCloud* psCloud = polyscope::registerPointCloud(name, positions);
        //psCloud->addVectorQuantity("impulse", impulses);
    }

    void AddParticlesToPolyscope(thrust::device_vector<Particle> particles_d, std::string name) {
        thrust::host_vector<Particle> particles = particles_d;

        std::vector<Vec> positions;
        std::vector<Vec> impulses;

        positions.reserve(particles.size());
        impulses.reserve(particles.size());

        for (int i = 0; i < particles.size(); i++) {
            auto p = particles[i].pos;
            auto v = particles[i].impulse;
            positions.emplace_back(p[0], p[1], p[2]);
            impulses.emplace_back(v[0], v[1], v[2]);
        }

        polyscope::PointCloud* psCloud = polyscope::registerPointCloud(name, positions);
        psCloud->addVectorQuantity("impulse", impulses);
    }

    void OutputTilesAsVTU(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const fs::path& path) {
        using Coord = typename Tile::CoordType;

        fmt::print("Output Leaf Tiles in DA Device Grid to vtu file: {}\n", path.string());
        // Create VTK unstructured grid
        vtkSmartPointer<vtkUnstructuredGrid> unstructuredGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();
        // Create VTK points
        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();


        int hex_offs[8][3] = {
            {0,0,0},{1,0,0},{1,1,0},{0,1,0},
            {0,0,1},{1,0,1},{1,1,1},{0,1,1} };

		auto& holder = *holder_ptr;
        auto c_acc = holder.coordAccessor();
        holder.iterateLeafTiles(
            [&](const HATileInfo<Tile>& info) {
                vtkNew<vtkHexahedron> hex;
                auto bbox = c_acc.tileBBox(info);
                for (int s = 0; s < 8; s++) {
                    Coord off(hex_offs[s][0], hex_offs[s][1], hex_offs[s][2]);
                    auto p = bbox.min() + Vec(off[0], off[1], off[2]) * bbox.dim();
                    auto idx = points->InsertNextPoint(p[0], p[1], p[2]);
                    hex->GetPointIds()->SetId(s, idx);
                }
                unstructuredGrid->InsertNextCell(hex->GetCellType(), hex->GetPointIds());
            }
        );

        unstructuredGrid->SetPoints(points);

        // Write to vtu file
        vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
        writer->SetFileName(path.string().c_str());
        writer->SetInputData(unstructuredGrid);
        writer->SetDataModeToBinary();
        writer->Write();
    }

    void OutputPoissonGridAsUnstructuredVTU(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const std::vector<std::pair<int, std::string>> scalar_channels, std::vector<std::pair<int, std::string>> vec_channels, const fs::path& path) {
        CPUTimer<std::chrono::milliseconds> timer;
        timer.start();

        auto& holder = *holder_ptr;

        std::vector<HATileInfo<Tile> > leaf_infos;
		for (int i = 0; i <= holder.mMaxLevel; i++) {
			for (auto& info : holder.mHostLevels[i]) {
				if (info.isLeaf()) {
                    leaf_infos.push_back(info);
				}
			}
		}
		int num_leafs = leaf_infos.size();
		int num_leaf_voxels = num_leafs * Tile::SIZE;
		int num_leaf_nodes = num_leafs * Tile::NODESIZE;

        using Coord = typename Tile::CoordType;

        auto acc = holder.coordAccessor();

        fmt::print("Start output Poisson Grid to vtu file: {}\n", path.string());


        vtkNew<vtkPoints> points;
        //load points
        {
            points->SetNumberOfPoints(num_leaf_nodes);
            for (int i = 0; i < leaf_infos.size(); i++) {
                auto& info = leaf_infos[i];
                int base_off = i * Tile::NODESIZE;
                for (int j = 0; j < Tile::NODESIZE; j++) {

                    auto r_ijk = acc.localNodeOffsetToCoord(j);

                    auto pos = acc.cellCorner(info, r_ijk);
                    points->SetPoint(base_off + j, pos[0], pos[1], pos[2]);
                }
            }
        }

        // Create VTK unstructured grid
        vtkNew<vtkIdTypeArray> hex_vert_ids;
        //load hex vert ids
        {
            hex_vert_ids->SetNumberOfValues(num_leafs * Tile::SIZE * (1 + 8));
            int hex_offs[8][3] = {
    {0,0,0},{1,0,0},{1,1,0},{0,1,0},
    {0,0,1},{1,0,1},{1,1,1},{0,1,1} };
            for (int i = 0; i < leaf_infos.size(); i++) {
                int tile_node_off = i * Tile::NODESIZE;
                for (int j = 0; j < Tile::SIZE; j++) {
                    Coord l_ijk = acc.localOffsetToCoord(j);
                    hex_vert_ids->SetValue(i * Tile::SIZE * 9 + j * 9 + 0, 8);

                    for (int s = 0; s < 8; s++) {
                        Coord off(hex_offs[s][0], hex_offs[s][1], hex_offs[s][2]);
                        auto idx = acc.localNodeCoordToOffset(l_ijk + off);
						hex_vert_ids->SetValue(i * Tile::SIZE * 9 + j * 9 + s + 1, tile_node_off + idx);
                        //hex_vert_ids->SetValue(i * Tile::SIZE * 8 + j * 8 + s, tile_node_off + idx);
                    }
                }
            }
        }

        vtkNew<vtkCellArray> hexagons;
        {
            hexagons->SetCells(num_leaf_voxels, hex_vert_ids);
        }
        vtkNew<vtkUnstructuredGrid> unstructuredGrid;
		unstructuredGrid->SetPoints(points);
		unstructuredGrid->SetCells(VTK_HEXAHEDRON, hexagons);


        std::vector<vtkSmartPointer<vtkFloatArray>> scalar_data;
        std::vector<vtkSmartPointer<vtkFloatArray>> vec_data;

        for (int i = 0; i < scalar_channels.size(); i++) {
            auto data = vtkSmartPointer<vtkFloatArray>::New();
            data->SetName(scalar_channels[i].second.c_str());
            data->SetNumberOfComponents(1);
            data->Allocate(num_leaf_voxels);
            scalar_data.push_back(data);
        }

        for (int i = 0; i < vec_channels.size(); i++) {
            auto data = vtkSmartPointer<vtkFloatArray>::New();
            data->SetName(vec_channels[i].second.c_str());
            data->SetNumberOfComponents(3);
            data->Allocate(num_leaf_voxels);
            vec_data.push_back(data);
        }

        for (int info_idx = 0; info_idx < leaf_infos.size(); info_idx++) {
			int node_base_off = info_idx * Tile::NODESIZE;
			int cell_base_off = info_idx * Tile::SIZE;

            auto& info = leaf_infos[info_idx];
            if (info.isLeaf()) {

                auto& tile = info.tile();
                for (int cell_idx = 0; cell_idx < Tile::SIZE; cell_idx++) {
					Coord l_ijk = acc.localOffsetToCoord(cell_idx);
                    auto g_ijk = acc.localToGlobalCoord(info, l_ijk);

                    //type_data->InsertNextValue(tile.type(l_ijk));
                    for (int i = 0; i < scalar_channels.size(); i++) {
                        int channel = scalar_channels[i].first;
                        T f = (channel == -1) ? tile.type(l_ijk) : tile(scalar_channels[i].first, l_ijk);
                        scalar_data[i]->InsertNextValue(f);
                    }
                    for (int i = 0; i < vec_channels.size(); i++) {
                        int u_channel = vec_channels[i].first;
                        auto u = tile(u_channel, l_ijk);
                        auto v = tile(u_channel + 1, l_ijk);
                        auto w = tile(u_channel + 2, l_ijk);
                        vec_data[i]->InsertNextTuple3(u, v, w);
                    }
                }
            }
        }

        for (int i = 0; i < scalar_channels.size(); i++) {
			unstructuredGrid->GetCellData()->AddArray(scalar_data[i]);
        }
        for (int i = 0; i < vec_channels.size(); i++) {
            unstructuredGrid->GetCellData()->AddArray(vec_data[i]);
        }

        //timer.stop("build output structure");
        //timer.start();


        // Write to vtu file
        vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
        writer->SetFileName(path.string().c_str());
        writer->SetInputData(unstructuredGrid);
        writer->SetDataModeToBinary();
        writer->SetCompressorTypeToZLib();
        writer->Write();

        timer.stop(fmt::format("Output grid to unstructured .vtu file: {}", path.string()));
    }

    void OutputPoissonGridAsStructuredVTI(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const std::vector<std::pair<int, std::string>> scalar_channels, std::vector<std::pair<int, std::string>> vec_channels, const fs::path& path)
    {
        CPUTimer<std::chrono::milliseconds> timer;
        timer.start();

        auto& holder = *holder_ptr;
        std::vector<HATileInfo<Tile>> leaf_infos;

        for (int level = 0; level <= holder.mMaxLevel; level++) {
            for (auto& info : holder.mHostLevels[level]) {
                if (info.isLeaf()) {
                    leaf_infos.push_back(info);
                }
            }
        }

        ASSERT(!leaf_infos.empty(), "OutputPoissonGridAsStructuredVTI: no leaf tiles to output");

        auto acc = holder.coordAccessor();
        const int max_level = holder.mMaxLevel;

        Coord global_min(
            std::numeric_limits<int>::max(),
            std::numeric_limits<int>::max(),
            std::numeric_limits<int>::max()
        );
        Coord global_max(
            std::numeric_limits<int>::lowest(),
            std::numeric_limits<int>::lowest(),
            std::numeric_limits<int>::lowest()
        );

        for (auto& info : leaf_infos) {
            const int level = info.mLevel;
            const int shift = max_level - level;

            for (Coord l_ijk : {Coord(0, 0, 0), Coord(Tile::DIM - 1, Tile::DIM - 1, Tile::DIM - 1)}) {
                Coord g_ijk = acc.localToGlobalCoord(info, l_ijk);

                Coord min_fine_coord(
                    g_ijk[0] << shift,
                    g_ijk[1] << shift,
                    g_ijk[2] << shift
                );
                Coord max_fine_coord(
                    (g_ijk[0] + 1) << shift,
                    (g_ijk[1] + 1) << shift,
                    (g_ijk[2] + 1) << shift
                );

                global_min[0] = std::min(global_min[0], min_fine_coord[0]);
                global_min[1] = std::min(global_min[1], min_fine_coord[1]);
                global_min[2] = std::min(global_min[2], min_fine_coord[2]);

                global_max[0] = std::max(global_max[0], max_fine_coord[0]);
                global_max[1] = std::max(global_max[1], max_fine_coord[1]);
                global_max[2] = std::max(global_max[2], max_fine_coord[2]);
            }
        }

        Coord cell_dims = global_max - global_min;

        ASSERT(cell_dims[0] > 0 && cell_dims[1] > 0 && cell_dims[2] > 0,
            "OutputPoissonGridAsStructuredVTI: invalid cell dims [{}, {}, {}], global_min={}, global_max={}",
            cell_dims[0], cell_dims[1], cell_dims[2], global_min, global_max);

        const int64_t nx = static_cast<int64_t>(cell_dims[0]);
        const int64_t ny = static_cast<int64_t>(cell_dims[1]);
        const int64_t nz = static_cast<int64_t>(cell_dims[2]);
        const int64_t num_cells = nx * ny * nz;

        ASSERT(num_cells > 0,
            "OutputPoissonGridAsStructuredVTI: num_cells must be positive, got {}", num_cells);

        vtkNew<vtkImageData> structuredGrid;
        structuredGrid->SetDimensions(cell_dims[0] + 1, cell_dims[1] + 1, cell_dims[2] + 1);
        structuredGrid->SetOrigin(global_min[0], global_min[1], global_min[2]);

        auto h = acc.voxelSize(max_level);
        structuredGrid->SetSpacing(h, h, h);

        std::vector<vtkSmartPointer<vtkFloatArray>> scalar_data;
        std::vector<vtkSmartPointer<vtkFloatArray>> vec_data;
        std::vector<float*> scalar_ptrs;
        std::vector<float*> vec_ptrs;

        scalar_data.reserve(scalar_channels.size());
        vec_data.reserve(vec_channels.size());
        scalar_ptrs.reserve(scalar_channels.size());
        vec_ptrs.reserve(vec_channels.size());

        for (const auto& scalar_channel : scalar_channels) {
            auto data = vtkSmartPointer<vtkFloatArray>::New();
            data->SetName(scalar_channel.second.c_str());
            data->SetNumberOfComponents(1);
            data->SetNumberOfTuples(static_cast<vtkIdType>(num_cells));
            data->Fill(0.0f);
            scalar_ptrs.push_back(data->GetPointer(0));
            scalar_data.push_back(data);
        }

        for (const auto& vec_channel : vec_channels) {
            auto data = vtkSmartPointer<vtkFloatArray>::New();
            data->SetName(vec_channel.second.c_str());
            data->SetNumberOfComponents(3);
            data->SetNumberOfTuples(static_cast<vtkIdType>(num_cells));
            data->FillComponent(0, 0.0);
            data->FillComponent(1, 0.0);
            data->FillComponent(2, 0.0);
            vec_ptrs.push_back(data->GetPointer(0));
            vec_data.push_back(data);
        }

        struct Vec3f {
            float x, y, z;
        };

        std::vector<float> scalar_vals(scalar_channels.size());
        std::vector<Vec3f> vec_vals(vec_channels.size());

        for (auto& info : leaf_infos) {
            const auto& tile = info.tile();
            const int level = info.mLevel;
            const int shift = max_level - level;

            ASSERT(shift >= 0, "OutputPoissonGridAsStructuredVTI: negative shift {}", shift);

            for (int cell_idx = 0; cell_idx < Tile::SIZE; cell_idx++) {
                Coord l_ijk = acc.localOffsetToCoord(cell_idx);
                Coord g_ijk = acc.localToGlobalCoord(info, l_ijk);

                const int x0 = (g_ijk[0] << shift) - global_min[0];
                const int y0 = (g_ijk[1] << shift) - global_min[1];
                const int z0 = (g_ijk[2] << shift) - global_min[2];
                const int span = 1 << shift;

                const int x1 = x0 + span;
                const int y1 = y0 + span;
                const int z1 = z0 + span;

                ASSERT(0 <= x0 && x0 < x1 && x1 <= nx, "x range invalid: [{}, {}) nx={}", x0, x1, nx);
                ASSERT(0 <= y0 && y0 < y1 && y1 <= ny, "y range invalid: [{}, {}) ny={}", y0, y1, ny);
                ASSERT(0 <= z0 && z0 < z1 && z1 <= nz, "z range invalid: [{}, {}) nz={}", z0, z1, nz);

                for (int i = 0; i < static_cast<int>(scalar_channels.size()); i++) {
                    int channel = scalar_channels[i].first;
                    T value;
                    if (channel == -1) value = static_cast<T>(tile.type(l_ijk));
                    else if (channel == -2) value = static_cast<T>(level);
                    else value = tile(channel, l_ijk);

                    ASSERT(std::isfinite(value),
                        "Non-finite scalar value at channel {}, l_ijk={}, level={}", channel, l_ijk, level);

                    scalar_vals[i] = static_cast<float>(value);
                }

                for (int i = 0; i < static_cast<int>(vec_channels.size()); i++) {
                    int u_channel = vec_channels[i].first;
                    T u = tile(u_channel, l_ijk);
                    T v = tile(u_channel + 1, l_ijk);
                    T w = tile(u_channel + 2, l_ijk);

                    ASSERT(std::isfinite(u) && std::isfinite(v) && std::isfinite(w),
                        "Non-finite vector value at channels [{}, {}, {}], l_ijk={}, level={}",
                        u_channel, u_channel + 1, u_channel + 2, l_ijk, level);

                    vec_vals[i] = Vec3f{ static_cast<float>(u), static_cast<float>(v), static_cast<float>(w) };
                }

                for (int z = z0; z < z1; ++z) {
                    const int64_t z_base = static_cast<int64_t>(z) * nx * ny;
                    for (int y = y0; y < y1; ++y) {
                        const int64_t row_base = static_cast<int64_t>(y) * nx + z_base;
                        const int64_t begin = row_base + x0;
                        const int64_t end = row_base + x1;

                        for (int i = 0; i < static_cast<int>(scalar_ptrs.size()); i++) {
                            std::fill(scalar_ptrs[i] + begin, scalar_ptrs[i] + end, scalar_vals[i]);
                        }

                        const int row_len = x1 - x0;
                        for (int i = 0; i < static_cast<int>(vec_ptrs.size()); i++) {
                            float* ptr = vec_ptrs[i] + 3 * begin;
                            const float vx = vec_vals[i].x;
                            const float vy = vec_vals[i].y;
                            const float vz = vec_vals[i].z;

                            for (int k = 0; k < row_len; ++k) {
                                ptr[0] = vx;
                                ptr[1] = vy;
                                ptr[2] = vz;
                                ptr += 3;
                            }
                        }
                    }
                }
            }
        }

        for (auto& data : scalar_data) {
            structuredGrid->GetCellData()->AddArray(data);
        }
        for (auto& data : vec_data) {
            structuredGrid->GetCellData()->AddArray(data);
        }

        vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
        writer->SetFileName(path.string().c_str());
        writer->SetInputData(structuredGrid);
        writer->SetDataModeToBinary();
        // writer->SetCompressorTypeToZLib();

        const int ok = writer->Write();
        ASSERT(ok == 1, "vtkXMLImageDataWriter failed to write {}", path.string());

        double elapsed = timer.stop();
        Pass("Finished writing Poisson grid to structured VTI file: {} ({} ms)", path.string(), elapsed);
    }

    void AddLeveledTilesToPolyscopeVolumetricMesh(HADeviceGrid<Tile>& grid, const uint8_t types, const std::string& base_name) {
        using Coord = typename Tile::CoordType;

        int hex_offs[8][3] = {
            {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
            {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
        };

        auto acc = grid.hostAccessor();

        for (int level = 0; level < grid.mNumLevels; ++level) {
            std::vector<Vec> vertices;
            std::vector<std::array<size_t, 8>> hexCells;
            std::vector<int> cellLevels;
            std::vector<Coord> cellCoords;
            std::vector<int> cellInterestFlags;

            // 用整数坐标做 key，避免浮点比较
            struct CoordKey {
                int x, y, z;
                bool operator==(const CoordKey& other) const {
                    return x == other.x && y == other.y && z == other.z;
                }
            };

            struct CoordKeyHash {
                size_t operator()(const CoordKey& k) const {
                    size_t h1 = std::hash<int>{}(k.x);
                    size_t h2 = std::hash<int>{}(k.y);
                    size_t h3 = std::hash<int>{}(k.z);
                    return h1 ^ (h2 << 1) ^ (h3 << 2);
                }
            };

            std::unordered_map<CoordKey, size_t, CoordKeyHash> vertexMap;

            for (int i = 0; i < grid.hNumTiles[level]; ++i) {
                auto& info = grid.hTileArrays[level][i];
                if (!(info.mType & types)) continue;

                auto bbox = acc.tileBBox(info);
                std::array<size_t, 8> hexCell;

                for (int s = 0; s < 8; ++s) {
                    Coord off(hex_offs[s][0], hex_offs[s][1], hex_offs[s][2]);

                    auto p = bbox.min() + Vec(off[0], off[1], off[2]) * bbox.dim();
                    Vec vertex(p[0], p[1], p[2]);

                    // 用 bbox 的角点整数索引作为 key
                    auto c = info.mTileCoord + off;
                    CoordKey key{ (int)c[0], (int)c[1], (int)c[2] };

                    auto it = vertexMap.find(key);
                    if (it != vertexMap.end()) {
                        hexCell[s] = it->second;
                    }
                    else {
                        size_t id = vertices.size();
                        vertices.push_back(vertex);
                        vertexMap[key] = id;
                        hexCell[s] = id;
                    }
                }

                hexCells.push_back(hexCell);
                cellLevels.push_back(level);
                cellCoords.push_back(info.mTileCoord);

                auto tile = info.getTile(DEVICE);
                cellInterestFlags.push_back(tile.mIsInterestArea);
            }

            if (hexCells.empty()) continue;

            std::string name = fmt::format("{}_level_{}", base_name, level);
            auto mesh = polyscope::registerVolumeMesh(name, vertices, hexCells);
            mesh->addCellScalarQuantity("Level", cellLevels);
            mesh->addCellVectorQuantity("Tile Coord", cellCoords);
            mesh->addCellScalarQuantity("Interest Flag", cellInterestFlags);
            mesh->setTransparency(0.2);
        }
    }

    void OutputPoissonGridAsAMR(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const std::vector<std::pair<int, std::string>>& scalar_channels, const std::vector<std::pair<int, std::string>>& vec_channels, const fs::path& path)
    {
        //should output .vthb
        ASSERT(path.extension() == ".vthb",
            "OutputPoissonGridAsAMR: path must end with .vthb, got {}", path.string());

        CPUTimer<std::chrono::milliseconds> timer;
        timer.start();

        auto& holder = *holder_ptr;

        auto acc = holder.coordAccessor();
        const int max_level = holder.mMaxLevel;
        const int num_levels = max_level + 1;

        // Collect leaf blocks per level.
        std::vector<std::vector<HATileInfo<Tile>>> leaf_infos_per_level(num_levels);
        std::vector<unsigned int> blocks_per_level(num_levels, 0);

        for (int level = 0; level <= max_level; ++level) {
            for (auto& info : holder.mHostLevels[level]) {
                if (info.isLeaf()) {
                    leaf_infos_per_level[level].push_back(info);
                }
            }
            blocks_per_level[level] = static_cast<unsigned int>(leaf_infos_per_level[level].size());
        }

        unsigned int total_blocks = 0;
        for (auto n : blocks_per_level) total_blocks += n;

        ASSERT(total_blocks > 0, "OutputPoissonGridAsAMR: no leaf tiles to output");

        vtkNew<vtkOverlappingAMR> amr;

#if VTK_VERSION_NUMBER >= 90520250724ULL
        amr->Initialize(blocks_per_level);
#else
        std::vector<int> blocks_per_level_int(num_levels);
        for (int i = 0; i < num_levels; ++i) {
            blocks_per_level_int[i] = static_cast<int>(blocks_per_level[i]);
        }
        amr->Initialize(num_levels, blocks_per_level_int.data());
#endif
        double amr_origin[3] = { 0.0, 0.0, 0.0 };
        amr->SetOrigin(amr_origin);

        for (int level = 0; level <= max_level; ++level) {
            double h = acc.voxelSize(level);
            double spacing[3] = { h, h, h };
            amr->SetSpacing(level, spacing);
        }



        // Optional but recommended: specify refinement ratio between adjacent levels.
        // Here we assume standard octree refinement ratio = 2 in each dimension.
        for (unsigned int level = 0; level + 1 < static_cast<unsigned int>(num_levels); ++level) {
            amr->SetRefinementRatio(level, 2);
        }

        // Build one vtkUniformGrid for each leaf tile.
        for (int level = 0; level <= max_level; ++level) {
            const auto h = acc.voxelSize(level);

            for (int block_id = 0; block_id < static_cast<int>(leaf_infos_per_level[level].size()); ++block_id) {
                const auto& info = leaf_infos_per_level[level][block_id];
                const auto& tile = info.tile();

                // Global cell coordinate of the tile's first cell at THIS level.
                Coord g0 = acc.localToGlobalCoord(info, Coord(0, 0, 0));

                // AMRBox uses cell-index bounds, inclusive on both ends.
                int lo[3] = {
                    static_cast<int>(g0[0]),
                    static_cast<int>(g0[1]),
                    static_cast<int>(g0[2])
                };
                int hi[3] = {
                    static_cast<int>(g0[0] + Tile::DIM - 1),
                    static_cast<int>(g0[1] + Tile::DIM - 1),
                    static_cast<int>(g0[2] + Tile::DIM - 1)
                };

                vtkAMRBox box(lo, hi);
                amr->SetAMRBox(level, block_id, box);

                vtkNew<vtkUniformGrid> ug;

                // UniformGrid dimensions are point dimensions = cell dims + 1.
                ug->SetDimensions(Tile::DIM + 1, Tile::DIM + 1, Tile::DIM + 1);

                // World-space origin of the block.
                // Since g0 is the first cell index on this AMR level, the min corner is g0 * h.
                ug->SetOrigin(
                    static_cast<double>(g0[0]) * h,
                    static_cast<double>(g0[1]) * h,
                    static_cast<double>(g0[2]) * h);

                ug->SetSpacing(h, h, h);

                // -------- Cell data arrays --------
                std::vector<vtkSmartPointer<vtkFloatArray>> scalar_data;
                std::vector<vtkSmartPointer<vtkFloatArray>> vec_data;
                std::vector<float*> scalar_ptrs;
                std::vector<float*> vec_ptrs;

                scalar_data.reserve(scalar_channels.size());
                vec_data.reserve(vec_channels.size());
                scalar_ptrs.reserve(scalar_channels.size());
                vec_ptrs.reserve(vec_channels.size());

                const vtkIdType num_cells = static_cast<vtkIdType>(Tile::SIZE);

                for (const auto& scalar_channel : scalar_channels) {
                    auto data = vtkSmartPointer<vtkFloatArray>::New();
                    data->SetName(scalar_channel.second.c_str());
                    data->SetNumberOfComponents(1);
                    data->SetNumberOfTuples(num_cells);
                    scalar_ptrs.push_back(data->GetPointer(0));
                    scalar_data.push_back(data);
                }

                for (const auto& vec_channel : vec_channels) {
                    auto data = vtkSmartPointer<vtkFloatArray>::New();
                    data->SetName(vec_channel.second.c_str());
                    data->SetNumberOfComponents(3);
                    data->SetNumberOfTuples(num_cells);
                    vec_ptrs.push_back(data->GetPointer(0));
                    vec_data.push_back(data);
                }

                // Fill per-cell data. Here cell ordering is tile-local ordering.
                for (int cell_idx = 0; cell_idx < Tile::SIZE; ++cell_idx) {
                    Coord l_ijk = acc.localOffsetToCoord(cell_idx);
                    const vtkIdType cid = static_cast<vtkIdType>(cell_idx);

                    for (int i = 0; i < static_cast<int>(scalar_channels.size()); ++i) {
                        int channel = scalar_channels[i].first;
                        T value;
                        if (channel == -1) value = static_cast<T>(tile.type(l_ijk));
                        else if (channel == -2) value = static_cast<T>(level);
                        else value = tile(channel, l_ijk);

                        ASSERT(std::isfinite(value),
                            "Non-finite scalar value at channel {}, l_ijk={}, level={}",
                            channel, l_ijk, level);

                        scalar_ptrs[i][cid] = static_cast<float>(value);
                    }

                    for (int i = 0; i < static_cast<int>(vec_channels.size()); ++i) {
                        int u_channel = vec_channels[i].first;
                        T u = tile(u_channel, l_ijk);
                        T v = tile(u_channel + 1, l_ijk);
                        T w = tile(u_channel + 2, l_ijk);

                        ASSERT(std::isfinite(u) && std::isfinite(v) && std::isfinite(w),
                            "Non-finite vector value at channels [{}, {}, {}], l_ijk={}, level={}",
                            u_channel, u_channel + 1, u_channel + 2, l_ijk, level);

                        float* ptr = vec_ptrs[i] + 3 * cid;
                        ptr[0] = static_cast<float>(u);
                        ptr[1] = static_cast<float>(v);
                        ptr[2] = static_cast<float>(w);
                    }
                }

                for (auto& data : scalar_data) {
                    ug->GetCellData()->AddArray(data);
                }
                for (auto& data : vec_data) {
                    ug->GetCellData()->AddArray(data);
                }

                amr->SetDataSet(level, block_id, ug);
            }
        }

        vtkNew<vtkXMLUniformGridAMRWriter> writer;
        writer->SetFileName(path.string().c_str());
        writer->SetInputData(amr);
        writer->SetDataModeToBinary();
        // writer->SetCompressorTypeToZLib(); // 先关掉，优先追求输出速度

        const int ok = writer->Write();
        ASSERT(ok == 1, "vtkXMLUniformGridAMRWriter failed to write {}", path.string());

        double elapsed = timer.stop();
        Pass("Finished writing Poisson grid to AMR file: {} ({} ms, {} blocks)",
            path.string(), elapsed, total_blocks);
    }

    void AddLeveledPoissonGridCellCentersToPolyscopePointCloud(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const std::vector<std::pair<int, std::string>> scalar_channels, std::vector<std::pair<int, std::string>> vec_channels, int level, const double invalid_value) {
        auto& holder = *holder_ptr;
        using Coord = typename Tile::CoordType;

        auto acc = holder.coordAccessor();
        auto add_data = [&](const int level, const uint8_t tile_types, const std::string name) {
            std::vector<Vec> points;
            std::vector<int> levels;
            std::vector<int> ttypes;
            std::vector<Coord> ijk_values;

            std::vector<std::vector<float>> scalar_data(scalar_channels.size());
            std::vector<std::vector<Vec>> vec_data(vec_channels.size());
			std::vector<std::vector<float>> vec_length(vec_channels.size());

            for (auto& info : holder.mHostLevels[level]) {
                if (!(info.mType & tile_types)) continue;
                auto& tile = info.tile();

                for (int i = 0; i < Tile::DIM; i++) {
                    for (int j = 0; j < Tile::DIM; j++) {
                        for (int k = 0; k < Tile::DIM; k++) {
                            Coord l_ijk(i, j, k);

                            auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
                            //if (!(g_ijk[0] == 53 && 44 <= g_ijk[1] && g_ijk[2] <= 46)) continue;
                            auto pos = acc.cellCenter(info, l_ijk); // 使用cellCenter代替cellCorner

                            
        //                    {
        //                        auto ref_center = acc.cellCenterGlobal(5, Coord(120, 112, 94));
        //                        if ((pos - ref_center).length() > 0.01) continue;


        ////                        //if (info.mLevel != 5) continue;
        ////                        Coord diff = g_ijk - Coord(232, 238, 368);
        ////                        if (!(info.mLevel == 6 && abs(diff[0]) + abs(diff[1]) + abs(diff[2]) <= 5)) continue;
								//////Info("entering tile {} type {}", info.mTileCoord, info.mType);
        //                    }



                            ijk_values.push_back(g_ijk);

                            

                            points.push_back(pos);
                            levels.push_back(level);
                            ttypes.push_back(info.mType);



                            for (int s = 0; s < scalar_channels.size(); s++) {
                                int channel = scalar_channels[s].first;
                                T f = (channel == -1) ? tile.type(l_ijk) : tile(scalar_channels[s].first, l_ijk);
                                if (!isfinite(f) || fabs(f) > invalid_value) f = invalid_value;
                                scalar_data[s].push_back(f);
                            }

                            for (int t = 0; t < vec_channels.size(); t++) {
                                int u_channel = vec_channels[t].first;
                                auto u = tile(u_channel, l_ijk);
                                auto v = tile(u_channel + 1, l_ijk);
                                auto w = tile(u_channel + 2, l_ijk);
                                vec_data[t].push_back({ u, v, w });

								auto len = std::sqrt(u * u + v * v + w * w);
                                if (!isfinite(len) || len > invalid_value) {
                                    Info("Non-finite or large vector length at level {}, g_ijk {}: ({}, {}, {})", level, g_ijk, u, v, w);
                                    len = invalid_value;
                                }
								vec_length[t].push_back(len);

                                if (len > 100) {
									Info("Large vector magnitude at level {}, g_ijk {}: ({}, {}, {}), len {}", level, g_ijk, u, v, w, len);
                                }
                            }
                        }
                    }
                }
            }

            if (points.empty()) return;
                
            auto ps = polyscope::registerPointCloud(name, points);
            ps->addScalarQuantity("Level", levels);
            ps->addVectorQuantity("IJK", ijk_values);
            ps->addScalarQuantity("Tile Type", ttypes);

            for (int s = 0; s < scalar_channels.size(); s++) {
                ps->addScalarQuantity(scalar_channels[s].second, scalar_data[s]);
            }

            for (int v = 0; v < vec_channels.size(); v++) {
                ps->addVectorQuantity(vec_channels[v].second, vec_data[v]);
				ps->addScalarQuantity(vec_channels[v].second + "_len", vec_length[v]);
            }
            };

        int beg = 0, end = holder.mMaxLevel;
		if (level != -1) beg = end = level;
        for (int i = beg; i <= end; i++) {
            add_data(i, LEAF, fmt::format("Level{}LEAF", i));
            add_data(i, GHOST, fmt::format("Level{}GHOST", i));
            add_data(i, NONLEAF, fmt::format("Level{}NONLEAF", i));
        }
    }

    void AddPoissonGridCellCentersToPolyscopePointCloud(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const std::vector<std::pair<int, std::string>> scalar_channels, std::vector<std::pair<int, std::string>> vec_channels, const double invalid_value) {
        auto& holder = *holder_ptr;
        using Coord = typename Tile::CoordType;

        std::vector<Vec> points;
        std::vector<int> levels;
        std::vector<Coord> ijk_values;

        std::vector<std::vector<float>> scalar_data(scalar_channels.size());
        std::vector<std::vector<Vec>> vec_data(vec_channels.size());
        std::vector<std::vector<float>> vec_length(vec_channels.size());

        auto acc = holder.coordAccessor();
        for (int level = 0; level <= holder.mMaxLevel; level++) {
            for (auto& info : holder.mHostLevels[level]) {
                if (info.isLeaf()) {
                    auto& tile = info.tile();

                    for (int i = 0; i < Tile::DIM; i++) {
                        for (int j = 0; j < Tile::DIM; j++) {
                            for (int k = 0; k < Tile::DIM; k++) {
                                Coord l_ijk(i, j, k);
                                auto pos = acc.cellCenter(info, l_ijk); // 使用cellCenter代替cellCorner
                                points.push_back(pos);
                                levels.push_back(level);

								auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
                                ijk_values.push_back(g_ijk);

                                for (int s = 0; s < scalar_channels.size(); s++) {
                                    int channel = scalar_channels[s].first;
                                    T f = (channel == -1) ? tile.type(l_ijk) : tile(scalar_channels[s].first, l_ijk);
                                    //if(channel!=-1) Info("channel {} l_ijk: {}, f: {}", channel, l_ijk, f);
                                    if (isnan(f) || isinf(f)) f = invalid_value;
                                    scalar_data[s].push_back(f);
                                }

                                for (int t = 0; t < vec_channels.size(); t++) {
                                    int u_channel = vec_channels[t].first;
                                    auto u = tile(u_channel, l_ijk);
                                    auto v = tile(u_channel + 1, l_ijk);
                                    auto w = tile(u_channel + 2, l_ijk);
                                    vec_data[t].push_back({ u, v, w });

									auto len = std::sqrt(u * u + v * v + w * w);
                                    if (!isfinite(len)) {
                                        Info("Non-finite vector length at level {}, g_ijk {}: ({}, {}, {})", level, g_ijk, u, v, w);
                                        len = invalid_value;
                                    }
                                    vec_length[t].push_back(len);

                                    //if (len > 5) {
                                    //    Info("Large vector magnitude at level {}, g_ijk {}: ({}, {}, {}), len {}", level, g_ijk, u, v, w, len);
                                    //}

          //                          if (g_ijk == Coord(5, 0, 1)) {
										//Info("building polyscope level {} g_ijk {} uvw ({}, {}, {}), len {}", level, g_ijk, u, v, w, len);
          //                          }

                                    if (len > 10) {
										Warn("Large vector magnitude at level {}, g_ijk {}: ({}, {}, {})", level, g_ijk, u, v, w);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        polyscope::registerPointCloud("Poisson Grid", points);
        polyscope::getPointCloud("Poisson Grid")->addScalarQuantity("Level", levels);
        polyscope::getPointCloud("Poisson Grid")->addVectorQuantity("IJK", ijk_values);

        for (int s = 0; s < scalar_channels.size(); s++) {
            polyscope::getPointCloud("Poisson Grid")->addScalarQuantity(scalar_channels[s].second, scalar_data[s]);
        }

        for (int v = 0; v < vec_channels.size(); v++) {
            polyscope::getPointCloud("Poisson Grid")->addVectorQuantity(vec_channels[v].second, vec_data[v]);
            polyscope::getPointCloud("Poisson Grid")->addScalarQuantity(vec_channels[v].second + "_len", vec_length[v]);
        }
    }

    void AddPoissonGridFaceCentersToPolyscopePointCloud(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, std::vector<std::pair<int, std::string>> vec_channels, const double invalid_value) {
        auto& holder = *holder_ptr;
        using Coord = typename Tile::CoordType;

        std::vector<Vec> points[3];
        std::vector<int> levels[3];
        std::vector<Coord> ijk_values[3];

   //     std::vector<std::vector<Vec>> vec_data[3];
   //     for (int axis : {0, 1, 2}) {
			//vec_data[axis].resize(vec_channels.size());
   //     }

        std::vector<std::vector<float>> face_center_data[3];
        for (int axis : {0, 1, 2}) {
            face_center_data[axis].resize(vec_channels.size());
        }

        auto acc = holder.coordAccessor();
        holder.iterateLeafCells([&](const HATileInfo<Tile>& info, const Coord& l_ijk) {
            auto& tile = info.tile();

            for (int axis : {0, 1, 2}) {
                auto pos = acc.faceCenter(axis, info, l_ijk);
                points[axis].push_back(pos);
                levels[axis].push_back(info.mLevel);

                auto g_ijk = holder.coordAccessor().localToGlobalCoord(info, l_ijk);
                ijk_values[axis].push_back(g_ijk);

                for (int t = 0; t < vec_channels.size(); t++) {
                    int u_channel = vec_channels[t].first;
     //               Vec vec(0, 0, 0);
					//vec[axis] = tile(u_channel + axis, l_ijk);
     //               vec_data[axis][t].push_back(vec);
                    face_center_data[axis][t].push_back(tile(u_channel + axis, l_ijk));
                }
            }
            });

        for (int axis : {0, 1, 2}) {
			auto pc = polyscope::registerPointCloud(fmt::format("Poisson Grid Face Centers Axis {}", axis), points[axis]);
			pc->addScalarQuantity("Level", levels[axis]);
			pc->addVectorQuantity("IJK", ijk_values[axis]);
			for (int v = 0; v < vec_channels.size(); v++) {
				//pc->addVectorQuantity(vec_channels[v].second, vec_data[axis][v]);
				pc->addScalarQuantity(vec_channels[v].second, face_center_data[axis][v]);
			}
        }
    }

    void AddPoissonGridNodesToPolyscope(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const std::vector<std::pair<int, std::string>> scalar_channels, std::vector<std::pair<int, std::string>> vec_channels, const double invalid_value) {
        auto& holder = *holder_ptr;
        using Coord = typename Tile::CoordType;

        auto acc = holder.coordAccessor();
        //polyscope::init();

        for (int level = 0; level <= holder.mMaxLevel; level++) {
            std::vector<Vec> points;
            std::vector<Coord> tile_ijks, local_ijks;

            std::vector<std::vector<float>> scalar_data(scalar_channels.size());
            std::vector<std::vector<Vec>> vec_data(vec_channels.size());

            std::vector<Vec> tile_vertices;
            std::vector<std::array<size_t, 4>> tile_faces;

            for (auto& info : holder.mHostLevels[level]) {
                if (info.isLeaf()) {
                    auto& tile = info.tile();

                    for (int i = 0; i <= Tile::DIM; i++) {
                        for (int j = 0; j <= Tile::DIM; j++) {
                            for (int k = 0; k <= Tile::DIM; k++) {
                                Coord r_ijk(i, j, k);
                                auto pos = acc.cellCorner(info, r_ijk);

                                {
                                    auto ref_point = acc.cellCenterGlobal(3, Coord(60, 23, 99));
                                    if ((pos - ref_point).length() > 0.03) continue;
                                }


                                points.push_back(pos);

                                tile_ijks.push_back(info.mTileCoord);
                                local_ijks.push_back(r_ijk);

                                for (int s = 0; s < scalar_channels.size(); s++) {
                                    int channel = scalar_channels[s].first;
                                    T f = tile.node(scalar_channels[s].first, r_ijk);
                                    if (isnan(f) || isinf(f)) f = invalid_value;
                                    scalar_data[s].push_back(f);
                                }

                                for (int t = 0; t < vec_channels.size(); t++) {
                                    int u_channel = vec_channels[t].first;
                                    auto u = tile.node(u_channel, r_ijk);
                                    auto v = tile.node(u_channel + 1, r_ijk);
                                    auto w = tile.node(u_channel + 2, r_ijk);
                                    vec_data[t].push_back({ u, v, w });
                                }
                            }
                        }
                    }

                    // Define the 8 vertices of the tile (cube)
                    std::vector<Vec> cube_vertices;
                    for (int dx = 0; dx <= 1; ++dx) {
                        for (int dy = 0; dy <= 1; ++dy) {
                            for (int dz = 0; dz <= 1; ++dz) {
                                Coord l_ijk(dx * Tile::DIM, dy * Tile::DIM, dz * Tile::DIM);
                                cube_vertices.push_back(acc.cellCorner(info, l_ijk));
                            }
                        }
                    }

                    // Add vertices to the global list
                    auto base_idx = tile_vertices.size();
                    tile_vertices.insert(tile_vertices.end(), cube_vertices.begin(), cube_vertices.end());

                    // Define the faces of the cube
                    std::array<size_t, 4> faces[6] = {
                        {base_idx + 0, base_idx + 1, base_idx + 3, base_idx + 2},
                        {base_idx + 4, base_idx + 5, base_idx + 7, base_idx + 6},
                        {base_idx + 0, base_idx + 1, base_idx + 5, base_idx + 4},
                        {base_idx + 2, base_idx + 3, base_idx + 7, base_idx + 6},
                        {base_idx + 0, base_idx + 2, base_idx + 6, base_idx + 4},
                        {base_idx + 1, base_idx + 3, base_idx + 7, base_idx + 5}
                    };

                    // Add faces to the global list
                    tile_faces.insert(tile_faces.end(), std::begin(faces), std::end(faces));

                }
            }

            if (points.empty()) continue;

            auto level_cloud_name = fmt::format("Level {} Nodes", level);
            auto pc = polyscope::registerPointCloud(level_cloud_name, points);
            pc->addVectorQuantity("b_ijk", tile_ijks);
			pc->addVectorQuantity("r_ijk", local_ijks);

            for (int s = 0; s < scalar_channels.size(); s++) {
                pc->addScalarQuantity(scalar_channels[s].second, scalar_data[s]);
            }

            for (int v = 0; v < vec_channels.size(); v++) {
                pc->addVectorQuantity(vec_channels[v].second, vec_data[v]);
            }

            // Register the tile mesh for this level
            //auto tile_mesh_name = fmt::format("Level {} Tiles", level);
            //polyscope::registerSurfaceMesh(tile_mesh_name, tile_vertices, tile_faces);
        }

        //polyscope::show();
    }

}