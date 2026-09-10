#include "PoissonIOFunc.h"
#include "VTKFileIO.h"

namespace IOFunc {
namespace {

constexpr std::uint8_t kVtkVertex = 1;
constexpr std::uint8_t kVtkHexahedron = 12;

template <class ParticleType>
void WriteParticles(const thrust::host_vector<ParticleType>& particles, const fs::path& path) {
    std::vector<float> points;
    std::vector<std::int64_t> connectivity;
    std::vector<std::int64_t> offsets;
    std::vector<std::uint8_t> types(particles.size(), kVtkVertex);
    points.reserve(particles.size() * 3);
    for (std::size_t i = 0; i < particles.size(); ++i) {
        const auto p = particles[i].pos;
        points.insert(points.end(), {
            static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2])});
        connectivity.push_back(static_cast<std::int64_t>(i));
        offsets.push_back(static_cast<std::int64_t>(i + 1));
    }
    VTKFileIO::WriteVTU(path, points, connectivity, offsets, types);
}

std::vector<HATileInfo<Tile>> CollectLeaves(HAHostTileHolder<Tile>& holder) {
    std::vector<HATileInfo<Tile>> leaves;
    for (int level = 0; level <= holder.mMaxLevel; ++level) {
        for (const auto& info : holder.mHostLevels[level]) {
            if (info.isLeaf()) leaves.push_back(info);
        }
    }
    return leaves;
}

std::vector<VTKFileIO::FloatArray> MakeArrays(
    const std::vector<std::pair<int, std::string>>& scalars,
    const std::vector<std::pair<int, std::string>>& vectors,
    std::size_t cells)
{
    std::vector<VTKFileIO::FloatArray> result;
    for (const auto& item : scalars)
        result.push_back({item.second, 1, std::vector<float>(cells, 0.0f)});
    for (const auto& item : vectors)
        result.push_back({item.second, 3, std::vector<float>(cells * 3, 0.0f)});
    return result;
}

template <class Accessor>
void FillTileArrays(
    const Tile& tile, int level,
    const std::vector<std::pair<int, std::string>>& scalars,
    const std::vector<std::pair<int, std::string>>& vectors,
    std::vector<VTKFileIO::FloatArray>& arrays,
    std::size_t base, const Accessor& acc)
{
    for (int cell = 0; cell < Tile::SIZE; ++cell) {
        const auto local = acc.localOffsetToCoord(cell);
        for (std::size_t i = 0; i < scalars.size(); ++i) {
            const int channel = scalars[i].first;
            const T value = channel == -1 ? static_cast<T>(tile.type(local))
                : channel == -2 ? static_cast<T>(level) : tile(channel, local);
            arrays[i].values[base + cell] = static_cast<float>(value);
        }
        for (std::size_t i = 0; i < vectors.size(); ++i) {
            const int channel = vectors[i].first;
            auto& values = arrays[scalars.size() + i].values;
            const std::size_t dst = 3 * (base + cell);
            values[dst] = static_cast<float>(tile(channel, local));
            values[dst + 1] = static_cast<float>(tile(channel + 1, local));
            values[dst + 2] = static_cast<float>(tile(channel + 2, local));
        }
    }
}

} // namespace

void OutputMarkerParticleSystemAsVTU(
    std::shared_ptr<thrust::host_vector<MarkerParticle>> particles, fs::path path)
{
    WriteParticles(*particles, path);
}

void OutputParticleSystemAsVTU(
    std::shared_ptr<thrust::host_vector<Particle>> particles, fs::path path)
{
    CPUTimer timer;
    timer.start();
    WriteParticles(*particles, path);
    Pass("Finished writing particle system to VTU file: {} ({} ms)",
        path.string(), timer.stop());
}

void OutputTilesAsVTU(
    std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const fs::path& path)
{
    std::vector<float> points;
    std::vector<std::int64_t> connectivity, offsets;
    std::vector<std::uint8_t> types;
    const int corners[8][3] = {
        {0,0,0},{1,0,0},{1,1,0},{0,1,0},
        {0,0,1},{1,0,1},{1,1,1},{0,1,1}};
    auto& holder = *holder_ptr;
    const auto acc = holder.coordAccessor();
    holder.iterateLeafTiles([&](const HATileInfo<Tile>& info) {
        const auto bbox = acc.tileBBox(info);
        const auto base = static_cast<std::int64_t>(points.size() / 3);
        for (const auto& corner : corners) {
            const auto p = bbox.min() + Vec(corner[0], corner[1], corner[2]) * bbox.dim();
            points.insert(points.end(), {
                static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2])});
        }
        for (int i = 0; i < 8; ++i) connectivity.push_back(base + i);
        offsets.push_back(static_cast<std::int64_t>(connectivity.size()));
        types.push_back(kVtkHexahedron);
    });
    VTKFileIO::WriteVTU(path, points, connectivity, offsets, types);
}

void OutputPoissonGridAsUnstructuredVTU(
    std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr,
    const std::vector<std::pair<int, std::string>> scalar_channels,
    std::vector<std::pair<int, std::string>> vec_channels,
    const fs::path& path)
{
    auto& holder = *holder_ptr;
    const auto acc = holder.coordAccessor();
    const auto leaves = CollectLeaves(holder);
    std::vector<float> points;
    std::vector<std::int64_t> connectivity, offsets;
    std::vector<std::uint8_t> types(leaves.size() * Tile::SIZE, kVtkHexahedron);
    const int corners[8][3] = {
        {0,0,0},{1,0,0},{1,1,0},{0,1,0},
        {0,0,1},{1,0,1},{1,1,1},{0,1,1}};
    for (std::size_t tile_idx = 0; tile_idx < leaves.size(); ++tile_idx) {
        const auto& info = leaves[tile_idx];
        for (int node = 0; node < Tile::NODESIZE; ++node) {
            const auto p = acc.cellCorner(info, acc.localNodeOffsetToCoord(node));
            points.insert(points.end(), {
                static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2])});
        }
        for (int cell = 0; cell < Tile::SIZE; ++cell) {
            const auto local = acc.localOffsetToCoord(cell);
            for (const auto& corner : corners) {
                const auto node = acc.localNodeCoordToOffset(
                    local + typename Tile::CoordType(corner[0], corner[1], corner[2]));
                connectivity.push_back(
                    static_cast<std::int64_t>(tile_idx * Tile::NODESIZE + node));
            }
            offsets.push_back(static_cast<std::int64_t>(connectivity.size()));
        }
    }
    auto arrays = MakeArrays(
        scalar_channels, vec_channels, leaves.size() * Tile::SIZE);
    for (std::size_t i = 0; i < leaves.size(); ++i)
        FillTileArrays(leaves[i].tile(), leaves[i].mLevel, scalar_channels,
            vec_channels, arrays, i * Tile::SIZE, acc);
    VTKFileIO::WriteVTU(path, points, connectivity, offsets, types, {}, arrays);
}

void OutputPoissonGridAsStructuredVTI(
    std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr,
    const std::vector<std::pair<int, std::string>> scalar_channels,
    std::vector<std::pair<int, std::string>> vec_channels,
    const fs::path& path)
{
    CPUTimer<std::chrono::milliseconds> timer;
    timer.start();
    auto& holder = *holder_ptr;
    const auto acc = holder.coordAccessor();
    const auto leaves = CollectLeaves(holder);
    using Coord = typename Tile::CoordType;
    Coord global_min(INT_MAX, INT_MAX, INT_MAX), global_max(INT_MIN, INT_MIN, INT_MIN);
    for (const auto& info : leaves) {
        const int shift = holder.mMaxLevel - info.mLevel;
        const Coord first = acc.localToGlobalCoord(info, Coord(0, 0, 0));
        const Coord last = acc.localToGlobalCoord(
            info, Coord(Tile::DIM - 1, Tile::DIM - 1, Tile::DIM - 1));
        for (int axis = 0; axis < 3; ++axis) {
            global_min[axis] = std::min(global_min[axis], first[axis] << shift);
            global_max[axis] = std::max(global_max[axis], (last[axis] + 1) << shift);
        }
    }
    const Coord dims = global_max - global_min;
    ASSERT(dims[0] > 0 && dims[1] > 0 && dims[2] > 0,
        "Cannot write empty structured VTI grid");
    const std::size_t count =
        static_cast<std::size_t>(dims[0]) * dims[1] * dims[2];
    auto arrays = MakeArrays(scalar_channels, vec_channels, count);
    for (const auto& info : leaves) {
        const auto& tile = info.tile();
        const int shift = holder.mMaxLevel - info.mLevel;
        const int span = 1 << shift;
        for (int cell = 0; cell < Tile::SIZE; ++cell) {
            const Coord local = acc.localOffsetToCoord(cell);
            const Coord global = acc.localToGlobalCoord(info, local);
            const int x0 = (global[0] << shift) - global_min[0];
            const int y0 = (global[1] << shift) - global_min[1];
            const int z0 = (global[2] << shift) - global_min[2];
            for (int z = z0; z < z0 + span; ++z)
                for (int y = y0; y < y0 + span; ++y)
                    for (int x = x0; x < x0 + span; ++x) {
                        const std::size_t dst = x + static_cast<std::size_t>(dims[0])
                            * (y + static_cast<std::size_t>(dims[1]) * z);
                        for (std::size_t i = 0; i < scalar_channels.size(); ++i) {
                            const int channel = scalar_channels[i].first;
                            const T value = channel == -1 ? static_cast<T>(tile.type(local))
                                : channel == -2 ? static_cast<T>(info.mLevel)
                                : tile(channel, local);
                            arrays[i].values[dst] = static_cast<float>(value);
                        }
                        for (std::size_t i = 0; i < vec_channels.size(); ++i) {
                            const int channel = vec_channels[i].first;
                            auto& values = arrays[scalar_channels.size() + i].values;
                            values[3 * dst] = static_cast<float>(tile(channel, local));
                            values[3 * dst + 1] = static_cast<float>(tile(channel + 1, local));
                            values[3 * dst + 2] = static_cast<float>(tile(channel + 2, local));
                        }
                    }
        }
    }
    const int extent[6] = {0, dims[0], 0, dims[1], 0, dims[2]};
    const double h = acc.voxelSize(holder.mMaxLevel);
    const double origin[3] = {
        global_min[0] * h, global_min[1] * h, global_min[2] * h};
    const double spacing[3] = {h, h, h};
    VTKFileIO::WriteVTI(path, extent, origin, spacing, arrays);
    Pass("Finished writing Poisson grid to structured VTI file: {} ({} ms)",
        path.string(), timer.stop());
}

void OutputPoissonGridAsAMR(
    std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr,
    const std::vector<std::pair<int, std::string>>& scalar_channels,
    const std::vector<std::pair<int, std::string>>& vec_channels,
    const fs::path& path)
{
    ASSERT(path.extension() == ".vthb", "AMR path must end with .vthb: {}", path.string());
    auto& holder = *holder_ptr;
    const auto acc = holder.coordAccessor();
    const auto stem = path.stem().string();
    const auto directory = path.parent_path() / (stem + "_blocks");
    std::filesystem::create_directories(directory);
    std::ostringstream xml;
    xml.precision(17);
    xml << "<?xml version=\"1.0\"?>\n"
        << "<VTKFile type=\"vtkOverlappingAMR\" version=\"1.1\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
        << "  <vtkOverlappingAMR origin=\"0 0 0\" grid_description=\"XYZ\">\n";
    std::size_t total = 0;
    for (int level = 0; level <= holder.mMaxLevel; ++level) {
        const double h = acc.voxelSize(level);
        xml << "    <Block level=\"" << level << "\" spacing=\""
            << h << ' ' << h << ' ' << h << "\">\n";
        int block = 0;
        for (const auto& info : holder.mHostLevels[level]) {
            if (!info.isLeaf()) continue;
            const auto global =
                acc.localToGlobalCoord(info, typename Tile::CoordType(0, 0, 0));
            auto arrays = MakeArrays(scalar_channels, vec_channels, Tile::SIZE);
            FillTileArrays(info.tile(), level, scalar_channels, vec_channels, arrays, 0, acc);
            const int extent[6] = {0, Tile::DIM, 0, Tile::DIM, 0, Tile::DIM};
            const double origin[3] = {global[0] * h, global[1] * h, global[2] * h};
            const double spacing[3] = {h, h, h};
            const auto filename = "level_" + std::to_string(level)
                + "_block_" + std::to_string(block) + ".vti";
            VTKFileIO::WriteVTI(directory / filename, extent, origin, spacing, arrays);
            xml << "      <DataSet index=\"" << block << "\" amr_box=\""
                << global[0] << ' ' << global[0] + Tile::DIM - 1 << ' '
                << global[1] << ' ' << global[1] + Tile::DIM - 1 << ' '
                << global[2] << ' ' << global[2] + Tile::DIM - 1
                << "\" file=\"" << VTKFileIO::EscapeXML(stem + "_blocks/" + filename)
                << "\"/>\n";
            ++block;
            ++total;
        }
        xml << "    </Block>\n";
    }
    xml << "  </vtkOverlappingAMR>\n</VTKFile>\n";
    VTKFileIO::EnsureParentDirectory(path);
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open AMR file: " + path.string());
    out << xml.str();
    if (!out) throw std::runtime_error("Failed while writing AMR file: " + path.string());
    Pass("Finished writing Poisson grid to AMR file: {} ({} blocks)", path.string(), total);
}

} // namespace IOFunc
