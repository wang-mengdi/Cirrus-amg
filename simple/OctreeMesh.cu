#include "SimpleMesh.h"
#include "PoissonTile.h"
#include "NativeOctreeAccess.cuh"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <set>
#include <filesystem>
#include <numeric>
#include <limits>
#include <nlohmann/json.hpp>
#include "ConstructionMemory.h"

namespace simple {
namespace {

struct NativeCell {
    int level;
    int tileIndex;
    Coord local;
};

std::uint64_t tileChecksum(const Tile& tile) {
    const auto* bytes=reinterpret_cast<const unsigned char*>(&tile);
    std::uint64_t hash=14695981039346656037ull;
    for(std::size_t i=0;i<sizeof(Tile);++i){hash^=bytes[i];hash*=1099511628211ull;}
    return hash;
}
struct NativeTileBacking {
    std::filesystem::path directory,file;
    std::vector<std::uint64_t> checksums;
    std::vector<std::vector<std::size_t>> ordinal;
    std::vector<std::size_t> cellBegin;
    std::vector<int> cellOrder;
    bool owned=false;
    ~NativeTileBacking() {
        if(owned){std::error_code error;std::filesystem::remove(file,error);std::filesystem::remove(directory,error);}
    }
    std::ifstream open() const {
        if(std::filesystem::file_size(file)!=checksums.size()*sizeof(Tile))
            throw std::runtime_error("Native host Tile backing length changed");
        std::ifstream input(file,std::ios::binary);
        if(!input)throw std::runtime_error("Cannot read native host Tile backing");
        return input;
    }
    void read(std::ifstream& input,std::size_t index,Tile& tile) const {
        input.read(reinterpret_cast<char*>(&tile),sizeof(Tile));
        if(!input||input.gcount()!=sizeof(Tile)||tileChecksum(tile)!=checksums.at(index))
            throw std::runtime_error("Native host Tile backing is incomplete or damaged");
    }
};

struct NativeStorage {
    std::shared_ptr<HADeviceGrid<Tile>> grid;
    std::shared_ptr<HAHostTileHolder<Tile>> holder;
    std::vector<NativeCell> mapping;
    std::unique_ptr<NativeTileBacking> backing;
};

void cudaCheck(cudaError_t code, const char* operation) {
    if (code != cudaSuccess)
        throw std::runtime_error(std::string("SIMPLE native octree: ") + operation + ": "
                                 + cudaGetErrorString(code));
}

} // namespace

Mesh makeOctreeChannel(int ny, bool adaptive, bool periodicX, bool periodicZ,
                      const std::string& geometryPath,const std::string& geometryMetadata,bool buildFaces) {
    constructionMemory("native_mesh.begin");
    if(!buildFaces && geometryPath.empty())
        throw std::invalid_argument("Deferred native faces require embedded geometry");
    if (ny < 8 || (ny & (ny - 1)) != 0)
        throw std::invalid_argument("SIMPLE octree ny must be a power of two >= 8 (native tiles contain 8^3 cells)");
    Vec3 extent(1., .125, .125);
    std::set<std::array<int, 3>> refineTiles;
    if (!geometryPath.empty()) {
        nlohmann::json data;
        if(!geometryMetadata.empty()) data=nlohmann::json::parse(geometryMetadata);
        else {
            std::ifstream in(geometryPath);
            if (!in) throw std::runtime_error("Cannot read embedded geometry: " + geometryPath);
            in >> data;
        }
        for (int d = 0; d < 3; ++d) extent[d] = data.at("extent").at(d).get<double>();
        if (adaptive)
            for (const auto& key : data.at("refine_root_tiles"))
                refineTiles.insert(key.get<std::array<int, 3>>());
    }
    const int ntY = ny / static_cast<int>(Tile::DIM);
    const int ntX = static_cast<int>(std::llround(extent[0] / extent[1] * ntY));
    const int ntZ = static_cast<int>(std::llround(extent[2] / extent[1] * ntY));
    if (ntX < 1 || ntZ < 1 || std::abs(double(ntX)/ntY - extent[0]/extent[1]) > 1e-12 ||
        std::abs(double(ntZ)/ntY - extent[2]/extent[1]) > 1e-12)
        throw std::runtime_error("Embedded domain must align to whole native root tiles");
    const int roots = ntX * ntY * ntZ;
    uint32_t logHash = 10;
    while ((1ull << logHash) < static_cast<unsigned long long>(roots) * 32ull)
        ++logHash;
    if (logHash > 26) throw std::invalid_argument("SIMPLE octree request is too large");
    const float h0 = static_cast<float>(extent[1] / ny);
    auto storage = std::make_shared<NativeStorage>();
    storage->grid = std::make_shared<HADeviceGrid<Tile>>(h0,
        std::initializer_list<uint32_t>{logHash, logHash});
    auto& grid = *storage->grid;

    // Initialize every channel and cell flag before invoking native refinement.
    Tile initial;
    for (int ch = 0; ch < Tile::num_channels; ++ch)
        for (int q = 0; q < Tile::CHNLSIZE; ++q) initial(ch, q) = 0.0f;
    for (int q = 0; q < Tile::SIZE; ++q) initial.type(q) = CellType::INTERIOR;
    initial.mSerialIdx = 0;
    for (int i = 0; i < ntX; ++i)
        for (int j = 0; j < ntY; ++j)
            for (int k = 0; k < ntZ; ++k) {
                const double mid = (i + .5) * h0 * Tile::DIM;
                initial.mSerialIdx = geometryPath.empty() ? (mid >= .25 && mid < .75) :
                    int(refineTiles.count({i, j, k}));
                grid.setTileHost(0, Coord(i, j, k), initial, LEAF);
            }
    grid.rebuild(false);
    cudaCheck(cudaGetLastError(), "construct root tiles");
    constructionMemory("native_mesh.roots_ready");
    if (adaptive) {
        grid.iterativeRefine([] __device__(const HATileAccessor<Tile>& acc,
                                           HATileInfo<Tile>& info) {
            return info.mLevel == 0 && info.tile().mSerialIdx == 1 ? 1 : 0;
        }, false);
        cudaCheck(cudaDeviceSynchronize(), "native octree refinement");
    }
    constructionMemory("native_mesh.hierarchy_ready");

    // This is deliberately a view of actual native leaf cells, not a second
    // independently generated mesh or an expansion to the finest uniform level.
    storage->holder = grid.getHostTileHolderForLeafs();
    cudaCheck(cudaGetLastError(), "download native leaf holder");
    constructionMemory("native_mesh.leaf_holder_ready");
    Mesh mesh;
    mesh.extent = extent;
    mesh.backend = "HADeviceGrid<Tile> leaf cells; CPU conservative subface view";
    storage->mapping.reserve(static_cast<std::size_t>(storage->holder->numberOfLeafTiles()) * Tile::SIZE);
    auto acc = storage->holder->coordAccessor();
    for (int level = 0; level <= storage->holder->mMaxLevel; ++level) {
        auto& tiles = storage->holder->mHostLevels[level];
        for (int tileIndex = 0; tileIndex < static_cast<int>(tiles.size()); ++tileIndex) {
            auto& info = tiles[tileIndex];
            if (!info.isLeaf()) continue;
            for (int i = 0; i < Tile::DIM; ++i) {
                for (int j = 0; j < Tile::DIM; ++j) {
                    for (int k = 0; k < Tile::DIM; ++k) {
                        const Coord local(i, j, k);
                        storage->mapping.push_back(NativeCell{level, tileIndex, local});
                    }
                }
            }
        }
    }
    // Native hash order can change; stable ordering makes state dumps comparable.
    // Sort only native identities; materialize the larger Cell values once.
    const auto keyOf = [&](const NativeCell& value) {
        const auto& info = storage->holder->mHostLevels[value.level][value.tileIndex];
        const auto q = acc.localToGlobalCoord(info, value.local);
        return std::array<int, 3>{q[0], q[1], q[2]};
    };
    std::sort(storage->mapping.begin(), storage->mapping.end(), [&](const NativeCell& a, const NativeCell& b) {
        if (a.level != b.level) return a.level < b.level;
        return keyOf(a) < keyOf(b);
    });
    constructionMemory("native_mesh.mappings_sorted");
    mesh.cells.reserve(storage->mapping.size());
    for (const auto& native : storage->mapping) {
        const auto& info = storage->holder->mHostLevels[native.level][native.tileIndex];
        const auto x = acc.cellCenter(info, native.local);
        Cell c;
        c.level = native.level;
        c.key = keyOf(native);
        c.center = Vec3(x[0], x[1], x[2]);
        c.h = static_cast<double>(acc.voxelSize(info));
        c.volume = c.h * c.h * c.h;
        mesh.cells.push_back(c);
    }
    constructionMemory("native_mesh.cells_ready");
    mesh.nativeStorage = storage;
    std::cout << "Native octree: leafTiles=" << grid.numTotalLeafTiles()
              << " allTiles=" << grid.numTotalTiles()
              << " maxLevel=" << grid.mMaxLevel << " coarseNy=" << ny << '\n';
    mesh.periodicZ = periodicZ;
    if(buildFaces) buildFacesAndValidate(mesh, periodicX, periodicZ);
    if (buildFaces && adaptive && mesh.coarseFineFaces == 0 && geometryPath.empty())
        throw std::runtime_error("SIMPLE adaptive channel unexpectedly has no coarse/fine interfaces");
    return mesh;
}

void retainNativeCells(Mesh& mesh, const std::vector<int>& oldIndices) {
    auto storage = std::static_pointer_cast<NativeStorage>(mesh.nativeStorage);
    if(storage->backing)throw std::runtime_error("Cannot change native cell mapping after host Tile offload");
    std::vector<NativeCell> retained;
    retained.reserve(oldIndices.size());
    for (int index : oldIndices) retained.push_back(storage->mapping.at(index));
    storage->mapping = std::move(retained);
}

void offloadNativeHostTiles(Mesh& mesh,const std::string& freshDirectory) {
    if(!mesh.nativeStorage||freshDirectory.empty())throw std::invalid_argument("Missing native host Tile backing destination");
    auto storage=std::static_pointer_cast<NativeStorage>(mesh.nativeStorage);
    if(storage->backing)throw std::runtime_error("Native host Tiles already offloaded");
    auto& holder=*storage->holder;const std::size_t count=holder.mHostTiles.size();
    if(!count||storage->mapping.size()!=mesh.cells.size()||mesh.cells.size()>std::size_t(std::numeric_limits<int>::max()))
        throw std::runtime_error("Invalid native host Tile mapping");
    auto backing=std::make_unique<NativeTileBacking>();
    backing->directory=std::filesystem::absolute(freshDirectory).lexically_normal();
    if(std::filesystem::exists(backing->directory.parent_path()/"native_host_storage.json"))
        throw std::runtime_error("Preserve existing native host storage metadata");
    if(!std::filesystem::create_directories(backing->directory))throw std::runtime_error("Native host Tile backing directory must be fresh");
    backing->file=backing->directory/"tiles.bin";backing->owned=true;
    backing->ordinal.resize(holder.mHostLevels.size());backing->checksums.reserve(count);
    backing->cellBegin.assign(count+1,0);
    std::size_t expected=0;
    for(std::size_t level=0;level<holder.mHostLevels.size();++level) {
        for(const auto& info:holder.mHostLevels[level]) {
            if(expected>=count||info.mTilePtr!=holder.mHostTiles.data()+expected)
                throw std::runtime_error("Native leaf Tile storage order differs");
            backing->ordinal[level].push_back(expected++);
        }
    }
    if(expected!=count)throw std::runtime_error("Native leaf Tile count differs");
    for(const auto& cell:storage->mapping)++backing->cellBegin.at(backing->ordinal.at(cell.level).at(cell.tileIndex)+1);
    std::partial_sum(backing->cellBegin.begin(),backing->cellBegin.end(),backing->cellBegin.begin());
    backing->cellOrder.resize(storage->mapping.size());
    {
        auto cursor=backing->cellBegin;
        for(std::size_t i=0;i<storage->mapping.size();++i) {
            const auto& cell=storage->mapping[i];const auto tile=backing->ordinal.at(cell.level).at(cell.tileIndex);
            backing->cellOrder[cursor[tile]++]=int(i);
        }
    }
    {
        std::ofstream output(backing->file,std::ios::binary);
        if(!output)throw std::runtime_error("Cannot create native host Tile backing");
        for(const auto& tile:holder.mHostTiles) {
            backing->checksums.push_back(tileChecksum(tile));
            output.write(reinterpret_cast<const char*>(&tile),sizeof(Tile));
            if(!output)throw std::runtime_error("Cannot write native host Tile backing");
        }
        output.flush();if(!output)throw std::runtime_error("Cannot flush native host Tile backing");
        output.close();if(output.fail())throw std::runtime_error("Cannot close native host Tile backing");
    }
    // Verify the stored bytes before releasing the only host copy.
    {
        auto input=backing->open();Tile tile;
        for(std::size_t i=0;i<count;++i)backing->read(input,i,tile);
    }
    const std::size_t before=holder.mHostTiles.capacity()*sizeof(Tile);
    for(auto& level:holder.mHostLevels)for(auto& info:level)info.mTilePtr=nullptr;
    std::vector<Tile>().swap(holder.mHostTiles);
    storage->backing=std::move(backing);
    const auto& retained=*storage->backing;
    nlohmann::json report{{"storage","checked_file_tiles"},{"leaf_tiles",count},{"tile_bytes",sizeof(Tile)},
        {"released_host_tile_capacity_bytes",before},{"resident_host_tile_bytes",holder.mHostTiles.capacity()*sizeof(Tile)},
        {"backing_bytes",count*sizeof(Tile)},{"cell_order_bytes",retained.cellOrder.capacity()*sizeof(int)},
        {"complete_tile_bytes_preserved",true},{"mapping_frozen",true},{"backing_file",retained.file.string()}};
    std::ofstream manifest(retained.directory.parent_path()/"native_host_storage.json");manifest<<report.dump(2)<<'\n';manifest.flush();
    if(!manifest)throw std::runtime_error("Cannot write native host storage metadata");
    std::cout<<"Native host Tile backing: leaves="<<count<<" released_bytes="<<before<<" cell_order_bytes="<<retained.cellOrder.capacity()*sizeof(int)<<std::endl;
    constructionMemory("native_mesh.host_tiles_offloaded");
}

HADeviceGrid<Tile>& nativeDeviceGrid(const Mesh& mesh) {
    if(!mesh.nativeStorage)throw std::invalid_argument("Native grid storage is missing");
    return *std::static_pointer_cast<NativeStorage>(mesh.nativeStorage)->grid;
}

void exportNativeFields(Mesh& mesh, const std::vector<Vec3>& velocity,
                        const std::vector<double>& pressure, const std::string& path,bool preserveOperatorMetadata,bool writeBinary) {
    if (!mesh.nativeStorage || velocity.size() != mesh.cells.size()
        || pressure.size() != mesh.cells.size())
        throw std::invalid_argument("SIMPLE native export field size or storage mismatch");
    auto storage = std::static_pointer_cast<NativeStorage>(mesh.nativeStorage);
    if(storage->backing) {
        auto& backing=*storage->backing;auto input=backing.open();auto hostAcc=storage->grid->hostAccessor();
        std::size_t index=0;Tile tile;
        for(const auto& level:storage->holder->mHostLevels)for(const auto& info:level) {
            backing.read(input,index,tile);
            for(std::size_t k=backing.cellBegin.at(index);k<backing.cellBegin.at(index+1);++k) {
                const int i=backing.cellOrder[k];const auto& m=storage->mapping[i];
                for(int a=0;a<3;++a)tile(a,m.local)=static_cast<float>(velocity[i][a]);
                tile(3,m.local)=static_cast<float>(pressure[i]);
            }
            const auto& deviceInfo=hostAcc.tileInfo(info.mLevel,info.mTileCoord);
            const std::size_t bytes=preserveOperatorMetadata?4*Tile::CHNLSIZE*sizeof(Tile::T):sizeof(Tile);
            cudaCheck(cudaMemcpy(deviceInfo.mTilePtr,&tile,bytes,cudaMemcpyHostToDevice),"write backed SIMPLE fields to original leaf tile");
            ++index;
        }
        if(index!=backing.checksums.size())throw std::runtime_error("Native host Tile export count changed");
    } else {
    for (int i = 0; i < static_cast<int>(mesh.cells.size()); ++i) {
        const auto& m = storage->mapping[i];
        auto& tile = storage->holder->mHostLevels[m.level][m.tileIndex].tile();
        for (int a = 0; a < 3; ++a) tile(a, m.local) = static_cast<float>(velocity[i][a]);
        tile(3, m.local) = static_cast<float>(pressure[i]);
    }
    auto hostAcc = storage->grid->hostAccessor();
    for (const auto& level : storage->holder->mHostLevels) {
        for (const auto& info : level) {
            const auto& deviceInfo = hostAcc.tileInfo(info.mLevel, info.mTileCoord);
            // A live GPU operator owns cell masks, serial indices and neighbor
            // pointers. Export only the four physical channels in that case.
            const size_t bytes=preserveOperatorMetadata?4*Tile::CHNLSIZE*sizeof(Tile::T):sizeof(Tile);
            cudaCheck(cudaMemcpy(deviceInfo.mTilePtr, info.mTilePtr, bytes,
                                  cudaMemcpyHostToDevice), "write SIMPLE fields to original leaf tile");
        }
    }
    }
    if(!writeBinary)return;
    std::ofstream output(path, std::ios::binary);
    if(!output)throw std::runtime_error("SIMPLE could not open native tile field dump: " + path);
    storage->grid->dumpBinaryStream(output);
    output.flush();
    if (!output.good()) throw std::runtime_error("SIMPLE could not write native tile field dump: " + path);
    output.close();
    if(output.fail())throw std::runtime_error("SIMPLE could not close native tile field dump: " + path);
}

} // namespace simple
