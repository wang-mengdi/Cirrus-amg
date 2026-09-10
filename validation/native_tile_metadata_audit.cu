#include "native_tile_metadata_audit.h"
#include "NativeCompactGpu.h"
#include "NativeOctreeAccess.cuh"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <type_traits>

namespace {
void check(cudaError_t error,const char* what) {
    if(error!=cudaSuccess)throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(error));
}
struct Fixture {
    std::vector<HATileInfo<Tile>> tiles;
    std::filesystem::path original;
    bool dirty=false;
    explicit Fixture(simple::Mesh& mesh,const std::filesystem::path& output):original(output/"original_tiles.bin") {
        auto& grid=simple::nativeDeviceGrid(mesh);
        for(int level=0;level<=grid.mMaxLevel;++level)for(int j=0;j<grid.hNumTiles[level];++j) {
            const auto info=grid.hTileArrays[level][j];
            if(info.mType&(LEAF|NONLEAF|GHOST))tiles.push_back(info);
        }
        save(original);
    }
    void save(const std::filesystem::path& path) const {
        std::ofstream out(path,std::ios::binary);
        for(const auto& info:tiles) {
            std::array<unsigned char,sizeof(Tile)> bytes;
            check(cudaMemcpy(bytes.data(),info.mTilePtr,bytes.size(),cudaMemcpyDeviceToHost),"snapshot native tile");
            out.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());
        }
        out.flush();if(!out.good())throw std::runtime_error("Native tile snapshot write failed");
    }
    void seed() {
        static_assert(std::is_trivially_copyable<Tile>::value,"Native byte audit requires copyable tiles");
        dirty=true;
        for(size_t i=0;i<tiles.size();++i) {
            Tile tile;check(cudaMemcpy(&tile,tiles[i].mTilePtr,sizeof(tile),cudaMemcpyDeviceToHost),"read tile before sentinel");
            for(int ch=0;ch<Tile::num_channels;++ch)for(int q=0;q<Tile::CHNLSIZE;++q)
                tile(ch,q)=float(int((i*131+ch*37+q*17)%8191)-4095)/8192.f;
            tile.mSerialIdx=-100000-int(i);
            for(int q=0;q<Tile::SIZE;++q)tile.type(q)=q%3==0?DIRICHLET:q%3==1?INTERIOR:NEUMANN;
            tile.mIsInterestArea=(i%2)!=0;tile.mIsLockedRefine=(i%3)!=0;
            check(cudaMemcpy(tiles[i].mTilePtr,&tile,sizeof(tile),cudaMemcpyHostToDevice),"write tile sentinel");
        }
    }
    void equal(const std::filesystem::path& path,size_t bytes) const {
        if(std::filesystem::file_size(path)!=tiles.size()*sizeof(Tile))throw std::runtime_error("Native snapshot size changed");
        std::ifstream in(path,std::ios::binary);
        for(const auto& info:tiles) {
            std::array<unsigned char,sizeof(Tile)> expected,actual;
            in.read(reinterpret_cast<char*>(expected.data()),expected.size());
            if(!in.good())throw std::runtime_error("Native snapshot read failed");
            check(cudaMemcpy(actual.data(),info.mTilePtr,actual.size(),cudaMemcpyDeviceToHost),"read tile for comparison");
            if(std::memcmp(actual.data(),expected.data(),bytes)!=0)
                throw std::runtime_error("Native tile bytes changed outside the metadata lease");
        }
    }
    void restore() {
        if(!dirty)return;
        if(std::filesystem::file_size(original)!=tiles.size()*sizeof(Tile))throw std::runtime_error("Original tile snapshot size changed");
        std::ifstream in(original,std::ios::binary);
        for(const auto& info:tiles) {
            std::array<unsigned char,sizeof(Tile)> bytes;
            in.read(reinterpret_cast<char*>(bytes.data()),bytes.size());
            if(!in.good())throw std::runtime_error("Original tile snapshot read failed");
            check(cudaMemcpy(info.mTilePtr,bytes.data(),bytes.size(),cudaMemcpyHostToDevice),"restore initial native tile");
        }
        equal(original,sizeof(Tile));dirty=false;
    }
    ~Fixture(){try{restore();}catch(...) {}}
};
}

nlohmann::json auditNativeTileMetadata(simple::Mesh& mesh,
    const std::filesystem::path& output,bool exerciseAmg) {
    std::filesystem::create_directories(output);Fixture fixture(mesh,output);
    fixture.seed();const auto seeded=output/"sentinel_tiles.bin";fixture.save(seeded);
    nlohmann::json cases=nlohmann::json::array();
    // A bad leaf address fails before any tile upload. Restore the mesh input
    // immediately; the comparison below checks every byte of every native tile.
    const auto key=mesh.cells.front().key;mesh.cells.front().key[0]=-1024;
    bool rejected=false;
    try{simple::NativeCompactGpu gpu(mesh);}catch(const std::runtime_error& e){rejected=std::string(e.what())=="Active cell is not an original native leaf";}
    catch(...){mesh.cells.front().key=key;throw;}
    mesh.cells.front().key=key;
    if(!rejected)throw std::runtime_error("Expected pre-upload metadata construction rejection");
    fixture.equal(seeded,sizeof(Tile));cases.push_back("failure_before_upload_all_tile_bytes_restored");

    // Face coefficients are checked after tile metadata and volumes are
    // uploaded, so this exercises destruction of a partially built Impl.
    const double area=mesh.faces.front().area;mesh.faces.front().area=0.;rejected=false;
    try{simple::NativeCompactGpu gpu(mesh);}catch(const std::runtime_error& e){rejected=std::string(e.what())=="Invalid native face coefficient";}
    catch(...){mesh.faces.front().area=area;throw;}
    mesh.faces.front().area=area;
    if(!rejected)throw std::runtime_error("Expected post-upload metadata construction rejection");
    fixture.equal(seeded,sizeof(Tile));cases.push_back("failure_after_upload_all_tile_bytes_restored");

    {
        simple::NativeCompactGpu gpu(mesh);
        fixture.equal(seeded,offsetof(Tile,mNeighbors));
        std::vector<double> values(mesh.cells.size());
        for(size_t c=0;c<values.size();++c)values[c]=.2+std::sin(6.283185307179586*mesh.cells[c].center[0]/mesh.extent[0]);
        gpu.apply(values,1.,0.,false);
        const auto rhs=gpu.apply(values,.01,200.,true);
        if(exerciseAmg) {
            gpu.configureNativeAmg(mesh,0,.01,200.);
            const auto result=gpu.solve(rhs,.01,200.,true,-1,1e-13);
            if(result.relativeResidual>1e-13)throw std::runtime_error("Metadata audit AMG solve failed");
        }
        fixture.equal(seeded,offsetof(Tile,mNeighbors));
        cases.push_back("all_numeric_channel_bytes_preserved_during_apply_and_amg");
    }
    fixture.equal(seeded,sizeof(Tile));cases.push_back("normal_destruction_all_tile_bytes_restored");
    fixture.restore();
    return {{"passed",true},{"tiles",fixture.tiles.size()},{"tile_bytes",sizeof(Tile)},
        {"numeric_channel_bytes_per_tile",offsetof(Tile,mNeighbors)},
        {"metadata_suffix_bytes_per_tile",sizeof(Tile)-offsetof(Tile,mNeighbors)},
        {"all_tile_bytes_compared_per_snapshot",fixture.tiles.size()*sizeof(Tile)},
        {"amg_apply_exercised",exerciseAmg},{"cases",cases},{"original_mesh_bytes_restored",true},
        {"scope","Complete native tile byte comparison around initialization failures, operator use, AMG application and destruction; streaming snapshots keep host memory bounded"}};
}
