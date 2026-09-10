#include "SimpleMesh.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <cstdint>
#include <limits>
#include <cstring>
#include <iostream>
#include "ConstructionMemory.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace simple {
namespace {
using Key=std::array<int,3>;
using FKey=std::array<int,4>;
using Json=nlohmann::json;
template<size_t N> using Record=std::array<double,N>;
#ifdef _WIN32
// File-backed read-only pages avoid a second, pagefile-backed copy of the
// immutable packed geometry. Keep the file open without write sharing for the
// lifetime of the records so their indexes cannot change underneath us.
struct PackedMapping {
    HANDLE file=INVALID_HANDLE_VALUE,object=nullptr;
    const void* view=nullptr;
    uint64_t bytes=0;
    PackedMapping()=default;
    PackedMapping(const PackedMapping&)=delete;
    PackedMapping& operator=(const PackedMapping&)=delete;
    ~PackedMapping() {
        if(view)UnmapViewOfFile(view);
        if(object)CloseHandle(object);
        if(file!=INVALID_HANDLE_VALUE)CloseHandle(file);
    }
    void open(const std::filesystem::path& path) {
        file=CreateFileW(path.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,
                         FILE_ATTRIBUTE_NORMAL,nullptr);
        LARGE_INTEGER length{};
        if(file==INVALID_HANDLE_VALUE || !GetFileSizeEx(file,&length) || length.QuadPart<24 ||
           uint64_t(length.QuadPart)>std::numeric_limits<size_t>::max())
            throw std::runtime_error("Invalid packed cut-geometry header or file size");
        bytes=uint64_t(length.QuadPart);
        object=CreateFileMappingW(file,nullptr,PAGE_READONLY,0,0,nullptr);
        if(!object)throw std::runtime_error("Cannot map packed cut geometry");
        view=MapViewOfFile(object,FILE_MAP_READ,0,0,0);
        if(!view)throw std::runtime_error("Cannot view packed cut geometry");
    }
};
#endif
template<size_t N> struct Records {
    std::vector<Record<N>> owned;
#ifdef _WIN32
    std::shared_ptr<PackedMapping> mapping;
#endif
    const Record<N>* data() const {
#ifdef _WIN32
        if(mapping)return reinterpret_cast<const Record<N>*>(static_cast<const char*>(mapping->view)+24);
#endif
        return owned.data();
    }
    size_t size() const {
#ifdef _WIN32
        if(mapping)return size_t((mapping->bytes-24)/sizeof(Record<N>));
#endif
        return owned.size();
    }
    const Record<N>& operator[](size_t i) const {return data()[i];}
    void validateFinite() const {
#ifdef _WIN32
        if(mapping) {
            // Check every value through a bounded buffer. Touching the entire
            // mapping here would retain all geometry pages while native tiles
            // are constructed, before random-access lookup needs those pages.
            // Use the same open read-only handle; its no-write-sharing lock
            // remains held for both validation and the later mapped accesses.
            LARGE_INTEGER first{};first.QuadPart=24;
            if(!SetFilePointerEx(mapping->file,first,nullptr,FILE_BEGIN))
                throw std::runtime_error("Cannot seek packed cut geometry");
            std::vector<Record<N>> buffer(std::min<size_t>(size(),65536));
            static_assert(65536ull*sizeof(Record<N>)<=MAXDWORD,"Packed validation read exceeds DWORD");
            for(size_t begin=0;begin<size();) {
                const size_t count=std::min(buffer.size(),size()-begin);
                const DWORD bytes=DWORD(count*sizeof(Record<N>));DWORD received=0;
                if(!ReadFile(mapping->file,buffer.data(),bytes,&received,nullptr) || received!=bytes)
                    throw std::runtime_error("Truncated packed cut geometry");
                for(size_t i=0;i<count;++i)for(double value:buffer[i])
                    if(!std::isfinite(value))throw std::runtime_error("Nonfinite cut geometry");
                begin+=count;
            }
            return;
        }
#endif
        for(const auto& row:owned)for(double value:row)
            if(!std::isfinite(value))throw std::runtime_error("Nonfinite cut geometry");
    }
};
template<size_t N> Vec3 vectorAt(const Record<N>& row,int start) {
    return Vec3(row.at(start),row.at(start+1),row.at(start+2));
}
// Immutable reference geometry needs ordered lookup, not a tree node per row.
// Keep the original records and index them by compact key/row pairs. Sorting
// ties by input row preserves std::map::emplace's first-record convention,
// including duplicate periodic faces and the order of volume accumulation.
template<size_t N,size_t D> class RecordIndex {
    using IndexKey=std::array<int,D>;
    using Entry=std::pair<IndexKey,uint32_t>;
    const Records<N>& records_;
    std::vector<Entry> entries_;
public:
    template<class KeyOf,class Duplicate>
    RecordIndex(const Records<N>& records,KeyOf keyOf,Duplicate duplicate):records_(records) {
        if(records.size()>std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("Reference geometry index exceeds 32-bit row capacity");
        entries_.reserve(records.size());
        for(size_t i=0;i<records.size();++i)entries_.emplace_back(keyOf(records[i]),uint32_t(i));
        std::sort(entries_.begin(),entries_.end());
        size_t count=0;
        for(const auto& entry:entries_) {
            if(count && entries_[count-1].first==entry.first)
                duplicate(records_[entries_[count-1].second],records_[entry.second]);
            else entries_[count++]=entry;
        }
        entries_.resize(count);
    }
    const Record<N>* find(const IndexKey& key) const {
        const auto it=std::lower_bound(entries_.begin(),entries_.end(),key,
            [](const Entry& entry,const IndexKey& value){return entry.first<value;});
        return it!=entries_.end() && it->first==key?&records_[it->second]:nullptr;
    }
    const std::vector<Entry>& entries() const {return entries_;}
};
template<size_t N>
Records<N> readRecords(Json& data,const char* name,const std::filesystem::path& parent) {
    static_assert(sizeof(Record<N>)==N*sizeof(double),"Packed records must be contiguous");
    Records<N> records;
    if(data.at("format")=="aphros_cut_geometry_v1") {
        const auto& source=data.at(name);records.owned.resize(source.size());
        for(size_t i=0;i<source.size();++i) {
            if(source[i].size()!=N)throw std::runtime_error("Unexpected geometry record width");
            for(size_t j=0;j<N;++j)records.owned[i][j]=source[i][j].template get<double>();
        }
        data.erase(name);
    } else {
        const auto& table=data.at("tables").at(name);
        const auto path=parent/table.at("file").get<std::string>();
        char magic[8]{};uint64_t rows=0;uint32_t columns=0,endian=0;
#ifdef _WIN32
        records.mapping=std::make_shared<PackedMapping>();
        records.mapping->open(path);
        const auto* header=static_cast<const char*>(records.mapping->view);
        std::memcpy(magic,header,8);std::memcpy(&rows,header+8,8);
        std::memcpy(&columns,header+16,4);std::memcpy(&endian,header+20,4);
        const bool headerRead=true;
        const uint64_t bytes=records.mapping->bytes;
#else
        std::ifstream stream(path,std::ios::binary);
        stream.read(magic,8);stream.read(reinterpret_cast<char*>(&rows),8);
        stream.read(reinterpret_cast<char*>(&columns),4);stream.read(reinterpret_cast<char*>(&endian),4);
        const bool headerRead=bool(stream);
        const uint64_t bytes=headerRead?std::filesystem::file_size(path):0;
#endif
        if(!headerRead || std::string(magic,8)!="CIRRCUT1" || endian!=0x01020304 || columns!=N ||
           rows!=table.at("rows").get<uint64_t>() || columns!=table.at("columns").get<uint32_t>() ||
           rows>records.owned.max_size() || rows>(std::numeric_limits<uint64_t>::max()-24)/sizeof(Record<N>) ||
           bytes!=24+rows*sizeof(Record<N>))
            throw std::runtime_error("Invalid packed cut-geometry header or file size");
#ifndef _WIN32
        records.owned.resize(size_t(rows));
        stream.read(reinterpret_cast<char*>(records.owned.data()),std::streamsize(rows*sizeof(Record<N>)));
        if(!stream)throw std::runtime_error("Truncated packed cut geometry");
#endif
    }
    records.validateFinite();
    return records;
}
Key cellKey(const Vec3& center,double h) {
    Key key{};
    for(int d=0;d<3;++d) key[d]=int(std::llround(center[d]/h-.5));
    return key;
}
}

Mesh makeEmbeddedOctree(const std::string& path,bool adaptive) {
    std::ifstream input(path); if(!input) throw std::runtime_error("Cannot open cut geometry");
    Json data; input>>data;
    if(data.at("format")!="aphros_cut_geometry_v1" && data.at("format")!="aphros_cut_geometry_v2")
        throw std::runtime_error("Unknown cut geometry format");
    const auto parent=std::filesystem::path(path).parent_path();
    constructionMemory("embedded_mesh.begin");
    auto cellRecords=readRecords<12>(data,"cells",parent);
    auto faceRecords=readRecords<8>(data,"faces",parent);
    const auto wallRecords=readRecords<11>(data,"walls",parent);
    constructionMemory("embedded_mesh.packed_records_ready");
    const double hf=data.at("finest_h");
    const int finestNy=data.at("finest_ny");
    Json topology{{"extent",data.at("extent")},{"refine_root_tiles",data.at("refine_root_tiles")}};
    auto mesh=makeOctreeChannel(adaptive?finestNy/2:finestNy,adaptive,true,false,path,topology.dump(),false);
    constructionMemory("embedded_mesh.native_topology_ready");
    const int nx=int(std::llround(mesh.extent[0]/hf));
    const auto keyOfCell=[](const auto& row){return Key{int(row[0]),int(row[1]),int(row[2])};};
    auto cells=std::make_unique<RecordIndex<12,3>>(cellRecords,keyOfCell,[](const auto&,const auto&){
        throw std::runtime_error("Duplicate reference cell");
    });
    const RecordIndex<11,3> walls(wallRecords,keyOfCell,[](const auto&,const auto&){});
    // Resolve wall ownership before dropping the full reference cell table.
    std::vector<int> fineToNew(cellRecords.size(),-1);
    std::vector<int> wallOwners;
    double referenceVolume=0;
    size_t expectedFaces=0,reservedFaces=0;
    {
    auto faces=std::make_unique<RecordIndex<8,4>>(faceRecords,[nx](const auto& row){
        FKey key{}; for(int d=0;d<4;++d) key[d]=int(row[d]);
        if(key[0]==0) key[1]%=nx;
        return key;
    },[hf](const auto& first,const auto& row){
        if(std::abs(first[7]-row[7])>1e-12*hf*hf)
            throw std::runtime_error("Inconsistent periodic face aperture");
    });
    constructionMemory("embedded_mesh.reference_indices_ready");
    Mesh background;
    background.extent=mesh.extent;
    background.periodicZ=mesh.periodicZ;
    background.cells=std::move(mesh.cells);
    const auto& originalCells=background.cells;
    mesh.cells.clear(); mesh.coarseFineFaces=0;
    std::vector<int> oldIndices,oldToNew(originalCells.size(),-1);
    // A fine reference row can map to at most one native fine leaf. This dense
    // integer map replaces another key/tree allocation for each retained cell.
    for(int old=0;old<int(originalCells.size());++old) {
        auto cell=originalCells[old];
        if(std::abs(cell.h-hf)<hf*1e-12) {
            auto key=cellKey(cell.center,hf); const auto* row=cells->find(key);
            if(!row) continue;
            cell.volume=row->at(7);
            cell.cut=row->at(8)!=0.;
            fineToNew[size_t(row-cellRecords.data())]=int(mesh.cells.size());
        } else {
            if(std::abs(cell.h-2*hf)>hf*1e-12) throw std::runtime_error("Only one refinement level is supported");
            int active=0; double volume=0;
            Key low{}; for(int d=0;d<3;++d) low[d]=int(std::llround((cell.center[d]-.5*cell.h)/hf));
            for(int i=0;i<2;++i) for(int j=0;j<2;++j) for(int k=0;k<2;++k) {
                const auto* row=cells->find(Key{low[0]+i,low[1]+j,low[2]+k});
                if(row) {++active;volume+=row->at(7);}
            }
            if(!active) continue;
            if(active!=8 || std::abs(volume-cell.volume)>1e-12*cell.volume)
                throw std::runtime_error("Wall refinement missed a partial coarse leaf cell");
        }
        oldToNew[old]=int(mesh.cells.size()); oldIndices.push_back(old); mesh.cells.push_back(cell);
    }
    retainNativeCells(mesh,oldIndices);
    constructionMemory("embedded_mesh.fluid_cells_ready");
    wallOwners.reserve(walls.entries().size());
    for(const auto& entry:walls.entries()) {
        const auto* cell=cells->find(entry.first);
        const int owner=cell?fineToNew[size_t(cell-cellRecords.data())]:-1;
        if(owner<0)throw std::runtime_error("Embedded wall is missing its fine native cell");
        wallOwners.push_back(owner);
    }
    // Preserve the original sorted accumulation order. These are the last
    // consumers of reference cell rows; faces need only the retained cells.
    for(const auto& entry:cells->entries())referenceVolume+=cellRecords[entry.second].at(7);
    cells.reset();cellRecords=Records<12>{};std::vector<int>().swap(fineToNew);
    constructionMemory("embedded_mesh.cell_records_released");
    // Count and cache only the selected face geometry before allocating the
    // final array. This lets the immutable reference face table and its index
    // die before the much wider native Face records are materialized.
    struct FacePatch {uint64_t traversal;Vec3 center;double area;};
    std::vector<FacePatch> patches;
    uint64_t traversal=0;
    size_t fluidFaces=0;
    const auto clippedFace=[&](const Face& originalFace,bool store) {
        auto f=originalFace;
        const int p=oldToNew[f.owner],n=f.neighbor>=0?oldToNew[f.neighbor]:-1;
        if(p<0 && n<0) return;
        if(f.neighbor<0) {
            FKey key{f.axis,0,0,0};
            for(int d=0;d<3;++d)key[d+1]=int(std::llround(f.center[d]/hf-(d==f.axis?0.:.5)));
            if(f.axis==0)key[1]%=nx;
            // A cut cube may touch the background box while its fluid aperture
            // on that face is empty. Only actual open fluid contact is forbidden.
            if(!faces->find(key))return;
            throw std::runtime_error("Twisted fluid has an open aperture on the outer nonperiodic box");
        }
        const double h=std::min(originalCells[f.owner].h,originalCells[f.neighbor].h);
        if(std::abs(h-hf)<hf*1e-12) {
            FKey key{f.axis,0,0,0};
            for(int d=0;d<3;++d) key[d+1]=int(std::llround(f.center[d]/hf-(d==f.axis?0.:.5)));
            if(f.axis==0) key[1]%=nx;
            const auto* row=faces->find(key);
            if(!row) return;
            if(p<0||n<0) throw std::runtime_error("Positive aperture has an excluded neighbor");
            const Vec3 oldCenter=f.center;
            f.center=vectorAt(*row,4);
            if(f.axis==0 && oldCenter[0]>.5*mesh.extent[0] && f.center[0]<hf*.1) f.center[0]+=mesh.extent[0];
            f.area=row->at(7);
            f.ownerOffset=f.center-originalCells[f.owner].center;
            f.neighborOffset=f.ownerOffset-f.delta;
        }
        if(p<0||n<0) throw std::runtime_error("Full coarse face crosses excluded fluid");
        f.owner=p; f.neighbor=n;
        if(store) {
            patches.push_back({traversal,f.center,f.area});
        } else ++fluidFaces;
    };
    visitFacesAndValidate(background,true,false,[&](const Face& f){clippedFace(f,false);});
    constructionMemory("embedded_mesh.faces_counted");
    if(walls.entries().size()>mesh.faces.max_size() || fluidFaces>mesh.faces.max_size()-walls.entries().size())
        throw std::length_error("Embedded face count exceeds storage capacity");
    expectedFaces=fluidFaces+walls.entries().size();
    patches.reserve(fluidFaces);
    visitFacesAndValidate(background,true,false,[&](const Face& f){clippedFace(f,true);++traversal;});
    if(patches.size()!=fluidFaces)throw std::runtime_error("Embedded face count changed between traversals");
    constructionMemory("embedded_mesh.faces_cached");
    faces.reset();faceRecords=Records<8>{};
    constructionMemory("embedded_mesh.face_records_released");
    mesh.faces.reserve(expectedFaces);reservedFaces=mesh.faces.capacity();
    uint64_t seen=0;size_t next=0;
    visitFacesAndValidate(background,true,false,[&](const Face& originalFace) {
        const uint64_t current=seen++;
        if(next==patches.size() || current<patches[next].traversal)return;
        if(current!=patches[next].traversal)throw std::runtime_error("Embedded cached face order changed");
        const auto& patch=patches[next++];auto f=originalFace;
        const int p=oldToNew[f.owner],n=f.neighbor>=0?oldToNew[f.neighbor]:-1;
        if(p<0||n<0)throw std::runtime_error("Embedded cached face lost its fluid neighbor");
        const double h=std::min(originalCells[f.owner].h,originalCells[f.neighbor].h);
        if(std::abs(h-hf)<hf*1e-12) {
            f.center=patch.center;f.area=patch.area;
            f.ownerOffset=f.center-originalCells[f.owner].center;
            f.neighborOffset=f.ownerOffset-f.delta;
        }
        f.owner=p;f.neighbor=n;
        if(mesh.cells[p].level!=mesh.cells[n].level)++mesh.coarseFineFaces;
        mesh.faces.push_back(f);
    });
    if(seen!=traversal || next!=patches.size() || mesh.faces.size()!=fluidFaces)
        throw std::runtime_error("Embedded cached face traversal changed");
    constructionMemory("embedded_mesh.fluid_faces_ready");
    } // Drop background cells, lookup/coverage workspaces and old native indices.
    size_t wallIndex=0;
    for(const auto& entry:walls.entries()) {
        const int owner=wallOwners.at(wallIndex++);
        const auto& row=wallRecords[entry.second];
        Face f; f.owner=owner; f.boundary=1;
        f.center=vectorAt(row,3); f.embeddedNormal=vectorAt(row,6);
        f.area=row.at(9);
        // Compact implicit coefficient distance used by Aphros. The actual wall
        // location is center/ownerOffset; its gradient uses a separate wall fit.
        f.distance=.5*hf;
        f.ownerOffset=f.center-mesh.cells[f.owner].center; f.delta=f.ownerOffset;
        if(!f.embeddedNormal.allFinite() || std::abs(f.embeddedNormal.norm()-1)>1e-10)
            throw std::runtime_error("Invalid embedded wall normal");
        mesh.faces.push_back(f);
    }
    if(mesh.faces.size()!=expectedFaces || mesh.faces.capacity()!=reservedFaces)
        throw std::runtime_error("Embedded face storage grew after exact reservation");
    constructionMemory("embedded_mesh.walls_ready");
    std::cout<<"Embedded face storage: size="<<mesh.faces.size()<<" capacity="<<mesh.faces.capacity()
             <<" bytes="<<mesh.faces.capacity()*sizeof(Face)<<'\n';
    double volume=0;
    for(const auto& c:mesh.cells) volume+=c.volume;
    if(std::abs(volume-referenceVolume)>1e-10*referenceVolume) throw std::runtime_error("Imported fluid volume differs from reference geometry");
    // Validate the imported control-volume geometry after clipping, not just
    // the original background cubes. Aperture and wall area vectors must close.
    std::vector<Vec3> closure(mesh.cells.size(),Vec3::Zero());
    std::vector<int> wallCount(mesh.cells.size(),0);
    for(const auto& f:mesh.faces) {
        if(!(f.area>0) || !std::isfinite(f.area) || !f.center.allFinite())
            throw std::runtime_error("Invalid embedded face geometry");
        const Vec3 normal=f.neighbor<0?f.embeddedNormal:Vec3(Vec3::Unit(f.axis)*f.sign);
        closure[f.owner]+=f.area*normal;
        if(f.neighbor>=0)closure[f.neighbor]-=f.area*normal;
        else ++wallCount[f.owner];
    }
    for(int i=0;i<int(mesh.cells.size());++i) {
        const auto& c=mesh.cells[i];
        if(!(c.volume>0) || !std::isfinite(c.volume) || c.volume>std::pow(c.h,3)*(1+1e-12))
            throw std::runtime_error("Invalid cut-cell volume");
        if(wallCount[i]!=int(c.cut))throw std::runtime_error("Cut-cell wall coverage is inconsistent");
        if(closure[i].norm()>1e-10*c.h*c.h)
            throw std::runtime_error("Embedded cell area vectors do not close");
    }
    mesh.embedded=true; mesh.geometrySource=path;
    mesh.backend="HADeviceGrid<Tile> native leaves with shared Aphros cut geometry";
    constructionMemory("embedded_mesh.geometry_ready");
    return mesh;
}
}
