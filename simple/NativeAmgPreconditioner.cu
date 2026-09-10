#include "NativeAmgPreconditioner.cuh"
#include "NativeOctreeAccess.cuh"
#include "NativeAmgTopology.h"
#include "ConstructionMemory.h"
#include "AMGSolver.h"
#include <array>
#include <map>
#include <stdexcept>
#include <cmath>
#include <sstream>

namespace simple {
namespace {
void check(cudaError_t status,const char* operation) {
    if(status!=cudaSuccess)throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}
using Key=std::array<int,4>;
struct HostTile {
    uint8_t type=0;
    std::array<uint8_t,512> cell{};
    using Coefficients=std::array<std::array<double,512>,4>;
    using Reaction=std::array<double,512>;
    std::unique_ptr<Coefficients> coefficient;
    std::unique_ptr<Reaction> reaction;
    HostTile() {cell.fill(NEUMANN);}
    double get(int channel,int q) const {return coefficient?(*coefficient)[channel][q]:0.;}
    void set(int channel,int q,double value) {
        // An absent array represents positive zero. Retain negative zero too,
        // so the storage change does not alter any uploaded coefficient bits.
        if(!coefficient&&(value!=0.||std::signbit(value)))coefficient=std::make_unique<Coefficients>();
        if(coefficient)(*coefficient)[channel][q]=value;
    }
    void add(int channel,int q,double value) {set(channel,q,get(channel,q)+value);}
    double getReaction(int q) const {return reaction?(*reaction)[q]:0.;}
    void setReaction(int q,double value) {
        if(!reaction&&(value!=0.||std::signbit(value)))reaction=std::make_unique<Reaction>();
        if(reaction)(*reaction)[q]=value;
    }
    void addReaction(int q,double value) {setReaction(q,getReaction(q)+value);}
};
void hostStorageMemory(const char* phase,const std::map<Key,HostTile>& data) {
    if(!std::getenv("SIMPLE_CONSTRUCTION_MEMORY"))return;
    size_t coefficients=0,reactions=0;
    for(const auto& item:data) {coefficients+=bool(item.second.coefficient);reactions+=bool(item.second.reaction);}
    std::printf("Native AMG host storage: phase=%s tiles=%llu coefficient_tiles=%llu reaction_tiles=%llu payload_bytes=%llu\n",
        phase,static_cast<unsigned long long>(data.size()),static_cast<unsigned long long>(coefficients),
        static_cast<unsigned long long>(reactions),static_cast<unsigned long long>(
            data.size()*sizeof(HostTile)+coefficients*sizeof(HostTile::Coefficients)+reactions*sizeof(HostTile::Reaction)));
    constructionMemory(phase);
}
struct MapCell {Tile* tile;int offset,slot;};
__global__ void transfer(const MapCell* map,int n,const double* rhs,double* result,bool upload) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;const auto m=map[c];
    if(upload) (*m.tile)(1,m.offset)=m.tile->type(m.offset)==INTERIOR?float(rhs[m.slot]):0.f;
    else result[m.slot]=m.tile->type(m.offset)==INTERIOR?double((*m.tile)(0,m.offset)):0.;
}
struct ClearWork {
    __device__ void operator()(HATileAccessor<Tile>&,HATileInfo<Tile>& info,const Coord& p) const {
        for(int ch=0;ch<5;++ch)info.tile()(ch,p)=0;
    }
};
struct Periodic {
    int nx;
    __device__ void operator()(HATileAccessor<Tile> acc,const int,HATileInfo<Tile>& info) const {
        const int count=nx<<info.mLevel;auto& tile=info.tile();
        if(info.mTileCoord[0]==0) {Coord other=info.mTileCoord;other[0]=count-1;tile.mNeighbors[0]=acc.tileInfo(info.mLevel,other);}
        if(info.mTileCoord[0]==count-1) {Coord other=info.mTileCoord;other[0]=0;tile.mNeighbors[3]=acc.tileInfo(info.mLevel,other);}
    }
};
int offset(const std::array<int,3>& p) {return (p[0]%8)*64+(p[1]%8)*8+p[2]%8;}
}
struct NativeAmgPreconditioner::Impl {
    std::unique_ptr<HADeviceGrid<Tile>> grid;
    AMGSolver amg{5,.5f,.5f,1.f};
    thrust::device_vector<MapCell> mapStorage;
    MapCell* mapping=nullptr;int count=0;
    explicit Impl(const Mesh& mesh,const std::vector<int>& slots,int gauge,double k,double mass,bool walls,const std::vector<double>& factors) {
        auto& original=nativeDeviceGrid(mesh);
        const int originalNy=int(std::llround(mesh.extent[1]/original.mH0));int shift=0;
        while((originalNy>>shift)>8)++shift;
        if((8<<shift)!=originalNy)throw std::runtime_error("Native AMG expects an integral power-of-two root resolution");
        const int maximum=original.mMaxLevel+shift;
        const float h0=std::ldexp(original.mH0,shift);
        const int rootCellsX=int(std::llround(mesh.extent[0]/h0));
        std::map<Key,HostTile> data;
        for(int level=0;level<=original.mMaxLevel;++level)for(int i=0;i<original.hNumTiles[level];++i) {
            const auto& info=original.hTileArrays[level][i];
            Key key{level+shift,info.mTileCoord[0],info.mTileCoord[1],info.mTileCoord[2]};
            data[key].type=info.mType;
        }
        for(int level=shift;level>0;--level) {
            std::vector<Key> parents;
            for(const auto& item:data)if(item.first[0]==level&&item.second.type!=GHOST) {
                const auto& q=item.first;parents.push_back({level-1,q[1]/2,q[2]/2,q[3]/2});
            }
            for(const auto& key:parents)data[key].type=NONLEAF;
        }
        // Native ghost spawning uses nonperiodic neighbor coordinates. The
        // periodic seam can join different leaf levels, so it also needs the
        // wrapped ghosts on BOTH sides, including the side whose coefficient
        // is stored on an existing fine leaf rather than on the ghost itself.
        completeNativeAmgPeriodicGhosts(data,maximum,rootCellsX/8,LEAF|NONLEAF,LEAF,GHOST);
        hostStorageMemory("amg.host_topology_ready",data);
        auto find=[&](int level,std::array<int,3> p)->std::pair<HostTile*,int> {
            if(level<0||level>maximum||p[1]<0||p[2]<0)return {nullptr,0};
            const int nx=rootCellsX<<level;p[0]=(p[0]%nx+nx)%nx;
            auto it=data.find({level,p[0]/8,p[1]/8,p[2]/8});
            return it==data.end()?std::make_pair(nullptr,0):std::make_pair(&it->second,offset(p));
        };
        auto get=[&](int level,std::array<int,3> p,int channel) {
            const auto entry=find(level,p);return entry.first?entry.first->get(channel,entry.second):0.;
        };
        for(size_t c=0;c<mesh.cells.size();++c) {
            const auto& cell=mesh.cells[c];const auto entry=find(cell.level+shift,cell.key);
            if(!entry.first||entry.first->type!=LEAF)throw std::runtime_error("Native AMG changed a physical leaf");
            entry.first->cell[entry.second]=int(c)==gauge?DIRICHLET:INTERIOR;
            entry.first->setReaction(entry.second,mass*cell.volume);
        }
        for(size_t j=0;j<mesh.faces.size();++j) {
            const auto& f=mesh.faces[j];
            const auto& p=mesh.cells[f.owner];const double w=k*f.area/f.distance*(factors.empty()?1.:factors[j]);
            if(f.neighbor<0) {if(walls) {const auto a=find(p.level+shift,p.key);a.first->addReaction(a.second,w);}continue;}
            const auto& n=mesh.cells[f.neighbor];int level=n.level+shift;auto key=n.key;double coefficient=-w;
            if(p.level!=n.level) {
                // The native ghost reconstruction contains a factor 1/2.
                // R_matrix=1/2 then gives the coarse face the sum of subfaces.
                coefficient*=2;
                if(p.level>n.level) {level=p.level+shift;key=p.key;++key[f.axis];}
            }
            const auto a=find(level,key);if(!a.first)throw std::runtime_error("Native AMG missing a required face/ghost tile");
            a.first->add(f.axis,a.second,coefficient);
        }
        hostStorageMemory("amg.host_face_coefficients_ready",data);
        // Ghost cell masks represent their physical coarse parent.
        for(auto& item:data)if(item.second.type==GHOST)for(int q=0;q<512;++q) {
            const auto& key=item.first;std::array<int,3> p{(key[1]*8+q/64)/2,(key[2]*8+(q/8)%8)/2,(key[3]*8+q%8)/2};
            const auto a=find(key[0]-1,p);item.second.cell[q]=a.first?a.first->cell[a.second]:NEUMANN;
        }
        // Coarsen masks and four negative-face coefficients in double precision.
        for(int level=maximum-1;level>=0;--level)for(auto& item:data)if(item.first[0]==level) {
            auto& tile=item.second;const auto& key=item.first;
            for(int q=0;q<512;++q) {
                std::array<int,3> base{2*(key[1]*8+q/64),2*(key[2]*8+(q/8)%8),2*(key[3]*8+q%8)};
                if(tile.type==NONLEAF) {
                    bool interior=false,dirichlet=false;
                    for(int i=0;i<2;++i)for(int j=0;j<2;++j)for(int l=0;l<2;++l) {
                        auto p=base;p[0]+=i;p[1]+=j;p[2]+=l;const auto a=find(level+1,p);
                        if(a.first) {interior|=a.first->cell[a.second]==INTERIOR;dirichlet|=a.first->cell[a.second]==DIRICHLET;}
                    }
                    tile.cell[q]=interior?INTERIOR:dirichlet?DIRICHLET:NEUMANN;
                }
                for(int axis=0;axis<3;++axis) {
                    double sum=0;for(int i=0;i<2;++i)for(int j=0;j<2;++j) {
                        auto p=base;p[(axis+1)%3]+=i;p[(axis+2)%3]+=j;auto before=p;--before[axis];
                        const auto a=find(level+1,p),b=find(level+1,before);
                        // A pinned child has no coarse unknown. Its conductance
                        // remains in the diagonal, not in a coarse off-diagonal.
                        if(a.first&&b.first&&a.first->cell[a.second]==INTERIOR&&b.first->cell[b.second]==INTERIOR)
                            sum+=get(level+1,p,axis);
                    }
                    tile.add(axis,q,.5*sum);
                }
            }
        }
        for(auto& item:data)if(item.second.type!=NONLEAF) {
            auto& tile=item.second;const auto& key=item.first;
            for(int q=0;q<512;++q)if(tile.cell[q]==INTERIOR) {
                double diagonal=tile.getReaction(q);std::array<int,3> p{key[1]*8+q/64,key[2]*8+(q/8)%8,key[3]*8+q%8};
                for(int axis=0;axis<3;++axis) {auto next=p;++next[axis];diagonal-=tile.get(axis,q)+get(key[0],next,axis);}
                tile.set(3,q,diagonal);
            }
            // Reaction terms are consumed only by the leaf/ghost diagonal.
            tile.reaction.reset();
        }
        for(int level=maximum-1;level>=0;--level)for(auto& item:data)if(item.first[0]==level&&item.second.type==NONLEAF) {
            auto& tile=item.second;const auto& key=item.first;
            for(int q=0;q<512;++q)if(tile.cell[q]==INTERIOR) {
                std::array<int,3> base{2*(key[1]*8+q/64),2*(key[2]*8+(q/8)%8),2*(key[3]*8+q%8)};double diagonal=0;
                for(int i=0;i<2;++i)for(int j=0;j<2;++j)for(int l=0;l<2;++l) {
                    auto p=base;p[0]+=i;p[1]+=j;p[2]+=l;diagonal+=get(level+1,p,3);
                }
                for(int axis=0;axis<3;++axis)for(int i=0;i<2;++i)for(int j=0;j<2;++j) {
                    auto p=base;p[axis]+=1;p[(axis+1)%3]+=i;p[(axis+2)%3]+=j;auto before=p;--before[axis];
                    const auto a=find(level+1,p),b=find(level+1,before);
                    if(a.first&&b.first&&a.first->cell[a.second]==INTERIOR&&b.first->cell[b.second]==INTERIOR)diagonal+=2*get(level+1,p,axis);
                }
                tile.set(3,q,.5*diagonal);
            }
        }
        hostStorageMemory("amg.host_hierarchy_ready",data);
        thrust::host_vector<uint32_t> hashes(maximum+1,10);
        for(int level=0;level<=maximum;++level) {
            size_t tiles=0;for(const auto& item:data)tiles+=item.first[0]==level;
            while((size_t(1)<<hashes[level])<32*tiles)++hashes[level];
        }
        grid=std::make_unique<HADeviceGrid<Tile>>(h0,hashes);
        for(auto it=data.begin();it!=data.end();) {
            const auto& item=*it;
            const auto& key=item.first;const auto& src=item.second;Tile tile;
            for(int ch=0;ch<Tile::num_channels;++ch)for(int q=0;q<Tile::CHNLSIZE;++q)tile(ch,q)=0;
            tile.mSerialIdx=0;
            for(int q=0;q<512;++q) {
                tile.type(q)=src.cell[q];
                if(src.type!=GHOST&&src.cell[q]==INTERIOR&&!(src.get(3,q)>0)) {
                    std::ostringstream message;message<<"Nonpositive native AMG diagonal at level "<<key[0]<<" voxel "<<q;throw std::runtime_error(message.str());
                }
                for(int ch=0;ch<4;++ch)tile(5+ch,q)=float(src.get(ch,q));
            }
            grid->setTileHost(key[0],Coord(key[1],key[2],key[3]),tile,src.type);
            // setTileHost copies synchronously; no subsequent host assembly
            // reads this map, so release each block in the same upload order.
            it=data.erase(it);
        }
        hostStorageMemory("amg.host_upload_released",data);
        // The copied native layout already contains its physical ghost tiles.
        // Added ancestors are complete root levels; no new physical leaves.
        grid->compressHost(false);grid->syncHostAndDevice();CalculateNeighborTiles(*grid);
        grid->launchTileFunc(Periodic{rootCellsX/8},-1,LEAF|NONLEAF|GHOST,LAUNCH_SUBTREE);
        std::vector<MapCell> map;map.reserve(slots.size());auto acc=grid->hostAccessor();
        for(size_t c=0;c<mesh.cells.size();++c) {
            const auto& cell=mesh.cells[c];const auto info=acc.tileInfo(cell.level+shift,Coord(cell.key[0]/8,cell.key[1]/8,cell.key[2]/8));
            if(info.empty()||!info.isLeaf())throw std::runtime_error("Native AMG active-leaf correspondence failed");
            map.push_back({info.mTilePtr,offset(cell.key),slots[c]});
        }
        count=int(map.size());mapStorage.assign(map.begin(),map.end());mapping=thrust::raw_pointer_cast(mapStorage.data());
        amg.omega=1.;check(cudaDeviceSynchronize(),"prepare native AMG hierarchy");
    }
    void apply(const double* source,double* result) {
        grid->launchVoxelFuncOnAllTiles(ClearWork{},LEAF|NONLEAF|GHOST);
        transfer<<<(count+255)/256,256>>>(mapping,count,source,result,true);
        amg.FASMuCycle(1,*grid,0,1,3,5,2,12);
        transfer<<<(count+255)/256,256>>>(mapping,count,source,result,false);
        check(cudaGetLastError(),"apply native AMG preconditioner");
    }
};
NativeAmgPreconditioner::NativeAmgPreconditioner(const Mesh& m,const std::vector<int>& s,int g,double k,double mass,bool walls,const std::vector<double>& factors):impl_(new Impl(m,s,g,k,mass,walls,factors)){}
NativeAmgPreconditioner::~NativeAmgPreconditioner()=default;
void NativeAmgPreconditioner::apply(const double* source,double* result){impl_->apply(source,result);}
int NativeAmgPreconditioner::levels() const{return impl_->grid->mMaxLevel+1;}
size_t NativeAmgPreconditioner::bytes() const{return size_t(impl_->grid->numTotalTiles())*sizeof(Tile)+size_t(impl_->count)*sizeof(MapCell);}
}
