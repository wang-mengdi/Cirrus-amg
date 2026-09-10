#include "NativeCompactGpu.h"
#include "NativeOctreeAccess.cuh"
#include "NativeAmgPreconditioner.cuh"
#include "PoissonGrid.h"
#include "ConstructionMemory.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <map>
#include <limits>
#include <stdexcept>
#include <cmath>
#include <chrono>
#include <sstream>
#include <fstream>
#include <cstdint>

namespace simple {
namespace {
// These separate double fields store cell values only. The original native
// float tiles retain their larger node-channel stride and unchanged layout.
constexpr int cellFieldStride=int(Tile::SIZE);
static_assert(cellFieldStride==Tile::DIM*Tile::DIM*Tile::DIM,"Cell-field stride must cover every local cell");
void checkedCuda(cudaError_t value,const char* label) {
    if(value!=cudaSuccess)throw std::runtime_error(std::string(label)+": "+cudaGetErrorString(value));
}
template<class T> struct DeviceArray {
    T* data=nullptr;size_t count=0;
    void allocate(size_t n) {
        if(data)throw std::runtime_error("GPU array already allocated");
        count=n;if(!n)return;
        checkedCuda(cudaMalloc(reinterpret_cast<void**>(&data),n*sizeof(T)),"allocate native operator field");
    }
    void upload(const std::vector<T>& values) {
        if(values.size()!=count)throw std::runtime_error("GPU upload size mismatch");
        if(count)checkedCuda(cudaMemcpy(data,values.data(),count*sizeof(T),cudaMemcpyHostToDevice),"upload native operator field");
    }
    ~DeviceArray(){if(data)cudaFree(data);}
};
struct Interface {int owner,neighbor;double weight;};
struct PressureInterface {int owner,neighbor,begin,end;};
struct PressureEntry {int slot;double weight;};
__global__ void addViscosityFaces(const PressureInterface* faces,size_t count,
                                 const PressureEntry* entries,const double* x,double* y,double scale) {
    const size_t j=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(j>=count)return;
    const auto f=faces[j];const double center=f.neighbor<0?0.:x[f.owner];double flux=0.;
    for(int k=f.begin;k<f.end;++k)flux+=entries[k].weight*(x[entries[k].slot]-center);
    flux*=scale;atomicAdd(y+f.owner,flux);
    if(f.neighbor>=0)atomicAdd(y+f.neighbor,-flux);
}
__global__ void addPressureInterfaces(const PressureInterface* faces,size_t count,
                                     const PressureEntry* entries,const double* x,double* y,double scale) {
    const size_t j=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(j>=count)return;
    const auto f=faces[j];const double center=x[f.owner];double flux=0.;
    // A local gradient annihilates constant pressure. Owner-relative values
    // preserve that identity exactly without changing its spatial stencil.
    for(int k=f.begin;k<f.end;++k)flux+=entries[k].weight*(x[entries[k].slot]-center);
    flux*=scale;atomicAdd(y+f.owner,flux);atomicAdd(y+f.neighbor,-flux);
}
__global__ void products(const int* slots,int count,const double* a,const double* b,double* out) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<count)out[c]=a[slots[c]]*b[slots[c]];
}
__global__ void initializePcg(const int* slots,int count,const double* b,const double* degree,
                            const double* volume,const double* wall,double* x,double* r,
                            double* diagonal,double k,double m,bool walls,int gauge) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=count)return;const int s=slots[c];
    x[s]=0;r[s]=b[s];diagonal[s]=c==gauge?1.:k*(degree[s]+(walls?wall[s]:0.))+m*volume[s];
}
__global__ void preconditionPcg(const int* slots,int count,const double* r,const double* diagonal,double* z) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<count) {const int s=slots[c];z[s]=r[s]/diagonal[s];}
}
__global__ void directionPcg(const int* slots,int count,const double* z,double* p,double beta) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<count) {const int s=slots[c];p[s]=z[s]+beta*p[s];}
}
__global__ void updatePcg(const int* slots,int count,double* x,double* r,const double* p,const double* ap,double alpha) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<count) {const int s=slots[c];x[s]+=alpha*p[s];r[s]-=alpha*ap[s];}
}
__global__ void residualPcg(const int* slots,int count,const double* b,const double* ax,double* r) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<count) {const int s=slots[c];r[s]=b[s]-ax[s];}
}
__global__ void residualPackedRhs(const int* slots,int count,const double* b,const double* ax,double* r) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<count) {const int s=slots[c];r[s]=b[c]-ax[s];}
}
__global__ void pinResult(double* y,const double* x,int slot) {y[slot]=x[slot];}
__global__ void packKrylov(const int* slots,int n,const double* field,double* packed,double scale) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<n)packed[c]=scale*field[slots[c]];
}
__global__ void unpackKrylov(const int* slots,int n,const double* packed,double* field) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<n)field[slots[c]]=packed[c];
}
__global__ void dotKrylov(const int* slots,int n,const double* field,const double* packed,double* product) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<n)product[c]=field[slots[c]]*packed[c];
}
__global__ void subtractKrylov(const int* slots,int n,double* field,const double* packed,double scale) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<n)field[slots[c]]-=scale*packed[c];
}
// One product reduction per (basis vector, cell block). Unlike MGS, CGS forms
// all coefficients from the same vector before subtracting their combination.
__global__ void krylovProductsMany(const int* slots,int n,const double* field,
                                    const double* packed,double* partials) {
    __shared__ double sums[256];
    const int c=int(blockIdx.x)*256+threadIdx.x;
    sums[threadIdx.x]=c<n?field[slots[c]]*packed[size_t(blockIdx.y)*n+c]:0.;
    __syncthreads();
    for(int offset=128;offset;offset>>=1) {
        if(threadIdx.x<offset)sums[threadIdx.x]+=sums[threadIdx.x+offset];
        __syncthreads();
    }
    if(threadIdx.x==0)partials[size_t(blockIdx.y)*gridDim.x+blockIdx.x]=sums[0];
}
__global__ void reduceKrylovProducts(const double* partials,int blocks,double* coefficients) {
    __shared__ double sums[256];double value=0.;
    for(int i=threadIdx.x;i<blocks;i+=256)value+=partials[size_t(blockIdx.x)*blocks+i];
    sums[threadIdx.x]=value;__syncthreads();
    for(int offset=128;offset;offset>>=1) {
        if(threadIdx.x<offset)sums[threadIdx.x]+=sums[threadIdx.x+offset];
        __syncthreads();
    }
    if(threadIdx.x==0)coefficients[blockIdx.x]=sums[0];
}
__global__ void subtractKrylovMany(const int* slots,int n,double* field,const double* packed,
                                  const double* coefficients,int count) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
    double correction=0.;for(int i=0;i<count;++i)correction+=coefficients[i]*packed[size_t(i)*n+c];
    field[slots[c]]-=correction;
}
__global__ void updateKrylov(const int* slots,int n,double* x,const double* images,const double* coefficients,int count) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;double update=0;
    for(int i=0;i<count;++i)update+=coefficients[i]*images[size_t(i)*n+c];x[slots[c]]+=update;
}
__device__ void twofoldAdd(double& high,double& low,double value) {
    const double sum=high+value,b=sum-high;
    const double error=(high-(sum-b))+(value-b);
    const double tail=low+error,merged=sum+tail,c=merged-sum;
    low=(sum-(merged-c))+(tail-c);high=merged;
}
__global__ void updateKrylovTwofold(const int* slots,int n,double* x,double* low,
                                  const double* images,const double* coefficients,int count) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
    double update=0,tail=0;
    for(int i=0;i<count;++i) {
        const double a=coefficients[i],b=images[size_t(i)*n+c],product=__dmul_rn(a,b);
        twofoldAdd(update,tail,product);twofoldAdd(update,tail,__fma_rn(a,b,-product));
    }
    const int s=slots[c];double high=x[s],error=low[s];
    twofoldAdd(high,error,update);twofoldAdd(high,error,tail);x[s]=high;low[s]=error;
}
__global__ void subtractMeanTwofold(const int* slots,int n,double* x,double* low,double mean) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;const int s=slots[c];
    double high=x[s],error=low[s];twofoldAdd(high,error,-mean);x[s]=high;low[s]=error;
}
__global__ void addLowImage(const int* slots,int n,double* y,const double* low) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<n) {const int s=slots[c];y[s]+=low[s];}
}
__global__ void sumValues(const int* slots,int n,const double* values,double* out) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<n)out[c]=values[slots[c]];
}
__global__ void subtractMean(const int* slots,int n,double* values,double mean) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<n)values[slots[c]]-=mean;
}
__global__ void addInterfaces(const Interface* faces,size_t count,const double* x,double* y,double scale) {
    const size_t j=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(j>=count)return;
    const auto f=faces[j];const double flux=scale*f.weight*(x[f.owner]-x[f.neighbor]);
    // One shared subface flux, with opposite signs in the adjacent cells.
    atomicAdd(y+f.owner,flux);atomicAdd(y+f.neighbor,-flux);
}
using Key=std::array<int,4>;
// Restore borrowed native metadata even when initialization throws after the
// first upload. Member destructors also run for an incomplete Impl constructor.
struct TileMetadataLease {
    static_assert(std::is_standard_layout<Tile>::value,"Native tile metadata requires standard layout");
    static constexpr size_t begin=offsetof(Tile,mNeighbors);
    static constexpr size_t bytes=sizeof(Tile)-begin;
    static_assert(begin==sizeof(Tile::mData),"Only numeric channels precede native metadata");
    static_assert(offsetof(Tile,mCellType)>=begin && offsetof(Tile,mSerialIdx)+sizeof(int)<=sizeof(Tile),
                  "Native mutable metadata must lie in the saved suffix");
    using Tail=std::array<unsigned char,bytes>;
    struct Snapshot {HATileInfo<Tile> info;Tail tail;};
    HADeviceGrid<Tile>& grid;
    std::vector<Snapshot> original;
    bool changed=false;
    explicit TileMetadataLease(HADeviceGrid<Tile>& value):grid(value) {
        size_t count=0;
        for(int level=0;level<=grid.mMaxLevel;++level)for(int j=0;j<grid.hNumTiles[level];++j)
            count+=(grid.hTileArrays[level][j].mType&(LEAF|NONLEAF|GHOST))!=0;
        original.reserve(count);
        // Same order and tile selection as getHostTileHolder, without copying
        // the fifteen numerical channels that this operator never changes.
        for(int level=0;level<=grid.mMaxLevel;++level)for(int j=0;j<grid.hNumTiles[level];++j) {
            const auto info=grid.hTileArrays[level][j];
            if(!(info.mType&(LEAF|NONLEAF|GHOST)))continue;
            original.emplace_back();auto& saved=original.back();saved.info=info;
            checkedCuda(cudaMemcpy(saved.tail.data(),address(info),bytes,cudaMemcpyDeviceToHost),
                        "save native operator tile metadata");
        }
        constructionMemory("native_gpu.tile_backup_ready");
    }
    static unsigned char* address(const HATileInfo<Tile>& info) {
        return reinterpret_cast<unsigned char*>(info.mTilePtr)+begin;
    }
    static void serial(Tail& tail,int value) {
        std::memcpy(tail.data()+offsetof(Tile,mSerialIdx)-begin,&value,sizeof(value));
    }
    static unsigned char* types(Tail& tail) {return tail.data()+offsetof(Tile,mCellType)-begin;}
    ~TileMetadataLease() {
        if(!changed)return;
        cudaDeviceSynchronize();
        for(const auto& saved:original)
            cudaMemcpy(address(saved.info),saved.tail.data(),bytes,cudaMemcpyHostToDevice);
    }
};
struct PeriodicNeighbors {
    int rootTilesX;
    __device__ void operator()(HATileAccessor<Tile> acc,const int,HATileInfo<Tile>& info) const {
        const int count=rootTilesX<<info.mLevel;auto& tile=info.tile();
        if(info.mTileCoord[0]==0) {
            Coord other=info.mTileCoord;other[0]=count-1;tile.mNeighbors[0]=acc.tileInfo(info.mLevel,other);
        }
        if(info.mTileCoord[0]==count-1) {
            Coord other=info.mTileCoord;other[0]=0;tile.mNeighbors[3]=acc.tileInfo(info.mLevel,other);
        }
    }
};
struct CompactCells {
    const double* values;double* result;
    const double* weights;const double* volume;const double* wall;
    size_t fieldSize;double diffusionScale,massScale;bool wallDirichlet;
    const unsigned char* skipFaces=nullptr;
    __device__ void operator()(HATileAccessor<Tile>& acc,HATileInfo<Tile>& info,const Coord& local) const {
        auto& tile=info.tile();if(tile.type(local)!=INTERIOR)return;
        const int offset=acc.localCoordToOffset(local),slot=tile.mSerialIdx*cellFieldStride+offset;
        const double center=values[slot];
        double value=massScale*volume[slot]*center;
        if(wallDirichlet)value+=diffusionScale*wall[slot]*center;
        for(int axis=0;axis<3;++axis)for(int sign=-1;sign<=1;sign+=2) {
            Coord other=local;other[axis]+=sign;const Tile* neighbor=&tile;
            if(other[axis]<0 || other[axis]>=Tile::DIM) {
                const auto adjacent=tile.mNeighbors[axis+(sign>0?3:0)];
                if(adjacent.empty())continue;
                neighbor=adjacent.mTilePtr;other[axis]=(sign<0?Tile::DIM-1:0);
            }
            const int next=neighbor->mSerialIdx*cellFieldStride+acc.localCoordToOffset(other);
            const size_t faceSlot=size_t(axis)*fieldSize+(sign<0?slot:next);
            if(skipFaces&&skipFaces[faceSlot])continue;
            const double weight=weights[faceSlot];
            if(weight!=0)value+=diffusionScale*weight*(center-values[next]);
        }
        result[slot]=value;
    }
};
}

struct NativeCompactGpu::Impl {
    std::shared_ptr<void> nativeStorage;
    HADeviceGrid<Tile>& grid;
    TileMetadataLease metadata;
    std::vector<int> slots;
    size_t size=0;
    DeviceArray<double> faceWeights,volumes,wallDiagonal,x,y;
    DeviceArray<double> degree,rhs,residual,z,direction,diagonal,scalar;
    DeviceArray<double> originalRhs;
    DeviceArray<double> solutionLow,solutionLowAx;
    DeviceArray<int> deviceSlots;
    DeviceReducer<double> reducer;
    std::unique_ptr<NativeAmgPreconditioner> pressureAmg,diffusionAmg;
    int amgGauge=-1;double amgViscosity=0,amgMass=0;
    bool meanZeroPressure=false;
    std::vector<double> cellVolumes,faceFactors;
    DeviceArray<double> basis,images,krylovCoefficients;
    DeviceArray<double> krylovPartials;
    bool batchedCgs2=false;
    int krylovDimension=20;
    DeviceArray<Interface> interfaces;
    bool fullPressure=false;
    DeviceArray<PressureInterface> pressureInterfaces;
    DeviceArray<PressureEntry> pressureEntries;
    bool fullViscosity=false;
    DeviceArray<PressureInterface> viscosityFaces;
    DeviceArray<PressureEntry> viscosityEntries;
    DeviceArray<unsigned char> viscositySkipFaces;

    explicit Impl(const Mesh& mesh,const std::vector<double>& factors):nativeStorage(mesh.nativeStorage),grid(nativeDeviceGrid(mesh)),metadata(grid),faceFactors(factors) {
        if(!mesh.embedded || mesh.cells.empty())throw std::runtime_error("Native GPU compact operator requires embedded fluid leaves");
        if(!faceFactors.empty()&&faceFactors.size()!=mesh.faces.size())throw std::runtime_error("Native face coefficient count differs from geometry");
        for(double value:faceFactors)if(!(value>0&&std::isfinite(value)))throw std::runtime_error("Native face coefficient must be positive and finite");
        if(mesh.periodicZ)throw std::runtime_error("Native embedded GPU operator currently supports x periodicity only");
        const int rootTilesX=int(std::llround(mesh.extent[0]/(grid.mH0*Tile::DIM)));
        if(metadata.original.size()>size_t(std::numeric_limits<int>::max()))throw std::runtime_error("Native tile index overflow");
        {
            // Only this scope needs the editable metadata copy and tile lookup.
            // The separate lease retains the original metadata for restoration.
            std::vector<TileMetadataLease::Tail> edited;
            edited.reserve(metadata.original.size());
            std::map<Key,size_t> tiles;
            // Tile zero is shared by every inactive tile. CompactCells may
            // inspect a neighbor's positive-face coefficient before checking
            // whether that neighbor has any fluid cells, so its serial must
            // remain valid and its weights/skip flags must remain zero.
            std::vector<int> serials(metadata.original.size(),0);
            for(size_t serial=0;serial<metadata.original.size();++serial) {
                const auto& saved=metadata.original[serial];const auto& info=saved.info;
                edited.push_back(saved.tail);auto& tail=edited.back();
                std::fill_n(TileMetadataLease::types(tail),Tile::SIZE,static_cast<unsigned char>(NEUMANN));
                tiles.emplace(Key{info.mLevel,info.mTileCoord[0],info.mTileCoord[1],info.mTileCoord[2]},serial);
            }
            slots.resize(mesh.cells.size());
            for(size_t c=0;c<mesh.cells.size();++c) {
                const auto& cell=mesh.cells[c];Key key{cell.level,cell.key[0]/8,cell.key[1]/8,cell.key[2]/8};
                auto it=tiles.find(key);
                if(it==tiles.end() || !metadata.original[it->second].info.isLeaf())throw std::runtime_error("Active cell is not an original native leaf");
                Coord local(cell.key[0]%8,cell.key[1]%8,cell.key[2]%8);
                const int offset=HACoordAccessor<Tile>::localCoordToOffset(local);
                slots[c]=int(it->second);serials[it->second]=-1;
                TileMetadataLease::types(edited[it->second])[offset]=INTERIOR;
            }
            size_t activeTiles=0;
            for(size_t j=0;j<serials.size();++j) {
                if(serials[j]<0)serials[j]=int(++activeTiles);
                TileMetadataLease::serial(edited[j],serials[j]);
            }
            size=(activeTiles+1)*cellFieldStride;
            if(size>size_t(std::numeric_limits<int>::max()))throw std::runtime_error("Native operator slot index overflow");
            for(size_t c=0;c<mesh.cells.size();++c) {
                const auto& key=mesh.cells[c].key;
                const int offset=HACoordAccessor<Tile>::localCoordToOffset(Coord(key[0]%8,key[1]%8,key[2]%8));
                slots[c]=serials[size_t(slots[c])]*cellFieldStride+offset;
            }
            if(std::getenv("SIMPLE_CONSTRUCTION_MEMORY"))
                std::printf("Native GPU double fields: original_tiles=%llu active_tiles=%llu storage_tiles=%llu zero_tile=0 field_values=%llu original_field_values=%llu\n",
                    static_cast<unsigned long long>(metadata.original.size()),static_cast<unsigned long long>(activeTiles),
                    static_cast<unsigned long long>(activeTiles+1),static_cast<unsigned long long>(size),
                    static_cast<unsigned long long>(metadata.original.size()*cellFieldStride));
            metadata.changed=true;
            for(size_t j=0;j<edited.size();++j)
                checkedCuda(cudaMemcpy(TileMetadataLease::address(metadata.original[j].info),edited[j].data(),
                                       TileMetadataLease::bytes,cudaMemcpyHostToDevice),"initialize native operator tile metadata");
            CalculateNeighborTiles(grid);
            // The benchmark is periodic in x. Wrap the original native neighbor
            // pointers at that seam; interior and transverse pointers are unchanged.
            grid.launchTileFunc(PeriodicNeighbors{rootTilesX},-1,LEAF|NONLEAF|GHOST,LAUNCH_SUBTREE);
            checkedCuda(cudaDeviceSynchronize(),"initialize native periodic neighbors");
        }
        constructionMemory("native_gpu.tile_workspace_released");
        {
            std::vector<double> v(size,0.);
            cellVolumes.resize(mesh.cells.size());
            for(size_t c=0;c<mesh.cells.size();++c)v[slots[c]]=cellVolumes[c]=mesh.cells[c].volume;
            volumes.allocate(size);volumes.upload(v);
        }
        std::vector<double> weights(3*size,0.),walls(size,0.),deg(size,0.);
        std::vector<Interface> cf;
        for(size_t j=0;j<mesh.faces.size();++j) {
            const auto& face=mesh.faces[j];const double factor=faceFactors.empty()?1.:faceFactors[j];
            if(!(face.area>0 && face.distance>0))throw std::runtime_error("Invalid native face coefficient");
            if(face.neighbor<0) {
                if(face.boundary!=1)throw std::runtime_error("Native embedded GPU operator requires no-slip external walls");
                walls[slots[face.owner]]+=face.area/face.distance*factor;continue;
            }
            if(face.sign!=1)throw std::runtime_error("Native compact operator expects positive shared-face orientation");
            const auto& p=mesh.cells[face.owner];const auto& n=mesh.cells[face.neighbor];
            const double weight=face.area/face.distance*factor;
            deg[slots[face.owner]]+=weight;deg[slots[face.neighbor]]+=weight;
            if(p.level!=n.level)cf.push_back({slots[face.owner],slots[face.neighbor],weight});
            else {
                auto expected=p.key;++expected[face.axis];
                expected[0]%=(rootTilesX<<p.level)*Tile::DIM;
                if(expected!=n.key)throw std::runtime_error("Same-level face does not match the native stencil");
                const size_t slot=size_t(face.axis)*size+slots[face.neighbor];
                if(weights[slot]!=0)throw std::runtime_error("Duplicate native regular face");
                weights[slot]=weight;
            }
        }
        faceWeights.allocate(weights.size());faceWeights.upload(weights);
        std::vector<double>().swap(weights);
        wallDiagonal.allocate(size);wallDiagonal.upload(walls);
        std::vector<double>().swap(walls);
        degree.allocate(size);degree.upload(deg);
        std::vector<double>().swap(deg);
        interfaces.allocate(cf.size());interfaces.upload(cf);
        std::vector<Interface>().swap(cf);
        deviceSlots.allocate(slots.size());deviceSlots.upload(slots);
        x.allocate(size);y.allocate(size);
    }
    void applyDevice(const double* values,double* result,double diffusionScale,double massScale,bool wallDirichlet,int gauge=-1) {
        checkedCuda(cudaMemset(result,0,size*sizeof(double)),"clear native operator result");
        const bool fullDiffusion=wallDirichlet&&fullViscosity;
        grid.launchVoxelFunc(CompactCells{values,result,faceWeights.data,volumes.data,wallDiagonal.data,size,diffusionScale,massScale,
                             wallDirichlet&&!fullDiffusion,fullDiffusion?viscositySkipFaces.data:nullptr},
                             -1,LEAF,LAUNCH_SUBTREE);
        if(fullDiffusion) {
            if(viscosityFaces.count)addViscosityFaces<<<unsigned((viscosityFaces.count+255)/256),256>>>(
                viscosityFaces.data,viscosityFaces.count,viscosityEntries.data,values,result,diffusionScale);
        } else if(fullPressure&&!wallDirichlet&&massScale==0) {
            if(pressureInterfaces.count)addPressureInterfaces<<<unsigned((pressureInterfaces.count+255)/256),256>>>(
                pressureInterfaces.data,pressureInterfaces.count,pressureEntries.data,values,result,diffusionScale);
        } else if(interfaces.count)addInterfaces<<<unsigned((interfaces.count+255)/256),256>>>(interfaces.data,interfaces.count,values,result,diffusionScale);
        if(gauge>=0)pinResult<<<1,1>>>(result,values,slots[gauge]);
        checkedCuda(cudaGetLastError(),"launch native compact operator");
    }
    std::vector<double> apply(const std::vector<double>& input,double diffusionScale,double massScale,bool wallDirichlet) {
        if(input.size()!=slots.size() || !std::isfinite(diffusionScale) || !std::isfinite(massScale))
            throw std::runtime_error("Invalid native compact apply input");
        std::vector<double> host(size,0.);
        for(size_t c=0;c<input.size();++c) {
            if(!std::isfinite(input[c]))throw std::runtime_error("Nonfinite native compact input");
            host[slots[c]]=input[c];
        }
        x.upload(host);applyDevice(x.data,y.data,diffusionScale,massScale,wallDirichlet);
        checkedCuda(cudaMemcpy(host.data(),y.data,size*sizeof(double),cudaMemcpyDeviceToHost),"download native compact result");
        std::vector<double> output(slots.size());
        for(size_t c=0;c<output.size();++c) {
            output[c]=host[slots[c]];
            if(!std::isfinite(output[c]))throw std::runtime_error("Nonfinite native operator result");
        }
        return output;
    }
    double dot(const double* a,const double* b) {
        products<<<unsigned((slots.size()+255)/256),256>>>(deviceSlots.data,int(slots.size()),a,b,reducer.data());
        reducer.sumAsyncTo(scalar.data);double value=0;
        checkedCuda(cudaMemcpy(&value,scalar.data,sizeof(double),cudaMemcpyDeviceToHost),"read native PCG reduction");
        if(!std::isfinite(value))throw std::runtime_error("Nonfinite native PCG reduction");
        return value;
    }
    double packedDot(const double* a,const double* packed) {
        dotKrylov<<<unsigned((slots.size()+255)/256),256>>>(deviceSlots.data,int(slots.size()),a,packed,reducer.data());
        reducer.sumAsyncTo(scalar.data);double value=0;
        checkedCuda(cudaMemcpy(&value,scalar.data,sizeof(double),cudaMemcpyDeviceToHost),"read native FGMRES reduction");
        if(!std::isfinite(value))throw std::runtime_error("Nonfinite native FGMRES reduction");return value;
    }
    void center(double* field) {
        const int n=int(slots.size());const unsigned blocks=unsigned((n+255)/256);
        sumValues<<<blocks,256>>>(deviceSlots.data,n,field,reducer.data());reducer.sumAsyncTo(scalar.data);
        double sum=0;checkedCuda(cudaMemcpy(&sum,scalar.data,sizeof(double),cudaMemcpyDeviceToHost),"read pressure nullspace component");
        if(!std::isfinite(sum))throw std::runtime_error("Nonfinite pressure nullspace component");
        subtractMean<<<blocks,256>>>(deviceSlots.data,n,field,sum/n);
    }
    void centerSolution() {
        const int n=int(slots.size());const unsigned blocks=unsigned((n+255)/256);
        double mean=0;
        for(const double* field:{x.data,solutionLow.data}) {
            sumValues<<<blocks,256>>>(deviceSlots.data,n,field,reducer.data());reducer.sumAsyncTo(scalar.data);
            double sum=0;checkedCuda(cudaMemcpy(&sum,scalar.data,sizeof(double),cudaMemcpyDeviceToHost),"read twofold pressure mean");
            if(!std::isfinite(sum))throw std::runtime_error("Nonfinite twofold pressure mean");mean+=sum/n;
        }
        subtractMeanTwofold<<<blocks,256>>>(deviceSlots.data,n,x.data,solutionLow.data,mean);
    }
    void applySolution(double k,double m,bool walls,int gauge,bool twofold) {
        applyDevice(x.data,y.data,k,m,walls,gauge);
        if(twofold) {
            applyDevice(solutionLow.data,solutionLowAx.data,k,m,walls,gauge);
            addLowImage<<<unsigned((slots.size()+255)/256),256>>>(deviceSlots.data,int(slots.size()),y.data,solutionLowAx.data);
        }
    }
    bool fgmres(NativeAmgPreconditioner& preconditioner,double k,double m,bool walls,int gauge,
                 double norm2,double tolerance,int maximum,NativeGpuSolveResult& answer,
                 double originalNorm2,double originalTolerance) {
        constexpr int maximumDimension=64;
        const int dimension=krylovDimension,n=int(slots.size());const unsigned blocks=unsigned((n+255)/256);
        if(!basis.data) {basis.allocate(size_t(n)*(dimension+1));images.allocate(size_t(n)*dimension);krylovCoefficients.allocate(dimension);}
        if(batchedCgs2&&!krylovPartials.data)krylovPartials.allocate(size_t(blocks)*dimension);
        const double target=tolerance*std::sqrt(norm2);
        const bool neumann=gauge<0&&!walls&&m==0;
        while(answer.iterations<maximum) {
            if(neumann)center(residual.data);
            const double beta=std::sqrt(dot(residual.data,residual.data));
            if(beta<=target) {answer.relativeResidual=beta/std::sqrt(norm2);return true;}
            packKrylov<<<blocks,256>>>(deviceSlots.data,n,residual.data,basis.data,1/beta);
            double h[maximumDimension+1][maximumDimension]{},cs[maximumDimension]{},sn[maximumDimension]{},g[maximumDimension+1]{};g[0]=beta;int used=0;
            for(int j=0;j<dimension&&answer.iterations<maximum;++j) {
                unpackKrylov<<<blocks,256>>>(deviceSlots.data,n,basis.data+size_t(j)*n,residual.data);
                preconditioner.apply(residual.data,z.data);
                if(neumann)center(z.data);
                packKrylov<<<blocks,256>>>(deviceSlots.data,n,z.data,images.data+size_t(j)*n,1.);
                applyDevice(z.data,y.data,k,m,walls,gauge);
                if(neumann)center(y.data);
                // Reorthogonalization protects tight double tolerances when
                // the float preconditioner produces nearly dependent vectors.
                for(int pass=0;pass<2;++pass) {
                    if(batchedCgs2) {
                        krylovProductsMany<<<dim3(blocks,unsigned(j+1)),256>>>(deviceSlots.data,n,y.data,basis.data,krylovPartials.data);
                        reduceKrylovProducts<<<j+1,256>>>(krylovPartials.data,int(blocks),krylovCoefficients.data);
                        double column[maximumDimension]{};
                        checkedCuda(cudaMemcpy(column,krylovCoefficients.data,size_t(j+1)*sizeof(double),cudaMemcpyDeviceToHost),
                                    "read batched FGMRES inner products");
                        ++answer.orthogonalizationTransfers;
                        for(int i=0;i<=j;++i) {
                            if(!std::isfinite(column[i]))throw std::runtime_error("Nonfinite batched FGMRES reduction");
                            h[i][j]+=column[i];
                        }
                        subtractKrylovMany<<<blocks,256>>>(deviceSlots.data,n,y.data,basis.data,krylovCoefficients.data,j+1);
                    } else for(int i=0;i<=j;++i) {
                        const double value=packedDot(y.data,basis.data+size_t(i)*n);h[i][j]+=value;
                        ++answer.orthogonalizationTransfers;
                        subtractKrylov<<<blocks,256>>>(deviceSlots.data,n,y.data,basis.data+size_t(i)*n,value);
                    }
                }
                const double next=std::sqrt(dot(y.data,y.data));h[j+1][j]=next;
                if(next>0)packKrylov<<<blocks,256>>>(deviceSlots.data,n,y.data,basis.data+size_t(j+1)*n,1/next);
                for(int i=0;i<j;++i) {const double a=h[i][j],b=h[i+1][j];h[i][j]=cs[i]*a+sn[i]*b;h[i+1][j]=-sn[i]*a+cs[i]*b;}
                const double diagonal=std::hypot(h[j][j],h[j+1][j]);
                if(!(diagonal>0))throw std::runtime_error("Native FGMRES Arnoldi breakdown");
                cs[j]=h[j][j]/diagonal;sn[j]=h[j+1][j]/diagonal;h[j][j]=diagonal;h[j+1][j]=0;
                g[j+1]=-sn[j]*g[j];g[j]*=cs[j];used=j+1;++answer.iterations;
                if(std::abs(g[j+1])<=target||next==0)break;
            }
            std::vector<double> coefficients(dimension,0.);
            for(int i=used-1;i>=0;--i) {double value=g[i];for(int j=i+1;j<used;++j)value-=h[i][j]*coefficients[j];coefficients[i]=value/h[i][i];}
            krylovCoefficients.upload(coefficients);
            if(neumann) {
                updateKrylovTwofold<<<blocks,256>>>(deviceSlots.data,n,x.data,solutionLow.data,images.data,krylovCoefficients.data,used);
                centerSolution();
            } else updateKrylov<<<blocks,256>>>(deviceSlots.data,n,x.data,images.data,krylovCoefficients.data,used);
            applySolution(k,m,walls,gauge,neumann);
            residualPcg<<<blocks,256>>>(deviceSlots.data,n,rhs.data,y.data,residual.data);
            answer.relativeResidual=std::sqrt(dot(residual.data,residual.data)/norm2);
            if(neumann) {
                // y is a fresh full stencil Ax, not an Arnoldi residual estimate.
                // Check the unmodified RHS at each restart. The quarter-tolerance
                // safety target can be below the arithmetic floor even after the
                // actual requested equation is solved. Keep the compatible Krylov
                // residual intact if the original equation still needs work.
                residualPackedRhs<<<blocks,256>>>(deviceSlots.data,n,originalRhs.data,y.data,direction.data);
                const double actual=std::sqrt(dot(direction.data,direction.data)/originalNorm2);
                ++answer.originalRhsChecks;
                if(actual<=originalTolerance) {
                    answer.relativeResidual=actual;answer.originalRhsAccepted=true;return true;
                }
            }
            if(answer.relativeResidual<=tolerance)return true;
            ++answer.restarts;
        }
        return false;
    }
    NativeGpuSolveResult solve(const std::vector<double>& b,double k,double m,bool walls,int gauge,double tolerance,int maximum) {
        const auto begin=std::chrono::steady_clock::now();
        if(fullPressure&&(!pressureAmg||!meanZeroPressure))
            throw std::runtime_error("Full pressure interfaces require native AMG/FGMRES and mean-zero pressure");
        if(fullViscosity&&!diffusionAmg)throw std::runtime_error("Full viscosity requires native AMG/FGMRES");
        const bool neumann=meanZeroPressure&&!walls&&m==0;
        if(b.size()!=slots.size()||!(k>0)||!(m>=0)||!std::isfinite(k)||!std::isfinite(m)||
           !(tolerance>0&&tolerance<1)||maximum<1||gauge < -1||gauge>=int(slots.size())||
           (gauge<0&&!walls&&m==0)||(gauge>=0&&!neumann&&b[gauge]!=0))
            throw std::runtime_error("Invalid native PCG system or pressure pin");
        double scale=0;for(double v:b) {if(!std::isfinite(v))throw std::runtime_error("Nonfinite native PCG rhs");scale=std::max(scale,std::abs(v));}
        NativeGpuSolveResult answer;answer.values.assign(b.size(),0.);
        if(scale==0)return answer;
        if(!rhs.data) {
            rhs.allocate(size);residual.allocate(size);z.allocate(size);direction.allocate(size);diagonal.allocate(size);
            scalar.allocate(1);reducer.resize(slots.size());
        }
        std::vector<double> host(size,0.);
        long double total=0,totalError=0,totalVolume=0;
        for(size_t c=0;c<b.size();++c) {
            const long double next=total+b[c];
            totalError+=std::abs(total)>=std::abs(b[c])?(total-next)+b[c]:(b[c]-next)+total;
            total=next;totalVolume+=cellVolumes[c];
        }
        total+=totalError;
        long double correction2=0,source2=0;
        for(size_t c=0;c<b.size();++c) {
            const double correction=neumann?double(total*cellVolumes[c]/totalVolume):0.;
            host[slots[c]]=(b[c]-correction)/scale;correction2+=static_cast<long double>(correction)*correction;source2+=static_cast<long double>(b[c])*b[c];
        }
        answer.compatibilityRelativeL2=std::sqrt(double(correction2/source2));
        if(answer.compatibilityRelativeL2>tolerance) {
            double divergence=0;for(size_t c=0;c<b.size();++c)divergence=std::max(divergence,std::abs(b[c])/cellVolumes[c]);
            std::ostringstream message;message<<"Pressure RHS is incompatible beyond the requested tolerance: compatibility="<<answer.compatibilityRelativeL2<<", divergence="<<divergence<<", rhs_sum="<<double(total);
            throw std::runtime_error(message.str());
        }
        rhs.upload(host);checkedCuda(cudaMemset(x.data,0,size*sizeof(double)),"clear native PCG solution");
        if(neumann) {
            if(!solutionLow.data) {solutionLow.allocate(size);solutionLowAx.allocate(size);}
            checkedCuda(cudaMemset(solutionLow.data,0,size*sizeof(double)),"clear twofold pressure solution");
        }
        checkedCuda(cudaMemset(direction.data,0,size*sizeof(double)),"clear native PCG direction");
        const int n=int(slots.size());const unsigned blocks=unsigned((n+255)/256);
        double originalNorm2=0;
        if(neumann) {
            std::vector<double> packed(b.size());for(size_t c=0;c<b.size();++c)packed[c]=b[c]/scale;
            if(!originalRhs.data)originalRhs.allocate(b.size());
            originalRhs.upload(packed);
            unpackKrylov<<<blocks,256>>>(deviceSlots.data,n,originalRhs.data,direction.data);
            originalNorm2=dot(direction.data,direction.data);
        }
        const int activeGauge=neumann?-1:gauge;
        initializePcg<<<blocks,256>>>(deviceSlots.data,n,rhs.data,degree.data,volumes.data,wallDiagonal.data,x.data,residual.data,diagonal.data,k,m,walls,activeGauge);
        const double norm2=dot(rhs.data,rhs.data),target=tolerance*tolerance*norm2;
        double previous=0;bool restart=true,converged=false;
        if(pressureAmg) {
            if((walls&&(gauge!=-1||k!=amgViscosity||m!=amgMass))||(!walls&&(gauge!=amgGauge||k!=1||m!=0)))
                throw std::runtime_error("Native AMG cached operator parameters differ");
            // Leave room for the roundoff-sized compatibility adjustment and
            // final evaluation against the ORIGINAL Neumann right-hand side.
            // The externally required true-residual tolerance is unchanged.
            converged=fgmres(walls?*diffusionAmg:*pressureAmg,k,m,walls,activeGauge,norm2,neumann?.25*tolerance:tolerance,maximum,answer,originalNorm2,tolerance);
        } else for(int iteration=0;iteration<maximum;++iteration) {
            preconditionPcg<<<blocks,256>>>(deviceSlots.data,n,residual.data,diagonal.data,z.data);
            const double rz=dot(residual.data,z.data);
            if(!(rz>0))throw std::runtime_error("Native PCG nonpositive preconditioned residual");
            directionPcg<<<blocks,256>>>(deviceSlots.data,n,z.data,direction.data,restart?0.:rz/previous);
            applyDevice(direction.data,y.data,k,m,walls,gauge);
            const double curvature=dot(direction.data,y.data);
            if(!(curvature>0))throw std::runtime_error("Native PCG operator lost positive definiteness");
            updatePcg<<<blocks,256>>>(deviceSlots.data,n,x.data,residual.data,direction.data,y.data,rz/curvature);
            previous=rz;restart=false;answer.iterations=iteration+1;
            const double estimated=dot(residual.data,residual.data);
            // Verify the actual residual before acceptance. Restart only when
            // the recurrence reports convergence but the explicit residual
            // disagrees; fixed-period restarts discard long-wavelength modes.
            if(estimated<=target||iteration+1==maximum) {
                applyDevice(x.data,y.data,k,m,walls,gauge);
                residualPcg<<<blocks,256>>>(deviceSlots.data,n,rhs.data,y.data,residual.data);
                const double actual=dot(residual.data,residual.data);
                answer.relativeResidual=std::sqrt(actual/norm2);
                if(actual<=target) {converged=true;break;}
                restart=true;++answer.restarts;
            }
        }
        if(neumann&&!answer.originalRhsAccepted) {
            // Acceptance uses the ORIGINAL compatible face-balance RHS, before
            // the roundoff-sized compatibility correction. No residual waiver.
            // A restart may already have verified exactly this original RHS and
            // a fresh Ax; do not repeat that evaluation when it passed.
            for(size_t c=0;c<b.size();++c)host[slots[c]]=b[c]/scale;
            rhs.upload(host);applySolution(k,m,walls,-1,true);
            residualPcg<<<blocks,256>>>(deviceSlots.data,n,rhs.data,y.data,residual.data);
            answer.relativeResidual=std::sqrt(dot(residual.data,residual.data)/dot(rhs.data,rhs.data));
            converged=answer.relativeResidual<=tolerance;
        }
        if(!converged) {
            std::vector<double> iterate(b.size()),actual(b.size()),low;
            checkedCuda(cudaMemcpy(host.data(),x.data,size*sizeof(double),cudaMemcpyDeviceToHost),"download failed native iterate");
            for(size_t c=0;c<b.size();++c)iterate[c]=host[slots[c]];
            checkedCuda(cudaMemcpy(host.data(),y.data,size*sizeof(double),cudaMemcpyDeviceToHost),"download failed native Ax");
            for(size_t c=0;c<b.size();++c)actual[c]=host[slots[c]];
            if(neumann) {
                low.resize(b.size());checkedCuda(cudaMemcpy(host.data(),solutionLow.data,size*sizeof(double),cudaMemcpyDeviceToHost),"download failed twofold tail");
                for(size_t c=0;c<b.size();++c)low[c]=host[slots[c]];
            }
            std::ostringstream message;message<<"Native GPU "<<(pressureAmg?"FGMRES":"PCG")<<" failed true residual: "<<answer.relativeResidual<<" after "<<answer.iterations<<" iterations; operator="<<(walls?"diffusion":"pressure")<<", tolerance="<<tolerance<<", compatibility="<<answer.compatibilityRelativeL2;
            throw NativeGpuConvergenceError(message.str(),answer,tolerance,walls,scale,std::move(iterate),std::move(actual),std::move(low));
        }
        checkedCuda(cudaMemcpy(host.data(),x.data,size*sizeof(double),cudaMemcpyDeviceToHost),"download native PCG solution");
        if(neumann)answer.lowValues.resize(b.size());
        for(size_t c=0;c<b.size();++c) {
            const double high=host[slots[c]];answer.values[c]=high*scale;
            if(neumann)answer.lowValues[c]=std::fma(high,scale,-answer.values[c]);
            if(!std::isfinite(answer.values[c]))throw std::runtime_error("Nonfinite native PCG solution");
        }
        if(neumann) {
            checkedCuda(cudaMemcpy(host.data(),solutionLow.data,size*sizeof(double),cudaMemcpyDeviceToHost),"download accepted twofold pressure tail");
            for(size_t c=0;c<b.size();++c) {
                answer.lowValues[c]+=host[slots[c]]*scale;
                if(!std::isfinite(answer.lowValues[c]))throw std::runtime_error("Nonfinite pressure solution tail");
            }
        }
        answer.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();return answer;
    }
};
NativeCompactGpu::NativeCompactGpu(const Mesh& mesh,const std::vector<double>& factors):impl_(new Impl(mesh,factors)){}
NativeCompactGpu::~NativeCompactGpu()=default;
void NativeCompactGpu::dumpPressureFaces(const Mesh& mesh,const std::filesystem::path& path) const {
    const auto& p=*impl_;
    if(!p.fullPressure || mesh.nativeStorage!=p.nativeStorage || mesh.cells.size()!=p.slots.size())
        throw std::runtime_error("Pressure face export requires the actual full operator mesh");
    if(std::filesystem::exists(path))throw std::runtime_error("Preserve earlier pressure face export");
    std::vector<double> weights(p.faceWeights.count);
    std::vector<PressureInterface> faces(p.pressureInterfaces.count);
    std::vector<PressureEntry> entries(p.pressureEntries.count);
    checkedCuda(cudaMemcpy(weights.data(),p.faceWeights.data,weights.size()*sizeof(double),cudaMemcpyDeviceToHost),"read back pressure face weights");
    if(!faces.empty())checkedCuda(cudaMemcpy(faces.data(),p.pressureInterfaces.data,faces.size()*sizeof(PressureInterface),cudaMemcpyDeviceToHost),"read back pressure interfaces");
    if(!entries.empty())checkedCuda(cudaMemcpy(entries.data(),p.pressureEntries.data,entries.size()*sizeof(PressureEntry),cudaMemcpyDeviceToHost),"read back pressure entries");
    std::vector<std::int32_t> cells(p.size,-1);
    for(size_t c=0;c<p.slots.size();++c)cells[p.slots[c]]=std::int32_t(c);
    std::uint64_t regular=0;
    for(const auto& face:mesh.faces)regular+=face.neighbor>=0 && mesh.cells[face.owner].level==mesh.cells[face.neighbor].level;
    std::ofstream out(path,std::ios::binary);
    const auto write=[&](auto value){out.write(reinterpret_cast<const char*>(&value),sizeof(value));};
    out.write("CIRRUSP1",8);write(std::uint64_t(mesh.cells.size()));write(regular);
    write(std::uint64_t(faces.size()));write(std::uint64_t(entries.size()));
    for(const auto& face:mesh.faces) {
        if(face.neighbor<0 || mesh.cells[face.owner].level!=mesh.cells[face.neighbor].level)continue;
        write(std::int32_t(face.owner));write(std::int32_t(face.neighbor));
        write(weights[size_t(face.axis)*p.size+p.slots[face.neighbor]]);
    }
    for(const auto& face:faces) {
        write(cells.at(face.owner));write(cells.at(face.neighbor));
        write(std::int32_t(face.begin));write(std::int32_t(face.end));
    }
    for(const auto& entry:entries) {write(cells.at(entry.slot));write(entry.weight);}
    out.flush();if(!out.good())throw std::runtime_error("Failed pressure face export");
}
std::vector<double> NativeCompactGpu::apply(const std::vector<double>& x,double k,double m,bool walls) {
    return impl_->apply(x,k,m,walls);
}
NativeGpuSolveResult NativeCompactGpu::solve(const std::vector<double>& rhs,double k,double m,bool walls,int gauge,double tolerance,int maximum) {
    return impl_->solve(rhs,k,m,walls,gauge,tolerance,maximum);
}
void NativeCompactGpu::configureNativeAmg(const Mesh& mesh,int gauge,double viscosity,double mass) {
    if(impl_->pressureAmg||impl_->diffusionAmg)throw std::runtime_error("Native AMG already configured");
    auto pressure=std::make_unique<NativeAmgPreconditioner>(mesh,impl_->slots,gauge,1.,0.,false,impl_->faceFactors);
    auto diffusion=std::make_unique<NativeAmgPreconditioner>(mesh,impl_->slots,-1,viscosity,mass,true,impl_->faceFactors);
    impl_->pressureAmg=std::move(pressure);impl_->diffusionAmg=std::move(diffusion);
    impl_->amgGauge=gauge;impl_->amgViscosity=viscosity;impl_->amgMass=mass;
}
void NativeCompactGpu::setKrylovDimension(int dimension) {
    if(dimension<2||dimension>64)throw std::runtime_error("Native FGMRES dimension must be between 2 and 64");
    if(!impl_->pressureAmg)throw std::runtime_error("Krylov dimension requires native AMG/FGMRES");
    if(impl_->basis.data)throw std::runtime_error("Set Krylov dimension before the first FGMRES solve");
    impl_->krylovDimension=dimension;
}
int NativeCompactGpu::amgLevels() const {return impl_->pressureAmg?impl_->pressureAmg->levels():0;}
void NativeCompactGpu::useMeanZeroPressure() {
    if(!impl_->pressureAmg)throw std::runtime_error("Mean-zero pressure currently requires native AMG/FGMRES");
    impl_->meanZeroPressure=true;
}
void NativeCompactGpu::configurePressureInterfaces(const Mesh& mesh,const std::vector<NativePressureFaceStencil>& stencils) {
    if(impl_->fullPressure)throw std::runtime_error("Full pressure interfaces already configured");
    if(mesh.nativeStorage!=impl_->nativeStorage||mesh.cells.size()!=impl_->slots.size()||stencils.size()!=impl_->interfaces.count)
        throw std::runtime_error("Full pressure interface topology differs from native compact topology");
    std::vector<unsigned char> seen(mesh.faces.size(),0);
    std::vector<PressureInterface> faces;std::vector<PressureEntry> entries;
    for(const auto& stencil:stencils) {
        if(stencil.face>=mesh.faces.size()||seen[stencil.face]++)throw std::runtime_error("Duplicate or invalid full pressure interface");
        const auto& face=mesh.faces[stencil.face];
        if(face.neighbor<0||mesh.cells[face.owner].level==mesh.cells[face.neighbor].level||stencil.gradient.empty())
            throw std::runtime_error("Expected a coarse/fine pressure gradient stencil");
        if(entries.size()>size_t(std::numeric_limits<int>::max()))throw std::runtime_error("Pressure stencil index overflow");
        const int begin=int(entries.size());double sum=0,norm=0;
        const double area=face.area*(impl_->faceFactors.empty()?1.:impl_->faceFactors[stencil.face]);
        for(const auto& term:stencil.gradient) {
            if(term.first<0||size_t(term.first)>=mesh.cells.size()||!std::isfinite(term.second))
                throw std::runtime_error("Invalid full pressure stencil entry");
            sum+=term.second;norm+=std::abs(term.second);
            if(term.first!=face.owner&&term.second!=0)entries.push_back({impl_->slots[term.first],-area*term.second});
        }
        if(!(norm>0)||std::abs(sum)>1e-11*norm||entries.size()>size_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("Full pressure gradient does not annihilate constants");
        faces.push_back({impl_->slots[face.owner],impl_->slots[face.neighbor],begin,int(entries.size())});
    }
    impl_->pressureInterfaces.allocate(faces.size());impl_->pressureInterfaces.upload(faces);
    impl_->pressureEntries.allocate(entries.size());impl_->pressureEntries.upload(entries);
    impl_->fullPressure=true;
}
size_t NativeCompactGpu::fullPressureInterfaceFaces() const {return impl_->pressureInterfaces.count;}
void NativeCompactGpu::configureViscosityFaces(const Mesh& mesh,const std::vector<NativeViscosityFaceStencil>& stencils) {
    if(impl_->fullViscosity)throw std::runtime_error("Full viscosity already configured");
    if(mesh.nativeStorage!=impl_->nativeStorage||mesh.cells.size()!=impl_->slots.size())
        throw std::runtime_error("Full viscosity topology differs from native compact topology");
    auto custom=[&](const Face& f) {
        const double h=mesh.cells[f.owner].h;
        return f.neighbor<0||mesh.cells[f.owner].level!=mesh.cells[f.neighbor].level||std::abs(f.area-h*h)>1e-12*h*h;
    };
    size_t expected=0;for(const auto& f:mesh.faces)expected+=custom(f);
    if(stencils.size()!=expected)throw std::runtime_error("Full viscosity face coverage differs");
    std::vector<unsigned char> seen(mesh.faces.size(),0),skip(3*impl_->size,0);
    std::vector<PressureInterface> faces;std::vector<PressureEntry> entries;
    for(const auto& stencil:stencils) {
        if(stencil.face>=mesh.faces.size()||seen[stencil.face]++)throw std::runtime_error("Duplicate or invalid full viscosity face");
        const auto& face=mesh.faces[stencil.face];
        if(!custom(face)||stencil.gradient.empty())throw std::runtime_error("Expected a cut, coarse/fine or wall viscosity gradient");
        if(entries.size()>size_t(std::numeric_limits<int>::max()))throw std::runtime_error("Viscosity stencil index overflow");
        const int begin=int(entries.size());double sum=0,norm=0;
        const double area=face.area*(impl_->faceFactors.empty()?1.:impl_->faceFactors[stencil.face]);
        for(const auto& term:stencil.gradient) {
            if(term.first<0||size_t(term.first)>=mesh.cells.size()||!std::isfinite(term.second))
                throw std::runtime_error("Invalid full viscosity stencil entry");
            sum+=term.second;norm+=std::abs(term.second);
            if((face.neighbor<0||term.first!=face.owner)&&term.second!=0)
                entries.push_back({impl_->slots[term.first],-area*term.second});
        }
        if(!(norm>0)||(face.neighbor>=0&&std::abs(sum)>1e-11*norm)||entries.size()>size_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("Invalid full viscosity gradient normalization");
        const int neighbor=face.neighbor<0?-1:impl_->slots[face.neighbor];
        faces.push_back({impl_->slots[face.owner],neighbor,begin,int(entries.size())});
        if(face.neighbor>=0&&mesh.cells[face.owner].level==mesh.cells[face.neighbor].level)
            skip[size_t(face.axis)*impl_->size+neighbor]=1;
    }
    impl_->viscosityFaces.allocate(faces.size());impl_->viscosityFaces.upload(faces);
    impl_->viscosityEntries.allocate(entries.size());impl_->viscosityEntries.upload(entries);
    impl_->viscositySkipFaces.allocate(skip.size());impl_->viscositySkipFaces.upload(skip);
    impl_->fullViscosity=true;
}
size_t NativeCompactGpu::fullViscosityFaces() const {return impl_->viscosityFaces.count;}
void NativeCompactGpu::useBatchedCgs2(bool enabled) {
    if(enabled&&!impl_->pressureAmg)throw std::runtime_error("Batched CGS2 requires native AMG/FGMRES");
    impl_->batchedCgs2=enabled;
}
size_t NativeCompactGpu::allocatedBytes() const {
    return impl_->size*(impl_->rhs.data?13:8)*sizeof(double)+impl_->interfaces.count*sizeof(Interface)+
        impl_->deviceSlots.count*sizeof(int)+impl_->scalar.count*sizeof(double)+
        impl_->originalRhs.count*sizeof(double)+
        (impl_->solutionLow.count+impl_->solutionLowAx.count)*sizeof(double)+
        impl_->pressureInterfaces.count*sizeof(PressureInterface)+impl_->pressureEntries.count*sizeof(PressureEntry)+
        impl_->viscosityFaces.count*sizeof(PressureInterface)+impl_->viscosityEntries.count*sizeof(PressureEntry)+impl_->viscositySkipFaces.count+
        impl_->reducer.d_data.size()*sizeof(double)+impl_->reducer.d_temp_sum.size()+impl_->reducer.d_temp_max.size()+
        (impl_->basis.count+impl_->images.count+impl_->krylovCoefficients.count+impl_->krylovPartials.count)*sizeof(double)+
        (impl_->pressureAmg?impl_->pressureAmg->bytes():0)+(impl_->diffusionAmg?impl_->diffusionAmg->bytes():0);
}
size_t NativeCompactGpu::interfaceFaces() const {return impl_->interfaces.count;}
}
