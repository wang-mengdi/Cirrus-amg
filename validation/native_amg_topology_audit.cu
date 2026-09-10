#include "native_amg_topology_audit.h"
#include "NativeOctreeAccess.cuh"
#include "NativeAmgTopology.h"
#include "NativeCompactGpu.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <map>
#include <iomanip>
#include <random>
#include <stdexcept>

nlohmann::json auditNativeAmgTopology(const simple::Mesh& mesh,
    const std::filesystem::path& output,bool completePeriodic) {
    using Key=std::array<int,4>;
    auto& original=simple::nativeDeviceGrid(mesh);
    const int ny=int(std::llround(mesh.extent[1]/original.mH0));int shift=0;
    while((ny>>shift)>8)++shift;
    if((8<<shift)!=ny)throw std::runtime_error("Topology audit requires dyadic native root resolution");
    const int rootX=int(std::llround(mesh.extent[0]/std::ldexp(original.mH0,shift)));
    struct Type {int type=0;};std::map<Key,Type> tiles;
    for(int level=0;level<=original.mMaxLevel;++level)for(int j=0;j<original.hNumTiles[level];++j) {
        const auto& info=original.hTileArrays[level][j];
        tiles[{level+shift,info.mTileCoord[0],info.mTileCoord[1],info.mTileCoord[2]}].type=info.mType;
    }
    const size_t originalCount=tiles.size();
    for(int level=shift;level>0;--level) {
        std::vector<Key> parents;
        for(const auto& item:tiles)if(item.first[0]==level&&item.second.type!=GHOST) {
            const auto& q=item.first;parents.push_back({level-1,q[1]/2,q[2]/2,q[3]/2});
        }
        for(const auto& p:parents)tiles[p].type=NONLEAF;
    }
    const auto oldTiles=tiles;
    std::vector<Key> added;
    if(completePeriodic)added=simple::completeNativeAmgPeriodicGhosts(
        tiles,original.mMaxLevel+shift,rootX/8,LEAF|NONLEAF,LEAF,GHOST);
    for(const auto& p:oldTiles)if(tiles.at(p.first).type!=p.second.type)
        throw std::runtime_error("Periodic ghost completion modified an original tile");
    auto canonical=[&](int level,std::array<int,3> p) {
        const int nx=rootX<<level;p[0]=(p[0]%nx+nx)%nx;return p;
    };
    auto tileKey=[&](int level,const std::array<int,3>& p) {return Key{level,p[0]/8,p[1]/8,p[2]/8};};
    auto type=[&](const Key& key) {auto it=tiles.find(key);return it==tiles.end()?0:it->second.type;};
    for(const auto& c:mesh.cells) {
        const int level=c.level+shift;
        if(type(tileKey(level,canonical(level,c.key)))!=LEAF)throw std::runtime_error("Topology audit lost a physical leaf");
    }
    std::filesystem::create_directories(output);
    std::ofstream tileDump(output/"amg_tiles.csv");tileDump<<"level,x,y,z,type,original\n";
    for(const auto& item:tiles) {
        for(int q:item.first)tileDump<<q<<',';
        tileDump<<item.second.type<<','<<(oldTiles.find(item.first)!=oldTiles.end())<<'\n';
    }
    tileDump.flush();if(!tileDump.good())throw std::runtime_error("AMG tile topology dump failed");
    std::ofstream dump(output/"missing_faces.csv");
    dump<<"face,axis,owner,neighbor,owner_level,owner_x,owner_y,owner_z,neighbor_level,neighbor_x,neighbor_y,neighbor_z,required_level,raw_x,raw_y,raw_z,canonical_x,canonical_y,canonical_z,parent_type,unwrapped_tile_type,periodic,fine_to_coarse,parent_matches_neighbor\n";
    size_t missing=0,coarseFine=0,periodicMissing=0,validCoarseParent=0;
    nlohmann::json samples=nlohmann::json::array();std::map<Key,size_t> absent;
    for(size_t j=0;j<mesh.faces.size();++j) {
        const auto& f=mesh.faces[j];if(f.neighbor<0)continue;
        const auto& p=mesh.cells[f.owner];const auto& n=mesh.cells[f.neighbor];
        coarseFine+=p.level!=n.level;
        int level=n.level+shift;auto key=n.key;
        if(p.level>n.level) {level=p.level+shift;key=p.key;++key[f.axis];}
        const auto mapped=canonical(level,key);const auto required=tileKey(level,mapped);
        if(type(required))continue;
        ++missing;++absent[required];const bool periodic=key!=mapped;periodicMissing+=periodic;
        std::array<int,3> parent=mapped;for(int& q:parent)q/=2;
        const int parentType=type(tileKey(level-1,parent));
        const bool fineToCoarse=p.level==n.level+1;
        const bool matches=fineToCoarse&&parent==n.key;
        validCoarseParent+=matches&&parentType==LEAF;
        dump<<j<<','<<f.axis<<','<<f.owner<<','<<f.neighbor<<','<<p.level;
        for(int q:p.key)dump<<','<<q;dump<<','<<n.level;
        for(int q:n.key)dump<<','<<q;dump<<','<<level;
        for(int q:key)dump<<','<<q;for(int q:mapped)dump<<','<<q;
        dump<<','<<parentType<<','<<type(tileKey(level,key))<<','<<periodic<<','<<fineToCoarse<<','<<matches<<'\n';
        if(samples.size()<32)samples.push_back({{"face",j},{"axis",f.axis},{"owner",f.owner},{"neighbor",f.neighbor},
            {"owner_level",p.level},{"owner_key",p.key},{"neighbor_level",n.level},{"neighbor_key",n.key},
            {"required_level",level},{"raw_required_voxel",key},{"canonical_required_voxel",mapped},
            {"required_tile",required},{"parent_voxel",parent},{"parent_type",parentType},
            {"unwrapped_tile_type",type(tileKey(level,key))},{"periodic_mapping_changed_key",periodic},
            {"fine_to_coarse",fineToCoarse},{"parent_matches_physical_neighbor",matches}});
    }
    dump.flush();if(!dump.good())throw std::runtime_error("Topology audit dump failed");
    if(coarseFine!=size_t(mesh.coarseFineFaces))throw std::runtime_error("Topology audit coarse/fine count differs");
    nlohmann::json absentTiles=nlohmann::json::array();
    for(const auto& item:absent)absentTiles.push_back({{"tile",item.first},{"faces",item.second}});
    return {{"scope","Enumerate every physical face against the existing native AMG tile addressing; no coefficient or flow solve"},
        {"passed",missing==0},{"fluid_cells",mesh.cells.size()},{"faces",mesh.faces.size()},
        {"coarse_fine_faces",coarseFine},{"original_tiles",originalCount},{"amg_tiles_with_ancestors",tiles.size()},
        {"shift",shift},{"missing_faces",missing},{"missing_periodic_faces",periodicMissing},
        {"periodic_ghost_completion_enabled",completePeriodic},{"added_periodic_ghost_tiles",added},{"original_tiles_unchanged",true},
        {"missing_faces_with_valid_coarse_leaf_parent",validCoarseParent},{"missing_tiles",absentTiles},{"samples",samples},
        {"type_values",{{"leaf",LEAF},{"nonleaf",NONLEAF},{"ghost",GHOST}}}};
}

nlohmann::json auditNativeAmgCompactSolve(const simple::Mesh& mesh,
    const std::filesystem::path& output) {
    const size_t n=mesh.cells.size();const double mu=.01,mass=200.;
    std::vector<double> diagonal(n,0.);
    for(const auto& f:mesh.faces)if(f.neighbor>=0) {
        const double w=f.area/f.distance;diagonal[f.owner]+=w;diagonal[f.neighbor]+=w;
    }
    const int gauge=int(std::max_element(diagonal.begin(),diagonal.end())-diagonal.begin());
    std::vector<double>().swap(diagonal);
    simple::NativeCompactGpu gpu(mesh);gpu.configureNativeAmg(mesh,gauge,mu,mass);
    gpu.setKrylovDimension(20);gpu.useBatchedCgs2();gpu.useMeanZeroPressure();
    nlohmann::json setup={{"amg_levels",gpu.amgLevels()},{"allocated_gpu_bytes",gpu.allocatedBytes()},
        {"cells",n},{"faces",mesh.faces.size()},{"gauge",gauge},{"preconditioner_setup_completed",true}};
    std::ofstream(output/"amg_setup.json")<<setup.dump(2)<<'\n';
    // Independent conservative CPU face loop, with no global sparse matrix.
    auto apply=[&](const std::vector<double>& x,bool viscous) {
        std::vector<double> y(n,0.);const double scale=viscous?mu:1.;
        if(viscous)for(size_t c=0;c<n;++c)y[c]=mass*mesh.cells[c].volume*x[c];
        for(const auto& f:mesh.faces) {
            const double w=scale*f.area/f.distance;
            if(f.neighbor<0) {if(viscous)y[f.owner]+=w*x[f.owner];}
            else {const double flux=w*(x[f.owner]-x[f.neighbor]);y[f.owner]+=flux;y[f.neighbor]-=flux;}
        }
        return y;
    };
    std::mt19937 random(271828);std::uniform_real_distribution<double> uniform(-1.,1.);
    std::vector<double> known(n);
    for(size_t c=0;c<n;++c)known[c]=std::sin(6.283185307179586*mesh.cells[c].center[0]/mesh.extent[0])+.1*uniform(random);
    nlohmann::json tests=nlohmann::json::array();
    for(bool viscous:{false,true}) {
        auto exact=known;
        if(!viscous) {long double sum=0;for(double x:exact)sum+=x;const double mean=double(sum/n);for(double& x:exact)x-=mean;}
        const auto rhs=apply(exact,viscous);
        const auto image=gpu.apply(exact,viscous?mu:1.,viscous?mass:0.,viscous);
        long double rhs2=0,imageError2=0;
        for(size_t c=0;c<n;++c) {rhs2+=(long double)rhs[c]*rhs[c];const double e=image[c]-rhs[c];imageError2+=(long double)e*e;}
        const double imageError=std::sqrt(double(imageError2/rhs2));
        if(!(imageError<1e-12))throw std::runtime_error("Compact AMG audit CPU/GPU face actions differ");
        const auto solution=gpu.solve(rhs,viscous?mu:1.,viscous?mass:0.,viscous,viscous?-1:gauge,1e-13);
        if(!viscous&&solution.lowValues.size()!=n)throw std::runtime_error("Compact AMG pressure audit lost the twofold tail");
        const auto highImage=apply(solution.values,viscous);
        const auto lowImage=solution.lowValues.empty()?std::vector<double>(n,0.):apply(solution.lowValues,viscous);
        long double residual2=0,error2=0,exact2=0;
        const auto path=output/(viscous?"compact_diffusion.csv":"compact_pressure.csv");
        std::ofstream dump(path);dump<<std::setprecision(17)<<"id,known,rhs,solution,solution_low,cpu_residual\n";
        for(size_t c=0;c<n;++c) {
            const double low=solution.lowValues.empty()?0.:solution.lowValues[c];
            const double residual=rhs[c]-(highImage[c]+lowImage[c]);
            const double error=(solution.values[c]-exact[c])+low;
            residual2+=(long double)residual*residual;error2+=(long double)error*error;exact2+=(long double)exact[c]*exact[c];
            dump<<c<<','<<exact[c]<<','<<rhs[c]<<','<<solution.values[c]<<','<<low<<','<<residual<<'\n';
        }
        dump.flush();if(!dump.good())throw std::runtime_error("Compact AMG solve dump failed");
        const double residual=std::sqrt(double(residual2/rhs2)),error=std::sqrt(double(error2/exact2));
        const bool pass=residual<1e-12&&error<1e-8&&solution.relativeResidual<=1e-13&&(viscous||solution.originalRhsAccepted);
        tests.push_back({{"operator",viscous?"implicit_diffusion":"pressure"},{"passed",pass},
            {"cpu_gpu_apply_relative_l2",imageError},{"cpu_face_relative_residual",residual},
            {"known_solution_relative_l2",error},{"gpu_true_relative_residual",solution.relativeResidual},
            {"original_rhs_accepted",solution.originalRhsAccepted},{"iterations",solution.iterations},{"seconds",solution.seconds}});
        std::ofstream(output/"compact_checks.json")<<nlohmann::json({{"passed",false},{"completed_operators",tests},{"setup",setup}}).dump(2)<<'\n';
        if(!pass)throw std::runtime_error("Compact AMG solve failed original manufactured-solution gates");
    }
    return {{"scope","Actual native AMG compact pressure/diffusion solves against an independent all-face CPU action; not full face reconstruction or physical flow"},
        {"passed",true},{"completed_operators",tests},{"setup",setup}};
}
