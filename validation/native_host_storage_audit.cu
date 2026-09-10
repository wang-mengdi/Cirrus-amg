#include "native_host_storage_audit.h"
#include "NativeOctreeAccess.cuh"
#include "NativeCompactGpu.h"
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>

namespace {
#include "native_binary_legacy_reference.inc"
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
template<class F> void rejects(F call,const char* message) {
    bool rejected=false;try{call();}catch(const std::exception&){rejected=true;}
    require(rejected,message);
}
std::vector<uint8_t> bytes(const std::filesystem::path& path) {
    std::ifstream input(path,std::ios::binary);require(bool(input),"Cannot read audit file");
    return {std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>()};
}
struct RejectingBuffer:std::streambuf {
    std::streamsize xsputn(const char*,std::streamsize) override {return 0;}
    int_type overflow(int_type) override {return traits_type::eof();}
};
void serializationCheck(HADeviceGrid<Tile>& grid,uint8_t types,int maximum) {
    const auto reference=legacyBinaryBlob(grid,types,maximum);
    require(reference==grid.dumpBinaryBlob(types,maximum),"Legacy and current vector native dump differ");
    std::ostringstream stream(std::ios::out|std::ios::binary);grid.dumpBinaryStream(stream,types,maximum);
    const auto text=stream.str();
    require(text.size()==reference.size()&&std::memcmp(text.data(),reference.data(),reference.size())==0,
            "Legacy and streaming native dump differ");
}
}

nlohmann::json auditNativeHostStorage(simple::Mesh& mesh,const std::filesystem::path& output) {
    std::filesystem::create_directories(output);auto& grid=simple::nativeDeviceGrid(mesh);
    const auto count=mesh.cells.size();nlohmann::json cases=nlohmann::json::array(),applyCheck;
    for(uint8_t type:{uint8_t(LEAF|GHOST|NONLEAF),uint8_t(LEAF),uint8_t(GHOST),uint8_t(NONLEAF),uint8_t(0)})
        for(int maximum:{-1,0})serializationCheck(grid,type,maximum);
    cases.push_back("legacy_vector_and_stream_exact_ten_selection_cases");
    RejectingBuffer buffer;std::ostream broken(&buffer);
    rejects([&]{grid.dumpBinaryStream(broken);},"Stream write failure was ignored");
    cases.push_back("stream_write_failure_rejected");
    const auto existing=output/"existing";std::filesystem::create_directory(existing);
    std::ofstream(existing/"foreign.txt")<<"preserve me";
    rejects([&]{simple::offloadNativeHostTiles(mesh,existing.string());},"Existing backing directory overwritten");
    require(bytes(existing/"foreign.txt")==std::vector<uint8_t>({'p','r','e','s','e','r','v','e',' ','m','e'}),"Foreign file changed");
    std::ofstream(output/"native_host_storage.json")<<"existing metadata";
    rejects([&]{simple::offloadNativeHostTiles(mesh,(output/"blocked").string());},"Existing storage metadata overwritten");
    require(!std::filesystem::exists(output/"blocked"),"Blocked offload created scratch");
    require(bytes(output/"native_host_storage.json")==std::vector<uint8_t>({'e','x','i','s','t','i','n','g',' ','m','e','t','a','d','a','t','a'}),"Existing metadata changed");
    std::filesystem::remove(output/"native_host_storage.json");
    cases.push_back("existing_directory_and_metadata_preserved");
    std::array<std::vector<simple::Vec3>,2> velocity;
    std::array<std::vector<double>,2> pressure;
    for(int pattern=0;pattern<2;++pattern) {
        velocity[pattern].resize(count);pressure[pattern].resize(count);
        for(std::size_t i=0;i<count;++i) {
            for(int a=0;a<3;++a)velocity[pattern][i][a]=(pattern?-2.:1.)*(std::sin(.071*(i+3*a))+.125*a);
            pressure[pattern][i]=(pattern?-.75:1.25)*std::cos(.027*i);
        }
    }
    std::array<std::vector<uint8_t>,2> full;
    for(int p=0;p<2;++p) {
        simple::exportNativeFields(mesh,velocity[p],pressure[p],"",false,false);
        full[p]=legacyBinaryBlob(grid);
    }
    const auto backing=output/"backing";const auto file=backing/"tiles.bin";
    {
        // Capture references and replay on the SAME live grid: raw Tile bytes
        // include device pointers, so cross-process raw comparisons are invalid.
        simple::NativeCompactGpu gpu(mesh);
        const auto before=gpu.apply(pressure[0],.01,200.,true);
        const auto repeat=gpu.apply(pressure[0],.01,200.,true);
        std::array<std::vector<uint8_t>,2> preserved;
        for(int p=0;p<2;++p) {
            simple::exportNativeFields(mesh,velocity[p],pressure[p],"",true,false);
            preserved[p]=legacyBinaryBlob(grid);
        }
        simple::offloadNativeHostTiles(mesh,backing.string());
        require(legacyBinaryBlob(grid)==preserved[1],"Offload changed device Tile bytes");
        for(int p:{0,1,0}) {
            const auto path=output/("preserved_"+std::to_string(p)+".bin");
            simple::exportNativeFields(mesh,velocity[p],pressure[p],path.string(),true,true);
            require(bytes(path)==preserved[p],"Backed metadata-preserving file differs");
            require(grid.dumpBinaryBlob()==preserved[p],"Backed metadata-preserving device state differs");
        }
        const auto after=gpu.apply(pressure[0],.01,200.,true);
        require(before.size()==after.size()&&repeat.size()==before.size(),"Native apply size changed");
        double scale=0,repeatError=0,exportError=0;
        std::ofstream diagnostic(output/"apply_repeat.csv");diagnostic.precision(17);
        diagnostic<<"id,before,repeat_before_offload,after_export\n";
        for(std::size_t i=0;i<count;++i) {
            scale=std::max(scale,std::abs(before[i]));
            repeatError=std::max(repeatError,std::abs(repeat[i]-before[i]));
            exportError=std::max(exportError,std::abs(after[i]-before[i]));
            diagnostic<<i<<','<<before[i]<<','<<repeat[i]<<','<<after[i]<<'\n';
        }
        diagnostic.flush();require(bool(diagnostic),"Cannot write operator repeat diagnostic");
        require(scale>0&&repeatError/scale<1e-12&&exportError/scale<1e-12,"Native apply changed beyond operator audit tolerance");
        applyCheck={{"repeat_before_offload_scaled_linf",repeatError/scale},{"after_export_scaled_linf",exportError/scale},
            {"repeat_bitwise_identical",std::memcmp(before.data(),repeat.data(),before.size()*sizeof(double))==0},
            {"export_bitwise_identical",std::memcmp(before.data(),after.data(),before.size()*sizeof(double))==0},
            {"tolerance",1e-12},{"note","Coarse/fine contributions use floating-point atomic addition; repeated apply can differ by roundoff even before offload. Raw Tile export checks remain exact."}};
        cases.push_back("live_operator_metadata_bytes_preserved_and_apply_within_cpu_oracle_audit_tolerance");
    }
    for(int p:{0,1,0}) {
        const auto path=output/("full_"+std::to_string(p)+".bin");
        simple::exportNativeFields(mesh,velocity[p],pressure[p],path.string(),false,true);
        require(bytes(path)==full[p],"Backed full Tile file differs");
        require(grid.dumpBinaryBlob()==full[p],"Backed full Tile device state differs");
    }
    cases.push_back("full_tile_repeated_export_matches_original_host_holder_exactly");
    simple::exportNativeFields(mesh,velocity[1],pressure[1],(output/"must_not_exist.bin").string(),false,false);
    require(!std::filesystem::exists(output/"must_not_exist.bin")&&grid.dumpBinaryBlob()==full[1],"No-binary export did not synchronize fields");
    cases.push_back("no_binary_mode_still_synchronizes_device");
    rejects([&]{simple::offloadNativeHostTiles(mesh,(output/"twice").string());},"Second offload accepted");
    rejects([&]{simple::retainNativeCells(mesh,{});},"Post-offload mapping change accepted");
    cases.push_back("duplicate_offload_and_remapping_rejected");
    const auto initialBytes=std::filesystem::file_size(file);
    {std::ofstream extra(file,std::ios::binary|std::ios::app);extra.put('x');}
    rejects([&]{simple::exportNativeFields(mesh,velocity[0],pressure[0],"",true,false);},"Backing length change ignored");
    require(grid.dumpBinaryBlob()==full[1],"Length rejection changed device state");
    std::filesystem::resize_file(file,initialBytes);
    char original=0;
    {std::fstream edit(file,std::ios::in|std::ios::out|std::ios::binary);edit.get(original);edit.seekp(0);edit.put(original^1);}
    rejects([&]{simple::exportNativeFields(mesh,velocity[0],pressure[0],"",true,false);},"Backing corruption ignored");
    require(grid.dumpBinaryBlob()==full[1],"First-Tile corruption rejection changed device state");
    {std::fstream edit(file,std::ios::in|std::ios::out|std::ios::binary);edit.put(original);}
    simple::exportNativeFields(mesh,velocity[0],pressure[0],"",false,false);
    require(grid.dumpBinaryBlob()==full[0],"Restored backing did not reproduce original fields");
    cases.push_back("backing_length_and_first_tile_checksum_failures_rejected_then_exact_recovery");
    nlohmann::json storage;std::ifstream(output/"native_host_storage.json")>>storage;
    require(storage.at("resident_host_tile_bytes")==0&&storage.at("backing_bytes")==initialBytes,
            "Host payload was not released");
    require(storage.at("released_host_tile_capacity_bytes").get<std::uint64_t>()>=initialBytes,"Released capacity below full payload");
    std::ofstream(backing/"foreign.txt")<<"preserve";
    mesh.nativeStorage.reset();
    require(!std::filesystem::exists(file)&&std::filesystem::exists(backing/"foreign.txt"),"Owned scratch cleanup removed foreign data or retained owned Tiles");
    cases.push_back("owned_tile_file_removed_and_foreign_file_preserved_at_destruction");
    return {{"passed",true},{"cells",count},{"cases",cases},{"storage",storage},{"operator_repeat",applyCheck},
        {"scope","Same-live-grid raw bytes against frozen legacy serialization and original host holder; metadata and full exports, failed IO and ownership; not physical convergence"}};
}
