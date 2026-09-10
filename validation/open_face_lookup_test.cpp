#include "OpenFaceLookup.h"
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {
void require(bool ok,const std::string& message) {if(!ok)throw std::runtime_error(message);}
template<class Visit>
void compare(size_t rows,int columns,Visit visit,const std::string& name) {
    std::vector<int> dense(rows*size_t(columns),-1);
    visit([&](size_t key,int face){dense.at(key)=face;});
    simple::OpenFaceLookup compact(rows,columns,visit);
    require(compact.size()==dense.size(),"Logical lookup size differs");
    size_t occupied=0;
    for(size_t key=0;key<dense.size();++key) {
        if(compact[key]!=dense[key])throw std::runtime_error("Lookup differs at "+std::to_string(key));
        occupied+=dense[key]>=0;
    }
    require(compact.occupied()==occupied,"Occupied lookup size differs");
    require(compact[dense.size()]==-1 && compact[std::numeric_limits<size_t>::max()]==-1,"Missing sentinel differs");
    std::cout<<"{\"case\":\""<<name<<"\",\"passed\":true,\"slots_checked\":"<<dense.size()
             <<",\"occupied\":"<<occupied<<",\"dense_bytes\":"<<dense.capacity()*sizeof(int)
             <<",\"compact_storage_bytes\":"<<compact.storageBytes()<<"}\n";
    simple::OpenFaceLookup().swap(compact);
    require(compact.size()==0 && compact.occupied()==0 && compact[0]==-1,"Lookup release failed");
}
void synthetic() {
    using Event=std::pair<size_t,int>;
    auto check=[](size_t rows,int columns,std::vector<Event> events,const std::string& name) {
        compare(rows,columns,[&](auto add){for(auto e:events)add(e.first,e.second);},name);
    };
    check(0,7,{},"zero_rows");check(8,7,{},"all_absent");
    check(8,7,{{55,9},{8,20},{2,8},{7,11},{0,0},{8,3},{7,4},{55,1},{2,90},{8,1}},"unsorted_duplicate_last_assignment");
    check(6,1,{{5,0},{0,2},{3,7},{3,1}},"single_column");
    size_t rejected=0;
    auto rejects=[&](auto run){bool caught=false;try {run();}catch(const std::exception&){caught=true;}require(caught,"Invalid lookup accepted");++rejected;};
    rejects([]{simple::OpenFaceLookup t(2,0,[](auto){});});
    rejects([]{simple::OpenFaceLookup t(std::numeric_limits<size_t>::max(),2,[](auto){});});
    rejects([]{simple::OpenFaceLookup t(2,3,[](auto add){add(6,0);});});
    rejects([]{simple::OpenFaceLookup t(2,3,[](auto add){add(0,-1);});});
    rejects([]{int call=0;simple::OpenFaceLookup t(2,3,[&](auto add){if(call++==0)add(0,1);});});
    rejects([]{int call=0;simple::OpenFaceLookup t(2,3,[&](auto add){add(0,1);if(call++==1)add(1,2);});});
    std::cout<<"{\"guard_rejections\":"<<rejected<<",\"passed\":true}\n";
}
void fixture(const std::filesystem::path& path) {
    std::ifstream input(path,std::ios::binary);require(bool(input),"Cannot open lookup fixture");
    char magic[8];std::uint64_t rows,columns,count;
    input.read(magic,8);input.read(reinterpret_cast<char*>(&rows),8);
    input.read(reinterpret_cast<char*>(&columns),8);input.read(reinterpret_cast<char*>(&count),8);
    require(bool(input) && std::memcmp(magic,"CIRRFACE",8)==0,"Invalid lookup fixture header");
    require(columns>0 && columns<=std::uint64_t(std::numeric_limits<int>::max()) &&
            rows<=std::numeric_limits<size_t>::max()/columns && count<=(std::numeric_limits<size_t>::max()-32)/12,"Lookup fixture capacity exceeded");
    require(std::filesystem::file_size(path)==32+12*count,"Lookup fixture file size differs");
    std::vector<unsigned char> records(size_t(12*count));
    input.read(reinterpret_cast<char*>(records.data()),std::streamsize(records.size()));require(bool(input),"Truncated lookup fixture");
    compare(size_t(rows),int(columns),[&](auto add){
        for(size_t i=0;i<size_t(count);++i) {
            std::uint64_t key;std::int32_t face;
            std::memcpy(&key,records.data()+12*i,8);std::memcpy(&face,records.data()+12*i+8,4);
            require(key<rows*columns && face>=0,"Invalid fixture event");add(size_t(key),int(face));
        }
    },path.filename().string());
}
} // namespace

int main(int argc,char** argv) {
    try {
        const std::uint16_t endian=1;require(*reinterpret_cast<const unsigned char*>(&endian)==1,"Fixture requires little endian");
        synthetic();require(argc>1,"Expected an actual mesh fixture");
        for(int i=1;i<argc;++i)fixture(argv[i]);
        std::cout<<"PASS: every dense lookup slot and duplicate assignment matched\n";
    }catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
