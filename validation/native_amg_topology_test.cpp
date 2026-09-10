#include "NativeAmgTopology.h"
#include <array>
#include <iostream>
#include <map>

using Key=std::array<int,4>;
struct Tile {int type=0;double sentinel=123.25;};
using Map=std::map<Key,Tile>;
void require(bool x){if(!x)throw std::runtime_error("Periodic AMG topology test failed");}
auto complete(Map& m,int level=1,int nx=2){return simple::completeNativeAmgPeriodicGhosts(m,level,nx,3,1,4);}
int main() {
    try {
        int cases=0;
        for(bool reverse:{false,true}) {
            Map m;
            m[{0,0,0,0}].type=reverse?1:2;m[{0,1,0,0}].type=reverse?2:1;
            const Key active{1,reverse?3:0,0,0},ghost{1,reverse?0:3,0,0};
            m[active].type=1;const Map original=m;const auto added=complete(m);
            require(added.size()==1&&added[0]==ghost&&m.at(ghost).type==4);
            for(const auto& x:original)require(m.at(x.first).type==x.second.type&&m.at(x.first).sentinel==x.second.sentinel);
            require(complete(m).empty());++cases;
        }
        {
            Map m;for(int x=0;x<2;++x)m[{0,x,0,0}].type=2;
            for(int x=0;x<4;++x)m[{1,x,0,0}].type=1;
            m[{1,-1,0,0}].type=4;m[{1,4,0,0}].type=4;
            const Map old=m;require(complete(m).empty()&&m.size()==old.size());++cases;
        }
        {
            Map m;for(int x=0;x<2;++x)m[{0,x,0,0}].type=2;
            for(int x=0;x<4;++x)m[{1,x,0,0}].type=x==3?2:1;
            m[{2,7,0,0}].type=1;
            auto added=complete(m,2);require(added.size()==1&&added[0]==Key{2,0,0,0});++cases;
        }
        {
            Map m;m[{1,1,0,0}].type=1;require(complete(m).empty());++cases;
        }
        for(int type:{0,2,4}) {
            Map m;m[{1,0,0,0}].type=1;
            if(type)m[{0,1,0,0}].type=type;
            bool rejected=false;try{complete(m);}catch(const std::runtime_error&){rejected=true;}
            require(rejected);++cases;
        }
        for(auto invalid:std::array<std::array<int,2>,4>{{{-1,2},{30,2},{1,0},{29,4}}}) {
            Map m;bool rejected=false;try{complete(m,invalid[0],invalid[1]);}catch(const std::runtime_error&){rejected=true;}
            require(rejected);++cases;
        }
        std::cout<<"{\"passed\":true,\"cases\":"<<cases<<",\"scope\":\"Both periodic orientations, existing tiles, multilevel closure, idempotence and invalid coarse-parent/extent rejection\"}\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
