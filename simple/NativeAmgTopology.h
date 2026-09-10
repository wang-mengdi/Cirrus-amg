#pragma once
#include <limits>
#include <stdexcept>
#include <vector>

namespace simple {
// Match native face-neighbor ghost construction across the x-periodic seam.
// Existing physical tiles and their type/coefficient storage are untouched.
// A ghost is a fine-level representation of an existing physical coarse leaf.
template<class TileMap>
std::vector<typename TileMap::key_type> completeNativeAmgPeriodicGhosts(
    TileMap& tiles,int maximum,int rootTilesX,int activeMask,int leafType,int ghostType) {
    using Key=typename TileMap::key_type;
    if(maximum<0||maximum>=30||rootTilesX<=0||rootTilesX>(std::numeric_limits<int>::max()>>maximum))
        throw std::runtime_error("Invalid native AMG periodic tile extent");
    std::vector<Key> added;
    for(int level=maximum;level>0;--level) {
        const int nx=rootTilesX<<level;
        std::vector<Key> pending;
        for(const auto& item:tiles) {
            const auto& key=item.first;
            if(key[0]!=level||!(item.second.type&activeMask))continue;
            if(key[1]!=0&&key[1]!=nx-1)continue;
            Key other=key;other[1]=key[1]==0?nx-1:0;
            if(tiles.find(other)!=tiles.end())continue;
            const Key parent{level-1,other[1]/2,other[2]/2,other[3]/2};
            const auto p=tiles.find(parent);
            if(p==tiles.end()||p->second.type!=leafType)
                throw std::runtime_error("Missing native AMG periodic neighbor is not backed by a coarse leaf");
            pending.push_back(other);
        }
        for(const auto& key:pending) {
            auto inserted=tiles.try_emplace(key);
            if(inserted.second) {inserted.first->second.type=ghostType;added.push_back(key);}
        }
    }
    return added;
}
} // namespace simple
