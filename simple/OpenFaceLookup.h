#pragma once
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace simple {

// Cartesian face keys retain their original linear indexing and -1 for an
// absent sample. Store only populated z entries within each (axis,x,y) row.
// The visitor must replay the same immutable (linear key, face ID) sequence.
class OpenFaceLookup {
    struct Entry {int z,face;};
    size_t size_=0;
    int columns_=0;
    std::vector<size_t> offsets_;
    std::vector<Entry> entries_;
public:
    OpenFaceLookup()=default;
    template<class Visit>
    OpenFaceLookup(size_t rows,int columns,Visit visit):columns_(columns) {
        if(columns<=0 || rows==std::numeric_limits<size_t>::max() ||
           rows>std::numeric_limits<size_t>::max()/size_t(columns))
            throw std::length_error("Open-face lattice dimensions exceed capacity");
        size_=rows*size_t(columns);offsets_.assign(rows+1,0);
        auto rowOf=[&](size_t key,int face) {
            if(key>=size_ || face<0)throw std::out_of_range("Invalid open-face lookup entry");
            return key/size_t(columns_);
        };
        visit([&](size_t key,int face){++offsets_[rowOf(key,face)+1];});
        for(size_t row=1;row<offsets_.size();++row) {
            if(offsets_[row]>entries_.max_size()-offsets_[row-1])
                throw std::length_error("Open-face lookup exceeds entry capacity");
            offsets_[row]+=offsets_[row-1];
        }
        entries_.resize(offsets_.back());
        {
            auto cursor=offsets_;
            visit([&](size_t key,int face) {
                const size_t row=rowOf(key,face);
                if(cursor[row]>=offsets_[row+1])throw std::runtime_error("Open-face lookup count changed");
                entries_[cursor[row]++]={int(key%size_t(columns_)),face};
            });
            for(size_t row=0;row<rows;++row)
                if(cursor[row]!=offsets_[row+1])throw std::runtime_error("Open-face lookup count changed");
        }
        size_t written=0;
        for(size_t row=0;row<rows;++row) {
            const size_t first=offsets_[row],last=offsets_[row+1],start=written;
            if(last>first)std::stable_sort(entries_.begin()+first,entries_.begin()+last,
                [](const Entry& a,const Entry& b){return a.z<b.z;});
            // Stable ordering preserves the old dense table's last assignment
            // for repeated keys, even when face IDs are not in ascending order.
            for(size_t j=first;j<last;++j) {
                const Entry entry=entries_[j];
                if(written>start && entries_[written-1].z==entry.z)entries_[written-1]=entry;
                else entries_[written++]=entry;
            }
            offsets_[row]=start;
        }
        offsets_.back()=written;entries_.resize(written);
    }
    size_t size() const {return size_;}
    size_t occupied() const {return entries_.size();}
    size_t storageBytes() const {return offsets_.capacity()*sizeof(size_t)+entries_.capacity()*sizeof(Entry);}
    int operator[](size_t key) const {
        if(key>=size_)return -1;
        const size_t row=key/size_t(columns_),first=offsets_[row],last=offsets_[row+1];
        if(first==last)return -1;
        const int z=int(key%size_t(columns_));
        const auto end=entries_.begin()+last;
        const auto it=std::lower_bound(entries_.begin()+first,end,z,
            [](const Entry& entry,int value){return entry.z<value;});
        return it!=end && it->z==z?it->face:-1;
    }
    void swap(OpenFaceLookup& other) noexcept {
        std::swap(size_,other.size_);std::swap(columns_,other.columns_);
        offsets_.swap(other.offsets_);entries_.swap(other.entries_);
    }
};

} // namespace simple
