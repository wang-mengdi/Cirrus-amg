// Read-only capture and exact scalar conversion of ORIGINAL Embed geometry.
// The isolated Embed header grants friendship; its layout and algorithms do not change.
#pragma once
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>

struct TwistedGeometryState {
  struct Stream {
    std::ifstream input;
    std::ofstream output;
    bool load;
    Stream(const std::string& path,bool reading):load(reading) {
      if(load)input.open(path,std::ios::binary);
      else {
        std::ifstream previous(path,std::ios::binary);
        if(previous.good())throw std::runtime_error("Preserve previous geometry snapshot");
        output.open(path,std::ios::binary);
      }
    }
    template<class T> void Raw(T& v) {
      if(load)input.read(reinterpret_cast<char*>(&v),sizeof(v));
      else output.write(reinterpret_cast<const char*>(&v),sizeof(v));
      if(load?!input.good():!output.good())throw std::runtime_error("Geometry snapshot I/O failed");
    }
    template<class T> void Value(T& v) {
      if constexpr(std::is_enum<T>::value) {
        int32_t x=load?0:int32_t(v);Raw(x);if(load)v=T(x);
      } else if constexpr(std::is_floating_point<T>::value) {
        double x=load?0:double(v);Raw(x);if(load)v=T(x);
      } else {
        for(size_t d=0;d<T::dim;++d)Value(v[d]);
      }
    }
    template<class T> void Value(std::vector<T>& v) {
      uint64_t n=v.size();Raw(n);if(n>64)throw std::runtime_error("Invalid snapshot polygon size");
      if(load)v.resize(size_t(n));for(auto& x:v)Value(x);
    }
    template<class T,class Idx> void Field(GField<T,Idx>& field,const std::string& name) {
      uint64_t length=name.size();Raw(length);if(length>100)throw std::runtime_error("Invalid snapshot label");
      std::string label=load?std::string(size_t(length),'\0'):name;
      for(auto& ch:label)Raw(ch);
      if(label!=name)throw std::runtime_error("Geometry snapshot field order differs");
      int64_t begin=field.GetRange().begin().operator*().raw(),end=field.GetRange().end().operator*().raw();
      int32_t halo=field.GetHalo();Raw(begin);Raw(end);Raw(halo);
      if(begin<0||end<begin||end>100000000||halo<0||halo>GField<T,Idx>::kMaxHalo)
        throw std::runtime_error("Invalid geometry field range");
      if(load) {field.Reinit(GRange<Idx>(Idx(begin),Idx(end)));field.SetHalo(halo);}
      // Include raw index storage as well as all valid halo slots.
      for(size_t i=0;i<field.size();++i)Value(field.data()[i]);
    }
  };
  template<class M> static void Transfer(Embed<M>& eb,const std::string& path,bool load) {
    Stream io(path,load);uint64_t magic=0x3154534f45475041ULL;io.Raw(magic);
    if(magic!=0x3154534f45475041ULL)throw std::runtime_error("Invalid geometry snapshot signature");
    for(size_t d=0;d<M::dim;++d) {
      int64_t count=eb.m.GetGlobalSize()[d];double h=double(eb.m.GetCellSize()[d]);io.Raw(count);io.Raw(h);
      if(count!=eb.m.GetGlobalSize()[d]||h!=double(eb.m.GetCellSize()[d]))throw std::runtime_error("Geometry snapshot mesh differs");
    }
    io.Field(eb.fnl_,"levelset");io.Field(eb.fft_,"face_type");io.Field(eb.ffpoly_,"face_polygon");
    io.Field(eb.ffs_,"face_area");io.Field(eb.ff_face_center_,"face_center");
    io.Field(eb.fct_,"cell_type");io.Field(eb.fcn_,"wall_normal");io.Field(eb.fca_,"wall_plane");
    io.Field(eb.fcs_,"wall_area");io.Field(eb.fcv_,"cell_volume");io.Field(eb.fc_face_center_,"wall_center");
    io.Field(eb.fc_cell_center_,"cell_center");io.Field(eb.fc_sdf_,"wall_distance");
    io.Field(eb.fc_cutdx_,"wall_displacement");io.Field(eb.fcvst3_,"stencil_volume");
    if(load&&io.input.peek()!=std::ifstream::traits_type::eof())throw std::runtime_error("Trailing geometry snapshot bytes");
    if(!load) {io.output.flush();if(!io.output.good())throw std::runtime_error("Geometry snapshot flush failed");}
  }
};
