// Release construction-only storage after the fixed validation driver has
// dumped geometry and Proj has copied its initial velocity. No operator changes.
#pragma once
#include <fstream>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

struct TwistedColdStorage {
  struct Memory {size_t working=0,private_bytes=0;};
  static Memory Observe() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX c{};c.cb=sizeof(c);
    if(!GetProcessMemoryInfo(GetCurrentProcess(),
        reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&c),sizeof(c)))
      throw std::runtime_error("Cold-storage memory query failed");
    return {c.WorkingSetSize,c.PrivateUsage};
#else
    return {};
#endif
  }
  template<class T,class I> static size_t DynamicBytes(const GField<T,I>&) {return 0;}
  template<class T,class I> static size_t DynamicBytes(const GField<std::vector<T>,I>& field) {
    size_t bytes=0;
    for(size_t i=0;i<field.size();++i)bytes+=field.data()[i].capacity()*sizeof(T);
    return bytes;
  }
  template<class T,class I> static void Drop(GField<T,I>& field,const char* name,std::ostream& out) {
    if(!field.owning())throw std::runtime_error("Cold-storage release requires an owned field");
    const size_t bytes=field.size()*sizeof(T),dynamic=DynamicBytes(field);
    const auto before=Observe();
    GField<T,I>().swap(field);
    const auto after=Observe();
    if(!field.empty())throw std::runtime_error("Cold field was not released");
    out<<name<<','<<bytes<<','<<dynamic<<','<<before.working<<','<<after.working<<','
       <<before.private_bytes<<','<<after.private_bytes<<'\n';
  }
  template<class M> static void Release(Embed<M>& eb,
      FieldNode<typename M::Scal>& initial_levelset,
      FieldCell<typename M::Vect>& initial_velocity) {
    std::ofstream out("cold_storage.csv");
    out<<"field,logical_bytes,dynamic_capacity_bytes,working_set_before,working_set_after,private_before,private_after\n";
    Drop(initial_levelset,"driver_initial_levelset",out);
    Drop(initial_velocity,"driver_initial_velocity",out);
    // These are construction/visualization or optional-distance caches. The
    // fixed Proj/BCG/diffusion path uses the retained areas, normals, cell
    // volumes, face centers, wall planes and stencil volumes. Guarded getters
    // in the isolated header reject any later use of a released cache.
    Drop(eb.fnl_,"geometry_levelset",out);
    Drop(eb.ffpoly_,"geometry_face_polygons",out);
    Drop(eb.fc_cell_center_,"geometry_cell_centroids",out);
    Drop(eb.fc_sdf_,"geometry_signed_distance",out);
    Drop(eb.fc_cutdx_,"geometry_cut_displacement",out);
    out.flush();if(!out.good())throw std::runtime_error("Cold-storage record write failed");
  }
};
