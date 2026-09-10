#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "geom/field.h"
#include <psapi.h>
#include <omp.h>

void Require(bool value,const char* message) {
  if(!value)throw std::runtime_error(message);
}

struct Tracked {
  static int alive;
  int value=19;
  Tracked(){++alive;}
  Tracked(const Tracked& x):value(x.value){++alive;}
  ~Tracked(){--alive;}
};
int Tracked::alive=0;

size_t WorkingSet() {
  PROCESS_MEMORY_COUNTERS_EX m{};m.cb=sizeof(m);
  Require(GetProcessMemoryInfo(GetCurrentProcess(),
      reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&m),sizeof(m))!=0,"memory query");
  return m.WorkingSetSize;
}

int main() {
  const bool enabled=std::getenv("APHROS_TWISTED_ZERO_PAGES")!=nullptr;
  using S=long double;
  using V=generic::Vect<S,3>;
  static_assert(std::is_trivially_default_constructible<V>::value,"vector default constructor");
  static_assert(std::is_trivially_copyable<V>::value,"vector copy representation");
  constexpr size_t n=32771; // Deliberately includes a partial final page.
  {
    Vector<S> source(n,S(0));
    Require(source.virtual_allocation()==enabled,"arithmetic allocation mode");
    Require(source.pristine_zero()==enabled,"fresh zero state");
    source[0]=S(-0.0L);source[255]=1.25L;source[256]=-2.5L;
    source[n/2]=std::numeric_limits<S>::quiet_NaN();source[n-1]=9.75L;
    Vector<S> copied(source);
    const auto& a=source;const auto& b=copied;
    Require(std::memcmp(a.data(),b.data(),n*sizeof(S))==0,"copy all representations including padding");
    Require(std::signbit(b[0]) && std::isnan(b[n/2]),"negative zero and NaN");
    source[255]=4;
    Require(b[255]==1.25L,"independent copy");
    Vector<S> moved(std::move(copied));
    Require(copied.size()==0 && moved[n-1]==9.75L,"move construction");
    Vector<S> assigned(1,S(7));assigned=std::move(moved);
    Require(moved.size()==0 && assigned[n-1]==9.75L,"move assignment");
    assigned=assigned;Require(assigned[n-1]==9.75L,"self copy assignment");
    assigned.resize(n+3);Require(assigned.size()==n+3,"resize");
  }
  {
    Vector<V> vectors(n,V(S(0)));
    Require(vectors.virtual_allocation()==enabled,"vector-valued allocation");
    vectors[255]=V(1,-2,3);vectors[n-1]=V(-0.0L,4,5);
    Vector<V> copied=vectors;
    const auto& a=vectors;const auto& b=copied;
    Require(std::memcmp(a.data(),b.data(),n*sizeof(V))==0,"vector-valued copy");
    Require(std::signbit(b[n-1][0]),"vector negative zero");
    Vector<S> negative(n,-0.0L);
    Require(!negative.pristine_zero() && std::signbit(static_cast<const Vector<S>&>(negative)[n-1]),"negative zero fill is not skipped");
  }
  {
    using F=FieldCell<S>;const F::Range range{IdxCell(n)};
    F f;f.Reinit(range,S(0));
    Require(f[IdxCell(0)]==0 && f[IdxCell(n-1)]==0,"fresh Reinit zero");
    auto* alias=f.data();alias[0]=42;f.Reinit(range,S(0));
    Require(alias==f.data() && alias[0]==0,"same-range Reinit preserves aliases and clears dirty values");
    f.Reinit(range,S(-0.0L));
    Require(std::signbit(f[IdxCell(n-1)]),"field signed-zero reset");
    std::vector<S> storage(n,S(5));
    {F external(storage.data(),range);external.Reinit(range,S(0));external[IdxCell(n-1)]=11;}
    Require(storage[0]==0 && storage[n-1]==11,"external storage lifetime and writes");
  }
  {
    Vector<S> parallel(n,S(0));
    omp_set_num_threads(4);
#pragma omp parallel for
    for(int i=0;i<int(n);++i)parallel[size_t(i)]=S(i%91)-45;
    Vector<S> copy(parallel);const auto& read=copy;
    for(size_t i=0;i<n;++i)Require(read[i]==S(i%91)-45,"parallel disjoint writes and subsequent copy");
  }
  {
    Vector<Tracked> objects(n);
    Require(!objects.virtual_allocation() && Tracked::alive==int(n),"nontrivial constructors retained");
    {Vector<Tracked> copied(objects);Require(Tracked::alive==2*int(n),"nontrivial copy lifetime");}
    Require(Tracked::alive==int(n),"nontrivial destruction");
  }
  Require(Tracked::alive==0,"all nontrivial instances destroyed");
  const size_t before=WorkingSet();
  Vector<S> large(4*1024*1024,S(0));
  const size_t allocated=WorkingSet();
  Vector<S> duplicate(large);
  const size_t copied=WorkingSet();
  const auto& read=duplicate;
  Require(read[0]==0 && read[read.size()-1]==0,"large zero copy endpoints");
  std::cout<<"{\"passed\":true,\"enabled\":"<<(enabled?"true":"false")
    <<",\"large_buffer_bytes\":"<<large.size()*sizeof(S)
    <<",\"working_set_before\":"<<before<<",\"working_set_after_zero_allocation\":"<<allocated
    <<",\"working_set_after_zero_copy\":"<<copied<<"}\n";
}
