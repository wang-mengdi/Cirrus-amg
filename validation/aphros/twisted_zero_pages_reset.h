// Optional storage-only optimization for an isolated Windows reference build.
// Numerical operators, logical field sizes, scalar types, and indices are intact.
#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace twisted_zero_pages {

template<class T> struct Supported : std::is_arithmetic<T> {};
template<class S,size_t D>
struct Supported<generic::Vect<S,D>> : std::is_arithmetic<S> {};

template<class T> bool PositiveZero(const T& value) {
  if constexpr(std::is_floating_point<T>::value)
    return value==T(0) && !std::signbit(value);
  else if constexpr(std::is_integral<T>::value)return value==T(0);
  else return false;
}
template<class S,size_t D> bool PositiveZero(const generic::Vect<S,D>& value) {
  for(size_t i=0;i<D;++i)if(!PositiveZero(value[i]))return false;
  return true;
}

template<class T> T* Allocate(size_t count,bool& virtual_allocation,std::atomic<bool>& pristine_zero) {
  virtual_allocation=false;pristine_zero.store(false,std::memory_order_relaxed);
  if(count>std::numeric_limits<size_t>::max()/sizeof(T))throw std::bad_array_new_length();
#ifdef _WIN32
  if constexpr(Supported<T>::value && std::is_trivially_copyable<T>::value &&
      std::is_trivially_default_constructible<T>::value &&
      std::is_trivially_destructible<T>::value && alignof(T)<=65536) {
    if(count*sizeof(T)>=65536 && std::getenv("APHROS_TWISTED_ZERO_PAGES")) {
      auto data=static_cast<T*>(VirtualAlloc(nullptr,count*sizeof(T),
          MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE));
      if(!data)throw std::bad_alloc();
      // Establish T object lifetimes with default initialization. These types
      // have trivial default constructors, which do not overwrite zero pages.
      std::uninitialized_default_construct_n(data,count);
      virtual_allocation=true;pristine_zero.store(true,std::memory_order_relaxed);
      return data;
    }
  }
#endif
  return new T[count];
}

// Reset only an owned VirtualAlloc region while retaining its reservation.
// Call between worker regions, with no concurrent readers/writers, just as for
// the original full-field fill. All prior pointers keep the same address.
template<class T> bool Reset(T* data,size_t count,bool virtual_allocation,
                            std::atomic<bool>& pristine_zero) {
#ifdef _WIN32
  if constexpr(Supported<T>::value && std::is_trivially_copyable<T>::value &&
      std::is_trivially_default_constructible<T>::value &&
      std::is_trivially_destructible<T>::value) {
    if(virtual_allocation) {
      // Never restore pristine=true: an existing mutable alias may write later
      // without another data()/operator[] call to invalidate that flag.
      pristine_zero.store(false,std::memory_order_relaxed);
      const size_t bytes=count*sizeof(T);
      if(!VirtualFree(data,bytes,MEM_DECOMMIT))throw std::bad_alloc();
      if(VirtualAlloc(data,bytes,MEM_COMMIT,PAGE_READWRITE)!=data)throw std::bad_alloc();
      std::uninitialized_default_construct_n(data,count);
      return true;
    }
  }
#else
  (void)data;(void)count;(void)virtual_allocation;(void)pristine_zero;
#endif
  return false;
}

template<class T> void Release(T* data,bool virtual_allocation) noexcept {
#ifdef _WIN32
  if(virtual_allocation) {
    if(!VirtualFree(data,0,MEM_RELEASE))std::terminate();
    return;
  }
#else
  (void)virtual_allocation;
#endif
  delete[] data;
}

// Destination is a fresh demand-zero allocation of trivial objects. Copy every
// nonzero byte page; leave all-zero pages backed by the OS zero-page mechanism.
// This also preserves negative zero, NaN payloads, and long-double padding.
template<class T> void Copy(const T* source,T* destination,size_t count,
                            bool virtual_allocation,std::atomic<bool>& pristine_zero) {
  if constexpr(std::is_trivially_copyable<T>::value) {
    if(virtual_allocation) {
      constexpr size_t chunk=4096;
      static constexpr std::array<unsigned char,chunk> zero{};
      const auto* src=reinterpret_cast<const unsigned char*>(source);
      auto* dst=reinterpret_cast<unsigned char*>(destination);
      const size_t bytes=count*sizeof(T);
      bool all_zero=true;
      for(size_t offset=0;offset<bytes;offset+=chunk) {
        const size_t length=std::min(chunk,bytes-offset);
        if(std::memcmp(src+offset,zero.data(),length)!=0) {
          std::memcpy(dst+offset,src+offset,length);all_zero=false;
        }
      }
      pristine_zero.store(all_zero,std::memory_order_relaxed);return;
    }
  }
  std::copy(source,source+count,destination);pristine_zero.store(false,std::memory_order_relaxed);
}

} // namespace twisted_zero_pages
