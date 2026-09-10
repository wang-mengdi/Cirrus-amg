#pragma once
#include "SimpleMesh.h"
#include <memory>
#include <vector>
namespace simple {
// Native float FAS cycles on an auxiliary hierarchy with the SAME physical
// leaves and additional coarser levels. This is an approximate inverse only;
// the outer double operator and true residual define the solved equation.
class NativeAmgPreconditioner {
public:
    NativeAmgPreconditioner(const Mesh&,const std::vector<int>& nativeSlots,
                           int gauge,double diffusion,double mass,bool walls,const std::vector<double>& faceFactors);
    ~NativeAmgPreconditioner();
    void apply(const double* source,double* result);
    int levels() const;
    size_t bytes() const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
}
