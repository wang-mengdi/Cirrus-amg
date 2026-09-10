#pragma once
#include "SimpleMesh.h"
#include <memory>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simple {
struct NativePressureFaceStencil {
    size_t face;
    std::vector<std::pair<int,double>> gradient;
};
struct NativeViscosityFaceStencil {
    size_t face;
    std::vector<std::pair<int,double>> gradient;
};
struct NativeGpuSolveResult {
    std::vector<double> values;
    // Mean-zero pressure retains sub-ulp updates as a twofold solution. Its
    // mathematical value is values + lowValues; consumers must retain both.
    std::vector<double> lowValues;
    int iterations=0,restarts=0;
    double relativeResidual=0,seconds=0;
    double compatibilityRelativeL2=0;
    int originalRhsChecks=0;
    int orthogonalizationTransfers=0;
    bool originalRhsAccepted=false;
};
// A failed numerical solve is distinguishable from CUDA, input and I/O errors.
// Only optional acceleration trials may recover from this exception.
struct NativeGpuConvergenceError : std::runtime_error {
    double relativeResidual,tolerance,compatibilityRelativeL2;
    int iterations;
    bool viscosity;
    double rhsScale;
    // Failed device iterate and the exact final Ax used for rejection, before
    // multiplying the solution by rhsScale. These are diagnostics, never a solution.
    std::vector<double> scaledIterate,scaledAx,scaledLowIterate;
    NativeGpuConvergenceError(const std::string& message,const NativeGpuSolveResult& result,
                              double requestedTolerance,bool diffusion,double scale,
                              std::vector<double> iterate,std::vector<double> ax,std::vector<double> low={})
        :std::runtime_error(message),relativeResidual(result.relativeResidual),
         tolerance(requestedTolerance),compatibilityRelativeL2(result.compatibilityRelativeL2),
         iterations(result.iterations),viscosity(diffusion),rhsScale(scale),
         scaledIterate(std::move(iterate)),scaledAx(std::move(ax)),scaledLowIterate(std::move(low)) {}
};
// Matrix-free FV operators on native HA tiles, with optional full coarse/fine
// pressure gradients. Regular faces use tile
// neighbor pointers; coarse/fine and optional cut/wall stencils have local connectivity.
// Double sidecar fields preserve the cut-cell validation precision while the
// original Cirrus float FAS cycles provide the optional AMG preconditioner.
// Optional full viscosity evaluates cut-face and wall gradients on the device.
// Stencil construction and transport remain on the CPU.
class NativeCompactGpu {
public:
    // Optional positive face conductances multiply area/distance without
    // changing geometric areas, volumes, or the physical octree.
    explicit NativeCompactGpu(const Mesh& mesh,const std::vector<double>& faceFactors={});
    ~NativeCompactGpu();
    NativeCompactGpu(const NativeCompactGpu&)=delete;
    NativeCompactGpu& operator=(const NativeCompactGpu&)=delete;
    // diffusionScale*K*x + massScale*V*x. K includes compact no-slip walls only
    // when wallDirichlet is true. Pressure uses false (impermeable Neumann wall).
    std::vector<double> apply(const std::vector<double>& x,double diffusionScale,
                              double massScale,bool wallDirichlet);
    // Read back actual device face weights and local stencils for an offline
    // arithmetic oracle. Diagnostic export only; no global matrix is assembled.
    void dumpPressureFaces(const Mesh&,const std::filesystem::path&) const;
    // Zero-start double PCG/Jacobi, or FGMRES/native AMG when configured. Vectors and Ax stay
    // on the device throughout each solve; only reduction scalars cross back.
    // A nonnegative gauge selects a symmetric pressure pin. Mean-zero mode
    // instead solves the unpinned operator and retains that pin only in AMG.
    NativeGpuSolveResult solve(const std::vector<double>& rhs,double diffusionScale,
                              double massScale,bool wallDirichlet,int gauge,
                              double tolerance,int maxIterations=4000);
    // Enables flexible GMRES with native float FAS cycles as a preconditioner.
    // The physical operator and convergence residual remain double precision.
    void configureNativeAmg(const Mesh&,int pressureGauge,double viscosity,double mass);
    // Optional two-pass classical Gram--Schmidt. All inner products in one
    // pass are reduced together; true-equation stopping remains unchanged.
    void useBatchedCgs2(bool enabled=true);
    // Configure restart storage before the first FGMRES solve. The original
    // default is 20; a larger space changes only the Krylov iteration strategy.
    void setKrylovDimension(int dimension);
    int amgLevels() const;
    void useMeanZeroPressure();
    // Replace compact coarse/fine pressure fluxes by the existing full local
    // gradient stencils. The global matrix is not assembled on the device.
    // Viscosity and the native AMG preconditioner retain their compact form.
    void configurePressureInterfaces(const Mesh&,const std::vector<NativePressureFaceStencil>&);
    size_t fullPressureInterfaceFaces() const;
    // Replace compact gradients on every cut, coarse/fine and wall face.
    // The homogeneous no-slip wall gradient is evaluated without centering.
    void configureViscosityFaces(const Mesh&,const std::vector<NativeViscosityFaceStencil>&);
    size_t fullViscosityFaces() const;
    size_t allocatedBytes() const;
    size_t interfaceFaces() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
