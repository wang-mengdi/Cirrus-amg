// Host-only C-ABI client. Include after the caller's Eigen/AMGCL headers so
// Windows macros do not affect those libraries. The default CPU path never
// constructs this client and has no CUDA runtime dependency.
#pragma once
#include <windows.h>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

inline std::wstring TwistedCudaEnvironment(const wchar_t* name) {
  const DWORD size=GetEnvironmentVariableW(name,nullptr,0);
  if(!size)throw std::runtime_error("Missing CUDA AMG library or dependency directory");
  std::wstring value(size,L'\0');
  const DWORD written=GetEnvironmentVariableW(name,&value[0],size);
  if(written!=size-1)throw std::runtime_error("CUDA AMG environment changed during lookup");
  value.resize(written);
  if(!std::filesystem::path(value).is_absolute())
    throw std::runtime_error("CUDA AMG library paths must be absolute");
  return value;
}

class TwistedCudaAmgLibrary {
  struct Directory {
    DLL_DIRECTORY_COOKIE value=nullptr;
    explicit Directory(const std::wstring& path):value(AddDllDirectory(path.c_str())) {
      if(!value)throw std::runtime_error("Cannot register CUDA dependency directory");
    }
    ~Directory() {if(value)RemoveDllDirectory(value);}
  };
  struct Module {
    HMODULE value=nullptr;
    explicit Module(const std::wstring& path):value(LoadLibraryExW(path.c_str(),nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)) {
      if(!value)throw std::runtime_error("Cannot load CUDA AMG library, Windows error "+std::to_string(GetLastError()));
    }
    ~Module() {if(value)FreeLibrary(value);}
  };
  // Dependency directory remains registered until the library is unloaded.
  Directory directory_;
  Module module_;
  template<class Function> Function symbol(const char* name) {
    auto address=GetProcAddress(module_.value,name);
    if(!address)throw std::runtime_error(std::string("Missing CUDA AMG symbol: ")+name);
    return reinterpret_cast<Function>(address);
  }
public:
  using Create=int (__cdecl*)(int32_t,int32_t,const int32_t*,const int32_t*,const double*,
      int32_t,void**,double*,char*,uint64_t);
  using Solve=int (__cdecl*)(void*,int32_t,const double*,double*,int32_t*,double*,double*,char*,uint64_t);
  using Destroy=void (__cdecl*)(void*);
  Create create;
  Solve solve;
  Destroy destroy;
  TwistedCudaAmgLibrary():directory_(TwistedCudaEnvironment(L"APHROS_TWISTED_CUDA_DEPENDENCY_DIR")),
      module_(TwistedCudaEnvironment(L"APHROS_TWISTED_CUDA_LIBRARY")),
      create(symbol<Create>("twisted_cuda_amg_create")),
      solve(symbol<Solve>("twisted_cuda_amg_solve")),
      destroy(symbol<Destroy>("twisted_cuda_amg_destroy")) {
    using Version=int (__cdecl*)();
    if(symbol<Version>("twisted_cuda_amg_abi_version")()!=1)
      throw std::runtime_error("Unsupported CUDA AMG C ABI version");
  }
  static std::shared_ptr<TwistedCudaAmgLibrary> Get() {
    static auto library=std::make_shared<TwistedCudaAmgLibrary>();
    return library;
  }
};

class TwistedCudaAmgClient {
  std::shared_ptr<TwistedCudaAmgLibrary> library_;
  void* handle_=nullptr;
  int32_t rows_=0,nonzeros_=0;
  bool symmetric_=false;
  void Trace(const char* stage,const std::string& system,int iterations,double residual,double seconds) const {
    static std::ofstream log("cuda_amg.csv");
    static size_t event=0;
    if(!event)log<<"event,stage,system,rows,nonzeros,symmetric,iterations,reported_residual,seconds\n";
    log<<std::setprecision(21)<<++event<<','<<stage<<','<<system<<','<<rows_<<','<<nonzeros_<<','
       <<int(symmetric_)<<','<<iterations<<','<<residual<<','<<seconds<<'\n';
    log.flush();if(!log.good())throw std::runtime_error("CUDA AMG diagnostic write failed");
  }
public:
  TwistedCudaAmgClient(const TwistedCudaAmgClient&)=delete;
  TwistedCudaAmgClient& operator=(const TwistedCudaAmgClient&)=delete;
  template<class Matrix>
  TwistedCudaAmgClient(const Matrix& matrix,bool symmetric,const std::string& system):
      library_(TwistedCudaAmgLibrary::Get()),symmetric_(symmetric) {
    static_assert(Matrix::IsRowMajor,"CUDA AMG expects CSR coefficients");
    static_assert(sizeof(typename Matrix::StorageIndex)==sizeof(int32_t),"CUDA AMG expects 32-bit CSR indices");
    if(matrix.rows()<=0 || matrix.rows()>std::numeric_limits<int32_t>::max() ||
       matrix.nonZeros()>std::numeric_limits<int32_t>::max())
      throw std::runtime_error("CUDA AMG matrix exceeds C ABI dimensions");
    rows_=int32_t(matrix.rows());nonzeros_=int32_t(matrix.nonZeros());
    char error[2048]{};double seconds=0.;
    const int code=library_->create(rows_,nonzeros_,matrix.outerIndexPtr(),matrix.innerIndexPtr(),
        matrix.valuePtr(),int32_t(symmetric),&handle_,&seconds,error,sizeof(error));
    if(code || !handle_)throw std::runtime_error(std::string("CUDA AMG construction: ")+error);
    try {Trace("create",system,0,0.,seconds);}
    catch(...) {library_->destroy(handle_);handle_=nullptr;throw;}
  }
  ~TwistedCudaAmgClient() {if(handle_)library_->destroy(handle_);}
  std::tuple<size_t,double> Solve(const std::vector<double>& source,std::vector<double>& answer,
                               const std::string& system) {
    if(source.size()!=size_t(rows_) || answer.size()!=size_t(rows_))
      throw std::runtime_error("CUDA AMG client vector size differs");
    char error[2048]{};int32_t iterations=0;double residual=0.,seconds=0.;
    const int code=library_->solve(handle_,rows_,source.data(),answer.data(),&iterations,&residual,&seconds,
        error,sizeof(error));
    if(code)throw std::runtime_error(std::string("CUDA AMG solve: ")+error);
    if(iterations<0 || iterations>500)throw std::runtime_error("Invalid CUDA AMG iteration count");
    Trace("solve",system,iterations,residual,seconds);
    return std::make_tuple(size_t(iterations),residual);
  }
};
