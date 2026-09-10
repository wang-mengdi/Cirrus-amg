#pragma once

#include <cstdio>
#include <cstdlib>
#ifdef _WIN32
#pragma push_macro("interface")
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
// Windows RPC headers define this common CFD variable name as a macro.
// Restore its pre-header meaning instead of changing numerical source names.
#pragma pop_macro("interface")
#endif

namespace simple {
// Opt-in process observations at construction boundaries. This does not trim
// working sets, synchronize the GPU, or change any numerical settings.
inline void constructionMemory(const char* phase) {
    if(!std::getenv("SIMPLE_CONSTRUCTION_MEMORY"))return;
#ifdef _WIN32
    using Query=BOOL (WINAPI*)(HANDLE,PPROCESS_MEMORY_COUNTERS,DWORD);
    static const auto query=reinterpret_cast<Query>(
        GetProcAddress(GetModuleHandleW(L"kernel32.dll"),"K32GetProcessMemoryInfo"));
    PROCESS_MEMORY_COUNTERS_EX counters{};counters.cb=sizeof(counters);
    MEMORYSTATUSEX memory{};memory.dwLength=sizeof(memory);
    if(query && query(GetCurrentProcess(),reinterpret_cast<PPROCESS_MEMORY_COUNTERS>(&counters),sizeof(counters)) &&
       GlobalMemoryStatusEx(&memory)) {
        std::printf("Construction memory: phase=%s working_set_bytes=%llu peak_working_set_bytes=%llu private_bytes=%llu available_physical_bytes=%llu\n",
            phase,static_cast<unsigned long long>(counters.WorkingSetSize),
            static_cast<unsigned long long>(counters.PeakWorkingSetSize),
            static_cast<unsigned long long>(counters.PrivateUsage),
            static_cast<unsigned long long>(memory.ullAvailPhys));
    } else std::printf("Construction memory: phase=%s unavailable\n",phase);
#else
    std::printf("Construction memory: phase=%s unavailable\n",phase);
#endif
    std::fflush(stdout);
}
} // namespace simple
