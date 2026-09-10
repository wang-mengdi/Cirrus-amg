add_rules("mode.debug", "mode.release", "mode.releasedbg")
set_languages("c++17")

if is_mode("debug") then
    set_symbols("debug")
    set_optimize("none")
    add_defines("CIRRUS_DEBUG")
    if is_plat("windows") then
        add_cxxflags("/RTC1")
    end
end

if is_mode("releasedbg") then
    set_symbols("debug")
    set_optimize("fast")
    add_defines("CIRRUS_DEBUG")
end

if is_mode("release") then
    set_optimize("fastest")
end

add_requires("fmt =12.1.0")
add_requireconfs("*.fmt", { override = true, version = "12.1.0" })


includes("./src/xmake.lua")

set_rundir("$(projectdir)")

add_requires("magic_enum >=0.9.7")
add_requires("tbb")
add_requires("libigl")

add_defines("FMT_UNICODE=0")

target("cirrus_cutcell")
    set_kind("binary")
    set_values("cuda.rdc", false)
    add_headerfiles("cirrus_cutcell/*.h")
    add_files("cirrus_cutcell/*.cpp", "cirrus_cutcell/*.cu")
    add_includedirs("cirrus_cutcell", {public = true})
    add_cugencodes("native")
    add_cuflags("-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "--allow-unsupported-compiler", {force = true})
    if is_mode("debug") then
        add_cuflags("-G", "-lineinfo", {force = true})
    elseif is_mode("releasedbg") then
        add_cuflags("-lineinfo", {force = true})
    end
    if is_plat("windows") then
        add_cxxflags("/utf-8")
    end
    add_packages("libigl", "tbb", "polyscope")
    add_deps("src")

    if is_plat("windows") then
        after_build(function (target)
            local userprofile = os.getenv("USERPROFILE")
            local pkgdir = path.join(userprofile, "AppData/Local/.xmake/packages")

            local pattern = path.join(pkgdir, "t/token/24.09.0", "*", "bin", "token.dll")

            local outdir = target:targetdir()
            os.mkdir(outdir)

            for _, dll in ipairs(os.files(pattern)) do
                os.cp(dll, outdir)
            end
        end)
    end


target("tests")
    set_kind("binary")
    set_values("cuda.rdc", false)
    add_headerfiles("tests/*.h")
    add_files("tests/*.cpp", "tests/*.cu")
    add_includedirs("tests", {public = true})
    add_cugencodes("native")
    add_cuflags("-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "--allow-unsupported-compiler", {force = true})
    if is_plat("windows") then
        add_cxxflags("/utf-8")
    end
    add_deps("src")
    add_packages("magic_enum")
    add_packages("tbb", "polyscope")

-- Accuracy-first SIMPLE implementation on the native HA octree leaf topology.
-- Double precision CPU matrices make operator dumps reproducible before GPU porting.
option("simple_amgcl_root")
    set_default("")
    set_showmenu(true)
    set_description("Optional AMGCL include root for SIMPLE pressure solves")
option_end()

target("simple_channel")
    set_kind("binary")
    set_values("cuda.rdc", false)
    add_files("simple/*.cpp", "simple/*.cu")
    add_headerfiles("simple/*.h")
    add_includedirs("simple")
    if get_config("simple_amgcl_root") ~= "" and get_config("simple_amgcl_root") ~= nil then
        add_includedirs(get_config("simple_amgcl_root"))
        add_defines("SIMPLE_HAVE_AMGCL")
        if is_plat("windows") then add_cxxflags("/openmp") end
    end
    add_cugencodes("native")
    add_cuflags("-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "--allow-unsupported-compiler", {force = true})
    if is_plat("windows") then
        add_cxxflags("/utf-8")
    end
    add_deps("src")

target("simple_anderson_test")
    set_kind("binary")
    add_files("simple/AndersonAcceleration.cpp", "validation/anderson_test.cpp")
    add_includedirs("simple")
    add_packages("eigen", "nlohmann_json")

target("simple_anderson_file_test")
    set_kind("binary")
    add_files("simple/AndersonAcceleration.cpp", "validation/anderson_file_test.cpp")
    add_includedirs("simple")
    add_packages("eigen", "nlohmann_json")

target("simple_anderson_file_scale_test")
    set_kind("binary")
    add_files("simple/AndersonAcceleration.cpp", "validation/anderson_file_scale_test.cpp")
    add_includedirs("simple")
    add_packages("eigen", "nlohmann_json")

target("native_compact_gpu_audit")
    set_kind("binary")
    set_values("cuda.rdc", false)
    add_files("simple/NativeCompactGpu.cu", "simple/NativeAmgPreconditioner.cu", "simple/OctreeMesh.cu", "simple/EmbeddedMesh.cpp", "simple/SimpleMesh.cpp", "validation/native_compact_gpu_audit.cpp")
    add_files("simple/EmbeddedOperators.cpp", "simple/QuadraticReconstruction.cpp")
    add_files("validation/native_tile_metadata_audit.cu")
    add_files("validation/native_host_storage_audit.cu")
    add_files("validation/native_amg_topology_audit.cu")
    add_includedirs("simple")
    add_cugencodes("native")
    add_cuflags("-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "--allow-unsupported-compiler", {force = true})
    if is_plat("windows") then add_cxxflags("/utf-8") end
    add_deps("src")

