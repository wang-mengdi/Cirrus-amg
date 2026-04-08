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

