add_rules("mode.debug", "mode.release", "mode.releasedbg")
set_languages("c++17")

if is_mode("debug") then
    set_symbols("debug")
    set_optimize("none")
    add_defines("CIRRUS_DEBUG")
    add_cxxflags("/RTC1")
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

--add_requires("openvdb =11.0.0", {configs = { nanovdb = false }})
--add_requireconfs("openvdb.boost", {override = true, version = "1.85.0", configs = {system = false}})

add_defines("FMT_UNICODE=0")


-- target("cirrus")
--     set_kind("binary")
--     add_headerfiles("cirrus/*.h")
--     add_files("cirrus/*.cpp", "cirrus/*.cu")
--     add_includedirs("cirrus", {public = true})
--     add_cugencodes("native")
--     add_cuflags("-extended-lambda --std=c++17")
--     add_deps("src")

--     after_build(function (target)
--         local userprofile = os.getenv("USERPROFILE")
--         local pkgdir = path.join(userprofile, "AppData/Local/.xmake/packages")

--         local pattern = path.join(pkgdir, "t/token/24.09.0", "*", "bin", "token.dll")

--         local outdir = target:targetdir()
--         os.mkdir(outdir)

--         for _, dll in ipairs(os.files(pattern)) do
--             os.cp(dll, outdir)
--         end
--     end)
    

-- target("sl_cutcell")
--     set_kind("binary")
--     add_headerfiles("sl_cutcell/*.h")
--     add_files("sl_cutcell/*.cpp", "sl_cutcell/*.cu")
--     add_includedirs("sl_cutcell", {public = true})
--     add_cugencodes("native")
--     --add_cuflags("-extended-lambda --std=c++17")
--     add_cuflags("-std=c++17 --expt-relaxed-constexpr --expt-extended-lambda")
--     add_cxxflags("/utf-8")
--     add_packages("libigl", "tbb", "polyscope")
--     add_deps("src")

--     after_build(function (target)
--         local userprofile = os.getenv("USERPROFILE")
--         local pkgdir = path.join(userprofile, "AppData/Local/.xmake/packages")

--         local pattern = path.join(pkgdir, "t/token/24.09.0", "*", "bin", "token.dll")

--         local outdir = target:targetdir()
--         os.mkdir(outdir)

--         for _, dll in ipairs(os.files(pattern)) do
--             os.cp(dll, outdir)
--         end
--     end)

-- target("impulse_cutcell")
--     set_kind("binary")
--     add_headerfiles("impulse_cutcell/*.h")
--     add_files("impulse_cutcell/*.cpp", "impulse_cutcell/*.cu")
--     add_includedirs("impulse_cutcell", {public = true})
--     add_cugencodes("native")
--     --add_cuflags("-extended-lambda --std=c++17")
--     add_cuflags("-std=c++17 --expt-relaxed-constexpr --expt-extended-lambda")
--     add_cxxflags("/utf-8")
--     add_packages("libigl", "tbb", "polyscope")
--     add_deps("src")

--     after_build(function (target)
--         local userprofile = os.getenv("USERPROFILE")
--         local pkgdir = path.join(userprofile, "AppData/Local/.xmake/packages")

--         local pattern = path.join(pkgdir, "t/token/24.09.0", "*", "bin", "token.dll")

--         local outdir = target:targetdir()
--         os.mkdir(outdir)

--         for _, dll in ipairs(os.files(pattern)) do
--             os.cp(dll, outdir)
--         end
--     end)

target("cirrus_cutcell")
    set_kind("binary")
    add_headerfiles("cirrus_cutcell/*.h")
    add_files("cirrus_cutcell/*.cpp", "cirrus_cutcell/*.cu")
    add_includedirs("cirrus_cutcell", {public = true})
    add_cugencodes("native")
    --add_cuflags("-extended-lambda --std=c++17")
    add_cuflags("-std=c++17 --expt-relaxed-constexpr --expt-extended-lambda")
    if is_mode("debug") then
        add_cuflags("-G -lineinfo")
    elseif is_mode("releasedbg") then
        add_cuflags("-lineinfo")
    end
    add_cxxflags("/utf-8")
    add_packages("libigl", "tbb", "polyscope")
    add_deps("src")

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


target("tests")
    set_kind("binary")
    add_headerfiles("tests/*.h")
    add_files("tests/*.cpp", "tests/*.cu")
    add_includedirs("tests", {public = true})
    add_cugencodes("native")
    add_cuflags("-std=c++17 --expt-relaxed-constexpr --expt-extended-lambda")
    add_cxxflags("/utf-8")
    add_deps("src")
    add_packages("magic_enum")
    add_packages("tbb", "polyscope")

