add_rules("mode.debug", "mode.release", "mode.releasedbg")
set_languages("c++17")

includes("./src/xmake.lua")

set_rundir("$(projectdir)")

add_requires("magic_enum >=0.9.7")

-- target("cirrus")
--     set_kind("binary")
--     add_headerfiles("cirrus/*.h")
--     add_files("cirrus/*.cpp", "cirrus/*.cu")
--     add_includedirs("cirrus", {public = true})
--     add_cugencodes("native")
--     add_cuflags("-extended-lambda --std=c++17")
--     add_deps("src")
    

target("impulse")
    set_kind("binary")
    add_headerfiles("impulse/*.h")
    add_files("impulse/*.cpp", "impulse/*.cu")
    add_includedirs("impulse", {public = true})
    add_cugencodes("native")
    --add_cuflags("-extended-lambda --std=c++17")
    add_cuflags("-std=c++17 --expt-relaxed-constexpr --expt-extended-lambda")
    add_deps("src")

target("tests")
    set_kind("binary")
    add_headerfiles("tests/*.h")
    add_files("tests/*.cpp", "tests/*.cu")
    add_includedirs("tests", {public = true})
    add_cugencodes("native")
    add_cuflags("-std=c++17 --expt-relaxed-constexpr --expt-extended-lambda")
    add_deps("src")
    add_packages("magic_enum")
