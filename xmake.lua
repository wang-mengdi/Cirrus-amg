add_rules("mode.debug", "mode.release", "mode.releasedbg")
set_languages("c++17")

includes("./common/xmake.lua")
includes("./src/xmake.lua")

set_rundir("$(projectdir)")

target("cirrus")
    set_kind("binary")
    add_files("main.cpp")
    add_includedirs(".", {public = true})
    if is_plat("windows") then
        set_values("build.vcxproj.includes", "$(CUDA_PATH)/include")
    end
    add_cugencodes("native")
    add_cuflags("-extended-lambda --std=c++17 -lineinfo")
    add_cuflags("-rdc=true")
    add_deps("common","src")

target("tests")
    set_kind("binary")
    add_headerfiles("tests/*.h")
    add_files("tests/*.cpp", "tests/*.cu")
    add_includedirs("tests", {public = true})
    if is_plat("windows") then
        set_values("build.vcxproj.includes", "$(CUDA_PATH)/include")
    end

    add_cugencodes("native")
    add_cuflags("-extended-lambda --std=c++17 -lineinfo")
    add_cuflags("-rdc=true")
    add_deps("common","src")
