add_rules("mode.debug", "mode.release", "mode.releasedbg")
set_languages("c++17")

includes("../common/xmake.lua")

add_requires("eigen >=3.4.0")
add_requires("cuda", {system = true})
add_requires("vtk >=9.5.1", {configs = {cuda = true}})
--add_requireconfs("vtk.fast_float", {override = true, version = "3.10.1"})
--add_requires("polyscope =2.3")
--add_requireconfs("polyscope.glm", {override = true, version = "0.9.9+8"})
--add_requireconfs("polyscope.imgui", {override = true, version = "1.91.1"})
add_requires("polyscope", {version = "2.5.0"})
add_requireconfs("polyscope.glm", {override = true, version = "0.9.9+8"})

add_requires("openvdb >=12.1.1")


add_defines("FMT_UNICODE=0")


target("src")
    set_kind("static")
    add_headerfiles("*.h","../ext/*.h")
    add_files("*.cpp","*.cu")
    add_includedirs(".", "../ext", {public=true})

    -- if is_plat("windows") then
    --     set_values("build.vcxproj.includes", "$(CUDA_PATH)/include")
    -- end
    -- add_includedirs("$(env CUDA_PATH)/include", "$(env CUDA_PATH)/include/cccl", {public = true})
    -- if is_plat("windows") then
    --     set_values("build.vcxproj.includes", "$(env CUDA_PATH)/include;$(env CUDA_PATH)/include/cccl")
    -- end

    add_cugencodes("native")
    add_cuflags("-extended-lambda --std=c++17 -lineinfo --allow-unsupported-compiler")
    add_cuflags("-rdc=true")


    add_packages("cuda", {public = true})
    add_packages("eigen", {public = true})
    add_packages("vtk", {public = true})
    add_packages("polyscope")
    add_packages("openvdb", {public = true})

    add_deps("common")
