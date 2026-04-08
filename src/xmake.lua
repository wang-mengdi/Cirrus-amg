add_rules("mode.debug", "mode.release", "mode.releasedbg")
set_languages("c++17")

includes("../common/xmake.lua")

add_requires("eigen >=3.4.0")
add_requires("cuda", {system = true})
add_requires("vtk >=9.5.1", {configs = {cuda = true}})
add_requires("polyscope", {version = "2.5.0"})
add_requireconfs("polyscope.glm", {override = true, version = "0.9.9+8"})

add_requires("openvdb >=12.1.1")


add_defines("FMT_UNICODE=0")


target("src")
    set_kind("static")
    add_headerfiles("*.h","../ext/*.h")
    add_files("*.cpp","*.cu")
    add_includedirs(".", "../ext", {public=true})
    
    add_cugencodes("native")
    add_cuflags("--expt-extended-lambda", "--std=c++17", "-lineinfo", "--allow-unsupported-compiler", {force = true})
    add_cuflags("-rdc=true", {force = true})


    add_packages("cuda", {public = true})
    add_packages("eigen", {public = true})
    add_packages("vtk", {public = true})
    add_packages("polyscope")
    add_packages("openvdb", {public = true})

    add_deps("common")
