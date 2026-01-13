set_languages("c++17")
add_cxflags("/utf-8", {tools = "cl"})


--add_requires("fmt =8.1.1")
add_requires("nlohmann_json >=3.10.5")



target("common")
    set_kind("static")
    add_headerfiles("*.h")
    add_files("*.cpp")
    add_includedirs(".",{public=true})
    add_packages("fmt",{public=true})
    add_packages("nlohmann_json",{public=true})
