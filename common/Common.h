//////////////////////////////////////////////////////////////////////////
// Common header
// Copyright (c) (2022-), Bo Zhu, Mengdi Wang
// This file is part of MESO, whose distribution is governed by the LICENSE file.
//////////////////////////////////////////////////////////////////////////
#pragma once

#include <iostream>
//#include <fstream>

#include <fmt/core.h>
#include <fmt/color.h>
//fmt/ranges.h will override the format of Vector<T,d>
//#include <fmt/ranges.h>
//ideally we don't want to use standard list, queue and array

#include <nlohmann/json.hpp>

//#include <boost/filesystem.hpp>
#include <filesystem>

//namespace bf = boost::filesystem;
namespace fs = std::filesystem;

using json = nlohmann::json;

//
//namespace FileFunc {
//    void CreateDirectory(const fs::path path);
//}

namespace CommonConstants {
    constexpr double pi = 3.14159265358979;
    constexpr double g = (double)9.81;
}

namespace PhysicalUnits {
    constexpr double m = 1.0;
    constexpr double cm = 0.01;
    constexpr double mm = 1e-3;
    constexpr double s = 1.0;
    constexpr double ms = 1e-3;
}

template<class... Args>
void Info(const char* fmt_str, const Args&...args) {
    fmt::print("#     ");
    //auto fst = fmt::format_string<Args...>(fmt_str);
    //fmt::print(fst, (args)...);
    fmt::print(fmt_str, args...);
    fmt::print("\n");
}
void Info(const std::string& str);

template<typename ...Args>
void Warn(const char* fmt_str, const Args&...args) {
    fmt::print(fg(fmt::color::yellow), "#     ");
    fmt::print(fg(fmt::color::yellow), fmt_str, args...);
    fmt::print("\n");
}
void Warn(const std::string& str);

template<typename ...Args>
void Error(const char* fmt_str, const Args&...args) {
    std::string msg = "#     " + fmt::format(fmt_str, args...) + "\n";
    fmt::print(fg(fmt::color::red), msg);
    throw msg;
}
//void Error(const std::string& str);

template<typename ...Args>
void Pass(const char* fmt_str, const Args&...args) {
    fmt::print(fg(fmt::color::green), "#     ");
    fmt::print(fg(fmt::color::green), fmt_str, args...);
    fmt::print("\n");
}
void Pass(const std::string& str);

//template <typename... Args>
//void ASSERT(const bool flg, const char* fmt_str = "", const Args &...args) {
//    if (!flg) {
//        Error(fmt_str, args...);
//    }
//}

#include <fmt/format.h>
#include <fmt/color.h>

inline void CpuAssertImpl(bool cond,
    const char* expr,
    const char* file,
    int line)
{
    if (!cond) {
        fmt::print(fmt::fg(fmt::color::red),
            "ASSERT FAILED: {}\n  at {}:{}\n",
            expr, file, line);

#if defined(_MSC_VER)
        __debugbreak();
#else
        __builtin_trap();
#endif
    }
}

template<typename... Args>
inline void CpuAssertMsgImpl(bool cond,
    const char* expr,
    const char* file,
    int line,
    const char* fmt_str,
    const Args&... args)
{
    if (!cond) {
        fmt::print(fmt::fg(fmt::color::red),
            "ASSERT FAILED: {}\n  at {}:{}\n",
            expr, file, line);

        fmt::print(fmt::fg(fmt::color::red), fmt_str, args...);
        fmt::print("\n");

#if defined(_MSC_VER)
        __debugbreak();
#else
        __builtin_trap();
#endif
    }
}

#define ASSERT_1(cond) \
    CpuAssertImpl((cond), #cond, __FILE__, __LINE__)

#define ASSERT_2(cond, fmt_str, ...) \
    CpuAssertMsgImpl((cond), #cond, __FILE__, __LINE__, fmt_str, ##__VA_ARGS__)

#define GET_ASSERT_MACRO(_1,_2,_3,NAME,...) NAME

#define ASSERT(...) \
    GET_ASSERT_MACRO(__VA_ARGS__, ASSERT_2, ASSERT_1)(__VA_ARGS__)


namespace Json {
	template<class T>
	T Value(json& j, const std::string key, const T default_value) {
		if (j.contains(key)) {
			T value = j.at(key);
			fmt::print(fg(fmt::color::green), "#     [=] Parse key ");
			fmt::print("{}", key);
			fmt::print(fg(fmt::color::green), " from json: ");
			fmt::print("{}\n", value);
			return value;
		}
		else {
			j[key] = default_value;
			fmt::print(fg(fmt::color::yellow), "#     [+] Can't parse key ");
			fmt::print("{}", key);
			fmt::print(fg(fmt::color::yellow), " in json, set to default value: ");
			fmt::print("{}\n", default_value);
			return default_value;
		}
	}

	template<class T>
	void Set_Non_Override(json& j, const std::string key, const T value) {
		if (!j.contains(key)) {
			j[key] = value;
		}
	}
}