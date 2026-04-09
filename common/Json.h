#pragma once

#include <fmt/core.h>
#include <fmt/color.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

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
