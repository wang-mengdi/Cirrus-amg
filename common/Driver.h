//////////////////////////////////////////////////////////////////////////
// Simulator Driver
// Copyright (c) (2022-), Bo Zhu, Mengdi Wang
// This file is part of MESO, whose distribution is governed by the LICENSE file.
//////////////////////////////////////////////////////////////////////////
#pragma once

#include "Simulator.h"
#include "MetaData.h"

class DriverTimer {
public:
	std::chrono::time_point<std::chrono::system_clock> total_start;
	std::chrono::time_point<std::chrono::system_clock> lap_start;
	std::chrono::time_point<std::chrono::system_clock> loop_start;

	DriverTimer() {
		Reset();
	}
	void Reset(void);
	//total time
	double Total_Time(const double unit = PhysicalUnits::s);
	//lap time, and reset the lap clock
	double Lap_Time(const double unit = PhysicalUnits::s);
};

namespace DriverFunc {	//will change timer
	double Print_Frame_Info(DriverTimer& frame_timer, const DriverMetaData& meta_data, int num_steps);
	//will change timer
	void Print_Iteration_Info(DriverTimer& iteration_timer, const double dt, const double running_cfl, const double current_time, const double frame_time);

	//simulate from first_frame to last_frame
	//output frames [first_frame, last_frame]
	void Advance(Simulator& simulator, DriverMetaData& meta_data);
};
