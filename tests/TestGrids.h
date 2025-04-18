#pragma once

#include <string>
#include "PoissonTile.h"

enum class TestGrids {
	uniform8 = 0,
	uniform32,
	uniform128,
	uniform256,
	uniform512,
	staircase12,
	staircase21,
	staircase34,
	staircase43,
	twosources67,
	twosources_deform,
	centersource,
};

std::string ToString(const TestGrids grid_name);

//enum class ConvergenceTestGridName {
//	uniform = 0,
//	sphere_shell_05, // R=0.5, center=(0.5, 0.5, 0.5)
//};
//
//std::string ToString(const ConvergenceTestGridName grid_name);

template<class FuncV>
__hostdev__ int CornerInteriorCount(FuncV phi, const nanovdb::BBox<Vec>& bbox, T isovalue = 0)
{
	// Count how many of the 8 corners of the bbox are inside the solid (phi < 0).
	const Vec& bmin = bbox.min();
	const Vec& bmax = bbox.max();

	int inside_cnt = 0;
	for (int di : {0, 1})
	{
		for (int dj : {0, 1})
		{
			for (int dk : {0, 1})
			{
				Vec vpos;
				vpos[0] = (di == 0) ? bmin[0] : bmax[0];
				vpos[1] = (dj == 0) ? bmin[1] : bmax[1];
				vpos[2] = (dk == 0) ? bmin[2] : bmax[2];
				if (phi(vpos) - isovalue < 0)
				{
					inside_cnt++;
				}
			}
		}
	}
	return inside_cnt;
}

class UniformGridCase {
public:
    __hostdev__ static int target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, int min_level, int max_level);
    __hostdev__ static uint8_t type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk);
};

class CenterPointGridCase {
public:
    __hostdev__ static int target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, int min_level, int max_level);
    __hostdev__ static uint8_t type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk);
};

class SphereShell05GridCase {
public:
    __hostdev__ static T phi(const Vec& pos);
    __hostdev__ static int target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, int min_level, int max_level);
    __hostdev__ static uint8_t type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk);
};

class SphereSolid05GridCase {
public:
    __hostdev__ static T phi(const Vec& pos);
    __hostdev__ static int target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, int min_level, int max_level);
    __hostdev__ static uint8_t type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk);
};

class StarShellGerrisGridCase {
public:
    __hostdev__ static T phi(const Vec& pos);
    __hostdev__ static int target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, int min_level, int max_level);
    __hostdev__ static uint8_t type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk);
};

std::shared_ptr<HADeviceGrid<Tile>> CreateTestGrid(const std::string grid_name, const int min_level, const int max_level);