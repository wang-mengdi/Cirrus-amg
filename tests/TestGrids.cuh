#pragma once

#include "TestGridsStr.h"
#include "PoissonTile.h"

template<class FuncV>
__hostdev__ int CornerInteriorCount(FuncV phi, const nanovdb::BBox<Vec>& bbox, T isovalue = 0)
{
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
    __hostdev__ static inline int target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, int min_level, int max_level) {
        return max_level;
    }
    __hostdev__ static inline uint8_t type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) {
        return CellType::INTERIOR;
    }
};

class CenterPointGridCase {
public:
    __hostdev__ static inline int target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, int min_level, int max_level) {
        auto bbox = acc.tileBBox(info);
        const double eps = 1e-6;
		bbox.min() = bbox.min() - Vec(eps, eps, eps);
		bbox.max() = bbox.max() + Vec(eps, eps, eps);
        Vec ctr(0.5, 0.5, 0.5);
        return bbox.isInside(ctr) ? max_level : min_level;
    }
    __hostdev__ static inline uint8_t type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) {
        return CellType::INTERIOR;
    }
};

class SphereEmptyGridCase {
public:
    __hostdev__ static inline T phi(const Vec& pos) {
        const Vec ctr(0.5, 0.5, 0.5);
        constexpr T radius = 0.5 / 2;
        return (pos - ctr).length() - radius;
    }
    __hostdev__ static inline int target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, int min_level, int max_level) {
        auto bbox = acc.tileBBox(info);
        int inside_cnt = CornerInteriorCount(phi, bbox);
        return (inside_cnt == 0 || inside_cnt == 8) ? min_level : max_level;
    }
    __hostdev__ static inline uint8_t type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) {
        return CellType::INTERIOR;
    }
};

class SphereSolidGridCase {
public:
    __hostdev__ static inline T phi(const Vec& pos) {
        const Vec ctr(0.5, 0.5, 0.5);
        constexpr T radius = 0.5 / 2;
        return (pos - ctr).length() - radius;
    }
    __hostdev__ static inline int target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, int min_level, int max_level) {
        auto bbox = acc.tileBBox(info);
        int inside_cnt0 = CornerInteriorCount(phi, bbox, 0.0);
        int inside_cnt1 = CornerInteriorCount(phi, bbox, 0.01);
        return (inside_cnt0 == 8 || inside_cnt1 == 0) ? min_level : max_level;
    }
    __hostdev__ static inline uint8_t type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) {
        auto bbox = acc.cellBBox(info, l_ijk);
        int inside_cnt = CornerInteriorCount(phi, bbox);
        return (inside_cnt == 8) ? CellType::NEUMANN : CellType::INTERIOR;
    }
};

class StarEmptyGridCase {
public:
    __hostdev__ static inline T phi(const Vec& pos) {
        T x = pos[0] - 0.5;
        T y = pos[1] - 0.5;
        T z = pos[2] - 0.5;
        T r = sqrt(x * x + y * y + z * z);
        if (r == 0) return T(-FLT_MAX);
        T theta = acos(z / r);
        T phi = atan2(y, x);
        T r0 = 0.237 + 0.079 * cos(6 * theta) * cos(6 * phi);
        return r - r0;
    }
    __hostdev__ static inline int target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, int min_level, int max_level) {
        auto bbox = acc.tileBBox(info);
        int inside_cnt = CornerInteriorCount(phi, bbox);
        return (inside_cnt == 0 || inside_cnt == 8) ? min_level : max_level;
    }
    __hostdev__ static inline uint8_t type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) {
        return CellType::INTERIOR;
    }
};

class StarSolidGridCase {
public:
    __hostdev__ static inline T phi(const Vec& pos) {
        T x = pos[0] - 0.5;
        T y = pos[1] - 0.5;
        T z = pos[2] - 0.5;
        T r = sqrt(x * x + y * y + z * z);
        if (r == 0) return T(-FLT_MAX);
        T theta = acos(z / r);
        T phi = atan2(y, x);
        T r0 = 0.237 + 0.079 * cos(6 * theta) * cos(6 * phi);
        return r - r0;
    }
    __hostdev__ static inline int target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, int min_level, int max_level) {
        auto bbox = acc.tileBBox(info);
        int inside_cnt0 = CornerInteriorCount(phi, bbox, 0.0);
        int inside_cnt1 = CornerInteriorCount(phi, bbox, 0.01);
        return (inside_cnt0 == 8 || inside_cnt1 == 0) ? min_level : max_level;
    }
    __hostdev__ static inline uint8_t type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) {
        auto bbox = acc.cellBBox(info, l_ijk);
        int inside_cnt = CornerInteriorCount(phi, bbox);
        return (inside_cnt == 8) ? CellType::NEUMANN : CellType::INTERIOR;
    }
};

template <class GridCase>
std::shared_ptr<HADeviceGrid<Tile>> CreateTestGrid(const int min_level, const int max_level)
{
    uint32_t scale = 8;
    float h = 1.0 / scale;
    auto grid_ptr = std::make_shared<HADeviceGrid<Tile>>(h, thrust::host_vector{ 16, 16, 16, 16, 16, 16, 20, 20, 16, 16 });
    auto& grid = *grid_ptr;
    grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
    grid.rebuild();
    grid.iterativeRefine([=] __device__(const HATileAccessor<Tile> &acc, HATileInfo<Tile> &info)
    {
        return GridCase::target(acc, info, min_level, max_level);
    }, false);
    int num_cells = grid.numTotalLeafTiles() * Tile::SIZE;
    int total_hash_bytes = grid.hashTableDeviceBytes();
    Info("Total {}M cells, hash table {}GB", num_cells / (1024.0 * 1024), total_hash_bytes / (1024.0 * 1024 * 1024));

    grid.launchVoxelFuncOnAllTiles(
        [=] __device__(HATileAccessor<Tile> &acc, HATileInfo<Tile> &info, const Coord & l_ijk)
    {
        auto& tile = info.tile();
        tile.type(l_ijk) = GridCase::type(acc, info, l_ijk);
    },
        LEAF);
    return grid_ptr;
}

std::shared_ptr<HADeviceGrid<Tile>> CreateTestGrid(const std::string grid_name, const int min_level, const int max_level);
