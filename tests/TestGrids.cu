#include "TestGrids.h"

__hostdev__ int UniformGridCase::target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, int min_level, int max_level) {
    return max_level;
}

__hostdev__ uint8_t UniformGridCase::type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) {
    return CellType::INTERIOR;
}

__hostdev__ int CenterPointGridCase::target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, int min_level, int max_level) {
    auto bbox = acc.tileBBox(info);
    const double eps = 1e-6;
    Vec ctr(0.5, 0.5, 0.5);
    ctr = ctr - Vec(eps, eps, eps);
    return bbox.isInside(ctr) ? max_level : min_level;
}

__hostdev__ uint8_t CenterPointGridCase::type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) {
    return CellType::INTERIOR;
}

__hostdev__ T SphereShell05GridCase::phi(const Vec& pos) {
    const Vec ctr(0.5, 0.5, 0.5);
    constexpr T radius = 0.5 / 2;
    return (pos - ctr).length() - radius;
}

__hostdev__ int SphereShell05GridCase::target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, int min_level, int max_level) {
    auto bbox = acc.tileBBox(info);
    int inside_cnt = CornerInteriorCount(phi, bbox);
    return (inside_cnt == 0 || inside_cnt == 8) ? min_level : max_level;
}

__hostdev__ uint8_t SphereShell05GridCase::type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) {
    return CellType::INTERIOR;
}

__hostdev__ T SphereSolid05GridCase::phi(const Vec& pos) {
    const Vec ctr(0.5, 0.5, 0.5);
    constexpr T radius = 0.5 / 2;
    return (pos - ctr).length() - radius;
}

__hostdev__ int SphereSolid05GridCase::target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, int min_level, int max_level) {
    auto bbox = acc.tileBBox(info);
    int inside_cnt0 = CornerInteriorCount(phi, bbox, 0.0);
    int inside_cnt1 = CornerInteriorCount(phi, bbox, 0.01);
    return (inside_cnt0 == 8 || inside_cnt1 == 0) ? min_level : max_level;
}

__hostdev__ uint8_t SphereSolid05GridCase::type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) {
    auto bbox = acc.cellBBox(info, l_ijk);
    int inside_cnt = CornerInteriorCount(phi, bbox);
    return (inside_cnt == 8) ? CellType::NEUMANN : CellType::INTERIOR;
}

__hostdev__ T StarShellGerrisGridCase::phi(const Vec& pos) {
    T x = pos[0] - 0.5;
    T y = pos[1] - 0.5;
    T z = pos[2] - 0.5;
    T r = sqrt(x * x + y * y + z * z);
    if (r == 0) return T(FLT_MAX);
    T theta = acos(z / r);
    T phi = atan2(y, x);
    T r0 = 0.237 + 0.079 * cos(6 * theta) * cos(6 * phi);
    return r - r0;
}

__hostdev__ int StarShellGerrisGridCase::target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, int min_level, int max_level) {
    auto bbox = acc.tileBBox(info);
    int inside_cnt = CornerInteriorCount(phi, bbox);
    return (inside_cnt == 0 || inside_cnt == 8) ? min_level : max_level;
}

__hostdev__ uint8_t StarShellGerrisGridCase::type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) {
    return CellType::INTERIOR;
}

//class SphereAir05GridCase
//{
//public:
//	__hostdev__ static int target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const int min_level, const int max_level)
//	{
//		auto bbox = acc.tileBBox(info);
//		auto bmin = bbox.min();
//		auto bmax = bbox.max();
//		const Vec ctr(0.5, 0.5, 0.5);
//		constexpr T radius = 0.5 / 2;
//		int inside_cnt = 0;
//		for (int di : {0, 1})
//		{
//			for (int dj : {0, 1})
//			{
//				for (int dk : {0, 1})
//				{
//					Vec vpos;
//					vpos[0] = bmin[0] + di * (bmax[0] - bmin[0]);
//					vpos[1] = bmin[1] + dj * (bmax[1] - bmin[1]);
//					vpos[2] = bmin[2] + dk * (bmax[2] - bmin[2]);
//					if ((vpos - ctr).length() < radius)
//					{
//						inside_cnt++;
//					}
//				}
//			}
//		}
//		if (inside_cnt == 0 || inside_cnt == 8)
//			return min_level;
//		else
//			return max_level;
//	}
//	__hostdev__ static uint8_t type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk)
//	{
//		auto pos = acc.cellCenter(info, l_ijk);
//		const Vec ctr(0.5, 0.5, 0.5);
//		constexpr T radius = 0.5 / 2;
//		if ((pos - ctr).length() < radius)
//			return CellType::DIRICHLET;
//		else
//			return CellType::INTERIOR;
//	}
//};


//class StarSolidGerrisGridCase
//{
//public:
//	__hostdev__ static T phi(const Vec& pos)
//	{
//		T x = pos[0] - 0.5;
//		T y = pos[1] - 0.5;
//		T z = pos[2] - 0.5;
//		T r = sqrt(x * x + y * y + z * z);
//		T theta = acos(z / r);
//		T phi = atan2(y, x);

//		T r0 = 0.237 + 0.079 * cos(6 * theta) * cos(6 * phi);
//		return r - r0;
//	}
//	__hostdev__ static int target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const int min_level, const int max_level)
//	{
//		auto bbox = acc.tileBBox(info);
//		auto bmin = bbox.min();
//		auto bmax = bbox.max();
//		int inside_cnt = 0;
//		for (int di : {0, 1})
//		{
//			for (int dj : {0, 1})
//			{
//				for (int dk : {0, 1})
//				{
//					Vec vpos;
//					vpos[0] = bmin[0] + di * (bmax[0] - bmin[0]);
//					vpos[1] = bmin[1] + dj * (bmax[1] - bmin[1]);
//					vpos[2] = bmin[2] + dk * (bmax[2] - bmin[2]);

//					if (phi(vpos) < 0)
//					{
//						inside_cnt++;
//					}
//				}
//			}
//		}
//		if (inside_cnt == 0 || inside_cnt == 8)
//			return min_level;
//		else
//			return max_level;
//	}

//	__hostdev__ static uint8_t type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk)
//	{
//		auto pos = acc.cellCenter(info, l_ijk);
//		auto dx = acc.voxelSize(info);
//		Vec bmin(pos[0] - 0.5 * dx, pos[1] - 0.5 * dx, pos[2] - 0.5 * dx);
//		const Vec ctr(0.5, 0.5, 0.5);
//		constexpr T radius = 0.5 / 2;
//		int inside_cnt = 0;
//		for (int di : {0, 1})
//		{
//			for (int dj : {0, 1})
//			{
//				for (int dk : {0, 1})
//				{
//					Vec vpos;
//					vpos[0] = bmin[0] + di * dx;
//					vpos[1] = bmin[1] + dj * dx;
//					vpos[2] = bmin[2] + dk * dx;

//					if (phi(vpos) < 0)
//					{
//						inside_cnt++;
//					}
//				}
//			}
//		}
//		if (inside_cnt == 8)
//			return CellType::NEUMANN;
//		else
//			return CellType::INTERIOR;
//	}
//};