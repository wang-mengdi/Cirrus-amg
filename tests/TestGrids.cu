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
	bbox.min() = bbox.min() - Vec(eps, eps, eps);
	bbox.max() = bbox.max() + Vec(eps, eps, eps);
    Vec ctr(0.5, 0.5, 0.5);
    //ctr = ctr - Vec(eps, eps, eps);
    return bbox.isInside(ctr) ? max_level : min_level;
}

__hostdev__ uint8_t CenterPointGridCase::type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) {
    return CellType::INTERIOR;
}

__hostdev__ T SphereEmptyGridCase::phi(const Vec& pos) {
    const Vec ctr(0.5, 0.5, 0.5);
    constexpr T radius = 0.5 / 2;
    return (pos - ctr).length() - radius;
}

__hostdev__ int SphereEmptyGridCase::target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, int min_level, int max_level) {
    auto bbox = acc.tileBBox(info);
    int inside_cnt = CornerInteriorCount(phi, bbox);
    return (inside_cnt == 0 || inside_cnt == 8) ? min_level : max_level;
}

__hostdev__ uint8_t SphereEmptyGridCase::type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) {
    return CellType::INTERIOR;
}

__hostdev__ T SphereSolidGridCase::phi(const Vec& pos) {
    const Vec ctr(0.5, 0.5, 0.5);
    constexpr T radius = 0.5 / 2;
    return (pos - ctr).length() - radius;
}

__hostdev__ int SphereSolidGridCase::target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, int min_level, int max_level) {
    auto bbox = acc.tileBBox(info);
    int inside_cnt0 = CornerInteriorCount(phi, bbox, 0.0);
    int inside_cnt1 = CornerInteriorCount(phi, bbox, 0.01);
    return (inside_cnt0 == 8 || inside_cnt1 == 0) ? min_level : max_level;
}

__hostdev__ uint8_t SphereSolidGridCase::type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) {
    auto bbox = acc.cellBBox(info, l_ijk);
    int inside_cnt = CornerInteriorCount(phi, bbox);
    return (inside_cnt == 8) ? CellType::NEUMANN : CellType::INTERIOR;
}

__hostdev__ T StarEmptyGridCase::phi(const Vec& pos) {
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

__hostdev__ int StarEmptyGridCase::target(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, int min_level, int max_level) {
    auto bbox = acc.tileBBox(info);
    int inside_cnt = CornerInteriorCount(phi, bbox);
    return (inside_cnt == 0 || inside_cnt == 8) ? min_level : max_level;
}

__hostdev__ uint8_t StarEmptyGridCase::type(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) {
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

template <class GridCase>
std::shared_ptr<HADeviceGrid<Tile>> CreateTestGrid(const int min_level, const int max_level)
{
    uint32_t scale = 8;
    float h = 1.0 / scale;
    // 0:8, 1:16, 2:32, 3:64, 4:128, 5:256, 6:512, 7:1024
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

std::shared_ptr<HADeviceGrid<Tile>> CreateTestGrid(const std::string grid_name, const int min_level, const int max_level)
{
    if (grid_name == "uniform")
    {
        return CreateTestGrid<UniformGridCase>(min_level, max_level);
    }
    else if (grid_name == "center_point") {
        return CreateTestGrid<CenterPointGridCase>(min_level, max_level);
    }
    else if (grid_name == "sphere_empty")
    {
        return CreateTestGrid<SphereEmptyGridCase>(min_level, max_level);
    }
    else if (grid_name == "sphere_solid")
    {
        return CreateTestGrid<SphereSolidGridCase>(min_level, max_level);
    }
    else if (grid_name == "star_empty")
    {
        return CreateTestGrid<StarEmptyGridCase>(min_level, max_level);
    }
    else
    {
        Assert(false, "grid_name {} not supported", grid_name);
		return nullptr;
    }
}