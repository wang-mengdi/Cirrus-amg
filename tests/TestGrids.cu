#include "TestGrids.h"

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
    else if (grid_name == "star_solid")
    {
		return CreateTestGrid<StarSolidGridCase>(min_level, max_level);
    }
    else
    {
        ASSERT(false, "grid_name {} not supported", grid_name);
		return nullptr;
    }
}
