#include "PoissonTile.h"

VelTile ExtractVelocityTile(const Tile& src, int u_channel)
{
    ASSERT(u_channel >= 0);
    ASSERT(u_channel + 2 < Tile::num_channels);

    VelTile dst{};

    for (int c = 0; c < 3; ++c) {
        std::memcpy(dst.mData[c], src.mData[u_channel + c], sizeof(float) * Tile::CHNLSIZE);
    }

    for (int i = 0; i < 6; i++) {//some empty thing
        dst.mNeighbors[i] = HATileInfo<VelTile>();
    }
    std::memcpy(dst.mCellType, src.mCellType, sizeof(src.mCellType));

    dst.mIsInterestArea = src.mIsInterestArea;
    dst.mIsLockedRefine = src.mIsLockedRefine;
    dst.mStatus = src.mStatus;
    dst.mSerialIdx = src.mSerialIdx;

    return dst;
}

template class HADeviceGrid<Tile>;