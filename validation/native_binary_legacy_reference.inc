// Frozen f285ab48 src/HAGrid.h dumpBinaryBlob; only member qualification and function signature adapted.
	std::vector<uint8_t> legacyBinaryBlob(const HADeviceGrid<Tile>& grid,uint8_t tile_types = (LEAF | GHOST | NONLEAF),
		int max_level = -1)
	{
		ASSERT(grid.mCompressedFlag, "dumpBinaryBlob requires compressed grid (call compressHost first).");
		if (max_level < 0) max_level = grid.mMaxLevel;

		using CoordT = Coord; using T = Tile::T;

		static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
		static_assert(std::is_trivially_copyable_v<CoordT>, "Coord must be trivially copyable");
		static_assert(std::is_trivially_copyable_v<Tile>, "Tile must be trivially copyable for raw dump");

		struct Header {
			uint32_t magic;      // 'HAGR'
			uint32_t version;    // 1
			uint32_t num_levels; // grid.mNumLevels
			int32_t  max_level;  // stored max level
			uint32_t tile_size;  // sizeof(Tile)
			uint32_t coord_size; // sizeof(Coord)
			T        h0;         // grid.mH0
			uint64_t tile_count; // #records
		};

		// Count tiles
		uint64_t tile_count = 0;
		for (int lv = 0; lv <= max_level; ++lv) {
			for (uint32_t j = 0; j < grid.hNumTiles[lv]; ++j) {
				const auto& info = grid.hTileArrays[lv][j];
				if (info.mType & tile_types) tile_count++;
			}
		}

		Header hdr{};
		hdr.magic = 0x52474148u; // 'HAGR' little-endian
		hdr.version = 1;
		hdr.num_levels = grid.mNumLevels;
		hdr.max_level = max_level;
		hdr.tile_size = (uint32_t)sizeof(Tile);
		hdr.coord_size = (uint32_t)sizeof(CoordT);
		hdr.h0 = grid.mH0;
		hdr.tile_count = tile_count;

		const size_t bytes_header = sizeof(Header);
		const size_t bytes_hashes = sizeof(uint32_t) * hdr.num_levels;
		const size_t bytes_record =
			sizeof(uint32_t) +  // level
			sizeof(uint8_t) +  // type
			sizeof(CoordT) +  // tileCoord
			sizeof(Tile);       // tile bytes

		const size_t total = bytes_header + bytes_hashes + (size_t)tile_count * bytes_record;

		std::vector<uint8_t> blob(total);
		uint8_t* p = blob.data();

		auto write_raw = [&](const void* src, size_t n) {
			std::memcpy(p, src, n);
			p += n;
			};

		write_raw(&hdr, sizeof(Header));

		// write grid.hLog2Hashes
		for (uint32_t lv = 0; lv < hdr.num_levels; ++lv) {
			uint32_t v = grid.hLog2Hashes[lv];
			write_raw(&v, sizeof(uint32_t));
		}

		// write tile records (from compressed tile arrays)
		for (int lv = 0; lv <= max_level; ++lv) {
			for (uint32_t j = 0; j < grid.hNumTiles[lv]; ++j) {
				const auto& info = grid.hTileArrays[lv][j];
				if (!(info.mType & tile_types)) continue;

				uint32_t level_u32 = (uint32_t)info.mLevel;
				uint8_t  type_u8 = info.mType;
				CoordT   coord = info.mTileCoord;

				// Copy device tile bytes to host
				Tile tile_host = info.getTile(DEVICE);

				write_raw(&level_u32, sizeof(uint32_t));
				write_raw(&type_u8, sizeof(uint8_t));
				write_raw(&coord, sizeof(CoordT));
				write_raw(&tile_host, sizeof(Tile));
			}
		}

		ASSERT((size_t)(p - blob.data()) == total, "dumpBinaryBlob size mismatch");
		return blob;
	}
