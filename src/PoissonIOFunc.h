#pragma once

#include "PoissonGrid.h"
#include "FMParticles.h"
#include "CPUTimer.h"
#include "MarkerParticles.h"

#include <fmt/ostream.h>

namespace IOFunc {
    template <class T>
    static void WritePod(std::ostream& os, const T& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        os.write(reinterpret_cast<const char*>(&v), sizeof(T));
    }

    template <class T>
    static void ReadPod(std::istream& is, T& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        is.read(reinterpret_cast<char*>(&v), sizeof(T));
    }

    template <class T>
    static void WriteVector(std::ostream& os, const std::vector<T>& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        uint64_t n = (uint64_t)v.size();
        WritePod(os, n);
        if (n) os.write(reinterpret_cast<const char*>(v.data()), sizeof(T) * (size_t)n);
    }

    template <class T>
    static void ReadVector(std::istream& is, std::vector<T>& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        uint64_t n = 0;
        ReadPod(is, n);
        v.resize((size_t)n);
        if (n) is.read(reinterpret_cast<char*>(v.data()), sizeof(T) * (size_t)n);
    }

    void WriteHAHostTileHolderToFile(const HAHostTileHolder<Tile>& holder, const fs::path& filepath);

    HAHostTileHolder<Tile> ReadHAHostTileHolderFromFile(const fs::path& filepath);
    template <typename T> void WriteHostVectorToBinary(const thrust::host_vector<T>& vec, const fs::path& filepath) {
        std::ofstream outFile(filepath.string(), std::ios::binary);
        if (!outFile) {
            throw std::runtime_error("Failed to open file for writing: " + filepath.string());
        }

        size_t size = vec.size();
        outFile.write(reinterpret_cast<const char*>(&size), sizeof(size));

        outFile.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(T));

        outFile.close();
    }

    template <typename T> thrust::host_vector<T> ReadHostVectorFromBinary(const fs::path& filepath) {
        std::ifstream inFile(filepath.string(), std::ios::binary);
        if (!inFile) {
            throw std::runtime_error("Failed to open file for reading: " + filepath.string());
        }

        size_t size;
        inFile.read(reinterpret_cast<char*>(&size), sizeof(size));

        thrust::host_vector<T> vec(size);
        inFile.read(reinterpret_cast<char*>(vec.data()), size * sizeof(T));

        inFile.close();
        return vec;
    }

    void OutputMarkerParticleSystemAsVTU(std::shared_ptr<thrust::host_vector<MarkerParticle>> particles_ptr, fs::path path);
    void OutputParticleSystemAsVTU(std::shared_ptr<thrust::host_vector<Particle>> particles_ptr, fs::path path);

    void OutputTilesAsVTU(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const fs::path& path);

    void OutputPoissonGridAsUnstructuredVTU(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const std::vector<std::pair<int, std::string>> scalar_channels, std::vector<std::pair<int, std::string>> vec_channels, const fs::path& path);
    void OutputPoissonGridAsStructuredVTI(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const std::vector<std::pair<int, std::string>> scalar_channels, std::vector<std::pair<int, std::string>> vec_channels, const fs::path& path);
    void OutputPoissonGridAsAMR(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const std::vector<std::pair<int, std::string>>& scalar_channels, const std::vector<std::pair<int, std::string>>& vec_channels, const fs::path& path);
}
