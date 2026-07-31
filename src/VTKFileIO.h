#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace VTKFileIO {

struct FloatArray {
    std::string name;
    int components = 1;
    std::vector<float> values;
};

inline std::string EscapeXML(const std::string& value) {
    std::string result;
    result.reserve(value.size());
    for (char ch : value) {
        switch (ch) {
        case '&': result += "&amp;"; break;
        case '<': result += "&lt;"; break;
        case '>': result += "&gt;"; break;
        case '"': result += "&quot;"; break;
        case '\'': result += "&apos;"; break;
        default: result += ch; break;
        }
    }
    return result;
}

inline void EnsureParentDirectory(const std::filesystem::path& path) {
    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

class AppendedWriter {
public:
    std::uint64_t Add(const void* data, std::uint64_t bytes) {
        const std::uint64_t offset = size_;
        blocks_.push_back({data, bytes});
        size_ += sizeof(std::uint64_t) + bytes;
        return offset;
    }

    template <class T>
    std::uint64_t Add(const std::vector<T>& values) {
        return Add(values.data(), static_cast<std::uint64_t>(values.size() * sizeof(T)));
    }

    void Write(const std::filesystem::path& path, const std::string& xml) const {
        EnsureParentDirectory(path);
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Failed to open VTK output file: " + path.string());
        }
        out << xml << "  <AppendedData encoding=\"raw\">\n_";
        for (const auto& block : blocks_) {
            out.write(reinterpret_cast<const char*>(&block.bytes), sizeof(block.bytes));
            if (block.bytes != 0) {
                out.write(reinterpret_cast<const char*>(block.data),
                    static_cast<std::streamsize>(block.bytes));
            }
        }
        out << "\n  </AppendedData>\n</VTKFile>\n";
        if (!out) {
            throw std::runtime_error("Failed while writing VTK output file: " + path.string());
        }
    }

private:
    struct Block {
        const void* data;
        std::uint64_t bytes;
    };
    std::vector<Block> blocks_;
    std::uint64_t size_ = 0;
};

inline void AppendFloatArrays(
    std::ostringstream& xml,
    AppendedWriter& appended,
    const std::vector<FloatArray>& arrays,
    const char* indent)
{
    for (const auto& array : arrays) {
        const auto offset = appended.Add(array.values);
        xml << indent << "<DataArray type=\"Float32\" Name=\"" << EscapeXML(array.name)
            << "\" NumberOfComponents=\"" << array.components
            << "\" format=\"appended\" offset=\"" << offset << "\"/>\n";
    }
}

inline void WriteVTI(
    const std::filesystem::path& path,
    const int extent[6],
    const double origin[3],
    const double spacing[3],
    const std::vector<FloatArray>& cell_data)
{
    AppendedWriter appended;
    std::ostringstream xml;
    xml.precision(17);
    xml << "<?xml version=\"1.0\"?>\n"
        << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
        << "  <ImageData WholeExtent=\""
        << extent[0] << ' ' << extent[1] << ' ' << extent[2] << ' '
        << extent[3] << ' ' << extent[4] << ' ' << extent[5]
        << "\" Origin=\"" << origin[0] << ' ' << origin[1] << ' ' << origin[2]
        << "\" Spacing=\"" << spacing[0] << ' ' << spacing[1] << ' ' << spacing[2] << "\">\n"
        << "    <Piece Extent=\""
        << extent[0] << ' ' << extent[1] << ' ' << extent[2] << ' '
        << extent[3] << ' ' << extent[4] << ' ' << extent[5] << "\">\n"
        << "      <PointData/>\n"
        << "      <CellData>\n";
    AppendFloatArrays(xml, appended, cell_data, "        ");
    xml << "      </CellData>\n"
        << "    </Piece>\n"
        << "  </ImageData>\n";
    appended.Write(path, xml.str());
}

inline void WriteVTU(
    const std::filesystem::path& path,
    const std::vector<float>& points,
    const std::vector<std::int64_t>& connectivity,
    const std::vector<std::int64_t>& offsets,
    const std::vector<std::uint8_t>& types,
    const std::vector<FloatArray>& point_data = {},
    const std::vector<FloatArray>& cell_data = {})
{
    AppendedWriter appended;
    const auto points_offset = appended.Add(points);
    const auto connectivity_offset = appended.Add(connectivity);
    const auto offsets_offset = appended.Add(offsets);
    const auto types_offset = appended.Add(types);

    std::ostringstream xml;
    xml << "<?xml version=\"1.0\"?>\n"
        << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
        << "  <UnstructuredGrid>\n"
        << "    <Piece NumberOfPoints=\"" << points.size() / 3
        << "\" NumberOfCells=\"" << offsets.size() << "\">\n"
        << "      <PointData>\n";
    AppendFloatArrays(xml, appended, point_data, "        ");
    xml << "      </PointData>\n"
        << "      <CellData>\n";
    AppendFloatArrays(xml, appended, cell_data, "        ");
    xml << "      </CellData>\n"
        << "      <Points>\n"
        << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"appended\" offset=\""
        << points_offset << "\"/>\n"
        << "      </Points>\n"
        << "      <Cells>\n"
        << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"appended\" offset=\""
        << connectivity_offset << "\"/>\n"
        << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"appended\" offset=\""
        << offsets_offset << "\"/>\n"
        << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"appended\" offset=\""
        << types_offset << "\"/>\n"
        << "      </Cells>\n"
        << "    </Piece>\n"
        << "  </UnstructuredGrid>\n";
    appended.Write(path, xml.str());
}

} // namespace VTKFileIO
