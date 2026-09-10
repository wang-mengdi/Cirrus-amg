// Geometry-only exporter using the original, separately compiled Aphros Embed.
// Each slab has global coordinates and analytic node values including halos.
// No flow state is constructed, and no Aphros geometry formula is copied here.
#include "solver/embed.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using Mesh = MeshCartesian<double, 3>;
using Vect = Mesh::Vect;
using MIdx = Mesh::MIdx;
using Geometry = Embed<Mesh>;

template <std::size_t Columns>
class Table {
    std::ofstream stream;
    std::uint64_t rows = 0;
public:
    explicit Table(const std::filesystem::path& path) : stream(path, std::ios::binary) {
        if (!stream) throw std::runtime_error("Cannot create geometry table");
        const std::uint32_t columns = Columns, endian = 0x01020304;
        stream.write("CIRRCUT1", 8);
        stream.write(reinterpret_cast<const char*>(&rows), 8);
        stream.write(reinterpret_cast<const char*>(&columns), 4);
        stream.write(reinterpret_cast<const char*>(&endian), 4);
    }
    void append(const std::array<double, Columns>& row) {
        for (double x : row) if (!std::isfinite(x))
            throw std::runtime_error("Nonfinite geometry value");
        stream.write(reinterpret_cast<const char*>(row.data()), sizeof(row));
        ++rows;
    }
    std::uint64_t finish() {
        stream.seekp(8);
        stream.write(reinterpret_cast<const char*>(&rows), 8);
        stream.flush();
        if (!stream.good()) throw std::runtime_error("Geometry table write failed");
        stream.close();
        return rows;
    }
};

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("Usage: aphros_geometry_slabs ny slab_depth new_output_directory");
        const int ny = std::stoi(argv[1]), depth = std::stoi(argv[2]);
        if (ny < 8 || ny > 512 || (ny & (ny-1)) || depth < 1 || depth > ny)
            throw std::runtime_error("Expected power-of-two ny in [8,512] and positive slab depth <= ny");
        const std::filesystem::path out(argv[3]);
        if (std::filesystem::exists(out) || !std::filesystem::create_directories(out))
            throw std::runtime_error("Output must be a fresh directory");
        Table<12> cells(out/"geometry.cells.bin");
        Table<8> faces(out/"geometry.faces.bin");
        Table<11> walls(out/"geometry.walls.bin");
        Table<8> polygons(out/"geometry.polygons.bin");
        const double h = .125/ny;
        const MIdx global(2*ny, ny, ny);
        std::ofstream progress(out/"slabs.csv");
        progress << "begin_z,end_z,fluid_cells,fluid_faces,wall_faces,init_stages,unused_derived_halo_requests\n";
        for (int begin = 0; begin < ny; begin += depth) {
            const int end = std::min(begin+depth, ny);
            Mesh m(MIdx(0,0,begin), MIdx(2*ny,ny,end-begin),
                   Rect<Vect>(Vect(0,0,begin*h), Vect(.25,.125,end*h)),
                   2, true, true, global, 0);
            m.flags.is_periodic[0] = true;
            m.flags.is_periodic[1] = false;
            m.flags.is_periodic[2] = false;
            FieldNode<double> levelset(m, 0.);
            // These expressions and double intermediates match the established
            // independent tube driver. Dyadic coordinates are exact on each slab.
            for (auto node : m.AllNodes()) {
                const auto x = m.GetNode(node);
                const double angle = 2*M_PI*x[0]/.25;
                const double y = x[1]-.0625-.015*std::sin(angle);
                const double z = x[2]-.0625-.015*std::cos(angle);
                levelset[node] = .035-std::sqrt(y*y+z*z);
            }
            levelset.SetHalo(2);
            Geometry eb(m,0);
            int stages = 0;
            std::size_t derivedHaloRequests = 0;
            do {
                eb.Init(levelset);
                // Original Init computes cut geometry for all halo cells from
                // supplied node values. Its three queued communications concern
                // only neighbor-volume sums, signed distance and cut displacement.
                // Those derived fields are unused by this geometry-only exporter;
                // this object must never be used for a flow solve.
                derivedHaloRequests += m.GetComm().size();
                m.ClearComm();
                if (++stages > 4) throw std::runtime_error("Unexpected Embed initialization stages");
            } while (m.Pending());
            if (derivedHaloRequests != 3)
                throw std::runtime_error("Unexpected original Embed communication layout");
            std::uint64_t nc=0, nf=0, nw=0;
            for (auto c : eb.Cells()) {
                const auto q=m.GetIndexCells().GetMIdx(c);
                const auto x=m.GetCenter(c), g=eb.GetCellCenter(c);
                cells.append({double(q[0]),double(q[1]),double(q[2]),x[0],x[1],x[2],
                              h,eb.GetVolume(c),double(eb.IsCut(c)),g[0],g[1],g[2]});
                ++nc;
            }
            for (auto f : eb.Faces()) {
                const auto qd=m.GetIndexFaces().GetMIdxDir(f);
                const auto q=qd.first;
                const auto axis=qd.second.raw();
                // A z face shared by two slabs is emitted by the upper slab.
                // Keep the true global upper boundary in the final slab.
                if (q[2] == end && end < ny) continue;
                const auto x=eb.GetFaceCenter(f);
                faces.append({double(axis),double(q[0]),double(q[1]),double(q[2]),
                              x[0],x[1],x[2],eb.GetArea(f)});
                int vertex=0;
                for (const auto p : eb.GetFacePoly(f))
                    polygons.append({double(axis),double(q[0]),double(q[1]),double(q[2]),
                                     double(vertex++),p[0],p[1],p[2]});
                ++nf;
            }
            for (auto c : eb.CFaces()) {
                const auto q=m.GetIndexCells().GetMIdx(c);
                const auto x=eb.GetFaceCenter(c), n=eb.GetNormal(c);
                walls.append({double(q[0]),double(q[1]),double(q[2]),x[0],x[1],x[2],
                              n[0],n[1],n[2],eb.GetArea(c),eb.GetAlpha(c)});
                int vertex=0;
                for (const auto p : eb.GetCutPoly(c))
                    polygons.append({3.,double(q[0]),double(q[1]),double(q[2]),double(vertex++),p[0],p[1],p[2]});
                ++nw;
            }
            progress << begin << ',' << end << ',' << nc << ',' << nf << ',' << nw
                     << ',' << stages << ',' << derivedHaloRequests << '\n';
            progress.flush();
            if (!progress.good()) throw std::runtime_error("Slab progress write failed");
            std::cout << "Completed z cells [" << begin << ',' << end << "): " << nc << " fluid cells\n" << std::flush;
        }
        const auto nc=cells.finish(), nf=faces.finish(), nw=walls.finish(), np=polygons.finish();
        std::ofstream report(out/"geometry_summary.json");
        report << "{\"ny\":" << ny << ",\"slab_depth\":" << depth << ",\"cells\":" << nc
               << ",\"faces\":" << nf << ",\"walls\":" << nw << ",\"polygon_vertices\":" << np
               << ",\"flow_computed\":false}\n";
        report.flush();
        if (!report.good()) throw std::runtime_error("Geometry summary write failed");
    } catch (const std::exception& e) {
        std::cerr << "Slab geometry error: " << e.what() << '\n';
        return 1;
    }
}
