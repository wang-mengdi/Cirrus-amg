#include "SimpleMesh.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace simple {
namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("SIMPLE mesh: " + message);
}

// Native cells are already ordered by (level,x,y,z). Index whole z rows instead
// of allocating a hash node and buckets for every background cell. Unordered
// callers keep their original ids through a compact sorted permutation.
class CellLookup {
    struct Row {
        std::array<int,3> key;
        int begin,end,firstZ;
        bool contiguous;
    };
    const std::vector<Cell>& cells_;
    std::vector<int> order_;
    std::vector<Row> rows_;
    int id(int sorted) const {return order_.empty()?sorted:order_[sorted];}
    const Cell& cell(int sorted) const {return cells_[id(sorted)];}
public:
    explicit CellLookup(const std::vector<Cell>& cells):cells_(cells) {
        require(cells.size()<=size_t(std::numeric_limits<int>::max()), "leaf index exceeds integer capacity");
        const auto less=[](const Cell& a,const Cell& b) {
            return a.level!=b.level?a.level<b.level:a.key<b.key;
        };
        if(!std::is_sorted(cells.begin(),cells.end(),less)) {
            order_.resize(cells.size());std::iota(order_.begin(),order_.end(),0);
            std::sort(order_.begin(),order_.end(),[&](int a,int b){return less(cells[a],cells[b]);});
        }
        for(int i=0;i<int(cells.size());++i) {
            const auto& c=cell(i);
            if(i)require(less(cell(i-1),c), "duplicate native leaf cell");
            const std::array<int,3> rowKey{c.level,c.key[0],c.key[1]};
            if(rows_.empty() || rows_.back().key!=rowKey)
                rows_.push_back(Row{rowKey,i,i+1,c.key[2],true});
            else {
                auto& row=rows_.back();row.end=i+1;
                row.contiguous=row.contiguous && int64_t(c.key[2])-row.firstZ==i-row.begin;
            }
        }
    }
    int find(int level,const std::array<int,3>& key) const {
        const std::array<int,3> rowKey{level,key[0],key[1]};
        const auto it=std::lower_bound(rows_.begin(),rows_.end(),rowKey,
            [](const Row& row,const auto& value){return row.key<value;});
        if(it==rows_.end() || it->key!=rowKey)return -1;
        if(it->contiguous) {
            const int64_t offset=int64_t(key[2])-it->firstZ;
            return offset>=0 && offset<it->end-it->begin?id(it->begin+int(offset)):-1;
        }
        int first=it->begin,last=it->end;
        while(first<last) {
            const int mid=first+(last-first)/2;
            if(cell(mid).key[2]<key[2])first=mid+1;else last=mid;
        }
        return first<it->end && cell(first).key[2]==key[2]?id(first):-1;
    }
};

// The retained-array and streaming paths share every geometry check. Only
// storage of generated faces differs; the accumulation order is unchanged.
class MeshValidation {
    const Mesh& mesh;
    int n;
    std::vector<std::array<double, 6>> coverage;
    std::vector<double> divergenceVolume;
    int interfaces = 0;
    std::size_t faceCount = 0;
public:
    explicit MeshValidation(const Mesh& value):mesh(value),n(static_cast<int>(value.cells.size())) {
        require(n > 0, "no cells");
        coverage.resize(n);
        divergenceVolume.assign(n, 0.0);
    }
    void add(const Face& f) {
        require(f.owner >= 0 && f.owner < n, "invalid face owner");
        require(f.axis >= 0 && f.axis < 3 && std::abs(f.sign) == 1.0,
                "invalid face normal");
        require(f.area > 0.0 && f.distance > 0.0 && std::isfinite(f.area)
                && std::isfinite(f.distance), "invalid face geometry");
        const int side = f.sign > 0.0 ? 1 : 0;
        coverage[f.owner][2 * f.axis + side] += f.area;
        divergenceVolume[f.owner] += f.sign * f.ownerOffset[f.axis] * f.area / 3.0;
        require(std::abs(f.distance - f.sign * f.delta[f.axis]) < 1e-12,
                "distance differs from normal displacement");
        if (f.neighbor >= 0) {
            require(f.neighbor < n && f.neighbor != f.owner && f.boundary == 0,
                    "invalid internal neighbor");
            require(std::abs(mesh.cells[f.owner].level - mesh.cells[f.neighbor].level) <= 1,
                    "non-2:1 interface");
            if (mesh.cells[f.owner].level != mesh.cells[f.neighbor].level) ++interfaces;
            coverage[f.neighbor][2 * f.axis + 1 - side] += f.area;
            divergenceVolume[f.neighbor] -= f.sign * f.neighborOffset[f.axis] * f.area / 3.0;
            require((f.ownerOffset - f.neighborOffset - f.delta).norm() < 1e-12,
                    "periodic-image offsets are inconsistent");
        } else {
            require(f.boundary >= 1 && f.boundary <= 3, "invalid boundary type");
        }
        ++faceCount;
    }
    void finish(int expectedInterfaces) const {
        double volume = 0.0;
        double maxCoverageError = 0.0;
        double maxClosureError = 0.0;
        for (int i = 0; i < n; ++i) {
            const auto& c = mesh.cells[i];
            require(c.h > 0.0 && c.volume > 0.0, "invalid leaf size");
            require(std::abs(c.volume - c.h * c.h * c.h) < 1e-12 * c.volume,
                    "leaf volume differs from native cell size");
            for (int side = 0; side < 6; ++side)
                maxCoverageError = std::max(maxCoverageError,
                    std::abs(coverage[i][side] / (c.h * c.h) - 1.0));
            maxClosureError = std::max(maxClosureError,
                std::abs(divergenceVolume[i] / c.volume - 1.0));
            volume += c.volume;
        }
        require(maxCoverageError < 1e-12, "six-direction area coverage does not close");
        require(maxClosureError < 1e-12, "Gauss cell-volume closure failed");
        require(std::abs(volume / mesh.extent.prod() - 1.0) < 1e-12,
                "leaf volumes do not fill channel");
        require(interfaces == expectedInterfaces, "coarse/fine face count inconsistent");
        std::cout << "Mesh sanity: cells=" << n << " faces=" << faceCount
                  << " coarseFineFaces=" << interfaces << " volume=" << volume
                  << " maxAreaCoverageError=" << maxCoverageError
                  << " maxVolumeClosureError=" << maxClosureError << '\n';
    }
};

Face makeInternal(const Mesh& mesh, int owner, int neighbor, int axis, bool seam) {
    const auto& a = mesh.cells.at(owner);
    const auto& b = mesh.cells.at(neighbor);
    Vec3 image = b.center;
    if (seam) image[axis] += mesh.extent[axis];
    Face f;
    f.owner = owner;
    f.neighbor = neighbor;
    f.axis = axis;
    f.area = std::min(a.h, b.h) * std::min(a.h, b.h);
    f.center = a.h <= b.h ? a.center : image;
    f.center[axis] = a.center[axis] + 0.5 * a.h;
    f.delta = image - a.center;
    f.distance = f.delta[axis];
    f.ownerOffset = f.center - a.center;
    f.neighborOffset = f.center - image;
    require(f.distance > 0.0, "non-positive owner-neighbor distance");
    return f;
}

Face makeBoundary(const Mesh& mesh, int owner, int axis, int sign, int boundary) {
    const auto& c = mesh.cells.at(owner);
    Face f;
    f.owner = owner;
    f.axis = axis;
    f.sign = static_cast<double>(sign);
    f.area = c.h * c.h;
    f.distance = 0.5 * c.h;
    f.center = c.center;
    f.center[axis] += sign * f.distance;
    f.ownerOffset = f.center - c.center;
    f.delta = f.ownerOffset;
    f.boundary = boundary;
    return f;
}

} // namespace

void buildFacesAndValidate(Mesh& mesh, bool periodicX, bool periodicZ) {
    mesh.periodicZ = periodicZ;
    mesh.faces.clear();
    mesh.coarseFineFaces = 0;
    mesh.faces.reserve(mesh.cells.size() * 4);
    mesh.coarseFineFaces = visitFacesAndValidate(mesh, periodicX, periodicZ,
        [&](const Face& face) { mesh.faces.push_back(face); });
}

int visitFacesAndValidate(const Mesh& mesh, bool periodicX, bool periodicZ,
                          const std::function<void(const Face&)>& visit) {
    require(!mesh.cells.empty(), "empty leaf set");
    require(bool(visit), "missing face visitor");
    MeshValidation validation(mesh);
    int interfaces = 0;
    const auto emit = [&](const Face& face) {
        if (face.neighbor >= 0 && mesh.cells[face.owner].level != mesh.cells[face.neighbor].level)
            ++interfaces;
        validation.add(face);
        visit(face);
    };
    const CellLookup lookup(mesh.cells);
    auto findCell = [&](int level, const std::array<int, 3>& key) {
        return lookup.find(level,key);
    };

    for (int owner = 0; owner < static_cast<int>(mesh.cells.size()); ++owner) {
        const auto& c = mesh.cells[owner];
        for (int axis = 0; axis < 3; ++axis) {
            const bool periodic = (axis == 0 && periodicX) || (axis == 2 && periodicZ);
            const int count = static_cast<int>(std::llround(mesh.extent[axis] / c.h));
            require(c.key[axis] >= 0 && c.key[axis] < count, "leaf outside channel");
            if (c.key[axis] == 0 && !periodic)
                emit(makeBoundary(mesh, owner, axis, -1, axis == 0 ? 2 : 1));

            auto q = c.key;
            ++q[axis];
            const bool seam = q[axis] == count;
            if (seam && !periodic) {
                emit(makeBoundary(mesh, owner, axis, 1, axis == 0 ? 3 : 1));
                continue;
            }
            if (seam) q[axis] = 0;
            int neighbor = findCell(c.level, q);
            if (neighbor >= 0) {
                emit(makeInternal(mesh, owner, neighbor, axis, seam));
                continue;
            }
            if (c.level > 0) {
                auto parent = q;
                for (int& x : parent) x /= 2;
                neighbor = findCell(c.level - 1, parent);
                if (neighbor >= 0) {
                    emit(makeInternal(mesh, owner, neighbor, axis, seam));
                    continue;
                }
            }
            // Only the four children touching the neighbor's negative face.
            // Each is a separate shared flux unknown and has area h_coarse^2/4.
            const int t0 = (axis + 1) % 3;
            const int t1 = (axis + 2) % 3;
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 2; ++k) {
                    auto child = q;
                    for (int& x : child) x *= 2;
                    child[t0] += j;
                    child[t1] += k;
                    neighbor = findCell(c.level + 1, child);
                    require(neighbor >= 0, "missing face neighbor or refinement jump beyond 2:1");
                    emit(makeInternal(mesh, owner, neighbor, axis, seam));
                }
            }
        }
    }
    validation.finish(interfaces);
    return interfaces;
}

void validateMesh(const Mesh& mesh) {
    MeshValidation validation(mesh);
    for (const auto& face : mesh.faces) validation.add(face);
    validation.finish(mesh.coarseFineFaces);
}

void dumpMeshCsv(const Mesh& mesh, const std::string& prefix) {
    std::ofstream cells(prefix + "_cells.csv");
    std::ofstream faces(prefix + "_faces.csv");
    require(cells.good() && faces.good(), "cannot open mesh CSV output");
    cells << std::setprecision(17) << "id,level,i,j,k,x,y,z,h,volume\n";
    for (int i = 0; i < static_cast<int>(mesh.cells.size()); ++i) {
        const auto& c = mesh.cells[i];
        cells << i << ',' << c.level;
        for (int q : c.key) cells << ',' << q;
        for (int a = 0; a < 3; ++a) cells << ',' << c.center[a];
        cells << ',' << c.h << ',' << c.volume << '\n';
    }
    faces << std::setprecision(17)
          << "id,owner,neighbor,axis,sign,boundary,area,distance,x,y,z,dx,dy,dz,owner_dx,owner_dy,owner_dz,neighbor_dx,neighbor_dy,neighbor_dz\n";
    for (int i = 0; i < static_cast<int>(mesh.faces.size()); ++i) {
        const auto& f = mesh.faces[i];
        faces << i << ',' << f.owner << ',' << f.neighbor << ',' << f.axis << ','
              << f.sign << ',' << f.boundary << ',' << f.area << ',' << f.distance;
        for (const Vec3* v : {&f.center, &f.delta, &f.ownerOffset, &f.neighborOffset})
            for (int a = 0; a < 3; ++a) faces << ',' << (*v)[a];
        faces << '\n';
    }
    require(cells.good() && faces.good(), "mesh CSV write failed");
}

} // namespace simple
