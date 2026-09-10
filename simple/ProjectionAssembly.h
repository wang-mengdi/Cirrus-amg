#pragma once
#include "SimpleMesh.h"
#include "QuadraticReconstruction.h"
#include <Eigen/Sparse>
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <utility>

namespace simple {
namespace projectionAssembly {
using Sparse=Eigen::SparseMatrix<double>;
using Rows=Eigen::SparseMatrix<double,Eigen::RowMajor>;

// Only interface faces need wide rows. Preserve the first value, subsequent
// addition order and explicit zero entries of Eigen::setFromTriplets.
struct Row {
    int face;
    std::vector<std::pair<int,double>> terms;
    void add(int column,double value) {
        auto it=std::lower_bound(terms.begin(),terms.end(),column,
            [](const auto& term,int key){return term.first<key;});
        if(it==terms.end()||it->first!=column)terms.insert(it,{column,value});
        else it->second+=value;
    }
};
inline Eigen::Index checkedCount(size_t count) {
    if(count>size_t(std::numeric_limits<Sparse::StorageIndex>::max()))
        throw std::length_error("Projection sparse operator exceeds index capacity");
    return Eigen::Index(count);
}
inline Rows consumeRows(std::vector<Row>& rows,int faces,int cells) {
    size_t count=0;for(const auto& row:rows)count+=row.terms.size();
    Rows result(faces,cells);result.reserve(checkedCount(count));
    size_t cursor=0;
    for(int face=0;face<faces;++face) {
        result.startVec(face);
        if(cursor<rows.size()&&rows[cursor].face==face) {
            for(const auto& term:rows[cursor].terms)result.insertBack(face,term.first)=term.second;
            ++cursor;
        }
    }
    if(cursor!=rows.size())throw std::logic_error("Projection assembly rows are not ordered unique faces");
    result.finalize();std::vector<Row>().swap(rows);return result;
}

// Build just one interface's correction and Taylor rows. Both assembly passes
// use this same arithmetic, retaining first terms and explicit zeros.
inline std::array<Row,3> interfaceRows(const Mesh& mesh,const Rows& faceGradient,
                                      const QuadraticReconstruction* quadratic,int face) {
    if(!quadratic)throw std::logic_error("Projection interface requires quadratic reconstruction");
    const auto& f=mesh.faces[face];
    std::array<Row,3> rows={Row{face,{}},Row{face,{}},Row{face,{}}};
    for(Rows::InnerIterator it(faceGradient,face);it;++it)rows[0].add(int(it.col()),it.value());
    rows[0].add(f.owner,1/f.distance);rows[0].add(f.neighbor,-1/f.distance);
    for(int side=0;side<2;++side) {
        auto& tr=rows[side+1];
        const int c=side?f.neighbor:f.owner;const Vec3 r=side?f.neighborOffset:f.ownerOffset;
        tr.add(c,1);
        const double w[9]={r[0],r[1],r[2],.5*r[0]*r[0],.5*r[1]*r[1],.5*r[2]*r[2],r[0]*r[1],r[0]*r[2],r[1]*r[2]};
        for(int d=0;d<9;++d)for(const auto& term:quadratic->derivativeRow(c,d))
            tr.add(term.first,w[d]*term.second);
    }
    return rows;
}

// Count exact support first, then write each face directly into its final
// storage. No global deferred/Taylor row lists or correction transpose are
// retained. The coefficients and their addition order are unchanged.
// Callers supply empty output matrices.
inline void assemble(const Mesh& mesh,const Rows& faceGradient,const QuadraticReconstruction* quadratic,
                     Sparse& divergence,Sparse& normalGradient,Sparse& correction,
                     Rows& taylorOwner,Rows& taylorNeighbor) {
    const int nc=int(mesh.cells.size()),nf=int(mesh.faces.size());
    Eigen::VectorXi counts=Eigen::VectorXi::Zero(nc);
    Eigen::VectorXi correctionCounts=Eigen::VectorXi::Zero(nc);
    size_t nonzeros=0,correctionNonzeros=0,taylorNonzeros[2]={0,0};
    for(int j=0;j<nf;++j) {
        const auto& f=mesh.faces[j];if(f.neighbor<0)continue;
        ++counts[f.owner];++nonzeros;
        if(f.neighbor!=f.owner) {++counts[f.neighbor];++nonzeros;}
        if(mesh.cells[f.owner].level==mesh.cells[f.neighbor].level)continue;
        const auto rows=interfaceRows(mesh,faceGradient,quadratic,j);
        for(const auto& term:rows[0].terms) {++correctionCounts[term.first];++correctionNonzeros;}
        for(int side=0;side<2;++side)taylorNonzeros[side]+=rows[side+1].terms.size();
    }
    divergence.resize(nc,nf);divergence.reserve(checkedCount(nonzeros));
    normalGradient.resize(nf,nc);normalGradient.reserve(counts);
    checkedCount(correctionNonzeros);
    correction.resize(nf,nc);correction.reserve(correctionCounts);
    taylorOwner.resize(nf,nc);taylorOwner.reserve(checkedCount(taylorNonzeros[0]));
    taylorNeighbor.resize(nf,nc);taylorNeighbor.reserve(checkedCount(taylorNonzeros[1]));
    Eigen::VectorXi().swap(counts);Eigen::VectorXi().swap(correctionCounts);
    for(int j=0;j<nf;++j) {
        divergence.startVec(j);
        taylorOwner.startVec(j);taylorNeighbor.startVec(j);
        const auto& f=mesh.faces[j];if(f.neighbor<0)continue;
        if(f.owner==f.neighbor) {
            divergence.insertBack(f.owner,j)=1.+(-1.);
            normalGradient.insertBackUncompressed(j,f.owner)=-1/f.distance+1/f.distance;
        } else {
            const bool ordered=f.owner<f.neighbor;
            divergence.insertBack(ordered?f.owner:f.neighbor,j)=ordered?1.:-1.;
            divergence.insertBack(ordered?f.neighbor:f.owner,j)=ordered?-1.:1.;
            // Faces arrive in ascending order, so each column's inner indices
            // are sorted even across periodic seams and coarse/fine subfaces.
            normalGradient.insertBackUncompressed(j,f.owner)=-1/f.distance;
            normalGradient.insertBackUncompressed(j,f.neighbor)=1/f.distance;
        }
        if(mesh.cells[f.owner].level==mesh.cells[f.neighbor].level)continue;
        const auto rows=interfaceRows(mesh,faceGradient,quadratic,j);
        for(const auto& term:rows[0].terms)correction.insertBackUncompressed(j,term.first)=term.second;
        for(const auto& term:rows[1].terms)taylorOwner.insertBack(j,term.first)=term.second;
        for(const auto& term:rows[2].terms)taylorNeighbor.insertBack(j,term.first)=term.second;
    }
    divergence.finalize();normalGradient.makeCompressed();
    correction.makeCompressed();taylorOwner.finalize();taylorNeighbor.finalize();
}
} // namespace projectionAssembly
} // namespace simple
