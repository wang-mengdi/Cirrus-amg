#pragma once
#include "ProjectionAssembly.h"

namespace simple {
namespace projectionAssembly {
// Retain only one column of -D*diag(area)*G. Match the CSC product's
// contribution order, including first contributions and cancellation zeros.
template<class Visit>
void pressureColumns(const Sparse& divergence,const Eigen::VectorXd& areas,
                     const Sparse& gradient,Visit visit) {
    if(divergence.cols()!=areas.size() || gradient.rows()!=areas.size() ||
       divergence.rows()!=gradient.cols())
        throw std::invalid_argument("Pressure diagnostic operator dimensions differ");
    for(int col=0;col<gradient.cols();++col) {
        Row column{col,{}};
        for(Sparse::InnerIterator g(gradient,col);g;++g) {
            const int face=int(g.row());
            for(Sparse::InnerIterator d(divergence,face);d;++d)
                column.add(int(d.row()),(-d.value()*areas[face])*g.value());
        }
        visit(column);
    }
}
struct PressureDiagnostics {
    Eigen::VectorXd diagonal,image;
};
inline PressureDiagnostics pressureDiagnostics(const Sparse& divergence,const Eigen::VectorXd& areas,
                                               const Sparse& gradient,const Eigen::VectorXd& probe) {
    const auto nc=gradient.cols();
    if(probe.size()!=0 && probe.size()!=nc)
        throw std::invalid_argument("Pressure diagnostic probe dimensions differ");
    PressureDiagnostics result;
    result.diagonal=Eigen::VectorXd::Zero(nc);
    if(probe.size())result.image=Eigen::VectorXd::Zero(nc);
    pressureColumns(divergence,areas,gradient,[&](const Row& column) {
        for(const auto& term:column.terms) {
            if(term.first==column.face)result.diagonal[column.face]=term.second;
            if(probe.size())result.image[term.first]+=term.second*probe[column.face];
        }
    });
    return result;
}
} // namespace projectionAssembly
} // namespace simple
